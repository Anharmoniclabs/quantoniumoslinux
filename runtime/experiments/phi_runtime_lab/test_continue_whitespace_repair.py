"""Migration safeguards; run with python -m unittest test_continue_whitespace_repair."""
import json
from pathlib import Path
import tempfile
import unittest

from sentencepiece import sentencepiece_model_pb2 as pb
from continue_whitespace_repair import atomic_json, run_lock, validate_migration
from repair_whitespace_corpus import sha


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old, self.new = self.root / "old.model", self.root / "fixed.model"
        proto = pb.ModelProto()
        for text in ("<unk>", "<|code|>", "</s>", "<|general|>"):
            piece = proto.pieces.add()
            piece.piece, piece.score = text, 0.0
        self.old.write_bytes(proto.SerializeToString())
        proto.normalizer_spec.remove_extra_whitespaces = False
        self.new.write_bytes(proto.SerializeToString())
        self.source = {
            "format": "PhiStateMoE-v1", "data_fingerprint": "expanded", "config": {"vocab": 4},
            "tokenizer": "bpe", "optimizer_state": {}, "train_args": {},
            "numpy_rng_state": {}, "python_rng_state": (), "torch_rng_state": None,
        }
        self.source_manifest = {"data_fingerprint": "expanded", "extension": {"base_data_fingerprint": "base"}}
        self.repaired = {"vocab": 4, "tokenizer": "bpe", "whitespace_repair": {
            "original_data_fingerprint": "base", "original_tokenizer_sha256": sha(self.old),
            "tokenizer_sha256": sha(self.new), "all_documents_roundtrip_exact": True,
            "all_original_streams_accounted_for": True,
        }}

    def validate(self):
        validate_migration(self.source, self.source_manifest, self.repaired, self.old, self.new)

    def test_compatible_normalizer_change(self):
        self.validate()

    def test_unrelated_parent_rejected(self):
        self.source_manifest["extension"]["base_data_fingerprint"] = "other"
        with self.assertRaisesRegex(RuntimeError, "unrelated"):
            self.validate()

    def test_missing_optimizer_rejected(self):
        self.source.pop("optimizer_state")
        with self.assertRaisesRegex(RuntimeError, "optimizer"):
            self.validate()

    def test_missing_rng_rejected(self):
        self.source.pop("numpy_rng_state")
        with self.assertRaisesRegex(RuntimeError, "RNG"):
            self.validate()

    def test_changed_token_id_rejected(self):
        proto = pb.ModelProto()
        proto.ParseFromString(self.new.read_bytes())
        proto.pieces[1].piece = "DIFFERENT"
        self.new.write_bytes(proto.SerializeToString())
        self.repaired["whitespace_repair"]["tokenizer_sha256"] = sha(self.new)
        with self.assertRaisesRegex(RuntimeError, "token IDs"):
            self.validate()

    def test_duplicate_worker_lock(self):
        with run_lock(self.root):
            with self.assertRaisesRegex(RuntimeError, "active worker"):
                with run_lock(self.root):
                    pass
        with run_lock(self.root):
            pass

    def test_failed_json_write_preserves_progress(self):
        path = self.root / "progress.json"
        atomic_json(path, {"saved_step": 60000})
        with self.assertRaises(ValueError):
            atomic_json(path, {"saved_step": float("nan")})
        self.assertEqual(json.loads(path.read_text()), {"saved_step": 60000})


if __name__ == "__main__":
    unittest.main(verbosity=2)
