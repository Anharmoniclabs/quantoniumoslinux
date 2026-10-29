"""CPU-only regression tests; no user data, downloads or model checkpoints."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from repair_whitespace_corpus import recover_domain, pack, run, sha, SPLITS, DOMAINS


class MembershipTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"text": "alpha", "tokens": pack([10, 11, 12], "uint16")},
            {"text": "bravo", "tokens": pack([20, 21, 22], "uint16")},
            {"text": "charlie", "tokens": pack([30, 31, 32], "uint16")},
        ]
        self.order = {"train": [2], "val": [0], "test": [1]}
        self.stats = {}
        for split, ids in self.order.items():
            self.stats[f"{split}_documents"] = len(ids)
            self.stats[f"{split}_bytes"] = sum(len(self.records[i]["text"]) for i in ids)

    def streams(self, prefix=b"", suffix=b""):
        return {split: b"".join(prefix + self.records[i]["tokens"] + suffix for i in ids)
                for split, ids in self.order.items()}

    def test_boundary_variants_preserve_original_order(self):
        for before, after in (([], []), ([1], []), ([], [2]), ([1], [2])):
            with self.subTest(before=before, after=after):
                prefix, suffix = pack(before, "uint16"), pack(after, "uint16")
                result = recover_domain(self.records, self.streams(prefix, suffix), self.stats,
                                        [("mode", prefix, suffix)], 2)
                self.assertEqual(result[3], self.order)

    def test_changed_token_rejected(self):
        streams = self.streams()
        streams["val"] = pack([10, 11, 99], "uint16")
        with self.assertRaisesRegex(RuntimeError, "Cannot prove"):
            recover_domain(self.records, streams, self.stats, [("none", b"", b"")], 2)

    def test_unaccounted_trailing_token_rejected(self):
        streams = self.streams()
        streams["test"] += pack([123], "uint16")
        with self.assertRaises(RuntimeError):
            recover_domain(self.records, streams, self.stats, [("none", b"", b"")], 2)

    def test_split_statistics_rejected(self):
        self.stats["val_bytes"] += 1
        with self.assertRaises(RuntimeError):
            recover_domain(self.records, self.streams(), self.stats, [("none", b"", b"")], 2)

    def test_tokenizer_collision_rejected(self):
        self.records.append({"text": "a  lpha", "tokens": self.records[0]["tokens"]})
        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            recover_domain(self.records, self.streams(), self.stats, [("none", b"", b"")], 2)

    def test_reused_document_rejected(self):
        streams = self.streams()
        streams["test"] = streams["val"]
        with self.assertRaises(RuntimeError):
            recover_domain(self.records, streams, self.stats, [("none", b"", b"")], 2)


class EndToEndTests(unittest.TestCase):
    def test_rebuild_idempotency_and_corruption(self):
        import sentencepiece as spm
        from sentencepiece import sentencepiece_model_pb2 as pb

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, repair, raw = root / "base", root / "repair", root / "raw"
            for directory in (base, repair, raw):
                directory.mkdir()
            texts = {
                "code": ["def alpha(x):\r\n    return x + 13\r\n",
                         "def bravo(y):\n\treturn y * 27\n",
                         "def charlie(z):\n    return z - 39\n"],
                "general": ["A red apple grows on a tree.\n\nA second line.",
                            "The ocean contains salt water.\nFish swim there.",
                            "The moon reflects sunlight.\nEarth has one moon."],
            }
            source = root / "training.txt"
            source.write_text("\n".join(t for group in texts.values() for t in group))
            spm.SentencePieceTrainer.train(
                input=str(source), model_prefix=str(base / "tokenizer"),
                model_type="bpe", vocab_size=340, byte_fallback=True,
                hard_vocab_limit=False, minloglevel=2,
            )
            old = spm.SentencePieceProcessor(model_file=str(base / "tokenizer.model"))
            proto = pb.ModelProto()
            proto.ParseFromString(old.serialized_model_proto())
            proto.normalizer_spec.name = "identity"
            proto.normalizer_spec.ClearField("precompiled_charsmap")
            proto.normalizer_spec.ClearField("normalization_rule_tsv")
            proto.normalizer_spec.remove_extra_whitespaces = False
            (repair / "tokenizer.model").write_bytes(proto.SerializeToString())
            manifest = {"format": "PhiMoECorpus-v1", "tokenizer": "bpe",
                        "dtype": "uint16", "vocab": old.vocab_size(),
                        "data_fingerprint": "fixture-original", "bins": {}, "documents": {}}
            order = {"train": [2], "val": [0], "test": [1]}
            for domain in DOMAINS:
                with (raw / f"{domain}.jsonl").open("w") as stream:
                    for text in texts[domain]:
                        stream.write(json.dumps({"text": text, "sha256": hashlib.sha256(text.encode()).hexdigest()}) + "\n")
                stats = {}
                for split, indices in order.items():
                    normalized = [texts[domain][i].replace("\r\n", "\n").replace("\r", "\n") for i in indices]
                    data = b"".join(pack(old.encode(t) + [old.eos_id()], "uint16") for t in normalized)
                    key = f"{split}_{domain}"
                    (base / f"{key}.bin").write_bytes(data)
                    manifest["bins"][key] = {"file": f"{key}.bin", "tokens": len(data)//2,
                                             "sha256": hashlib.sha256(data).hexdigest()}
                    stats[f"{split}_documents"] = len(indices)
                    stats[f"{split}_bytes"] = sum(len(t.encode()) for t in normalized)
                manifest["documents"][domain] = stats
            (base / "manifest.json").write_text(json.dumps(manifest))
            recovery = {"raw_directory": str(raw), "original_data_fingerprint": "fixture-original",
                        "file_sha256": {d: sha(raw / f"{d}.jsonl") for d in DOMAINS}}
            (repair / "raw_recovery_complete.json").write_text(json.dumps(recovery))
            originals = {p.name: sha(p) for p in base.iterdir()}
            output = repair / "corpus_repaired_v1"
            with contextlib.redirect_stdout(io.StringIO()):
                run(base, repair, output)
                run(base, repair, output)
            self.assertEqual(originals, {p.name: sha(p) for p in base.iterdir()})
            fixed = spm.SentencePieceProcessor(model_file=str(output / "tokenizer.model"))
            for domain in DOMAINS:
                for split, indices in order.items():
                    ids = np.fromfile(output / f"{split}_{domain}.bin", dtype="uint16").tolist()
                    expected = texts[domain][indices[0]].replace("\r\n", "\n")
                    self.assertEqual(fixed.decode(ids[:-1]), expected)
            before = sha(output / "train_code.bin")
            with (output / "train_code.bin").open("ab") as stream:
                stream.write(b"\x00\x00")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "Bad output length"):
                run(base, repair, output)
            self.assertNotEqual(before, sha(output / "train_code.bin"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
