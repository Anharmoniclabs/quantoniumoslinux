"""Check stream integrity, window matching and diagnostic calculations on CPU."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import phistate_base_windows as w


def fixture():
    manifest = {
        "data_fingerprint": "fixture",
        "vocab": 4,
        "whitespace_repair": {"tokenizer_sha256": "tokenizer-fixture"},
    }
    bins = {
        f"{s}_{d}": (np.arange(4096, dtype=np.uint16) + i) % 4
        for i, (s, d) in enumerate((s, d) for s in ("train", "val") for d in w.DOMAINS)
    }
    return w.Corpus(manifest, None, bins, np.array([True, False, False, False]))


class LocalNext(torch.nn.Module):
    def forward(self, ids):
        return 3.0 * torch.nn.functional.one_hot((ids + 1) % 4, num_classes=4).float()


class WindowTests(unittest.TestCase):
    def test_manifest_ignores_sealed_streams_and_requires_every_train_val_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {
                "format": "PhiMoECorpus-v1",
                "tokenizer": "bpe",
                "vocab": 8192,
                "dtype": "uint16",
                "bins": {},
                "whitespace_repair": {},
            }
            for key, values in fixture().bins.items():
                file = root / f"{key}.bin"
                values.astype("<u2").tofile(file)
                manifest["bins"][key] = {
                    "tokens": len(values),
                    "file": file.name,
                    "sha256": w.sha_file(file),
                }
            # The sealed test paths do not even exist. Nothing may open them.
            for d in w.DOMAINS:
                manifest["bins"][f"test_{d}"] = {
                    "file": "unavailable.bin",
                    "tokens": 1,
                    "sha256": "unused",
                }
            tokfile = root / "tokenizer.model"
            tokfile.write_bytes(b"test-only")
            toksha = w.sha_file(tokfile)
            manifest["whitespace_repair"]["tokenizer_sha256"] = toksha
            fingerprint = w.digest(manifest)
            manifest["data_fingerprint"] = fingerprint
            (root / "manifest.json").write_text(json.dumps(manifest))
            with (
                patch.object(w, "FINGERPRINT", fingerprint),
                patch.object(w, "TOKENIZER_SHA", toksha),
                patch.object(w.spm, "SentencePieceProcessor") as proc,
            ):
                proc.return_value.vocab_size.return_value = 8192
                proc.return_value.id_to_piece.return_value = "word"
                self.assertEqual(set(w.open_corpus(root).bins), set(fixture().bins))
                file = root / "train_code.bin"
                file.write_bytes(b"\x01\x01" + file.read_bytes()[2:])
                with self.assertRaisesRegex(ValueError, "hash mismatch for train_code"):
                    w.open_corpus(root)

    def test_identical_shifted_windows_and_disjoint_schedules(self):
        corpus = fixture()
        plan = w.schedule(corpus, 9, 5, 16, 12, 0.55)
        self.assertEqual(plan, w.schedule(corpus, 9, 5, 16, 12, 0.55))
        self.assertNotEqual(plan["fingerprint"], w.schedule(corpus, 9, 5, 16, 13, 0.55)["fingerprint"])
        self.assertEqual(25, sum(d["domain"] == "code" for b in plan["steps"] for d in b))
        used = {}
        for batch in plan["steps"]:
            x, y = w.materialize(corpus, batch, "cpu")
            self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))
            self.assertEqual(tuple(x.shape), (5, 16))
            for row, desc in zip(y.tolist(), batch):
                self.assertEqual(desc["split"], "train")
                start = desc["offset"]
                self.assertEqual(
                    row, corpus.bins["train_" + desc["domain"]][start + 1 : start + 17].tolist()
                )
                previous = used.setdefault(desc["domain"], set())
                new = set(range(start, start + 17))
                self.assertFalse(previous & new)
                previous.update(new)
        val = w.validation_plan(corpus, 16, 16, 12)
        self.assertTrue(all(row["split"] == "val" for rows in val.values() for row in rows))
        with self.assertRaises(ValueError):
            w.schedule(corpus, 10000, 128, 128, 12)
        for desc in (
            {"split": "test", "domain": "code", "offset": 0, "context": 16},
            {"split": "val", "domain": "code", "offset": 4080, "context": 16},
        ):
            with self.assertRaises(ValueError):
                w.materialize(corpus, [desc], "cpu")

    def test_context_gain_and_token_partition(self):
        corpus = fixture()
        model = LocalNext()
        samples = w.validation_plan(corpus, 16, 12, 126)
        value = w.evaluate(model, corpus, samples, "cpu", batch_size=5)
        expected = math.log2(1 + 3 * math.exp(-3))
        for domain in w.DOMAINS:
            metrics = value[domain]
            self.assertEqual(metrics["context_probe_tokens"], 12 * 8)
            self.assertEqual(metrics["whitespace_tokens"] + metrics["other_tokens"], 192)
            self.assertAlmostEqual(expected, metrics["all_bpt"], places=6)
            self.assertAlmostEqual(expected, metrics["whitespace_bpt"], places=6)
            self.assertAlmostEqual(expected, metrics["other_bpt"], places=6)
            self.assertEqual(metrics["context_gain_bpt"], 0.0)
        self.assertTrue(model.training)
        model.eval()
        w.evaluate(model, corpus, samples, "cpu", batch_size=3)
        self.assertFalse(model.training)

    def test_wrong_history_fixes_absolute_position_and_last_input(self):
        corpus = fixture()
        for arr in corpus.bins.values():
            arr.fill(0)
        descriptors = {
            domain: [
                {"split": "val", "domain": domain, "context": 16, "offset": start} for start in (0, 17)
            ]
            for domain in w.DOMAINS
        }

        class Position(torch.nn.Module):
            def forward(self, ids):
                # Position zero predicts the wrong token independent of input identity.
                labels = (torch.arange(ids.size(1), device=ids.device) == 0).long().expand_as(ids)
                return 4.0 * torch.nn.functional.one_hot(labels, 4).float()

        result = w.evaluate(Position(), corpus, descriptors, "cpu", batch_size=1)
        self.assertGreater(result["code"]["context_gain_bpt"], 0)
        self.assertEqual(result["code"]["wrong_history_gain_bpt"], 0.0)
        # A history-sensitive model detects the other window while absolute
        # positions, each target, and the last eight true inputs stay identical.
        for domain in w.DOMAINS:
            corpus.bins["val_" + domain][17:34] = 2

        class First(torch.nn.Module):
            def forward(self, ids):
                return 4.0 * torch.nn.functional.one_hot(ids[:, :1].expand_as(ids), 4).float()

        result = w.evaluate(First(), corpus, descriptors, "cpu", batch_size=1)
        self.assertGreater(result["code"]["wrong_history_gain_bpt"], 5)
        self.assertEqual(result["code"]["wrong_history_tokens"], 16)
        self.assertEqual(result["code"]["wrong_history_retained_suffix"], 8)
        one = {d: [rows[0]] for d, rows in descriptors.items()}
        one_result = w.evaluate(First(), corpus, one, "cpu", batch_size=1)
        self.assertIsNone(one_result["code"]["wrong_history_gain_bpt"])

    def test_whitespace_pieces(self):
        for token in ("▁", "▁▁", "<0x09>", "<0x0A>", "<0x0D>", "<0x20>", "\n", "\u0085"):
            self.assertTrue(w.whitespace_piece(token), token)
        for token in ("▁def", "<0xA0>", "<0x32>", "<s>", "<|code|>", "word", ""):
            self.assertFalse(w.whitespace_piece(token), token)


if __name__ == "__main__":
    unittest.main()
