"""Verify orchestration and sealed-result reuse without requiring a GPU allocation."""

import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import a100_coherence_journal as j


class JournalTest(unittest.TestCase):
    def test_failed_screen_never_opens_final(self):
        screen = {"eligible": False, "grader_sha256": j.quality.GRADER_SHA}
        decision, identity = j.decide(None, {"broad": {}}, screen, None, None, None, None, None, None)
        self.assertEqual(decision["status"], "screen_failed")
        self.assertIsNone(identity)

    def test_sealed_pair_and_broad_results_bound_and_retryable(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            latest = out / "candidate.pt"
            latest.write_bytes(b"checkpoint identity")
            dev = {"qualified": True, "split": "dev", "grader_sha256": j.quality.GRADER_SHA}
            screen = {"eligible": True, "grader_sha256": j.quality.GRADER_SHA}
            sealed = {
                "split": "final",
                "grader_sha256": j.quality.GRADER_SHA,
                "pair_rate": 1.0,
                "decode_errors": 0,
                "buckets": {f: {"pairs_passed": 48, "pairs_total": 48} for f in j.quality.FAMILIES},
                "strata": {
                    f"input={i},wording={w}": {"passed": 48, "total": 48}
                    for i in ("familiar", "novel")
                    for w in ("familiar", "novel")
                },
            }
            args = (
                out,
                {"broad": dev, "dialogue": {"pair_rate": 1.0}},
                screen,
                latest,
                None,
                ["sealed pairs"],
                {"fingerprint": "broad"},
                "cpu",
                Mock(),
            )
            with (
                patch.object(j, "ckpt_load", return_value=(Mock(), {})) as loader,
                patch.object(j.quality, "evaluate_pairs", return_value=sealed) as pairs,
                patch.object(j.dialogue, "evaluate", return_value={"pair_rate": 1.0}) as chat,
                patch.object(
                    j.quality,
                    "evaluate_suite",
                    side_effect=[
                        RuntimeError("disconnected"),
                        {**dev, "split": "final", "decode_errors": 0},
                    ],
                ) as broad,
            ):
                with self.assertRaisesRegex(RuntimeError, "disconnected"):
                    j.decide(*args)
                for _ in range(2):
                    decision, _ = j.decide(*args)
                    self.assertTrue(decision["eligible_for_native_export"])
                self.assertEqual(
                    (pairs.call_count, broad.call_count, chat.call_count, loader.call_count), (1, 2, 1, 2)
                )
                latest.write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "Immutable"):
                    j.decide(*args)

    def test_normal_failed_path_fallback_and_idempotent_completion(self):
        # Mock hardware and costly components only. Drive guard, control flow, screen,
        # saved report and the STOP behavior are exercised through the public run().
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            parent = root / "source.pt"
            parent.write_bytes(b"fixture")
            archive = lambda *a, **k: Mock()

            def sub(obj, name, **kw):
                return stack.enter_context(patch.object(obj, name, **kw))

            sub(j.torch.cuda, "is_available", return_value=True)
            sub(j.torch.cuda, "get_device_name", return_value="A100 fixture")
            for name in ("reset_peak_memory_stats", "max_memory_allocated", "empty_cache"):
                sub(j.torch.cuda, name, return_value=0)
            sub(j, "PARENT_SHA", new=j.sha(parent))
            sub(j, "source_request", return_value=({"frozen": "fixture"}, parent, root, root))
            sub(j, "verify_source", return_value={"context": 512})
            sub(j, "run_preflight", return_value={"passed": True})
            sub(
                j.phistate_a100_diagnostics,
                "run",
                return_value={"status": "complete", "optimizer_steps_taken": 0},
            )
            sub(j.spm, "SentencePieceProcessor", side_effect=archive)
            sub(j, "load_rows", return_value={"train": {}, "val": {}})
            sub(
                j,
                "suites",
                return_value=({"dev": {"fingerprint": "d"}, "final": {"fingerprint": "f"}}, set()),
            )
            sub(j.paired_instruction_v2, "build", return_value={"train": [], "dev": [], "final": []})
            sub(j.plans, "schedules", return_value={"broad": [], "paired": []})
            sub(j.plans, "describe", return_value={"schedule_sha256": "s"})
            metrics = {
                "paired": {"rows": [], "pairs_passed": 0, "pairs_total": 192},
                "broad": {"rows": [], "buckets": {"knowledge": {"passed": 0, "total": 8}}},
                "answer_validation": {"val_answer_bpt": 5.0},
                "dialogue": {"rows": [], "pairs_passed": 0, "pairs_total": 8, "pair_rate": 0.0},
            }

            def arm(**kw):
                return {
                    "status": "complete",
                    "phase_step": 256,
                    "supervised_tokens": 32768,
                    "latest": str(parent),
                    "development_curve": [{"metrics": metrics}] * 2,
                }

            train = sub(j.data_comparison_train, "run_arm", side_effect=arm)
            sub(
                j.quality,
                "screen_candidate",
                return_value={
                    "eligible": False,
                    "grader_sha256": j.quality.GRADER_SHA,
                    "checks": {},
                    "reasons": [],
                },
            )
            fallback = sub(
                j.a100_foundation_control,
                "run",
                return_value={"status": "complete", "outcome": "inconclusive"},
            )
            out = root / j.OUTPUT
            out.mkdir()
            stop = out / "STOP"
            stop.touch()
            self.assertEqual(j.run(root)["status"], "stopped")
            self.assertTrue(stop.is_file())
            stop.unlink()
            result = j.run(root)
            self.assertEqual(
                (result["status"], result["outcome"], result["foundation_outcome"]),
                ("complete", "screen_failed", "inconclusive"),
            )
            self.assertIsNone(result["native_export"])
            self.assertFalse((out / "sealed_pairs.json").exists())
            self.assertEqual(j.run(root), result)
            self.assertEqual(train.call_count, 2)
            self.assertEqual(fallback.call_count, 1)


if __name__ == "__main__":
    unittest.main()
