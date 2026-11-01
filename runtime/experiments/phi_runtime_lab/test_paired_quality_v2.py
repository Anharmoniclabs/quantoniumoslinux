"""Regression checks for V2 free-text scoring, paired pass@1 and sealed final."""

import sys
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import coherence_quality as broad
import paired_quality_v2 as v2


def fixture_pairs(split="dev"):
    result = []
    for family, n, ir, wr in [
        ("arithmetic", 0, "familiar", "novel"),
        ("string", 1, "novel", "familiar"),
    ]:
        pid = f"{split}/{family}"
        strata = {"input_regime": ir, "wording_regime": wr}
        ops = ["+", "-"] if family == "arithmetic" else ["upper", "reverse"]
        golds = ["12", "-3"] if family == "arithmetic" else ["APPLE", "elppa"]
        items = [
            {
                "id": f"{pid}/{s}",
                "prompt": "question " + str(n * 2 + s),
                "bucket": family,
                "split": split,
                "pair_id": pid,
                "strata": dict(strata),
                "max_tokens": 12,
                "operation": ops[s],
                "rubric": {"kind": "exact", "expected": gold},
            }
            for s, gold in enumerate(golds)
        ]
        result.append(
            {
                "pair_id": pid,
                "family": family,
                "items": items,
                "strata": strata,
                "features": {"operations": ops, "add_carry": True if n == 0 else None},
            }
        )
    return result


def endpoint(pairs=10, counts=(1, 2, 3, 4), bpt=5.0):
    return {
        "paired": {
            "suite_fingerprint": "pairs",
            "split": "dev",
            "grader_sha256": v2.GRADER_SHA,
            "pairs_passed": pairs,
            "pairs_total": 192,
            "decode_errors": 0,
        },
        "broad": {
            "suite_fingerprint": "broad",
            "split": "dev",
            "grader_sha256": v2.GRADER_SHA,
            "qualified": False,
            "decode_errors": 0,
            "buckets": {k: {"passed": v} for k, v in zip(broad.BUCKETS, counts)},
        },
        "answer_validation": {"rows_fingerprint": "val", "val_answer_bpt": bpt},
    }


class PairedQualityTests(unittest.TestCase):
    def test_eight_concepts_with_paraphrases_and_wrong_claims(self):
        cases = [
            (
                "It's the computer's scratchpad for the task it is working on.",
                "It permanently deletes data used by running programs.",
            ),
            (
                "It keeps files even when the machine is switched off.",
                "It deletes every file when the power goes off.",
            ),
            (
                "The processor carries out a program's instructions.",
                "The processor stores instructions instead of executing them.",
            ),
            ("Python is a high-level programming language.", "Python is an operating system."),
            ("A variable is a name that refers to a value.", "A variable always stays constant."),
            (
                "It manages the hardware resources used by programs.",
                "It supplies electricity to the hardware.",
            ),
            ("Plants use sunlight to make sugar.", "Plants burn sugar when they get sunlight."),
            ("Gravity attracts objects toward one another.", "Gravity repels objects away from Earth."),
        ]
        for item, (good, bad) in zip(broad._bank("dev")[:8], cases):
            with self.subTest(item=item["id"]):
                self.assertTrue(v2.grade_answer(item, good)["passed"])
                self.assertFalse(v2.grade_answer(item, bad)["passed"])

    def test_positive_action_subject_polarity_qualifiers(self):
        item = broad._bank("dev")[0]
        for good in (
            "RAM stores data that the CPU needs right now.",
            "It keeps the programs and data you are using right now.",
            "A computer’s scratchpad for what it’s working on right now.",
            "RAM temporarily holds data used by running programs.",
            "RAM is not permanent. It temporarily holds data.",
            "RAM doesn't store files permanently. It keeps current data.",
            "RAM is working memory. Its contents are lost when power is off.",
        ):
            with self.subTest(good=good):
                self.assertTrue(v2.grade_answer(item, good)["passed"])
        for bad in (
            "RAM temporarily deletes information.",
            "The CPU temporarily holds data.",
            "RAM does not keep data for running programs.",
            "RAM permanently stores temporary data.",
            "RAM is working memory but it also deletes data permanently.",
            "RAM is working memory. It cooks pasta.",
        ):
            with self.subTest(bad=bad):
                result = v2.grade_answer(item, bad)
                self.assertFalse(result["passed"])
                self.assertTrue(result["review"])

    def test_exact_arithmetic_and_functional_code(self):
        arithmetic = broad._bank("dev")[16]
        self.assertTrue(v2.grade_answer(arithmetic, "27")["passed"])
        for bad in ("027", "27.0", "The result is 27.", "-27"):
            self.assertFalse(v2.grade_answer(arithmetic, bad)["passed"])
        code = broad._bank("dev")[24]
        self.assertTrue(v2.grade_answer(code, "def add(a,b):\n    return b+a")["passed"])
        for bad in ("def add(a,b):\n    return a-b", 'def add(a,b):\n    return eval("a+b")'):
            self.assertFalse(v2.grade_answer(code, bad)["passed"])

    def test_one_sample_eos_errors_and_pair_transfer_groups(self):
        answers = [
            {"text": s, "eos": e, "tokens": 1} for s, e in [("12", True), ("-3", True), ("APPLE", False)]
        ]
        answers.append(ValueError("invalid tokenizer output"))
        with patch.object(broad, "generate_answer", side_effect=answers) as decoder:
            r = v2.evaluate_pairs(None, None, fixture_pairs())
        self.assertEqual(decoder.call_count, 4)
        self.assertEqual((r["pairs_passed"], r["items_passed"], r["decode_errors"]), (1, 2, 1))
        self.assertEqual(r["strata"]["input=familiar,wording=novel"], {"passed": 1, "total": 1})
        self.assertEqual(r["feature_pair_scores"]["arithmetic/add_carry=True"]["total"], 1)
        self.assertFalse(r["rows"][1]["items"][0]["passed"])
        self.assertIn("invalid tokenizer", r["rows"][1]["items"][1]["decode_error"])
        with (
            patch.object(broad, "generate_answer", side_effect=RuntimeError("out of memory")),
            self.assertRaisesRegex(RuntimeError, "out of memory"),
        ):
            v2.evaluate_pairs(None, None, fixture_pairs())

    def test_cuda_fallback_is_recorded_in_decode_receipt(self):
        class Answers(list):
            decode_backend = "serial_after_batch_parity_mismatch"
            mismatch_index = 2

        class FakeGPU:
            def parameters(self):
                return iter([SimpleNamespace(device=SimpleNamespace(type="cuda"))])

        def generate(model, tokenizer, items, **settings):
            self.assertEqual(settings, {"batch_size": 16, "verify": 2, "serial_on_mismatch": True})
            return Answers({"text": i["rubric"]["expected"], "tokens": 1, "eos": True} for i in items)

        with patch.dict(sys.modules, {"a100_greedy_batch": SimpleNamespace(generate_batch=generate)}):
            result = v2.evaluate_pairs(FakeGPU(), None, fixture_pairs())
        self.assertEqual(result["pairs_passed"], 2)
        self.assertEqual(result["decoding"]["backend"], "serial_after_batch_parity_mismatch")
        self.assertEqual(result["decoding"]["batch_parity_mismatch_index"], 2)

    def test_invalid_metadata_raises_before_generation(self):
        pairs = fixture_pairs()
        pairs[1]["items"][1]["strata"] = {"input_regime": "bogus"}
        with patch.object(broad, "generate_answer") as generate:
            with self.assertRaisesRegex(ValueError, "Malformed pair"):
                v2.evaluate_pairs(None, None, pairs)
            generate.assert_not_called()

    def test_broad_rescore_and_blind_review(self):
        suite = {"split": "dev", "items": broad._bank("dev")}
        suite["fingerprint"] = broad.fingerprint(suite)

        def answer(model, tokenizer, prompt, max_tokens):
            text = "It's the computer's scratchpad for the current task." if "RAM" in prompt else ""
            return {"text": text, "tokens": 9, "eos": True}

        with patch.object(broad, "generate_answer", side_effect=answer):
            r = v2.evaluate_suite(None, None, suite)
        self.assertEqual(r["buckets"]["knowledge"]["passed"], 1)
        self.assertEqual(r["code_pass_at_1"], 0)
        pkt = v2.blind_review_packet({"base": r, "candidate": deepcopy(r)})
        self.assertEqual(len(pkt["packet"]), 8)
        for row in pkt["packet"]:
            self.assertEqual(set(row), {"id", "prompt", "answer"})
            self.assertEqual(len(pkt["key"][row["id"]]), 2)

    def test_screen_and_heldout_boundaries(self):
        baseline, control, cand = endpoint(), endpoint(11), endpoint(21)
        self.assertTrue(v2.screen_candidate(baseline, control, cand)["eligible"])
        for bad in [endpoint(20), endpoint(21, (0, 3, 3, 4)), endpoint(21, bpt=5.51)]:
            self.assertFalse(v2.screen_candidate(baseline, control, bad)["eligible"])
        bad = deepcopy(cand)
        bad["paired"]["decode_errors"] = 1
        self.assertFalse(v2.screen_candidate(baseline, control, bad)["eligible"])
        bad = deepcopy(cand)
        bad["paired"]["grader_sha256"] = "v1"
        with self.assertRaisesRegex(ValueError, "grader"):
            v2.screen_candidate(baseline, control, bad)

    def test_dev_rejects_final_prompts_before_decoding(self):
        from unittest.mock import Mock

        model = Mock(training=True)
        with patch.object(broad, "generate_answer") as generate:
            with self.assertRaisesRegex(ValueError, "dev pairs"):
                v2.evaluate_dev(model, None, fixture_pairs("final"), {"split": "dev"}, {})
            generate.assert_not_called()

    def test_final_only_after_screen_and_broad_development(self):
        yes = {"eligible": True, "grader_sha256": v2.GRADER_SHA}
        no = {"eligible": False, "grader_sha256": v2.GRADER_SHA}
        dev = endpoint()["broad"]
        final = {
            "split": "final",
            "grader_sha256": v2.GRADER_SHA,
            "decode_errors": 0,
            "pair_rate": 0.7,
            "buckets": {k: {"pairs_passed": 2, "pairs_total": 4} for k in v2.FAMILIES},
            "strata": {
                f"input={i},wording={w}": {"passed": 2, "total": 4}
                for i in ("familiar", "novel")
                for w in ("familiar", "novel")
            },
        }
        self.assertEqual(v2.final_decision(no, None, None)["status"], "screen_failed")
        with self.assertRaisesRegex(ValueError, "ineligible"):
            v2.final_decision(no, dev, final)
        self.assertEqual(v2.final_decision(yes, dev, final)["status"], "paired_transfer_only")
        dev["qualified"] = True
        self.assertEqual(v2.final_decision(yes, dev, final)["status"], "broad_final_required")
        broadfinal = {
            "split": "final",
            "grader_sha256": v2.GRADER_SHA,
            "qualified": True,
            "decode_errors": 0,
        }
        self.assertTrue(v2.final_decision(yes, dev, final, broadfinal)["eligible_for_native_export"])
        final["strata"]["input=novel,wording=novel"]["passed"] = 1
        self.assertEqual(v2.final_decision(yes, dev, final)["status"], "final_pairs_failed")
        with self.assertRaisesRegex(ValueError, "outside"):
            v2.final_decision(yes, dev, final, broadfinal)


if __name__ == "__main__":
    unittest.main()
