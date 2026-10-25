import difflib
import json
from pathlib import Path
import tempfile
import unittest
from evaluate import FAIR_FIELDS, TASKS, compare, evidence_metrics, grade_patch, grade_predictions, prepare, suite_hash


def patch(tid, source):
    return ''.join(difflib.unified_diff(TASKS[tid]['source'].splitlines(True), source.splitlines(True),
                                       'a/solution.py', 'b/solution.py'))


def manifest():
    out = dict.fromkeys(FAIR_FIELDS, 'fixed-control')
    out.update(suite_sha256=suite_hash(), attempts_per_task=1, harness_revision='test-only')
    return out


class EvaluationTests(unittest.TestCase):
    def test_prepare_batch_contract_and_no_hidden_tests(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'suite'
            prepare(root)
            rows = [json.loads(line) for line in (root / 'tasks.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), len(TASKS))
            for row in rows:
                snapshot = root / row['snapshot']
                self.assertEqual({p.name for p in snapshot.iterdir()}, {'solution.py', 'README.md'})
            with self.assertRaises(FileExistsError): prepare(root)

    def test_references_pass(self):
        for tid, task in TASKS.items():
            with self.subTest(task=tid):
                result = grade_patch(tid, patch(tid, task['reference']))
                self.assertTrue(result['verified'], result)

    def test_original_buggy_implementations_fail(self):
        for tid, task in TASKS.items():
            with self.subTest(task=tid):
                # A cosmetic edit ensures code, not merely empty patch rejection, is checked.
                result = grade_patch(tid, patch(tid, task['source'] + '\n# no fix\n'))
                self.assertEqual(result['failure'], 'tests_failed', result)

    def test_status_claim_does_not_count(self):
        report = grade_predictions([{'instance_id': tid, 'status': 'verified', 'model_patch': ''} for tid in TASKS], manifest())
        self.assertEqual(report['summary']['verified_solves'], 0)
        self.assertIsNone(report['summary']['totals']['completion_tokens'])
        self.assertIsNone(report['summary']['invalid_tool_rate'])

    def test_duplicate_prediction_rejected(self):
        row = {'instance_id': next(iter(TASKS))}
        with self.assertRaises(ValueError): grade_predictions([row, row], manifest())

    def test_missing_tasks_count_as_failures(self):
        report = grade_predictions([], manifest())
        self.assertEqual(report['summary']['failure_counts'], {'missing_prediction': 4})

    def test_patch_scope(self):
        tid = next(iter(TASKS))
        bad = patch(tid, TASKS[tid]['reference']).replace('solution.py', '../outside.py')
        self.assertEqual(grade_patch(tid, bad)['failure'], 'patch_scope')
        bad = 'diff --git a/solution.py b/solution.py\nold mode 100644\nnew mode 120000\n' + patch(tid, TASKS[tid]['reference'])
        self.assertFalse(grade_patch(tid, bad)['verified'])

    def test_timeout(self):
        tid = next(iter(TASKS))
        result = grade_patch(tid, patch(tid, 'while True: pass\n'), timeout=.05)
        self.assertEqual(result['failure'], 'grader_timeout')

    def test_exit_zero_is_not_success(self):
        tid = next(iter(TASKS))
        result = grade_patch(tid, patch(tid, 'raise SystemExit(0)\n'))
        self.assertFalse(result['verified'])

    def test_fair_pair_and_control_mismatch(self):
        empty = grade_predictions([], manifest())
        solved = grade_predictions([{'instance_id': tid, 'model_patch': patch(tid, t['reference'])}
                                    for tid, t in TASKS.items()], manifest())
        self.assertEqual(len(compare(empty, solved)['gained']), 4)
        solved['manifest']['max_steps'] = 'different'
        with self.assertRaises(ValueError): compare(empty, solved)
        del solved['manifest']['max_steps']
        with self.assertRaises(ValueError): compare(empty, solved)

    def test_telemetry_complete_and_legacy(self):
        events = [{'event': 'start', 'telemetry_schema': 1},
                  {'event': 'model', 'calls': [{}, {}], 'prompt_tokens': 10, 'completion_tokens': 4},
                  {'event': 'protocol_error', 'errors': ['', 'ERROR bad args']},
                  {'event': 'end'}]
        self.assertEqual(evidence_metrics(events), {'prompt_tokens': 10, 'completion_tokens': 4,
                                                   'tool_calls': 2, 'invalid_tool_calls': 1})
        self.assertIsNone(evidence_metrics(events[:-1])['completion_tokens'])
        events[0].pop('telemetry_schema')
        self.assertIsNone(evidence_metrics(events)['invalid_tool_calls'])
        self.assertIsNone(evidence_metrics(events)['prompt_tokens'])
        events[1]['completion_tokens'] = None
        self.assertIsNone(evidence_metrics(events)['completion_tokens'])

    def test_metrics_coverage_and_invalid_rate(self):
        rows = [{'instance_id': tid, 'model_patch': '', 'seconds': 1, 'steps': 2,
                 'prompt_tokens': 5, 'completion_tokens': 3,
                 'tool_calls': 4, 'invalid_tool_calls': 1} for tid in TASKS]
        summary = grade_predictions(rows, manifest())['summary']
        self.assertEqual(summary['totals']['completion_tokens'], 12)
        self.assertEqual(summary['invalid_tool_rate'], .25)
        rows[0]['completion_tokens'] = None
        summary = grade_predictions(rows, manifest())['summary']
        self.assertIsNone(summary['totals']['completion_tokens'])
        self.assertEqual(summary['metric_coverage_tasks']['completion_tokens'], 3)

if __name__ == '__main__': unittest.main()
