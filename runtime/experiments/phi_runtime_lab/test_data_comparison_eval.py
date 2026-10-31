"""Check the paired metric and fixed endpoint screening rule without training."""
import copy
import unittest
from unittest.mock import Mock, patch

import data_comparison_eval as evaluation


def probes():
    def item(name, answer, bucket):
        return dict(id=name, bucket=bucket, prompt='prompt ' + name, max_tokens=16,
                    rubric={'kind': 'exact', 'expected': answer})
    return [dict(pair_id='colors', items=[item('a', 'red', 'copy'), item('b', 'blue', 'copy')]),
            dict(pair_id='sums', items=[item('c', '4', 'math'), item('d', '9', 'math')])]


def endpoint(pairs, broad, bpt):
    return dict(paired=dict(suite_fingerprint='paired-dev', pairs_passed=pairs),
                broad=dict(suite_fingerprint='broad-dev', split='dev', buckets={'copy': {'passed': broad}}),
                answer_validation={'rows_fingerprint': 'val', 'val_answer_bpt': bpt})


class PairedEvaluationTests(unittest.TestCase):
    def test_requires_both_answers_and_keeps_actual_generations_and_buckets(self):
        texts = ['red', 'red', '4', '9']
        outputs = [dict(text=text, tokens=1, eos=True) for text in texts]
        with patch.object(evaluation, 'generate_answer', side_effect=outputs) as generate:
            result = evaluation.evaluate_pairs(None, None, probes())
        self.assertEqual(generate.call_count, 4)
        self.assertEqual(result['pairs_passed'], 1)
        self.assertEqual(result['pair_rate'], .5)
        self.assertEqual(result['item_rate'], .75)
        self.assertEqual(result['buckets']['copy']['items_passed'], 1)
        self.assertEqual(result['buckets']['copy']['pairs_passed'], 0)
        self.assertEqual(result['rows'][0]['items'][1]['text'], 'red')

    def test_duplicates_and_incomplete_pair_fail_before_generating(self):
        malformed = probes()
        malformed[1]['items'].pop()
        duplicate_id = probes()
        duplicate_id[1]['items'][0]['id'] = 'a'
        for value in [[], malformed, duplicate_id]:
            with patch.object(evaluation, 'generate_answer') as generate, self.assertRaises(ValueError):
                evaluation.evaluate_pairs(None, None, value)
            generate.assert_not_called()

    def test_fixed_retention_order_and_restore_model_training_mode(self):
        rows = {name: [dict(id=str(i), input_ids=[i, i + 1], labels=[-100, i + 1])
                       for i in reversed(range(9))] for name in evaluation.SOURCES}
        model = Mock(training=True)
        with patch.object(evaluation, 'evaluate_pairs', return_value={'pair_rate': .5}), \
             patch.object(evaluation, 'evaluate_suite', return_value={'macro_score': .25}), \
             patch.object(evaluation, 'evaluate_answers', side_effect=lambda *a: {'val_answer_bpt': 5.}) as score:
            first = evaluation.evaluate_dev(model, None, probes(), {}, rows)
            reordered = {k: list(reversed(v)) for k, v in rows.items()}
            second = evaluation.evaluate_dev(model, None, probes(), {}, reordered)
            absent = evaluation.evaluate_dev(model, None, probes(), {})
        self.assertIsNone(absent['answer_validation'])
        self.assertEqual(first, second)
        selected, limit = score.call_args.args[1:]
        self.assertEqual(limit, 8)
        self.assertEqual([r['id'] for r in selected['math']], list('01234567'))
        model.train.assert_called_with(True)

    def test_threshold_relative_to_both_references(self):
        baseline, control, candidate = endpoint(1, 8, 5.), endpoint(3, 7, 4.8), endpoint(5, 8, 5.3)
        self.assertTrue(evaluation.screen_candidate(baseline, control, candidate)['eligible'])
        for weak in [endpoint(4, 8, 5.3), endpoint(5, 7, 5.3), endpoint(5, 8, 5.31)]:
            self.assertFalse(evaluation.screen_candidate(baseline, control, weak)['eligible'])
        candidate['answer_validation'] = None
        self.assertFalse(evaluation.screen_candidate(baseline, control, candidate)['eligible'])

    def test_screen_rejects_different_probes_validation_or_final_candidates(self):
        base = endpoint(0, 0, 4.)
        for key, field in [('paired', 'suite_fingerprint'), ('broad', 'suite_fingerprint'),
                           ('answer_validation', 'rows_fingerprint'), ('broad', 'split')]:
            candidate = copy.deepcopy(base)
            candidate[key][field] = 'final'
            with self.assertRaises(ValueError):
                evaluation.screen_candidate(base, base, candidate)


if __name__ == '__main__':
    unittest.main()
