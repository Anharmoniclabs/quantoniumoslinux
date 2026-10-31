"""Offline orchestration tests: immutable budget, isolated source and final gate.

Trainer, model evaluation and native export are mocked. These tests verify run
control; they make no claim about language-model quality or native parity.
"""
import json
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
import coherence_recovery as recovery
import coherence_recovery_train as trainer
import coherence_quality as quality
import chat_recovery_cpp as native
import device_preflight


class RecoveryControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'broad_continuation_v1/latest.pt'
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b'original checkpoint is immutable input')
        for folder in ['corpus_repaired_v1', 'chat_recovery_v1/data']:
            path = self.root / folder
            path.mkdir(parents=True)
            (path / 'manifest.json').write_text('{"fixture": true}\n')
        (self.root / 'corpus_repaired_v1/tokenizer.model').write_bytes(b'fixture tokenizer')
        self.output = self.root / 'coherence_execution_v1'
        self.original_hash = recovery.sha(self.source)
        self.arm_results = {
            'original': {'qualified': True, 'macro_score': .65},
            'protected': {'qualified': True, 'macro_score': .80},
        }
        self.final_qualified = True
        self.stop_policy = None
        self.mutate_source_policy = None
        self.training_calls = []
        self.training_updates = 0
        self.final_loads = []
        self.model_loads = []
        self.exports = []
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.preflight = stack.enter_context(patch.object(
            device_preflight, 'run_preflight', return_value={'passed': True}))
        stack.enter_context(patch.object(trainer, 'run', self.fake_train))
        stack.enter_context(patch.object(quality, 'build_suites', self.fake_build))
        stack.enter_context(patch.object(quality, 'load_suite', self.fake_load_suite))
        stack.enter_context(patch.object(quality, 'evaluate_suite', self.fake_evaluate))
        stack.enter_context(patch.object(recovery, 'load_model', self.fake_model))
        stack.enter_context(patch.object(recovery, 'release_device', lambda: None))
        stack.enter_context(patch.object(native, 'export', self.fake_export))

    def fake_build(self, prepared, output):
        output.mkdir(parents=True, exist_ok=True)
        for split in ['dev', 'final']:
            (output / (split + '.json')).write_text(json.dumps({'split': split}))

    def fake_train(self, args):
        self.training_calls.append(args)
        self.assertEqual(Path(args.checkpoint), self.output / 'source.pt')
        self.assertEqual(recovery.sha(args.checkpoint), self.original_hash)
        self.assertEqual(Path(args.dev_suite).name, 'dev.json')
        self.assertEqual(Path(args.parent_stop), self.output / 'STOP')
        arm = Path(args.run_dir)
        arm.mkdir(exist_ok=True)
        progress_path = arm / 'progress.json'
        if progress_path.exists():
            prior = json.loads(progress_path.read_text())
            if prior['status'] == 'complete':
                return prior
        status = 'stopped' if args.policy == self.stop_policy else 'complete'
        if status == 'stopped':
            Path(args.parent_stop).touch()
        progress = {'status': status, 'completed_steps': 0 if status == 'stopped' else args.steps}
        (arm / 'progress.json').write_text(json.dumps(progress))
        if status == 'complete':
            self.training_updates += args.steps
            candidate = arm / 'candidate.pt'
            if not candidate.exists():
                torch.save({'quality': self.arm_results[args.policy],
                            'last_validation': {'val_answer_bpt': 4.0}}, candidate)
        if args.policy == self.mutate_source_policy:
            self.source.write_bytes(b'externally modified source')
        return progress

    def fake_load_suite(self, path):
        path = Path(path)
        self.assertEqual(path.name, 'final.json')
        self.final_loads.append(path)
        return json.loads(path.read_text())

    def fake_model(self, checkpoint, tokenizer, device):
        self.model_loads.append(Path(checkpoint))
        return object(), object()

    def fake_evaluate(self, model, tokenizer, suite):
        self.assertEqual(suite['split'], 'final')
        return {'qualified': self.final_qualified, 'macro_score': .75}

    def fake_export(self, folder, prepared):
        self.exports.append(Path(folder))
        self.assertTrue((Path(folder) / 'best.pt').is_file())
        return Path(folder) / 'native_report.json'

    def run_experiment(self, **kwargs):
        options = dict(root=self.root, output=self.output, steps=2, device='cpu',
                       eval_every=1, save_every=1)
        options.update(kwargs)
        return recovery.run(**options)

    def test_failed_preflight_starts_no_arm_and_preserves_source(self):
        self.preflight.return_value = {'passed': False}
        with self.assertRaisesRegex(RuntimeError, 'Device preflight failed'):
            self.run_experiment()
        self.assertEqual(self.training_calls, [])
        self.assertEqual(self.training_updates, 0)
        self.assertEqual(self.final_loads, [])
        self.assertEqual(self.exports, [])
        self.assertEqual(recovery.sha(self.source), self.original_hash)

    def test_only_dev_selected_candidate_touches_final_once(self):
        result = self.run_experiment()
        self.assertEqual(result['status'], 'qualified')
        self.assertEqual(result['selected'], 'protected')
        self.assertEqual([a.policy for a in self.training_calls], ['original', 'protected'])
        self.assertEqual(self.model_loads, [self.output / 'protected/candidate.pt'])
        self.assertEqual(len(self.final_loads), 1)
        self.assertEqual(len(self.exports), 1)
        self.assertEqual(recovery.sha(self.source), self.original_hash)
        self.assertTrue(result['source_unchanged'])

    def test_completed_rerun_does_not_train_evaluate_or_export(self):
        first = self.run_experiment()
        calls = (len(self.training_calls), len(self.final_loads), len(self.exports))
        second = self.run_experiment()
        self.assertEqual(first, second)
        self.assertEqual(calls, (len(self.training_calls), len(self.final_loads), len(self.exports)))

    def test_budget_is_immutable_even_after_terminal_result(self):
        self.run_experiment()
        with self.assertRaisesRegex(RuntimeError, 'immutable'):
            self.run_experiment(steps=3)
        self.assertEqual(len(self.training_calls), 2)

    def test_source_change_blocks_resume_before_training(self):
        self.run_experiment()
        self.source.write_bytes(b'changed checkpoint')
        with self.assertRaisesRegex(RuntimeError, 'immutable'):
            self.run_experiment()
        self.assertEqual(len(self.training_calls), 2)

    def test_no_export_and_no_final_when_dev_quality_fails(self):
        for result in self.arm_results.values():
            result['qualified'] = False
        result = self.run_experiment()
        self.assertEqual(result['status'], 'failed_quality')
        self.assertEqual(self.final_loads, [])
        self.assertEqual(self.exports, [])
        self.run_experiment()
        self.assertEqual(len(self.training_calls), 2)

    def test_final_failure_does_not_fall_back_to_other_candidate(self):
        self.final_qualified = False
        result = self.run_experiment()
        self.assertEqual(result['status'], 'failed_final')
        self.assertEqual(self.model_loads, [self.output / 'protected/candidate.pt'])
        self.assertEqual(len(self.final_loads), 1)
        self.assertEqual(self.exports, [])
        self.run_experiment()
        self.assertEqual(len(self.final_loads), 1)

    def test_parent_stop_propagates_from_first_arm(self):
        self.stop_policy = 'original'
        result = self.run_experiment()
        self.assertEqual(result['status'], 'stopped')
        self.assertEqual([a.policy for a in self.training_calls], ['original'])
        self.assertEqual(self.final_loads, [])
        self.assertEqual(self.exports, [])
        self.run_experiment()
        self.assertEqual(len(self.training_calls), 1)

    def test_stop_in_second_arm_prevents_final_and_export(self):
        self.stop_policy = 'protected'
        result = self.run_experiment()
        self.assertEqual(result['status'], 'stopped')
        self.assertEqual([a.policy for a in self.training_calls], ['original', 'protected'])
        self.assertEqual(self.final_loads, [])
        self.assertEqual(self.exports, [])

    def test_native_failure_retries_export_without_additional_updates(self):
        with patch.object(native, 'export', side_effect=RuntimeError('parity failed')):
            with self.assertRaisesRegex(RuntimeError, 'parity failed'):
                self.run_experiment()
        report = json.loads((self.output / 'report.json').read_text())
        self.assertEqual(report['status'], 'native_export_failed')
        updates_before_retry = self.training_updates
        second = self.run_experiment()
        self.assertEqual(second['status'], 'qualified')
        # Completed trainers may be invoked for read-only resume checks.
        self.assertEqual(len(self.training_calls), 4)
        self.assertEqual(self.training_updates, updates_before_retry)
        self.assertEqual(len(self.final_loads), 1)
        self.assertEqual(len(self.model_loads), 1)
        self.assertEqual(len(self.exports), 1)

    def test_source_mutation_during_comparison_prevents_final_and_export(self):
        self.mutate_source_policy = 'protected'
        with self.assertRaisesRegex(RuntimeError, 'Source checkpoint changed'):
            self.run_experiment()
        self.assertEqual(self.final_loads, [])
        self.assertEqual(self.exports, [])

    def test_final_cache_reused_after_interruption_before_terminal_report(self):
        self.run_experiment()
        # Simulate interruption after final-result persistence but before a
        # terminal report. Training mocks preserve their existing candidates.
        (self.output / 'report.json').unlink()
        result = self.run_experiment()
        self.assertEqual(result['status'], 'qualified')
        self.assertEqual(len(self.final_loads), 1)
        self.assertEqual(len(self.model_loads), 1)

    def test_used_final_suite_refuses_changed_candidate(self):
        self.run_experiment()
        (self.output / 'report.json').unlink()
        candidate = self.output / 'protected/candidate.pt'
        torch.save({'quality': self.arm_results['protected'],
                    'last_validation': {'val_answer_bpt': 3.5}}, candidate)
        with self.assertRaisesRegex(RuntimeError, 'already used for a different candidate'):
            self.run_experiment()
        self.assertEqual(len(self.final_loads), 1)
        self.assertEqual(len(self.exports), 1)


if __name__ == '__main__':
    unittest.main()
