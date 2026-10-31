"""Offline runner tests with real mini-model updates and mocked evaluations.

The mocks test orchestration and checkpoint semantics, not model capability.
"""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import ExitStack

import torch

import chat_recovery_data as data
import chat_recovery_train as old_train
import coherence_recovery_train as train
import moe_train
from optimizer_policy import protect_structured_parameters
from test_chat_recovery import fixture, adapter

REPLAY = {'val_mean_bpt': 6., 'val_code_bpt': 5., 'val_general_bpt': 7., 'val_count_tokens': 4096}
LOSS = {'val_answer_bpt': 5., 'per_source': {}}
ZERO = {'macro_score': 0., 'qualified': False, 'rows': []}


class RecoveryTrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.tokenizer, initial = fixture(cls.root)
        stub = cls.root/'upstream.md'; stub.write_text('offline fixture')
        cls.sft = cls.root/'sft'
        with ExitStack() as stack:
            stack.enter_context(patch.object(data, 'download_source', lambda *_: (stub, data.sha(stub), stub)))
            for source in data.SOURCES:
                stack.enter_context(patch.object(data, source+'_rows', adapter(source)))
            data.build(cls.root, cls.sft, cls.root/'cache', 256)
        # Produce a real source SFT state with moments, orders and RNG streams.
        args = SimpleNamespace(run_dir=str(cls.root/'source_run'), checkpoint=str(initial),
            sft_dir=str(cls.sft), data_dir=str(cls.root), steps=1, context=256, accum=4,
            device='cpu', eval_every=1, save_every=1, val_examples=1, patience=6, clear_stop=False)
        with patch.object(old_train, 'generate', return_value=[]), \
             patch.object(old_train, 'evaluate', return_value=LOSS), \
             patch.object(moe_train, 'evaluate_both', return_value=REPLAY):
            old_train.run(args)
        cls.source = cls.root/'source_run/latest.pt'
        cls.source_hash = data.sha(cls.source)
        cls.dev = cls.root/'dev.json'; cls.dev.write_text('{"split":"dev"}')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def args(self, name, policy='original', steps=2, eval_every=1):
        return SimpleNamespace(checkpoint=str(self.source), data_dir=str(self.root),
            sft_dir=str(self.sft), run_dir=str(self.root/name), policy=policy, steps=steps,
            dev_suite=str(self.dev), eval_every=eval_every, save_every=1,
            device='cpu', clear_stop=False, parent_stop=None)

    def mocks(self, quality=None, retention=None):
        stack = ExitStack()
        stack.enter_context(patch.object(train, 'load_suite', return_value={'split': 'dev'}))
        stack.enter_context(patch.object(train, 'evaluate', return_value=LOSS))
        def zero(*_):
            # Verify evaluation RNG activity does not perturb training streams.
            torch.rand(3)
            return copy.deepcopy(ZERO)
        stack.enter_context(patch.object(train, 'evaluate_suite', side_effect=quality or zero))
        stack.enter_context(patch.object(moe_train, 'evaluate_both',
                                        side_effect=retention if retention else None,
                                        return_value=REPLAY))
        return stack

    def checkpoint(self, args):
        return torch.load(Path(args.run_dir)/'latest.pt', weights_only=True)

    def test_failed_quality_never_promotes_and_coverage_is_actual(self):
        a = self.args('failure')
        with self.mocks():
            result = train.run(a)
        self.assertEqual(result['status'], 'completed_failed_quality')
        self.assertFalse((Path(a.run_dir)/'candidate.pt').exists())
        self.assertTrue((Path(a.run_dir)/'best_loss.pt').exists())
        saved = self.checkpoint(a)
        self.assertEqual(sum(v['draws'] for v in saved['coverage'].values()), 8)
        self.assertEqual(sum(v['input_tokens'] for v in saved['coverage'].values()), saved['phase_tokens'])
        self.assertEqual(sum(v['answer_tokens'] for v in saved['coverage'].values()), saved['answer_tokens'])
        for v in saved['coverage'].values():
            self.assertEqual(v['unique_indices'], sorted(set(v['unique_indices'])))
        self.assertEqual(data.sha(self.source), self.source_hash)
        self.assertEqual(saved['source_sft_steps'], 1)
        self.assertEqual(saved['sft_steps'], 3)

    def test_exact_resume_both_policies(self):
        for policy in ('original', 'protected'):
            with self.subTest(policy=policy), self.mocks():
                full = self.args('full_'+policy, policy, steps=3)
                resumed = self.args('resume_'+policy, policy, steps=3)
                train.run(full)
                writer = train.atomic_json
                def stop_after_one(path, value):
                    writer(path, value)
                    if path.name == 'progress.json' and value.get('phase_step') == 1:
                        (path.parent/'STOP').touch()
                with patch.object(train, 'atomic_json', side_effect=stop_after_one):
                    stopped = train.run(resumed)
                self.assertEqual(stopped['status'], 'stopped')
                resumed.clear_stop = True
                train.run(resumed)
                x, y = self.checkpoint(full), self.checkpoint(resumed)
                self.assertEqual(x['phase_step'], y['phase_step'])
                for name in x['state_dict']:
                    self.assertTrue(torch.equal(x['state_dict'][name], y['state_dict'][name]), name)
                for key in ('coverage', 'orders', 'positions', 'python_rng_state', 'numpy_rng_state'):
                    self.assertEqual(x[key], y[key])
                self.assertTrue(torch.equal(x['torch_rng_state'], y['torch_rng_state']))

    def test_budget_locked_and_completed_progress_reconciled(self):
        a = self.args('budget')
        with self.mocks():
            result = train.run(a)
            latest_hash = data.sha(Path(a.run_dir)/'latest.pt')
            (Path(a.run_dir)/'progress.json').write_text('{"status":"stale"}')
            again = train.run(a)
            self.assertEqual(again['status'], result['status'])
            self.assertEqual(data.sha(Path(a.run_dir)/'latest.pt'), latest_hash)
            a.steps = 3
            with self.assertRaisesRegex(RuntimeError, 'budget/settings'):
                train.run(a)

    def test_protected_baseline_preserves_all_optimizer_state(self):
        a = self.args('protected_initial', 'protected')
        writer = train.atomic_json
        def stop_at_baseline(path, value):
            writer(path, value)
            if path.name == 'progress.json':
                (path.parent/'STOP').touch()
        with self.mocks(), patch.object(train, 'atomic_json', side_effect=stop_at_baseline):
            train.run(a)
        source = torch.load(self.source, weights_only=True)
        saved = self.checkpoint(a)
        self.assertEqual(saved['phase_step'], 0)
        models, opts = [], []
        for state, protected in ((source, False), (saved, True)):
            model = moe_train.PhiStateMoELM(moe_train.MoEConfig(**state['config']))
            opt = moe_train.optimizer_for(SimpleNamespace(**state['train_args']), model)
            if protected:
                protect_structured_parameters(opt, model)
            opt.load_state_dict(state['optimizer_state'])
            models.append(model); opts.append(opt)
        for p, q in zip(models[0].parameters(), models[1].parameters()):
            for key, old in opts[0].state[p].items():
                new = opts[1].state[q][key]
                self.assertTrue(torch.equal(old, new) if torch.is_tensor(old) else old == new)

    def test_candidate_tracks_best_qualified_dev_not_latest(self):
        a = self.args('qualified')
        qualities = [ZERO, {'macro_score': .9, 'qualified': True},
                     {'macro_score': .8, 'qualified': True}]
        with self.mocks(quality=qualities):
            result = train.run(a)
        self.assertEqual(result['status'], 'completed_qualified')
        candidate = torch.load(Path(a.run_dir)/'candidate.pt', weights_only=True)
        self.assertEqual(candidate['phase_step'], 1)
        self.assertEqual(candidate['quality']['macro_score'], .9)
        self.assertEqual(candidate['sft_steps'], 2)
        self.assertEqual(self.checkpoint(a)['phase_step'], 2)

    def test_quality_plateau_stops_after_three_postbaseline_evaluations(self):
        a = self.args('plateau', steps=6)
        with self.mocks():
            result = train.run(a)
        self.assertEqual(result['status'], 'early_stopped_failed_quality')
        self.assertEqual(result['phase_step'], 3)

    def test_retention_failure_prevents_quality_promotion(self):
        a = self.args('retention')
        qualities = [ZERO, {'macro_score': 1., 'qualified': True}]
        with self.mocks(quality=qualities, retention=[REPLAY, {**REPLAY, 'val_mean_bpt': 7.}]):
            result = train.run(a)
        self.assertEqual(result['status'], 'retention_stopped')
        self.assertFalse((Path(a.run_dir)/'candidate.pt').exists())

    def test_parent_stop_and_final_suite_guard(self):
        a = self.args('parent_stop')
        a.parent_stop = str(self.root/'coordinator_STOP')
        Path(a.parent_stop).touch()
        with self.mocks():
            result = train.run(a)
        self.assertEqual(result['status'], 'stopped')
        self.assertEqual(result['phase_step'], 0)
        Path(a.parent_stop).unlink()
        b = self.args('final_guard')
        with self.mocks(), patch.object(train, 'load_suite', return_value={'split': 'final'}):
            with self.assertRaisesRegex(ValueError, 'sealed'):
                train.run(b)


if __name__ == '__main__':
    unittest.main()
