"""CPU fixture verifies exact target masking, objective, optimizer lineage and recovery."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

import data_comparison_train as train
import moe_train
from chat_recovery_data import TEMPLATE, sha
from test_chat_recovery import fixture


class DataComparisonTrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tmp = TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.tok, cls.parent = fixture(cls.root)
        s = torch.load(cls.parent, weights_only=True)
        s.update(training_stage='chat_recovery_v1', sft_steps=4000, step=69000,
                 chat_settings={'context': 128}, chat_template=TEMPLATE,
                 tokenizer='bpe', tokenizer_sha256=sha(cls.root/'tokenizer.model'))
        torch.save(s, cls.parent)
        cls.parent_hash = sha(cls.parent)
        eos = cls.tok.eos_id()
        cls.pool = [dict(id='a', source='alpha', input_ids=[11,12,13,14,15], labels=[-100,13,-100,15,eos]),
                    dict(id='b', source='beta', input_ids=[17,18,19,20], labels=[-100,19,20,eos])]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def args(self, suffix, **kwargs):
        return dict(parent=self.parent, tokenizer=self.tok, pool=self.pool, output=self.root/suffix,
                    policy_name='broad', device='cpu', steps=4, tokens_per_step=2,
                    seed=321, evaluate_callback=lambda model: {'small_metric': float(torch.rand(()))},
                    identity={'suite': 'unit-fixture'}, **kwargs)

    def assert_tensor_dict_equal(self, a, b):
        self.assertEqual(set(a), set(b))
        for name in a:
            self.assertTrue(torch.equal(a[name], b[name]), name)

    def test_exact_masked_stream_and_resume(self):
        params = self.args('full')
        before_rng = torch.get_rng_state().clone()
        done = train.run_arm(**params)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertEqual(done['supervised_tokens'], 8)
        self.assertEqual(done['stream_cursor'], [2,2])
        self.assertEqual(done['per_source']['alpha']['supervised_tokens'], 5)
        self.assertEqual(done['per_source']['beta']['supervised_tokens'], 3)
        self.assertEqual([r['step'] for r in done['development_curve']], [0,2,4])
        stop = self.root/'coordinator_STOP'
        params = self.args('paused', stop_path=stop)
        scorer = params['evaluate_callback']
        evaluations = []
        def stop_at_midpoint(model):
            evaluations.append(1)
            if len(evaluations) == 2:
                stop.touch()
            return scorer(model)
        params['evaluate_callback'] = stop_at_midpoint
        stopped = train.run_arm(**params)
        self.assertEqual((stopped['status'], stopped['phase_step']), ('stopped',2))
        self.assertEqual(stopped['stream_cursor'], [1,1])
        stop.unlink()
        params['evaluate_callback'] = scorer
        resumed = train.run_arm(**params)
        x = torch.load(done['latest'], weights_only=True)
        y = torch.load(resumed['latest'], weights_only=True)
        self.assert_tensor_dict_equal(x['state_dict'], y['state_dict'])
        self.assertEqual(x['optimizer_state']['param_groups'], y['optimizer_state']['param_groups'])
        for i in x['optimizer_state']['state']:
            self.assert_tensor_dict_equal(x['optimizer_state']['state'][i], y['optimizer_state']['state'][i])
        self.assertEqual(x['update_metrics'], y['update_metrics'])
        self.assertEqual(x['development_curve'], y['development_curve'])
        self.assertEqual(x['per_source'], y['per_source'])
        self.assertEqual(sha(self.parent), self.parent_hash)

    def test_one_step_matches_explicit_token_weighted_reference(self):
        a = self.args('objective')
        a.update(steps=1, tokens_per_step=5, tokenizer=self.root/'tokenizer.model')
        result = train.run_arm(**a)
        saved = torch.load(result['latest'], weights_only=True)
        source = torch.load(self.parent, weights_only=True)
        model = moe_train.PhiStateMoELM(moe_train.MoEConfig(**source['config']))
        model.load_state_dict(source['state_dict'])
        optimizer = moe_train.optimizer_for(SimpleNamespace(**source['train_args']), model)
        optimizer.load_state_dict(source['optimizer_state'])
        optimizer.zero_grad(set_to_none=True)
        # Three targets from the gapped first row, two from the second. The first
        # EOS is scored; the second row's EOS remains pending for the next step.
        a_logits = model(torch.tensor([[11,12,13,14,15]]))[0,[1,3,4]]
        a_penalty = model.router_penalty()[0]
        a_loss = F.cross_entropy(a_logits, torch.tensor([13,15,self.tok.eos_id()]), reduction='sum')
        ((a_loss + 3*a_penalty) / 5).backward()
        b_logits = model(torch.tensor([[17,18,19]]))[0,[1,2]]
        b_penalty = model.router_penalty()[0]
        b_loss = F.cross_entropy(b_logits, torch.tensor([19,20]), reduction='sum')
        ((b_loss + 2*b_penalty) / 5).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        self.assert_tensor_dict_equal(saved['state_dict'], model.state_dict())
        self.assertEqual(saved['stream_cursor'], [1,2])
        self.assertEqual(saved['sft_steps'], 4001)

    def test_completed_readback_and_locked_budget_schedule_identity(self):
        a = self.args('locked')
        a['steps'] = 1
        first = train.run_arm(**a)
        ckpt_hash = sha(first['latest'])
        self.assertEqual(train.run_arm(**a), first)
        self.assertEqual(sha(first['latest']), ckpt_hash)
        for key, value in [('steps',2), ('pool',list(reversed(self.pool))), ('identity',{'suite':'changed'})]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'settings differ'):
                train.run_arm(**{**a,key:value})

    def test_invalid_causal_target_rejected(self):
        a = self.args('invalid')
        a['pool'] = [{**self.pool[0], 'labels': [-100,55,-100,15,self.tok.eos_id()]}]
        with self.assertRaisesRegex(ValueError, 'next-token'):
            train.run_arm(**a)
        self.assertFalse((a['output']/'latest.pt').exists())


if __name__ == '__main__':
    unittest.main()
