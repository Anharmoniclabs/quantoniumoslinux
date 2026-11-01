"""Offline structural, causal, and exact resume tests for the scratch binding control."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import sentencepiece as spm
import torch

import a100_foundation_control as control


def tokenizer_fixture(root):
    input_file = root / 'text.txt'
    input_file.write_text(('ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz 0123456789\n'
                           'List Given Find Value for Read Look up\n') * 8)
    spm.SentencePieceTrainer.train(input=str(input_file), model_prefix=str(root / 'tokenizer'),
        model_type='bpe', vocab_size=400, hard_vocab_limit=False, byte_fallback=True,
        normalization_rule_name='identity', add_dummy_prefix=False, remove_extra_whitespaces=False,
        minloglevel=2)
    return spm.SentencePieceProcessor(model_file=str(root / 'tokenizer.model'))


class ControlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.tok = tokenizer_fixture(cls.root)
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_split_and_counterfactuals(self):
        train, dev = control._bank(self.tok, False), control._bank(self.tok, True)
        self.assertEqual(dev, control._bank(self.tok, True))
        keys = control._symbols(self.tok, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        vals = control._symbols(self.tok, '0123456789abcdefghijklmnopqrstuvwxyz')
        def bindings(bank):
            result = set()
            for (first, target, foil), (second, reverse, foil2) in bank:
                self.assertEqual((target, foil), (foil2, reverse))
                self.assertNotEqual(target, foil)
                self.assertEqual(sum(x != y for x, y in zip(first, second)), 2)
                prompt = self.tok.decode(first)
                self.assertIn(self.tok.decode([target]), prompt)
                for tokens in (first, second):
                    # Literal words can contain raw symbol pieces, so count actual
                    # bindings using the counterfactual's two changed positions.
                    positions = [i for i in range(len(first)) if first[i] != second[i]]
                    for p in positions:
                        previous_key = next(k for k in reversed(tokens[:p]) if k in keys)
                        self.assertIn(tokens[p], vals)
                        result.add((previous_key, tokens[p]))
            return result
        self.assertFalse(bindings(train) & bindings(dev))

    def test_dense_is_causal(self):
        torch.manual_seed(123)
        m = control.DenseReference(self.tok.vocab_size()).eval()
        ids = torch.tensor([[15, 16, 17], [15, 16, 20]])
        with torch.inference_mode():
            scores = m(ids)
        torch.testing.assert_close(scores[0, :2], scores[1, :2])

    def test_interruption_is_equivalent(self):
        run = control.run
        stop = self.root/'STOP'
        with patch.multiple(control, STEPS=3, BATCH_PAIRS=2, DEV_PAIRS=8, SAVE_EVERY=2):
            complete = run(self.tok, self.root/'complete', 'cpu', stop_path=stop)
            original_step = torch.optim.AdamW.step
            def stop_after_step(optimizer, *args, **kwargs):
                result = original_step(optimizer, *args, **kwargs)
                stop.touch()
                return result
            with patch.object(torch.optim.AdamW, 'step', new=stop_after_step):
                partial = run(self.tok, self.root/'resume', 'cpu', stop_path=stop)
                self.assertEqual(partial['status'], 'stopped')
                self.assertEqual(partial['arms']['dense']['updates'], 1)
            stop.unlink()
            resumed = run(self.root/'tokenizer.model', self.root/'resume', 'cpu', stop_path=stop)
            self.assertEqual(complete, resumed)
            self.assertEqual(resumed, run(self.tok, self.root/'resume', 'cpu', stop_path=stop))
            for name in control.ARMS:
                a = torch.load(self.root/'complete'/f'{name}.pt', weights_only=True)['model']
                b = torch.load(self.root/'resume'/f'{name}.pt', weights_only=True)['model']
                self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
            self.assertEqual(resumed['arms']['dense']['supervised_tokens'], 12)
            self.assertEqual(resumed['arms']['dense']['context_tokens'], resumed['arms']['phistate']['context_tokens'])
            with patch.object(control, 'STEPS', 4), self.assertRaisesRegex(RuntimeError, 'manifest changed'):
                run(self.tok, self.root/'resume', 'cpu', stop_path=stop)
