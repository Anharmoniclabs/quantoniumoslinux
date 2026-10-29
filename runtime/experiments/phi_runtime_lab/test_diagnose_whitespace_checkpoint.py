import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import sentencepiece as spm
import torch

import diagnose_whitespace_checkpoint as diagnostic
from moe_model import PhiStateMoELM, config_for
from repair_whitespace_corpus import sha


class DiagnosticTests(unittest.TestCase):
    def test_whitespace_classification(self):
        for piece in ('▁', '▁▁', '<0x09>', '<0x0A>', '<0x20>'):
            self.assertTrue(diagnostic.whitespace_piece(piece), piece)
        for piece in ('▁return', '<|code|>', '<0x41>', '<0xA0>', ''):
            self.assertFalse(diagnostic.whitespace_piece(piece), piece)

    def test_parity_detects_broken_step(self):
        torch.set_num_threads(2)
        model = PhiStateMoELM(config_for('mini', 256)).eval()
        self.assertTrue(diagnostic.parity(model, list(range(16)))['allclose_atol_1e_3_rtol_1e_4'])
        step = model.step
        def broken(*args):
            logits, state = step(*args)
            return logits + 1, state
        model.step = broken
        self.assertFalse(diagnostic.parity(model, list(range(16)))['allclose_atol_1e_3_rtol_1e_4'])

    def test_complete_run_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            text = root / 'text.txt'
            text.write_text('def add(a, b):\n    return a + b\nThe operating system manages memory.\n' * 20)
            spm.SentencePieceTrainer.train(input=str(text), model_prefix=str(root / 'tokenizer'),
                vocab_size=300, model_type='bpe', byte_fallback=True, hard_vocab_limit=False,
                user_defined_symbols=['<|code|>', '<|general|>'], minloglevel=2)
            tok = spm.SentencePieceProcessor(model_file=str(root / 'tokenizer.model'))
            tokens = np.array(tok.encode(text.read_text()), dtype=np.uint16)
            bins = {}
            for domain in ('code', 'general'):
                file = root / f'val_{domain}.bin'
                tokens.tofile(file)
                bins[f'val_{domain}'] = {'file': file.name, 'sha256': sha(file)}
            (root / 'manifest.json').write_text(json.dumps({'data_fingerprint': 'fixture',
                'dtype': 'uint16', 'bins': bins}))
            model = PhiStateMoELM(config_for('mini', tok.vocab_size()))
            checkpoint = root / 'latest.pt'
            torch.save({'step': 61000, 'config': vars(model.cfg), 'state_dict': model.state_dict(),
                        'train_args': {'context': 16}, 'data_fingerprint': 'fixture',
                        'corpus_migration': {'tokenizer_sha256': sha(root / 'tokenizer.model')}}, checkpoint)
            before = sha(checkpoint)
            diagnostic.run(SimpleNamespace(checkpoint=checkpoint, data_dir=root,
                                           expected_step=61000, device='cpu', windows=2))
            self.assertEqual(before, sha(checkpoint))
            report = json.loads(next(root.glob('diagnostic_*.json')).read_text())
            self.assertTrue(report['checkpoint_unchanged'])
            for row in report['validation'].values():
                self.assertEqual(row['all']['tokens'], 32)
                self.assertEqual(row['whitespace_only']['tokens'] + row['other']['tokens'], 32)
                weighted = sum(row[k]['tokens'] * (row[k]['bpt'] or 0)
                               for k in ('whitespace_only', 'other')) / 32
                self.assertAlmostEqual(weighted, row['all']['bpt'], places=5)


if __name__ == '__main__':
    unittest.main()
