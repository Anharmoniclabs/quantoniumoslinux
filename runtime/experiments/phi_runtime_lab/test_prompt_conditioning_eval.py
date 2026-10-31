"""Focused regressions for masked targets, comparison coverage and report caching."""
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
import chat_recovery_data as data
import prompt_conditioning_eval as audit
from test_chat_recovery import fixture


class PositionModel(torch.nn.Module):
    def forward(self, x):
        # Each input ID is the uniquely favored next token.
        return torch.nn.functional.one_hot(x, num_classes=16).float()*4


class ConditioningTests(unittest.TestCase):
    def test_alignment_uses_final_prompt_position_and_every_target(self):
        model = PositionModel()
        result = audit._score(model, [15, 1, 6], [6, 6, 6, 6, 6, 6, 6, 6, 1], 'cpu')
        good = math.log2(math.exp(4) + 15) - 4/math.log(2)
        bad = math.log2(math.exp(4) + 15)
        self.assertAlmostEqual(result['first8_bpt'], good, places=5)
        self.assertAlmostEqual(result['answer_bpt'], (8*good+bad)/9, places=5)
        wrong = audit._score(model, [6, 6, 1], [6, 6, 6, 6, 6, 6, 6, 6, 1], 'cpu')
        self.assertGreater(wrong['answer_bpt'], result['answer_bpt'])

    def test_real_model_pairing_no_truncation_and_cache(self):
        torch.set_num_threads(2)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer, checkpoint = fixture(root)
            # Build validation-only input. The diagnostic must never require train or final data.
            sft = root/'prepared'
            sft.mkdir()
            (sft/'tokenizer.model').write_bytes((root/'tokenizer.model').read_bytes())
            rows = []
            for source in data.SOURCES:
                for i in range(3):
                    messages = [dict(role='user', content=f'First turn {i}'), dict(role='assistant', content='A first reply.'),
                                dict(role='user', content=f'Second turn {source} {i}?'),
                                dict(role='assistant', content=f'Example {i} uses numbers and Python.')]
                    x, y = data.encode_example(messages, tokenizer, 256)
                    rows.append(dict(id=data.digest(data.canonical(messages)), split='val', source=source,
                                     input_ids=x, labels=y, messages=messages))
            (sft/'val.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            token_hash = data.sha(sft/'tokenizer.model')
            manifest = dict(context=256, template=data.TEMPLATE, tokenizer_sha256=token_hash,
                            base_data_fingerprint='synthetic-repaired-fixture', files={'val.jsonl':data.sha(sft/'val.jsonl')})
            manifest['data_fingerprint'] = data.digest(data.canonical(manifest))
            (sft/'manifest.json').write_text(json.dumps(manifest))
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            state.update(sft_data_fingerprint=manifest['data_fingerprint'], chat_template=data.TEMPLATE,
                         tokenizer_sha256=token_hash, sft_steps=1)
            torch.save(state, checkpoint)
            checkpoint_hash = data.sha(checkpoint)
            rng = torch.get_rng_state().clone()
            report = audit.run(checkpoint, sft, root/'reports', limit=3)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(data.sha(checkpoint), checkpoint_hash)
            for source in data.SOURCES:
                group = report['per_source'][source]
                self.assertEqual(group['scored'], 3)
                self.assertEqual({r['id'] for r in group['examples']}, {r['donor_id'] for r in group['examples']})
                for result in group['examples']:
                    self.assertNotEqual(result['id'], result['donor_id'])
                    self.assertEqual((result['correct_turns'], result['wrong_turns']), (3, 3))
                    row = next(r for r in rows if r['id'] == result['id'])
                    targets = [y for y in row['labels'] if y != -100]
                    self.assertEqual(result['target_sha256'], data.digest(data.canonical(targets)))
                self.assertEqual(group['comparisons']['wrong_minus_correct_answer_bpt']['examples'], 3)
            with patch.object(audit, 'ckpt_load', side_effect=AssertionError('should use cached report')):
                self.assertEqual(audit.run(checkpoint, sft, root/'reports', limit=3), report)
            # Make every alternative prompt infeasible. Equality of the target sequences
            # must not be achieved by silently truncating the reference answer.
            prepared = [audit._targets(r, tokenizer, 256) for r in rows[:3]]
            limited = audit._pair(prepared, tokenizer, context=2)
            self.assertEqual(limited[0], {})
            (sft/'val.jsonl').write_text((sft/'val.jsonl').read_text()+'\n')
            with self.assertRaisesRegex(ValueError, 'validation rows or tokenizer changed'):
                audit.run(checkpoint, sft, root/'reports', limit=3)


if __name__ == '__main__':
    unittest.main()
