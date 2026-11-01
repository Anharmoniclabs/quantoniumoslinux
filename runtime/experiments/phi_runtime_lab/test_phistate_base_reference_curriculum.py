import unittest

import torch

from phistate_base_reference import _probe, _synthetic_windows
from phistate_base_reference_curriculum import make_data


class Symbols:
    def encode(self, s):
        return list(map(ord, s))

    def decode(self, ids):
        return ''.join(map(chr, ids))

    def eos_id(self):
        return 2


class ScoringFixture(torch.nn.Module):
    """Test the scorer with a known output under a changed prefix."""
    def forward(self, ids):
        logits = torch.zeros(ids.shape[0], ids.shape[1], 128)
        logits[torch.arange(len(ids)), -2, ids[:, 2]] = 12.
        return logits


class CurriculumTests(unittest.TestCase):
    def test_fresh_exact_membership_and_correct_label_positions(self):
        tok = Symbols()
        new, old = make_data(tok), _synthetic_windows(tok)
        as_set = lambda t: set(map(tuple,t.tolist()))
        pilot = as_set(old['train']) | as_set(old['verify'])
        self.assertFalse(as_set(new['train']) & as_set(new['verify']))
        for split in new.values():
            self.assertFalse(as_set(split) & pilot)
            self.assertTrue(split[:,2].equal(split[:,-2]))
            self.assertTrue((split[:,-3] == ord('?')).all())

    def test_context_swap_actually_changes_correct_answer(self):
        data = make_data(Symbols())['verify'][:32]
        score = _probe(ScoringFixture(), data)
        self.assertEqual(score['value_accuracy'], 1.)
        self.assertEqual(score['accuracy_after_changing_only_earlier_value'], 0.)


if __name__=='__main__':
    unittest.main()
