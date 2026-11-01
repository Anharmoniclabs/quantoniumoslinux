import unittest

import torch

from phistate_base_reference import DenseCausalLM


class DenseReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)
        self.model = DenseCausalLM(19, width=16, layers=2, heads=2, context=8, hidden=32)

    def test_causality_and_preceding_context(self):
        ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 9, 7], [6, 8, 1, 4, 5]])
        logits = self.model(ids)
        self.assertEqual(logits.shape, (3, 5, 19))
        torch.testing.assert_close(logits[0, :3], logits[1, :3], atol=1e-6, rtol=1e-5)
        self.assertFalse(torch.allclose(logits[0, -1], logits[2, -1]))
        torch.testing.assert_close(logits[0, 2], self.model(ids[:1, :3])[0, -1], atol=1e-6, rtol=1e-5)

    def test_shape_and_tied_parameter_count(self):
        w, h, n = 16, 32, 2
        self.assertEqual(self.model.parameter_count(), 19*w+n*(4*w*w+3*w*h+2*w)+w)
        with self.assertRaises(ValueError):
            self.model(torch.zeros((1, 9), dtype=torch.long))

    def test_causal_shift_backpropagates_all_layers(self):
        x = torch.randint(19, (4, 6))
        logits = self.model(x[:, :-1])
        torch.nn.functional.cross_entropy(logits.transpose(1, 2), x[:, 1:]).backward()
        for p in self.model.parameters():
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
