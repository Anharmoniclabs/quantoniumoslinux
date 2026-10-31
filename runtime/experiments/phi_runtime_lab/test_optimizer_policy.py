"""Behavior checks for opt-in grouping; no pretrained checkpoint is changed."""
import copy
import math
import unittest

import torch
from torch import nn

from optimizer_policy import protect_structured_parameters
from phistate import GoldenState, LocalAttention, RMSNorm, CanonicalChannelMixer


class Fixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory = GoldenState(8, "dft")
        self.attn = LocalAttention(8, 2, 4)
        self.norm = RMSNorm(8)
        self.mixer = CanonicalChannelMixer(4, "dft")
        self.linear = nn.Linear(8, 8)


def make_optimizer(kind, model):
    return kind(model.parameters(), lr=.01, weight_decay=.1, foreach=False)


def half_life(model):
    return math.log(2) / (torch.nn.functional.softplus(model.memory.decay_raw.detach()) + 1e-4)


class OptimizerPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(20261031)

    def kinds(self):
        return (torch.optim.AdamW, torch.optim.Adafactor)

    def test_zero_gradient_decay_preserves_memory_and_norms(self):
        for kind in self.kinds():
            with self.subTest(optimizer=kind.__name__):
                old = Fixture()
                candidate = copy.deepcopy(old)
                old_opt, new_opt = make_optimizer(kind, old), make_optimizer(kind, candidate)
                before = half_life(candidate).clone()
                matrix = candidate.linear.weight.detach().clone()
                norm = candidate.norm.weight.detach().clone()
                protect_structured_parameters(new_opt, candidate)
                for model, opt in ((old, old_opt), (candidate, new_opt)):
                    for p in model.parameters():
                        p.grad = torch.zeros_like(p)
                    opt.step()
                self.assertTrue(torch.equal(half_life(candidate), before))
                self.assertTrue(bool((half_life(old) < before).all()))
                self.assertTrue(torch.equal(candidate.norm.weight, norm))
                self.assertTrue(torch.allclose(candidate.linear.weight, matrix * .999))

    def test_moments_counters_options_and_parameter_identity_preserved(self):
        for kind in self.kinds():
            with self.subTest(optimizer=kind.__name__):
                model = Fixture()
                opt = make_optimizer(kind, model)
                opt.param_groups[0]["param_names"] = [name for name, _ in model.named_parameters()]
                for p in model.parameters():
                    p.grad = torch.full_like(p, .2)
                opt.step()
                state = opt.state
                values = {id(p): copy.deepcopy(s) for p, s in state.items()}
                options = {k: v for k, v in opt.param_groups[0].items()
                           if k not in ("params", "weight_decay", "param_names")}
                identities = {id(p) for p in model.parameters()}
                protect_structured_parameters(opt, model)
                self.assertIs(opt.state, state)
                self.assertEqual({id(p) for g in opt.param_groups for p in g["params"]}, identities)
                for p, current in opt.state.items():
                    for key, value in current.items():
                        prior = values[id(p)][key]
                        self.assertTrue(torch.equal(value, prior) if torch.is_tensor(value) else value == prior)
                named = dict(model.named_parameters())
                for group in opt.param_groups:
                    for name, p in zip(group["param_names"], group["params"]):
                        self.assertIs(named[name], p)
                    for key, value in options.items():
                        self.assertEqual(group[key], value)

    def test_converted_state_reload_has_identical_next_update(self):
        for kind in self.kinds():
            with self.subTest(optimizer=kind.__name__):
                model = Fixture()
                opt = make_optimizer(kind, model)
                for p in model.parameters():
                    p.grad = torch.full_like(p, .2)
                opt.step()
                protect_structured_parameters(opt, model)
                restored = copy.deepcopy(model)
                restored_opt = make_optimizer(kind, restored)
                protect_structured_parameters(restored_opt, restored)
                restored_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
                for m, o in ((model, opt), (restored, restored_opt)):
                    for p in m.parameters():
                        p.grad = torch.full_like(p, .1)
                    o.step()
                for a, b in zip(model.parameters(), restored.parameters()):
                    self.assertTrue(torch.equal(a, b))

    def test_predictive_gradient_still_updates_protected_memory(self):
        for kind in self.kinds():
            with self.subTest(optimizer=kind.__name__):
                model = Fixture()
                opt = make_optimizer(kind, model)
                protect_structured_parameters(opt, model)
                before = model.memory.decay_raw.detach().clone()
                x = torch.randn(2, 12, 8)
                target = torch.randn_like(x)
                (model.memory(x) - target).square().mean().backward()
                self.assertGreater(float(model.memory.decay_raw.grad.abs().max()), 0.)
                opt.step()
                self.assertFalse(torch.equal(before, model.memory.decay_raw))

    def test_two_dimensional_attention_bias_is_protected_and_policy_idempotent(self):
        model = Fixture()
        opt = make_optimizer(torch.optim.AdamW, model)
        protect_structured_parameters(opt, model)
        original = [(g["weight_decay"], [id(p) for p in g["params"]]) for g in opt.param_groups]
        bias_group = next(g for g in opt.param_groups if any(p is model.attn.bias for p in g["params"]))
        self.assertEqual(bias_group["weight_decay"], 0.)
        protect_structured_parameters(opt, model)
        self.assertEqual(original, [(g["weight_decay"], [id(p) for p in g["params"]]) for g in opt.param_groups])

    def test_foreign_parameter_rejected_without_mutation(self):
        model = Fixture()
        foreign = nn.Parameter(torch.ones(2, 2))
        opt = torch.optim.AdamW([{"params": list(model.parameters())}, {"params": [foreign]}])
        groups = list(opt.param_groups)
        with self.assertRaisesRegex(ValueError, "outside"):
            protect_structured_parameters(opt, model)
        self.assertEqual(len(groups), len(opt.param_groups))
        for a, b in zip(groups, opt.param_groups):
            self.assertIs(a, b)


if __name__ == "__main__":
    unittest.main()
