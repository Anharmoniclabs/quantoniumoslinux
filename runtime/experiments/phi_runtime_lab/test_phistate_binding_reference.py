import unittest

import torch

from phistate_binding_data import cases, make_bank, wording
from phistate_binding_reference import Dense, LearnedSlots, _prepared


class ReferenceTests(unittest.TestCase):
    def test_split_edges_and_factorials(self):
        bank = make_bank()
        edges = {}
        for split, count in [("train", 24), ("dev", 16)]:
            self.assertEqual(len(bank[split]), count)
            edges[split] = set()
            prompts = set()
            for group in bank[split]:
                rows = cases(group)
                self.assertEqual(len(rows), 4)
                self.assertEqual(len({r["query"] for r in rows}), 2)
                self.assertEqual(len({r["value1"] for r in rows}), 2)
                for r in rows:
                    edges[split].update(((r["key1"], r["value1"]), (r["key2"], r["value2"])))
                    self.assertEqual(
                        r["answer"], dict(zip(group["keys"], [r["value1"], r["value2"]]))[r["query"]]
                    )
                    self.assertNotEqual(r["answer"], r["foil"])
                    for regime in ("familiar", "new"):
                        self.assertNotIn(wording(r, regime), prompts)
                        prompts.add(wording(r, regime))
            self.assertEqual(len(edges[split]), 32)
        self.assertFalse(edges["train"] & edges["dev"])

    def test_reference_inputs_and_decisions(self):
        pools, vocab, copies = _prepared()
        self.assertEqual(len(copies["dev"][0]), 32)
        x, slots, gold, foil = pools["dev", "familiar"]
        self.assertEqual((len(x), len(slots)), (64, 64))
        model = Dense(vocab)
        # The generic model sees only text tokens; passing corrupted parsed slots cannot influence it.
        with torch.inference_mode():
            self.assertTrue(torch.equal(model(x, slots), model(x, torch.zeros_like(slots))))
        parsed = LearnedSlots()
        with torch.inference_mode():
            a = parsed(x, slots)
            b = parsed(torch.ones_like(x), slots)
            self.assertTrue(torch.equal(a, b))
            self.assertEqual(a.shape, (64, 10))
        self.assertTrue((gold != foil).all())


if __name__ == "__main__":
    unittest.main()
