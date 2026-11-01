"""Offline updates use a CPU fixture; exported run_arm always requires a real A100."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from moe_model import PhiStateMoELM, config_for
from torch import nn

import phistate_base_cuda_train as train


class Tiny(nn.Module):
    def __init__(self, vocab=18):
        super().__init__()
        self.embed = nn.Embedding(vocab, 16)
        self.readout = nn.Linear(16, vocab)

    def forward(self, ids):
        return self.readout(self.embed(ids))


class TrainTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.x = torch.tensor([[2, 3, 4, 5], [5, 7, 9, 2]])
        self.y = torch.tensor([[3, 4, 5, 1], [7, 9, 2, 4]])

    def tearDown(self):
        torch.set_num_threads(self.threads)
        self.tmp.cleanup()

    def _run(self, out, *, arm="dense", stop=None, halt=False, warm_source=None):
        out.mkdir(exist_ok=True)
        torch.manual_seed(17)
        m = (
            PhiStateMoELM(config_for("mini", 256, mode="dft", width=16, layers=1, window=8))
            if arm == "phistate"
            else Tiny()
        )
        device = torch.device("cpu")
        opt = train._make_optimizer(m, arm, warm_source, device)
        if warm_source:
            m.load_state_dict(warm_source["state_dict"])
        req = {
            "steps": 4,
            "batch": 2,
            "context": 4,
            "arm": arm,
            "seed": 17,
            "gpu": "fixture",
            "data_fingerprint": "fixture",
            "model_config": vars(m.cfg) if arm == "phistate" else {"fixture": True},
        }

        def evaluate(model, corpus, validation, device):
            with torch.no_grad():
                loss = torch.nn.functional.cross_entropy(model(self.x).flatten(0, 1), self.y.flatten())
            if halt and (out / "latest.pt").is_file():
                stop.touch()
            return {"mean_bpt": float(loss) / 0.69314718056}

        meta = {"step": 66000, "trained_tokens": 33792000} if warm_source else {}
        with (
            patch.object(train.windows, "materialize", return_value=(self.x, self.y)),
            patch.object(train.windows, "evaluate", side_effect=evaluate),
        ):
            return train._train(
                m,
                opt,
                None,
                {"steps": [["fixture"] * 2] * 4},
                {},
                out,
                req,
                device,
                meta,
                stop,
                lambda: None,
            )

    def test_stop_resume_matches_complete_and_keeps_parent(self):
        stop = self.root / "STOP"
        whole = self.root / "whole"
        resumed = self.root / "resume"
        expected = self._run(whole)
        self.assertEqual(self._run(resumed, stop=stop, halt=True)["status"], "stopped")
        stop.unlink()
        self.assertEqual(self._run(resumed, stop=stop)["curve"], expected["curve"])
        a = torch.load(whole / "latest.pt", weights_only=True)
        b = torch.load(resumed / "latest.pt", weights_only=True)
        self.assertEqual(a["history"], b["history"])
        self.assertEqual(a["phase_trained_tokens"], 32)
        for key, t in a["state_dict"].items():
            self.assertTrue(torch.equal(t, b["state_dict"][key]))
        self.assertGreater(expected["curve"][-1]["weight_update"]["all"]["relative_l2"], 0)
        self.assertEqual(expected["device_receipt"]["optimizer_moment_device"], "cpu")
        self.assertEqual([r["step"] for r in expected["curve"]], [0, 2, 4])
        self.assertEqual(self._run(whole), expected)

    def test_phistate_router_and_saved_adafactor(self):
        phi = self._run(self.root / "phi", arm="phistate")
        self.assertGreater(phi["curve"][-1]["weight_update"]["groups"]["router"]["relative_l2"], 0)
        torch.manual_seed(41)
        model = Tiny()
        opt = torch.optim.Adafactor(model.parameters(), lr=0.0003, foreach=False)
        model(self.x).sum().backward()
        opt.step()
        source = {"optimizer_state": opt.state_dict(), "state_dict": model.state_dict()}
        info = self._run(self.root / "warm", arm="warm", warm_source=source)
        s = torch.load(info["latest"], weights_only=True)
        self.assertEqual(s["step"], 66004)
        self.assertEqual(s["trained_tokens"], 33792032)
        self.assertTrue(all(int(v["step"]) == 5 for v in s["optimizer_state"]["state"].values()))
        self.assertEqual(s["format"], "PhiStateMoE-v1")

    def test_a100_guard_before_data_or_writes(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            self.assertRaisesRegex(RuntimeError, "A100"),
        ):
            train.run_arm(
                self.root / "source",
                None,
                self.root / "out",
                arm="dense",
                seed=1,
                plan={},
                validation={},
            )
        self.assertFalse((self.root / "out").exists())
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "get_device_name", return_value="T4"),
            self.assertRaisesRegex(RuntimeError, "A100"),
        ):
            train.require_a100()


if __name__ == "__main__":
    unittest.main()
