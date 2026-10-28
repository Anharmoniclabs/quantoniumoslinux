#!/usr/bin/env python3
"""End-to-end CUDA training/validation/checkpoint/resume smoke test.

Uses only disposable synthetic tokens and a tiny freshly initialized MoE.
Does not change or inspect private user checkpoints, corpora, or tokenizer.
A passing run demonstrates device-aware real training loop compatibility;
it does not measure language quality or throughput on a real dataset.
"""
from pathlib import Path
import json
import subprocess
import sys
import tempfile

import numpy as np
import torch


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; select a Colab GPU runtime.")
    from moe_train import ckpt_load
    with tempfile.TemporaryDirectory(prefix="phistate_cuda_smoke_") as scratch:
        base = Path(scratch)
        data = base / "dataset"
        data.mkdir()
        bins = {}
        rng = np.random.default_rng(1729)
        for name in ("train_code", "train_general", "val_code", "val_general"):
            filename = f"{name}.bin"
            values = rng.integers(0, 256, size=1024, dtype=np.uint16)
            values.tofile(data / filename)
            bins[name] = {"file": filename, "tokens": len(values)}
        manifest = {
            "format": "PhiMoECorpus-v1",
            "dtype": "uint16",
            "vocab": 256,
            "tokenizer": "raw",
            "data_fingerprint": "CUDA_SMOKE_SYNTHETIC_NOT_FOR_PRETRAINING",
            "bins": bins,
        }
        (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        checkpoint = base / "smoke.pt"
        common = [
            sys.executable, str(Path(__file__).with_name("moe_train.py")), "train",
            "--data-dir", str(data), "--out", str(checkpoint),
            "--profile", "mini", "--mode", "dft", "--mix-basis", "dft",
            "--context", "32", "--batch", "2", "--accum", "1",
            "--eval-batch", "2", "--eval-batches", "1",
            "--eval-every", "1", "--optimizer", "adamw",
            "--device", "cuda", "--threads", "2", "--save-optimizer",
        ]
        print("CUDA SMOKE: fresh tiny model, synthetic training, CUDA validation", flush=True)
        subprocess.run(common + ["--steps", "2"], check=True)
        latest = base / "smoke.last.pt"
        model, state = ckpt_load(str(latest))
        assert state["step"] == 2 and "optimizer_state" in state
        assert next(model.parameters()).device.type == "cpu"
        print("CUDA SMOKE: reload optimizer state and resume on CUDA", flush=True)
        subprocess.run(common + ["--steps", "3", "--resume", str(latest)], check=True)
        _, resumed = ckpt_load(str(latest))
        assert resumed["step"] == 3 and "optimizer_state" in resumed
        assert resumed["data_fingerprint"] == manifest["data_fingerprint"]
        print("CUDA TRAIN/VALIDATE/CHECKPOINT/RESUME SMOKE PASS",flush=True)
        print("Synthetic smoke only: do not use these weights or metrics as real model evidence.",flush=True)


if __name__ == "__main__":
    main()
