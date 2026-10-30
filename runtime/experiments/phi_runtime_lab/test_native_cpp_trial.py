"""Numerical and format checks for the RFTMW PhiState adapter."""
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch

from moe_model import PhiStateMoELM, config_for
from native_cpp_trial import compile_native, export_model, native_run, parity


def main():
    torch.set_num_threads(2)
    os.environ["RFTMW_THREADS"] = "2"
    results = []
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        binary, weights = root / "native", root / "model.phimoe"
        compile_native(binary)
        for mode, topk, shared in (("dft", 2, 1), ("phi", 2, 1), ("local", 2, 1),
                                   ("dft", 1, 0), ("phi", 1, 1), ("local", 4, 0)):
            torch.manual_seed(100 + topk)
            model = PhiStateMoELM(config_for("mini", 256, mode=mode, top_k=topk, shared_ratio=shared))
            # Exercise nonidentity mixers, imaginary readouts, attention biases,
            # and learned routing rather than relying on initialized constants.
            with torch.no_grad():
                for b in model.blocks:
                    b.attn.bias.normal_(0, 0.2)
                    if b.memory is not None:
                        b.memory.phase_delta.normal_(0, 0.2)
                        b.memory.read_im.normal_(0, 0.1)
                        b.mixer.gain_raw.normal_(0, 0.5)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            for _ in range(3):
                x = torch.randint(256, (2, 24))
                y = model(x)
                penalty, _ = model.router_penalty()
                loss = torch.nn.functional.cross_entropy(y.reshape(-1, 256), x.roll(-1, 1).reshape(-1)) + penalty
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            export_model(model, weights)
            ids = torch.randint(256, (129,)).tolist()
            actual, stats = native_run(binary, weights, ids, 256, root, 8)
            result = parity(model, ids, actual)
            assert result["passed"], (mode, topk, shared, result)
            trajectory = ids + stats["generated_ids"]
            actual, _ = native_run(binary, weights, trajectory, 256, root)
            assert parity(model, trajectory, actual)["passed"]
            results.append({"mode": mode, "topk": topk, "shared": shared, **result})
            print(results[-1], flush=True)

        # Test the user's reported architecture size, with random local weights.
        # This is explicitly not a test of the user's inaccessible Drive checkpoint.
        torch.manual_seed(20261030)
        model = PhiStateMoELM(config_for("35m", 8192, mode="dft")).eval()
        export_model(model, weights)
        ids = torch.randint(8192, (128,)).tolist()
        actual, _ = native_run(binary, weights, ids, 8192, root)
        result = parity(model, ids, actual)
        assert result["passed"], result
        print({"profile": "35m", "parameters": sum(p.numel() for p in model.parameters()), **result}, flush=True)

        invalid = root / "invalid.phimoe"
        invalid.write_bytes(b"PHIMOE1\n")
        idsfile = root / "bad_tokens.u32"
        np.asarray([8192], dtype="<u4").tofile(idsfile)
        for bad_weights in (invalid, weights):
            p = subprocess.run([str(binary), str(bad_weights), str(idsfile), str(root / "out"), "0", "-1"],
                               capture_output=True, text=True)
            assert p.returncode != 0, "Invalid input accepted"
        print("PASS: six trained mini variants, 38,287,392-parameter random model, invalid input checks", flush=True)


if __name__ == "__main__":
    main()
