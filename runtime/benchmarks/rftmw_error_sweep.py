# SPDX-License-Identifier: AGPL-3.0-or-later
"""
RFTMW error-sweep benchmark (offline, reproducible).

Purpose (Track B):
- Treat Hybrid FFT/RFT as the production operator.
- Show bounded reconstruction error as a function of keep_ratio (and quantization),
  on both structured and unstructured signals.

Run:
  python benchmarks/rftmw_error_sweep.py --impl hybrid
  python benchmarks/rftmw_error_sweep.py --impl canonical
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import List

import numpy as np

from quantonium_os_src.engine.rftmw_memory import _compress_rft, _decompress_rft_blob


PHI = (1.0 + np.sqrt(5.0)) / 2.0


def _phi_periodic_signal(n: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / n
    sig = np.zeros(n, dtype=np.float64)
    for k in range(1, 6):
        sig += np.sin(2 * np.pi * (PHI**k) * t) / k
    sig = sig / (np.linalg.norm(sig) + 1e-15)
    return sig.astype(np.float64)


def _random_signal(n: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sig = rng.standard_normal(n).astype(np.float64)
    sig = sig / (np.linalg.norm(sig) + 1e-15)
    return sig


@dataclass
class Row:
    impl: str
    signal: str
    n: int
    keep_ratio: float
    mag_bits: int
    phase_bits: int
    compressed_bytes: int
    recon_rel_error: float


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["canonical", "hybrid"], default="hybrid")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    os.environ["QUANTONIUM_RFTMW_RFT_IMPL"] = args.impl

    keep_ratios = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00]
    rows: List[Row] = []

    for signal_name, x in [
        ("phi_periodic", _phi_periodic_signal(args.n)),
        ("random", _random_signal(args.n, seed=args.seed)),
    ]:
        for keep in keep_ratios:
            blob, _ = _compress_rft(
                x,
                keep_ratio=float(keep),
                mag_bits=12,
                phase_bits=10,
            )
            recon = _decompress_rft_blob(blob)[: args.n]
            err = float(np.linalg.norm(x - recon) / (np.linalg.norm(x) + 1e-15))
            rows.append(
                Row(
                    impl=args.impl,
                    signal=signal_name,
                    n=int(args.n),
                    keep_ratio=float(keep),
                    mag_bits=12,
                    phase_bits=10,
                    compressed_bytes=int(len(blob)),
                    recon_rel_error=err,
                )
            )

    report = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "impl": args.impl,
        "n": int(args.n),
        "seed": int(args.seed),
        "rows": [asdict(r) for r in rows],
    }

    os.makedirs("results", exist_ok=True)
    out_path = args.out.strip() or os.path.join("results", f"rftmw_error_sweep_{args.impl}_{int(time.time())}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

