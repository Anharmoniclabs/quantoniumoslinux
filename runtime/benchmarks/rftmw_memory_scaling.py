# SPDX-License-Identifier: AGPL-3.0-or-later
"""
RFTMW memory-layer scalability benchmark (offline, reproducible).

Measures:
- ingest (compression) time
- on-demand decompression time (get_weight)
- compression ratio
- reconstruction error

This benchmark is designed to support the "reduced datacenter compute" narrative
with concrete latency/RAM/ratio numbers on commodity hardware, without requiring
network downloads.

Run:
  python benchmarks/rftmw_memory_scaling.py --impl canonical
  python benchmarks/rftmw_memory_scaling.py --impl hybrid
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np

from quantonium_os_src.engine.rftmw_memory import RFTMWMemoryLayer


class _NumpyParam:
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def numel(self):
        return self._arr.size

    def element_size(self):
        return self._arr.dtype.itemsize


class SyntheticModel:
    """Duck-typed model with named_parameters() returning numpy-backed params."""

    def __init__(self, *, shapes: Dict[str, Tuple[int, ...]], seed: int = 0):
        rng = np.random.default_rng(seed)
        self._params: Dict[str, np.ndarray] = {}
        for name, shape in shapes.items():
            # Mix of structured-ish and random tensors.
            n = int(np.prod(shape))
            if "embed" in name or "wte" in name:
                t = np.arange(n, dtype=np.float64) / max(1, n)
                sig = np.zeros(n, dtype=np.float64)
                phi = (1.0 + np.sqrt(5.0)) / 2.0
                for k in range(1, 6):
                    sig += np.sin(2 * np.pi * (phi**k) * t) / k
                arr = sig.reshape(shape).astype(np.float32)
            else:
                arr = rng.standard_normal(n).reshape(shape).astype(np.float32)
            self._params[name] = arr

    def named_parameters(self):
        for name, w in self._params.items():
            yield name, _NumpyParam(w)


@dataclass
class Row:
    name: str
    numel: int
    original_bytes: int
    compressed_bytes: int
    method: str
    spectral_entropy: float
    reconstruction_error: float
    decompress_seconds: float


@dataclass
class Report:
    impl: str
    seed: int
    ingest_seconds: float
    total_original_bytes: int
    total_compressed_bytes: int
    compression_ratio: float
    rows: List[Row]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["canonical", "hybrid"], default="canonical")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    os.environ["QUANTONIUM_RFTMW_RFT_IMPL"] = args.impl

    # Transformer-ish shapes (kept small enough for laptop runs).
    shapes = {
        "transformer.wte.weight": (2048, 256),
        "transformer.h.0.attn.c_proj.weight": (256, 256),
        "transformer.h.0.mlp.c_fc.weight": (256, 1024),
        "transformer.h.0.ln_1.bias": (256,),
    }
    model = SyntheticModel(shapes=shapes, seed=args.seed)

    # Keep the cap high enough that we can exercise the RFT path in this benchmark.
    mem = RFTMWMemoryLayer(max_rft_elements=1_000_000)

    t0 = time.time()
    mem.ingest_model(model, verbose=False)
    ingest_s = time.time() - t0

    rows: List[Row] = []
    for name, param in model.named_parameters():
        slot = mem._weights[name]  # internal but stable for benchmarking
        t1 = time.time()
        restored = mem.get_weight(name)
        dt = time.time() - t1
        # sanity check: touch to avoid lazy optimizations
        _ = float(np.sum(restored[: min(8, restored.size)]))
        rows.append(
            Row(
                name=name,
                numel=int(param.numel()),
                original_bytes=int(param.numel() * param.element_size()),
                compressed_bytes=int(slot.compressed_bytes),
                method=str(slot.method.name),
                spectral_entropy=float(slot.spectral_entropy),
                reconstruction_error=float(slot.reconstruction_error),
                decompress_seconds=float(dt),
            )
        )

    total_orig = int(mem._report.total_original_bytes)
    total_comp = int(mem._report.total_compressed_bytes)
    ratio = float(total_comp / max(1, total_orig))

    report = Report(
        impl=args.impl,
        seed=int(args.seed),
        ingest_seconds=float(ingest_s),
        total_original_bytes=total_orig,
        total_compressed_bytes=total_comp,
        compression_ratio=ratio,
        rows=rows,
    )

    os.makedirs("results", exist_ok=True)
    out_path = args.out.strip() or os.path.join("results", f"rftmw_memory_scaling_{args.impl}_{int(time.time())}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, sort_keys=True)

    print(f"Wrote: {out_path}")
    print(f"ingest_seconds={ingest_s:.3f}  ratio={ratio:.3f}  orig={total_orig/1e6:.1f}MB  comp={total_comp/1e6:.1f}MB")
    for r in rows:
        print(f"- {r.name:40s} {r.method:10s} err={r.reconstruction_error:.3f} decomp={r.decompress_seconds*1e3:.1f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
