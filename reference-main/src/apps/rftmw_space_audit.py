# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""RFTMW space audit for model parameter capacity planning.

Purpose
- Measure compressed footprint with RFTMW memory layer.
- Estimate how many parameters can fit into fixed memory budgets.
- Emit a reproducible JSON report for build decisions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

from transformers import AutoModelForCausalLM

# Support both:
# - python -m src.apps.rftmw_space_audit
# - python src/apps/rftmw_space_audit.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantonium_os_src.engine.rftmw_memory import RFTMWMemoryLayer


def parse_budgets(csv: str) -> List[float]:
    vals: List[float] = []
    for part in csv.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError("At least one memory budget is required.")
    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local model path")
    ap.add_argument("--out", default="ai/training/reports/rftmw_space_audit.json")
    ap.add_argument("--local_only", action="store_true")
    ap.add_argument("--layer_limit", type=int, default=None)
    ap.add_argument("--entropy_threshold", type=float, default=0.40)
    ap.add_argument("--keep_ratio", type=float, default=0.30)
    ap.add_argument("--max_rft_error", type=float, default=0.08)
    ap.add_argument(
        "--memory_budgets_gb",
        default="8,16,24,32,48,64,96,128",
        help="Comma-separated memory budgets in GB.",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    budgets_gb = parse_budgets(args.memory_budgets_gb)

    print(f"Loading model: {args.model}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=args.local_only,
        )
    except Exception as e:
        raise SystemExit(
            "Failed to load model for space audit. "
            "If you are offline/firewalled, pre-cache the model and pass --local_only. "
            f"Details: {type(e).__name__}: {e}"
        ) from e
    model.eval()

    param_count = int(sum(p.numel() for p in model.parameters()))
    fp32_bytes = int(sum(p.numel() * 4 for p in model.parameters()))

    mem = RFTMWMemoryLayer(
        entropy_threshold=args.entropy_threshold,
        weight_keep_ratio=args.keep_ratio,
        max_rft_error=args.max_rft_error,
    )
    report = mem.ingest_model(model, layer_limit=args.layer_limit, verbose=args.verbose)
    rows = mem.layer_report()

    compressed_bytes = int(report.total_compressed_bytes)
    ratio = float(fp32_bytes / max(compressed_bytes, 1))

    # Weighted average per-layer reconstruction error by original layer size.
    weighted_err_num = 0.0
    weighted_err_den = 0.0
    for r in rows:
        orig_mb = float(r.get("original_mb", 0.0))
        weight = orig_mb * 1024.0 * 1024.0
        weighted_err_num += float(r.get("reconstruction_error", 0.0)) * weight
        weighted_err_den += weight
    weighted_err = float(weighted_err_num / max(weighted_err_den, 1.0))

    capacity = []
    for gb in budgets_gb:
        budget_bytes = int(gb * (1024 ** 3))
        fp32_fit = int(param_count * (budget_bytes / max(fp32_bytes, 1)))
        rftmw_fit = int(param_count * (budget_bytes / max(compressed_bytes, 1)))
        capacity.append(
            {
                "memory_gb": gb,
                "fit_params_fp32": fp32_fit,
                "fit_params_rftmw": rftmw_fit,
                "fit_multiplier_vs_fp32": float(rftmw_fit / max(fp32_fit, 1)),
            }
        )

    payload: Dict[str, object] = {
        "model": args.model,
        "layer_limit": args.layer_limit,
        "entropy_threshold": args.entropy_threshold,
        "keep_ratio": args.keep_ratio,
        "max_rft_error": args.max_rft_error,
        "parameter_count": param_count,
        "fp32_bytes": fp32_bytes,
        "compressed_bytes": compressed_bytes,
        "compression_ratio_vs_fp32": ratio,
        "weighted_reconstruction_error": weighted_err,
        "rft_layers": int(report.rft_layers),
        "int8_layers": int(report.int8_layers),
        "skip_layers": int(report.skip_layers),
        "capacity_by_budget": capacity,
        "notes": [
            "Capacity numbers are estimated by scaling measured compression footprint.",
            "Inference throughput impact is separate from memory fit and should be benchmarked independently.",
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved RFTMW space audit: {out}")
    print(f"Compression ratio vs fp32: {ratio:.2f}x")


if __name__ == "__main__":
    main()
