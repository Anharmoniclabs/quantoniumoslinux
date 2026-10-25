# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Evidence Pack generator for Track B (Hybrid-as-Production).

Creates a single, reproducible bundle under `results/` containing:
- canonical vs hybrid difference report
- RFTMW memory scaling reports (canonical + hybrid)
- error sweep report (hybrid by default)
- a markdown index with exact commands and artifact paths

Run:
  python scripts/evidence_pack_track_b.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List


@dataclass
class Artifact:
    name: str
    path: str


def _run(cmd: List[str], *, env: Dict[str, str]) -> None:
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"  cmd={' '.join(cmd)}\n"
            f"  stdout={proc.stdout}\n"
            f"  stderr={proc.stderr}\n"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", type=str, default="")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)

    ts = int(time.time())
    prefix = args.out_prefix.strip() or f"track_b_{ts}"

    results = root / "results"
    results.mkdir(exist_ok=True)

    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")

    artifacts: List[Artifact] = []

    bridge_out = results / f"{prefix}_bridge_canonical_vs_hybrid.json"
    _run([sys.executable, "scripts/bridge_canonical_vs_hybrid.py", "--sizes", "8", "16", "32", "64", "128", "--out", str(bridge_out)], env=env)
    artifacts.append(Artifact("bridge_canonical_vs_hybrid", str(bridge_out)))

    scaling_can = results / f"{prefix}_rftmw_memory_scaling_canonical.json"
    _run([sys.executable, "benchmarks/rftmw_memory_scaling.py", "--impl", "canonical", "--out", str(scaling_can)], env=env)
    artifacts.append(Artifact("rftmw_memory_scaling_canonical", str(scaling_can)))

    scaling_hyb = results / f"{prefix}_rftmw_memory_scaling_hybrid.json"
    _run([sys.executable, "benchmarks/rftmw_memory_scaling.py", "--impl", "hybrid", "--out", str(scaling_hyb)], env=env)
    artifacts.append(Artifact("rftmw_memory_scaling_hybrid", str(scaling_hyb)))

    sweep_hyb = results / f"{prefix}_rftmw_error_sweep_hybrid.json"
    _run([sys.executable, "benchmarks/rftmw_error_sweep.py", "--impl", "hybrid", "--out", str(sweep_hyb)], env=env)
    artifacts.append(Artifact("rftmw_error_sweep_hybrid", str(sweep_hyb)))

    meta = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": str(root),
        "artifacts": [asdict(a) for a in artifacts],
    }
    meta_path = results / f"{prefix}_evidence_pack.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    artifacts.append(Artifact("evidence_pack_meta", str(meta_path)))

    md_lines = [
        "# Track B Evidence Pack",
        "",
        f"- Generated: `{meta['ts']}`",
        f"- Python: `{meta['python']}`",
        f"- Platform: `{meta['platform']}`",
        "",
        "## Commands",
        "",
        "- `python scripts/bridge_canonical_vs_hybrid.py --sizes 8 16 32 64 128`",
        "- `python benchmarks/rftmw_memory_scaling.py --impl canonical`",
        "- `python benchmarks/rftmw_memory_scaling.py --impl hybrid`",
        "- `python benchmarks/rftmw_error_sweep.py --impl hybrid`",
        "",
        "## Artifacts",
        "",
    ]
    for a in artifacts:
        md_lines.append(f"- `{a.name}`: `{a.path}`")
    md_path = results / f"{prefix}_evidence_pack.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

