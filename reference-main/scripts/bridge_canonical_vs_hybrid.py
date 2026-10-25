# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Bridge evidence: Canonical (Gram-normalized) RFT vs Hybrid FFT/RFT (fast form).

This script is intended to produce *reproducible* numeric evidence that:
1) Both operators are unitary (within numerical tolerance), and
2) They are generally *not the same operator* (and do not appear to converge
   to the same operator as N increases), despite sharing φ-phase philosophy.

Outputs a JSON report under `results/` for attachment to dev/patent evidence.

Run:
  python scripts/bridge_canonical_vs_hybrid.py
  python scripts/bridge_canonical_vs_hybrid.py --sizes 16 32 64 128
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from typing import Any, Dict, List

import numpy as np

from algorithms.rft.core.resonant_fourier_transform import PHI, rft_basis_matrix


def _unitary_fft_matrix(n: int) -> np.ndarray:
    eye = np.eye(n, dtype=np.complex128)
    return np.fft.fft(eye, axis=0) / np.sqrt(float(n))


def _hybrid_operator(n: int) -> np.ndarray:
    # Hybrid definition used in docs/HYBRID_FFT_RFT_ALGORITHM.md:
    #   Y = E ⊙ FFT(x) / sqrt(N)
    # where E[k] = exp(i*2π*frac((k+1)*φ)).
    k = np.arange(n, dtype=np.float64)
    theta = 2.0 * np.pi * np.mod((k + 1.0) * PHI, 1.0)
    e = np.exp(1j * theta).astype(np.complex128, copy=False)
    d = np.diag(e)
    f = _unitary_fft_matrix(n)
    return d @ f


def _spectral_norm(a: np.ndarray) -> float:
    # Largest singular value (operator 2-norm).
    s = np.linalg.svd(a, compute_uv=False)
    return float(s[0]) if s.size else 0.0


def _phi_periodic_signal(n: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / n
    sig = np.zeros(n, dtype=np.float64)
    for k in range(1, 6):
        sig += np.sin(2 * np.pi * (PHI**k) * t) / k
    sig = sig / (np.linalg.norm(sig) + 1e-15)
    return sig.astype(np.complex128)


@dataclasses.dataclass
class SizeReport:
    n: int
    canonical_unitarity_fro: float
    hybrid_unitarity_fro: float
    op_diff_fro: float
    op_diff_spectral: float
    op_diff_rel_fro: float
    max_entry_abs_diff: float
    action_diff_random_rel: float
    action_diff_phi_rel: float
    seconds: float


def _analyze_size(n: int, *, rng: np.random.Generator) -> SizeReport:
    t0 = time.time()

    canonical = rft_basis_matrix(n, n, use_gram_normalization=True)
    hybrid = _hybrid_operator(n)

    ident = np.eye(n, dtype=np.complex128)
    can_u = float(np.linalg.norm(canonical.conj().T @ canonical - ident, ord="fro"))
    hyb_u = float(np.linalg.norm(hybrid.conj().T @ hybrid - ident, ord="fro"))

    diff = canonical - hybrid
    diff_fro = float(np.linalg.norm(diff, ord="fro"))
    can_fro = float(np.linalg.norm(canonical, ord="fro")) + 1e-15
    diff_rel = diff_fro / can_fro
    diff_spec = _spectral_norm(diff)
    max_entry = float(np.max(np.abs(diff)))

    # Action difference on (1) random unit vector and (2) φ-periodic signal.
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex128)
    x = x / (np.linalg.norm(x) + 1e-15)
    y_can = canonical.conj().T @ x
    y_hyb = hybrid @ x
    action_random = float(np.linalg.norm(y_can - y_hyb) / (np.linalg.norm(y_can) + 1e-15))

    x_phi = _phi_periodic_signal(n)
    y_can_phi = canonical.conj().T @ x_phi
    y_hyb_phi = hybrid @ x_phi
    action_phi = float(np.linalg.norm(y_can_phi - y_hyb_phi) / (np.linalg.norm(y_can_phi) + 1e-15))

    return SizeReport(
        n=n,
        canonical_unitarity_fro=can_u,
        hybrid_unitarity_fro=hyb_u,
        op_diff_fro=diff_fro,
        op_diff_spectral=diff_spec,
        op_diff_rel_fro=diff_rel,
        max_entry_abs_diff=max_entry,
        action_diff_random_rel=action_random,
        action_diff_phi_rel=action_phi,
        seconds=float(time.time() - t0),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="*", type=int, default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    rows: List[Dict[str, Any]] = []
    for n in args.sizes:
        rep = _analyze_size(int(n), rng=rng)
        rows.append(dataclasses.asdict(rep))
        print(
            f"N={n:5d}  "
            f"unitary(can)={rep.canonical_unitarity_fro:.2e}  "
            f"unitary(hyb)={rep.hybrid_unitarity_fro:.2e}  "
            f"||diff||F={rep.op_diff_fro:.3e}  "
            f"rel={rep.op_diff_rel_fro:.3e}  "
            f"act(rand)={rep.action_diff_random_rel:.3e}  "
            f"act(phi)={rep.action_diff_phi_rel:.3e}"
        )

    report = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "phi": float(PHI),
        "seed": int(args.seed),
        "sizes": [int(x) for x in args.sizes],
        "notes": [
            "Canonical = rft_basis_matrix(N,N,use_gram_normalization=True).",
            "Hybrid = diag(exp(i*2π*frac((k+1)*φ))) @ unitary FFT.",
            "This report quantifies operator-level and action-level differences.",
        ],
        "rows": rows,
    }

    os.makedirs("results", exist_ok=True)
    out_path = args.out.strip() or os.path.join("results", f"bridge_canonical_vs_hybrid_{int(time.time())}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
