#!/usr/bin/env python3
"""
Phase 4 (first probe): does canonical U_N show a recursive N/2 structure?

This is NOT a proof; it's a quick structural sanity check.

We build:
  U_N   = canonical RFT basis (Gram-normalized) at size N
  U_M   = canonical RFT basis at size M=N/2

Then we compare U_M against each quadrant of U_N:
  Q00, Q01, Q10, Q11 (each MxM)

If any quadrant strongly correlates with U_M (up to a global complex phase),
it suggests a possible recursive split analogous to FFT butterflies.
"""

from __future__ import annotations

import argparse

import numpy as np

from algorithms.rft.core.resonant_fourier_transform import rft_basis_matrix


def _corr_fro(A: np.ndarray, B: np.ndarray) -> float:
    a = A.ravel()
    b = B.ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-24)
    if denom == 0:
        return 0.0
    inner = np.vdot(a, b)
    return float(np.abs(inner) / denom)


def _phase_align_corr(A: np.ndarray, B: np.ndarray) -> tuple[float, complex]:
    a = A.ravel()
    b = B.ravel()
    inner = np.vdot(a, b)
    if inner == 0:
        return 0.0, 1.0 + 0j
    phase = inner / np.abs(inner)
    return _corr_fro(A, B / phase), phase


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512, help="N (must be even; default 512)")
    args = ap.parse_args()

    N = int(args.n)
    if N <= 0 or (N % 2) != 0:
        raise SystemExit("N must be positive and even.")
    M = N // 2

    print(f"Building canonical U_N for N={N} ...")
    UN = rft_basis_matrix(N, N, use_gram_normalization=True).astype(np.complex128, copy=False)
    print(f"Building canonical U_M for M={M} ...")
    UM = rft_basis_matrix(M, M, use_gram_normalization=True).astype(np.complex128, copy=False)

    quads = {
        "Q00": UN[:M, :M],
        "Q01": UN[:M, M:],
        "Q10": UN[M:, :M],
        "Q11": UN[M:, M:],
    }

    print()
    print("| Quadrant | phase-aligned corr (Fro) |")
    print("|---|---:|")
    for name, Q in quads.items():
        c, _phase = _phase_align_corr(Q, UM)
        print(f"| {name} | {c:.4f} |")


if __name__ == "__main__":
    main()

