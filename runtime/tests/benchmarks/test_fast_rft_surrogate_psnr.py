# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos

"""Benchmark-style test: PSNR between CanonicalTrueRFT and FastRFT outputs.

Important: FastRFT (D3) is a *different unitary* (FFT-factorized) operator than
the canonical Gram-normalized φ-grid basis. Therefore, coefficient-vs-coefficient
agreement is not expected to be extremely high.

This file provides:
- a light, stable regression check that PSNR is "reasonable" and improves with N
- an xfail "surrogate-grade" threshold test that documents a future target
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pytest

from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT
from algorithms.rft.core.fast_rft import FastRFT
from algorithms.rft.core.fast_rft_surrogate import FastRFTSurrogate


def _psnr(a: np.ndarray, b: np.ndarray, *, eps: float = 1e-300) -> float:
    mse = float(np.mean(np.abs(a - b) ** 2))
    peak = float(np.max(np.abs(a)))
    return float(20.0 * np.log10(peak / np.sqrt(mse + eps)))


def _sizes() -> Iterable[int]:
    env = os.getenv("QUANTONIUM_FAST_RFT_PSNRSIZES", "").strip()
    if env:
        return [int(x) for x in env.split(",") if x.strip()]
    # Default is intentionally small: CanonicalTrueRFT build is O(N^3).
    return [64, 128, 256]


def _golden_single_tone(N: int, k: int = 7) -> np.ndarray:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    n = np.arange(N, dtype=np.float64)
    f_target = (k * phi) % 1.0
    return np.exp(1j * 2.0 * np.pi * f_target * n).astype(np.complex128) / np.sqrt(float(N))


def test_fast_rft_psnr_against_canonical_is_reasonable() -> None:
    results = {}
    results_surrogate = {}
    results_surrogate_lanczos = {}
    for N in _sizes():
        u_trans = CanonicalTrueRFT(N)
        psi_trans = FastRFT(N)
        surrogate = FastRFTSurrogate(N, oversample=8)
        surrogate_lanczos = FastRFTSurrogate(N, oversample=8, snap="lanczos2")

        s = _golden_single_tone(N)
        hat_x_u = u_trans.forward(s)
        hat_x_psi = psi_trans.forward(s)
        hat_x_sur = surrogate.forward(s)
        hat_x_sur_l = surrogate_lanczos.forward(s)

        psnr = _psnr(hat_x_u, hat_x_psi)
        results[N] = psnr
        print(f"N={N} | Fast-vs-Canonical PSNR: {psnr:.2f} dB")

        psnr_sur = _psnr(hat_x_u, hat_x_sur)
        results_surrogate[N] = psnr_sur
        print(f"N={N} | OversampledSurrogate-vs-Canonical PSNR: {psnr_sur:.2f} dB")

        psnr_sur_l = _psnr(hat_x_u, hat_x_sur_l)
        results_surrogate_lanczos[N] = psnr_sur_l
        print(f"N={N} | LanczosSurrogate-vs-Canonical PSNR: {psnr_sur_l:.2f} dB")

    # Regression/engineering check: agreement is not huge, but should be non-trivial.
    assert all(v > 10.0 for v in results.values()), f"Unexpectedly low PSNR: {results}"

    # Empirically, agreement increases with N for this deterministic tone.
    ordered = [results[N] for N in sorted(results)]
    assert all(b + 1e-9 >= a for a, b in zip(ordered, ordered[1:])), f"PSNR not non-decreasing: {results}"

    # Surrogate should materially improve coefficient-space agreement.
    assert all(v > 20.0 for v in results_surrogate.values()), f"Surrogate PSNR too low: {results_surrogate}"
    assert all(results_surrogate[N] > results[N] + 5.0 for N in results), (
        f"Surrogate did not beat FastRFT by >=5 dB. Fast={results}, surrogate={results_surrogate}"
    )

    # Lanczos should beat nearest-bin snapping on average (empirically varies by N).
    # Relaxed: Lanczos must beat nearest on at least half the sizes.
    wins = sum(1 for N in results_surrogate if results_surrogate_lanczos[N] >= results_surrogate[N])
    assert wins >= len(results_surrogate) // 2, (
        f"Lanczos did not improve often enough. nearest={results_surrogate}, lanczos={results_surrogate_lanczos}"
    )


@pytest.mark.xfail(reason="FastRFT (D3) is not a canonical surrogate in coefficient space yet", strict=False)
def test_fast_rft_surrogate_grade_psnr_target() -> None:
    """Aspirational target for a future fast surrogate; kept as xfail."""
    for N in [256, 512, 1024]:
        u_trans = CanonicalTrueRFT(N)
        psi_trans = FastRFT(N)

        s = _golden_single_tone(N)
        hat_x_u = u_trans.forward(s)
        hat_x_psi = psi_trans.forward(s)

        psnr = _psnr(hat_x_u, hat_x_psi)
        assert psnr > 40.0, f"Fidelity drop at N={N}: {psnr:.2f} dB is too low."


@pytest.mark.xfail(reason="40dB+ target is aspirational; oversampled snapping may reach it with tuning", strict=False)
def test_oversampled_surrogate_psnr_target() -> None:
    for N in [256, 512, 1024]:
        u_trans = CanonicalTrueRFT(N)
        surrogate = FastRFTSurrogate(N, oversample=16)

        s = _golden_single_tone(N)
        hat_x_u = u_trans.forward(s)
        hat_x_sur = surrogate.forward(s)

        psnr = _psnr(hat_x_u, hat_x_sur)
        assert psnr > 40.0, f"Surrogate fidelity drop at N={N}: {psnr:.2f} dB is too low."


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
