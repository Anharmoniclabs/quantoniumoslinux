#!/usr/bin/env python3
"""
Test: Theorem 15 (Exact sparsity in the raw φ-grid frame vs. dense DFT leakage)

This test validates the finite-N, algebraic core:
- If s = Φ c with a K-sparse coefficient vector c (Φ is the raw φ-grid basis),
  then the exact dual-frame coefficients recover c (up to numerical tolerance).
- The unitary DFT coefficients of a single φ-grid tone are dense (no exact zeros).

Note: The *canonical* (Gram-normalized) RFT basis is unitary but mixes the raw
columns, so raw Φ-sparsity does not imply sparsity in the canonical RFT
coordinates. Theorem 15 is specifically about the raw φ-grid frame Φ and its
dual-frame coefficient extraction map.

FIX (v2): rft_forward_frame now requires dual_frame=True when the caller passes
a raw (non-Gram-normalized) frame matrix. Without it, the function computes
Φᴴ s (adjoint-only), which is NOT the dual-frame operator (ΦᴴΦ)^{-1} Φᴴ s
needed for exact coefficient recovery.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from algorithms.rft.core.resonant_fourier_transform import rft_basis_matrix, rft_forward_frame


def _k_for_energy(coeffs: np.ndarray, threshold: float = 0.999999) -> int:
    """Return smallest k capturing `threshold` fraction of coefficient energy."""
    energy = np.abs(coeffs) ** 2
    sorted_energy = np.sort(energy)[::-1]
    cumulative = np.cumsum(sorted_energy)
    total = cumulative[-1]
    return int(np.argmax(cumulative >= threshold * total) + 1)


@pytest.mark.parametrize("K_sparse", [1, 5, 12])
def test_theorem_15_exact_sparsity_raw_phi_frame_vs_dft_density(K_sparse: int) -> None:
    N = 256

    # Raw φ-grid basis Φ (not Gram-normalized).
    Phi = rft_basis_matrix(N, N, use_gram_normalization=False)

    # Build a K-sparse coefficient vector c and synthesize s = Φ c.
    rng = np.random.default_rng(42 + K_sparse)
    support = rng.choice(N, size=K_sparse, replace=False)
    c = np.zeros(N, dtype=np.complex128)
    c[support] = rng.standard_normal(K_sparse) + 1j * rng.standard_normal(K_sparse)

    s = Phi @ c
    norm_s = np.linalg.norm(s)
    assert norm_s > 0
    s /= norm_s
    c_true = c / norm_s

    # Dual-frame coefficient recovery: ĉ = (ΦᴴΦ)^{-1} Φᴴ s.
    # dual_frame=True is required because Φ is raw (not Gram-normalized);
    # the default adjoint-only path (Φᴴ s) does not recover c exactly.
    c_hat = rft_forward_frame(s, Phi, dual_frame=True)

    # Exact recovery (numerically).
    np.testing.assert_allclose(c_hat, c_true, rtol=0, atol=1e-10)

    # RFT side: K coefficients capture ~all energy.
    k_req_phi = _k_for_energy(c_hat, threshold=0.999999)
    assert k_req_phi <= K_sparse, f"Expected ≤{K_sparse} coefficients, got {k_req_phi}"

    # DFT side: coefficients are dense for a φ-grid tone (and typically near-dense for K>1).
    X_dft = np.fft.fft(s) / np.sqrt(N)  # unitary scaling

    if K_sparse == 1:
        assert np.count_nonzero(X_dft) == N, "Single φ-grid tone should have no exact-zero DFT bins"

    # Engineering-friendly density check: needs most bins to capture 99.9999% energy.
    k_req_dft = _k_for_energy(X_dft, threshold=0.999999)
    assert k_req_dft > int(0.8 * N), f"Expected dense leakage; only needed {k_req_dft}/{N}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
