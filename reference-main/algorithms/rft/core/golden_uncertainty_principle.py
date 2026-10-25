# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Golden-RFT Uncertainty Principle (Theorem 9).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from algorithms.rft.core.resonant_fourier_transform import rft_basis_matrix

PHI = (1 + np.sqrt(5)) / 2


# ---------------------------------------------------------------------------
# Spread functionals
# ---------------------------------------------------------------------------

def time_spread(x: np.ndarray) -> float:
    """Normalised RMS time spread in [0, 1]."""
    x = np.asarray(x, dtype=complex)
    N = len(x)
    energy = np.abs(x) ** 2
    total = energy.sum()
    if total < 1e-30:
        return 0.0
    t = np.arange(N, dtype=float) / N
    mean_t = np.sum(t * energy) / total
    var_t = np.sum((t - mean_t) ** 2 * energy) / total
    return float(np.sqrt(var_t))


def frequency_spread(X: np.ndarray) -> float:
    """Normalised RMS frequency spread in [0, 1]."""
    X = np.asarray(X, dtype=complex)
    N = len(X)
    energy = np.abs(X) ** 2
    total = energy.sum()
    if total < 1e-30:
        return 0.0
    f = np.arange(N, dtype=float) / N
    mean_f = np.sum(f * energy) / total
    var_f = np.sum((f - mean_f) ** 2 * energy) / total
    return float(np.sqrt(var_f))


def phi_frequencies(N: int, skip_dc: bool = False) -> np.ndarray:
    """
    Return the phi-grid: {frac(k*phi)} for k=0..N-1.

    Parameters
    ----------
    N : int
        Number of grid points.
    skip_dc : bool
        If True, return frequencies for k=1..N (skipping k=0 whose fractional
        part is identically 0, i.e. the DC component). Used by irrationality
        proofs that assert f != m/N for any integer m.
    """
    if skip_dc:
        k = np.arange(1, N + 1, dtype=float)
    else:
        k = np.arange(N, dtype=float)
    return np.mod(k * PHI, 1.0)


def golden_frequency_spread(X: np.ndarray) -> float:
    """Frequency spread using the phi-grid."""
    X = np.asarray(X, dtype=complex)
    N = len(X)
    energy = np.abs(X) ** 2
    total = energy.sum()
    if total < 1e-30:
        return 0.0
    f = phi_frequencies(N)
    mean_f = np.sum(f * energy) / total
    var_f = np.sum((f - mean_f) ** 2 * energy) / total
    return float(np.sqrt(var_f))


# ---------------------------------------------------------------------------
# Mutual coherence
# ---------------------------------------------------------------------------

def mutual_coherence(U: np.ndarray, V: np.ndarray | None = None) -> float:
    """Compute mutual coherence max|<u_i, v_j>|."""
    U = np.asarray(U, dtype=complex)
    if V is None:
        return float(np.max(np.abs(U)))
    V = np.asarray(V, dtype=complex)
    G = U.conj().T @ V
    return float(np.max(np.abs(G)))


def rft_dft_coherence(N: int) -> float:
    U_rft = rft_basis_matrix(N, N, use_gram_normalization=True)
    F = np.fft.fft(np.eye(N), axis=0, norm='ortho')
    G = U_rft.conj().T @ F
    return float(np.max(np.abs(G)))


# ---------------------------------------------------------------------------
# Uncertainty bounds
# ---------------------------------------------------------------------------

def golden_uncertainty_bound(N: int) -> Tuple[float, float]:
    """Return (heisenberg_bound, golden_bound)."""
    heisenberg = 1.0 / (4 * np.pi)
    U_rft = rft_basis_matrix(N, N, use_gram_normalization=True)
    mu = mutual_coherence(U_rft)
    golden = heisenberg * (1 - mu ** 2)
    return heisenberg, golden


# ---------------------------------------------------------------------------
# Uncertainty measurement
# ---------------------------------------------------------------------------

@dataclass
class UncertaintyMeasurement:
    product_dft: float
    product_rft: float
    uncertainty_ratio: float  # rft/dft


def _k99(coeffs: np.ndarray) -> int:
    energy = np.abs(coeffs) ** 2
    total = energy.sum()
    if total == 0:
        return 1
    idx = np.argsort(energy)[::-1]
    cumsum = np.cumsum(energy[idx])
    return int(np.searchsorted(cumsum, 0.99 * total) + 1)


def k99(coeffs: np.ndarray) -> int:
    return _k99(coeffs)


def measure_uncertainty(x: np.ndarray) -> UncertaintyMeasurement:
    x = np.asarray(x, dtype=complex)
    N = len(x)
    norm = np.linalg.norm(x)
    if norm > 0:
        x = x / norm
    X_dft = np.fft.fft(x, norm='ortho')
    dt = time_spread(x)
    df_dft = frequency_spread(X_dft)
    prod_dft = dt * df_dft
    U_rft = rft_basis_matrix(N, N, use_gram_normalization=True)
    X_rft = U_rft.conj().T @ x
    df_rft = golden_frequency_spread(X_rft)
    prod_rft = dt * df_rft
    ratio = prod_rft / max(prod_dft, 1e-30)
    return UncertaintyMeasurement(product_dft=prod_dft, product_rft=prod_rft, uncertainty_ratio=ratio)


# ---------------------------------------------------------------------------
# Phi frequency irrationality helpers
# ---------------------------------------------------------------------------

def phi_frequencies_nondc(N: int) -> np.ndarray:
    """
    Return {frac(k*phi)} for k=1..N  (DC component k=0 excluded).

    Used by irrationality proofs: f_k = {k*phi} is irrational for all k>=1
    because phi is a quadratic irrational (Weyl / three-distance theorem).
    k=0 trivially gives 0 = 0/N, which is rational, so it is excluded from
    those assertions.
    """
    return phi_frequencies(N, skip_dc=True)


def assert_phi_frequencies_irrational(N: int, tol: float = 1e-10) -> None:
    """
    Assert that none of the non-DC phi frequencies equals a rational m/N.

    Raises AssertionError with an informative message on the first failure.
    """
    freqs = phi_frequencies_nondc(N)
    for k_idx, f in enumerate(freqs, start=1):
        for m in range(N):
            diff = abs(f - m / N)
            assert diff > tol, (
                f"k={k_idx}: phi-frequency f={f:.6f} is too close to "
                f"{m}/{N}={m/N:.6f} (diff={diff:.2e} < tol={tol:.2e})"
            )


# ---------------------------------------------------------------------------
# Test signals
# ---------------------------------------------------------------------------

def gaussian_signal(N: int, center: float = 0.5, width: float = 0.05) -> np.ndarray:
    t = np.arange(N, dtype=float) / N
    x = np.exp(-0.5 * ((t - center) / width) ** 2).astype(complex)
    x /= np.linalg.norm(x)
    return x


def golden_quasiperiodic_signal(N: int, f0: float = 0.3, a: float = 0.5) -> np.ndarray:
    n = np.arange(N, dtype=float)
    x = np.exp(2j * np.pi * (f0 * n + a * np.mod(n * PHI, 1.0)))
    x /= np.linalg.norm(x)
    return x


def harmonic_signal(N: int, k: int = 3) -> np.ndarray:
    n = np.arange(N, dtype=float)
    x = np.exp(2j * np.pi * k * n / N)
    x /= np.linalg.norm(x)
    return x


def chirp_signal(N: int, f0: float = 0.1, f1: float = 0.4) -> np.ndarray:
    t = np.arange(N, dtype=float) / N
    x = np.exp(2j * np.pi * (f0 * t + 0.5 * (f1 - f0) * t ** 2) * N).astype(complex)
    x /= np.linalg.norm(x)
    return x


# ---------------------------------------------------------------------------
# Concentration-uncertainty duality
# ---------------------------------------------------------------------------

@dataclass
class ConcentrationDuality:
    k99_dft: int
    k99_rft: int


def concentration_uncertainty_duality(x: np.ndarray) -> ConcentrationDuality:
    x = np.asarray(x, dtype=complex)
    N = len(x)
    norm = np.linalg.norm(x)
    if norm > 0:
        x = x / norm
    X_dft = np.fft.fft(x, norm='ortho')
    U_rft = rft_basis_matrix(N, N, use_gram_normalization=True)
    X_rft = U_rft.conj().T @ x
    return ConcentrationDuality(k99_dft=_k99(X_dft), k99_rft=_k99(X_rft))
