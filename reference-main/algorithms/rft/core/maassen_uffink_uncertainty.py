# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Maassen-Uffink Entropic Uncertainty Principle (Theorem 9 â€” correct formulation).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from algorithms.rft.core.resonant_fourier_transform import rft_basis_matrix

PHI = (1 + np.sqrt(5)) / 2


# ---------------------------------------------------------------------------
# Entropy & coherence
# ---------------------------------------------------------------------------

def shannon_entropy(p: np.ndarray) -> float:
    """H(p) = -sum p_i log p_i, treating 0*log(0)=0."""
    p = np.asarray(p, dtype=float)
    mask = p > 0
    return float(-np.sum(p[mask] * np.log(p[mask])))


def signal_entropy(x: np.ndarray) -> float:
    """Entropy of the probability distribution |x|^2 / ||x||^2."""
    energy = np.abs(np.asarray(x, dtype=complex)) ** 2
    total = energy.sum()
    if total < 1e-30:
        return 0.0
    return shannon_entropy(energy / total)


def mutual_coherence(U: np.ndarray) -> float:
    """mu(U) = max |U_{jk}|."""
    return float(np.max(np.abs(U)))


# ---------------------------------------------------------------------------
# Maassen-Uffink bound
# ---------------------------------------------------------------------------

@dataclass
class MUBound:
    N: int
    mu: float
    entropy_bound: float  # -2 log(mu)


def compute_maassen_uffink_bound(U: np.ndarray) -> MUBound:
    N = U.shape[0]
    mu = mutual_coherence(U)
    bound = -2.0 * np.log(mu)
    return MUBound(N=N, mu=mu, entropy_bound=float(bound))


def dft_maassen_uffink_bound(N: int) -> MUBound:
    F = np.fft.fft(np.eye(N), axis=0, norm='ortho')
    return compute_maassen_uffink_bound(F)


def rft_maassen_uffink_bound(N: int) -> MUBound:
    U = rft_basis_matrix(N, N, use_gram_normalization=True)
    return compute_maassen_uffink_bound(U)


# ---------------------------------------------------------------------------
# Uncertainty measurement
# ---------------------------------------------------------------------------

@dataclass
class EntropicUncertainty:
    dft_sum: float
    rft_sum: float
    dft_bound: float
    rft_bound: float


def measure_entropic_uncertainty(x: np.ndarray) -> EntropicUncertainty:
    x = np.asarray(x, dtype=complex)
    N = len(x)
    norm = np.linalg.norm(x)
    if norm > 0:
        x = x / norm
    F = np.fft.fft(np.eye(N), axis=0, norm='ortho')
    U = rft_basis_matrix(N, N, use_gram_normalization=True)

    X_dft = F.conj().T @ x
    X_rft = U.conj().T @ x

    H_x = signal_entropy(x)

    dft_sum = H_x + signal_entropy(X_dft)
    rft_sum = H_x + signal_entropy(X_rft)

    dft_b = dft_maassen_uffink_bound(N)
    rft_b = rft_maassen_uffink_bound(N)

    return EntropicUncertainty(
        dft_sum=dft_sum, rft_sum=rft_sum,
        dft_bound=dft_b.entropy_bound, rft_bound=rft_b.entropy_bound,
    )


# ---------------------------------------------------------------------------
# k99 helper
# ---------------------------------------------------------------------------

def k99(coeffs: np.ndarray) -> int:
    energy = np.abs(coeffs) ** 2
    total = energy.sum()
    if total == 0:
        return 1
    idx = np.argsort(energy)[::-1]
    cumsum = np.cumsum(energy[idx])
    return min(int(np.searchsorted(cumsum, 0.99 * total) + 1), len(coeffs))


# ---------------------------------------------------------------------------
# Concentration measurement
# ---------------------------------------------------------------------------

@dataclass
class ConcentrationMeasurement:
    k99_dft: int
    k99_rft: int
    entropy_dft: float
    entropy_rft: float
    rft_wins_k99: bool
    rft_wins_entropy: bool


def measure_concentration(x: np.ndarray) -> ConcentrationMeasurement:
    x = np.asarray(x, dtype=complex)
    N = len(x)
    norm = np.linalg.norm(x)
    if norm > 0:
        x = x / norm
    F = np.fft.fft(np.eye(N), axis=0, norm='ortho')
    U = rft_basis_matrix(N, N, use_gram_normalization=True)
    X_dft = F.conj().T @ x
    X_rft = U.conj().T @ x
    k_dft = k99(X_dft)
    k_rft = k99(X_rft)
    H_dft = signal_entropy(X_dft)
    H_rft = signal_entropy(X_rft)
    return ConcentrationMeasurement(
        k99_dft=k_dft, k99_rft=k_rft,
        entropy_dft=H_dft, entropy_rft=H_rft,
        rft_wins_k99=(k_rft <= k_dft),
        rft_wins_entropy=(H_rft <= H_dft),
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


def delta_signal(N: int, pos: int = 0) -> np.ndarray:
    x = np.zeros(N, dtype=complex)
    x[pos] = 1.0
    return x


def uniform_signal(N: int) -> np.ndarray:
    x = np.ones(N, dtype=complex) / np.sqrt(N)
    return x


# ---------------------------------------------------------------------------
# verify_theorem_9
# ---------------------------------------------------------------------------

def verify_theorem_9(N: int, x: np.ndarray) -> Tuple[bool, str]:
    """Verify that the Maassen-Uffink bound holds for x."""
    eu = measure_entropic_uncertainty(x)
    tol = 1e-9
    if eu.dft_sum < eu.dft_bound - tol:
        return False, f'DFT bound violated: {eu.dft_sum:.6f} < {eu.dft_bound:.6f}'
    if eu.rft_sum < eu.rft_bound - tol:
        return False, f'RFT bound violated: {eu.rft_sum:.6f} < {eu.rft_bound:.6f}'
    return True, 'OK'
