# =============================================================================
# SPDX-License-Identifier: LicenseRef-SPX-Proprietary
# Copyright (c) 2024-2026 Luis M. Minier. All rights reserved.
#
# Patent Pending: USPTO Application 19/169,399
# "Hybrid Computational Framework for Quantum and Resonance Simulation"
#
# ⚠️  LEGACY / FAST-PATH MODULE — NOT THE CANONICAL RFT  ⚠️
#
# Implements: Fast Phi-RFT (Ψ = D_φ C_σ F) — O(N log N) FFT-factorized form.
# This is mathematically DISTINCT from the Canonical RFT (U = Gram-normalized φ-grid).
#
# Use this for:  Speed-critical paths where exact Canonical match is not required.
# Do NOT use for: Proofs, precision work, or claims about "the RFT".
#
# Theorems 1, 2, 8, 9 apply to this Fast form (Ψ).
# Theorems 3, 4, 5, 6 apply to the Canonical form (U) in canonical_true_rft.py.
#
# See docs/RFT_TERMINOLOGY_GLOSSARY.md for authoritative definitions.
#
# PERMITTED: View for peer review and academic verification only.
# NOT PERMITTED: Copy, modify, redistribute, or use commercially.
# =============================================================================

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Optional

from .resonant_fourier_transform import PHI, PHI_INV


def _phi_phase_kernel(N: int) -> NDArray[np.complex128]:
    """E[k] = exp(i·2π·frac((k+1)·φ)), k = 0…N-1."""
    k = np.arange(1, N + 1, dtype=np.float64)
    return np.exp(2j * np.pi * np.modf(k * PHI)[0])


def fast_rft(x: NDArray, inverse: bool = False) -> NDArray[np.complex128]:
    """
    Fast RFT — O(N log N) via FFT + φ-phase modulation.

    Forward:  Y[k] = E[k] · FFT(x)[k] / √N
    Inverse:  x[n] = IFFT(E*[k] · √N · Y[k])[n]

    where E[k] = exp(i·2π·frac((k+1)·φ)).

    Note: This is the legacy/fast variant. For exact reconstruction
    use CanonicalTrueRFT (Gram-normalised basis).

    Parameters
    ----------
    x       : input signal (1-D array, real or complex)
    inverse : if True, perform inverse fast RFT

    Returns
    -------
    Y or x_hat : NDArray[complex128]
    """
    x = np.asarray(x, dtype=np.complex128)
    N = len(x)
    E = _phi_phase_kernel(N)
    if not inverse:
        return (E * np.fft.fft(x) / np.sqrt(N)).astype(np.complex128)
    else:
        return np.fft.ifft(E.conj() * x * np.sqrt(N)).astype(np.complex128)


class FastRFT:
    """
    Object-oriented O(N log N) RFT using the φ-phase FFT kernel.

    Parameters
    ----------
    N : int — transform / signal length
    """

    def __init__(self, N: int):
        self.N = N
        self._E: Optional[NDArray[np.complex128]] = None

    @property
    def kernel(self) -> NDArray[np.complex128]:
        if self._E is None:
            self._E = _phi_phase_kernel(self.N)
        return self._E

    def forward(self, x: NDArray) -> NDArray[np.complex128]:
        """Y = E * FFT(x) / √N"""
        x = np.asarray(x, dtype=np.complex128)
        return (self.kernel * np.fft.fft(x) / np.sqrt(self.N)).astype(np.complex128)

    def inverse(self, Y: NDArray) -> NDArray[np.complex128]:
        """x = IFFT(E* * √N * Y)"""
        Y = np.asarray(Y, dtype=np.complex128)
        return np.fft.ifft(self.kernel.conj() * Y * np.sqrt(self.N)).astype(np.complex128)

    def energy_ratio(self, x: NDArray) -> float:
        """‖fast_rft(x)‖² / ‖x‖² — should equal 1.0 (Parseval)."""
        x = np.asarray(x, dtype=np.complex128)
        Y = self.forward(x)
        return float(np.sum(np.abs(Y) ** 2) / (np.sum(np.abs(x) ** 2) + 1e-300))

    def __repr__(self) -> str:
        return f"FastRFT(N={self.N})"


__all__ = ["fast_rft", "FastRFT"]
