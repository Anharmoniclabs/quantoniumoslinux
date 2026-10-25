# =============================================================================
# SPDX-License-Identifier: LicenseRef-SPX-Proprietary
# Copyright (c) 2024-2026 Luis M. Minier. All rights reserved.
#
# Patent Pending: USPTO Application 19/169,399
# "Hybrid Computational Framework for Quantum and Resonance Simulation"
#
# Implements: Canonical RFT with Gram-normalised (Löwdin) basis.
#   Φ̃ = Φ(ΦᴴΦ)^{-1/2}  — unitarity error < 5e-15 at machine precision.
#
# PERMITTED: View for peer review and academic verification only.
# NOT PERMITTED: Copy, modify, redistribute, or use commercially.
# =============================================================================

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Optional

from .resonant_fourier_transform import (
    PHI, rft_basis_matrix, rft_forward_frame, rft_inverse_frame
)


class CanonicalTrueRFT:
    """
    Canonical RFT using the Gram / Löwdin-normalised φ-basis.

    The canonical definition (USPTO 19/169,399):
        Φ_{n,k}  = (1/√N) exp(j·2π·frac((k+1)·φ)·n)
        Φ̃        = Φ·(ΦᴴΦ)^{-1/2}   (Löwdin orthogonalisation)
        Forward  : X = Φ̃ᴴ·x
        Inverse  : x̂ = Φ̃·X

    This gives a tight frame with near-unitary properties and
    unitarity error < 5e-15 at machine precision.

    Parameters
    ----------
    N : int  — transform size (number of input samples)
    M : int  — number of basis vectors (default = N)
    """

    def __init__(self, N: int, M: Optional[int] = None):
        self.N = N
        self.M = M if M is not None else N
        self._Phi_tilde: Optional[NDArray[np.complex128]] = None

    @property
    def phi(self) -> float:
        return PHI

    @property
    def basis(self) -> NDArray[np.complex128]:
        """Gram-normalised basis matrix Φ̃ (N×M), built lazily."""
        if self._Phi_tilde is None:
            self._Phi_tilde = rft_basis_matrix(
                self.N, self.M, use_gram_normalization=True
            )
        return self._Phi_tilde

    def forward(self, x: NDArray) -> NDArray[np.complex128]:
        """Canonical forward RFT: X = Φ̃ᴴ·x"""
        return rft_forward_frame(
            np.asarray(x, dtype=np.complex128), self.basis
        )

    def inverse(self, X: NDArray) -> NDArray[np.complex128]:
        """Canonical inverse RFT: x̂ = Φ̃·X"""
        return rft_inverse_frame(
            np.asarray(X, dtype=np.complex128), self.basis
        )

    def unitarity_error(self) -> float:
        """
        Compute ‖Φ̃ᴴ·Φ̃ - I‖_F  (should be < 5e-15 after Gram normalisation).
        """
        G = self.basis.conj().T @ self.basis
        return float(np.linalg.norm(G - np.eye(self.M), 'fro'))

    def reconstruction_error(self, x: NDArray) -> float:
        """
        Relative reconstruction error ‖x̂ - x‖ / ‖x‖ for a round-trip.
        """
        x = np.asarray(x, dtype=np.complex128)
        x_hat = self.inverse(self.forward(x))
        return float(np.linalg.norm(x_hat - x) / (np.linalg.norm(x) + 1e-300))

    def __repr__(self) -> str:
        return f"CanonicalTrueRFT(N={self.N}, M={self.M})"


__all__ = ["CanonicalTrueRFT"]
