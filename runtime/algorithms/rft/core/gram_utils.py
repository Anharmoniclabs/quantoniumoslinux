# =============================================================================
# SPDX-License-Identifier: LicenseRef-SPX-Proprietary
# Copyright (c) 2024-2026 Luis M. Minier. All rights reserved.
#
# Patent Pending: USPTO Application 19/169,399
# "Hybrid Computational Framework for Quantum and Resonance Simulation"
#
# Implements the Gram-normalisation (Löwdin / symmetric orthogonalisation)
# kernel used throughout the RFT framework:
#   Φ̃ = Φ·(ΦᴴΦ)^{-1/2}
#
# PERMITTED: View for peer review and academic verification only.
# NOT PERMITTED: Copy, modify, redistribute, or use commercially.
# =============================================================================

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def gram_normalize(
    Phi: NDArray[np.complex128],
    reg: float = 1e-300,
) -> NDArray[np.complex128]:
    """
    Apply Gram / Löwdin symmetric orthogonalisation to a frame matrix Φ.

        Φ̃ = Φ · (ΦᴴΦ)^{-1/2}

    This is the canonical normalisation step for the RFT basis
    (USPTO 19/169,399, Claim 1).

    Parameters
    ----------
    Phi : NDArray[complex128], shape (N, M)
        Input frame matrix (need not be square).
    reg : float
        Tikhonov regularisation added to eigenvalues before inversion
        to prevent division-by-zero for near-degenerate frames.

    Returns
    -------
    Phi_tilde : NDArray[complex128], shape (N, M)
        Gram-normalised frame satisfying Φ̃ᴴΦ̃ ≈ I_M.
    """
    Phi = np.asarray(Phi, dtype=np.complex128)
    G = Phi.conj().T @ Phi          # M×M Gram matrix
    eigvals, eigvecs = np.linalg.eigh(G)
    eigvals = np.maximum(eigvals, 0.0)  # clamp floating-point negatives
    inv_sqrt_diag = 1.0 / np.sqrt(eigvals + reg)
    G_inv_sqrt = (eigvecs * inv_sqrt_diag[np.newaxis, :]) @ eigvecs.conj().T
    return (Phi @ G_inv_sqrt).astype(np.complex128)


def gram_matrix(Phi: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Return the M×M Gram matrix G = ΦᴴΦ."""
    Phi = np.asarray(Phi, dtype=np.complex128)
    return (Phi.conj().T @ Phi).astype(np.complex128)


def gram_inverse_sqrt(
    Phi: NDArray[np.complex128],
    reg: float = 1e-300,
) -> NDArray[np.complex128]:
    """
    Return (ΦᴴΦ)^{-1/2} — the Gram inverse square-root matrix.

    This is the core normalisation kernel of the canonical RFT
    (USPTO 19/169,399).  The Gram-normalised frame is obtained via

        Φ̃ = Φ · gram_inverse_sqrt(Φ)

    Parameters
    ----------
    Phi : NDArray[complex128], shape (N, M)
        Input frame matrix.
    reg : float
        Tikhonov regularisation on eigenvalues (default 1e-300).

    Returns
    -------
    G_inv_sqrt : NDArray[complex128], shape (M, M)
        The matrix (ΦᴴΦ)^{-1/2}.
    """
    Phi = np.asarray(Phi, dtype=np.complex128)
    G = Phi.conj().T @ Phi
    eigvals, eigvecs = np.linalg.eigh(G)
    eigvals = np.maximum(eigvals, 0.0)
    inv_sqrt_diag = 1.0 / np.sqrt(eigvals + reg)
    return ((eigvecs * inv_sqrt_diag[np.newaxis, :]) @ eigvecs.conj().T).astype(np.complex128)


def frame_bounds(Phi: NDArray[np.complex128]) -> tuple[float, float]:
    """
    Return the tight-frame bounds (A, B) where:
        A·‖x‖² ≤ ‖ΦᴴΦ x‖² ≤ B·‖x‖²  for all x.
    A = B = 1 for a tight (Parseval) frame.
    """
    G = gram_matrix(Phi)
    eigvals = np.linalg.eigvalsh(G)
    return float(eigvals.min()), float(eigvals.max())


def unitarity_error(Phi: NDArray[np.complex128]) -> float:
    """
    Return ‖ΦᴴΦ - I‖_F — how far Φ is from being unitary/isometric.
    After gram_normalize this should be < 5e-15.
    """
    G = gram_matrix(Phi)
    return float(np.linalg.norm(G - np.eye(G.shape[0]), 'fro'))


__all__ = [
    "gram_normalize",
    "gram_matrix",
    "gram_inverse_sqrt",
    "frame_bounds",
    "unitarity_error",
]
