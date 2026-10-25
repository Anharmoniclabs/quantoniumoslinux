# =============================================================================
# SPDX-License-Identifier: LicenseRef-SPX-Proprietary
# Copyright (c) 2024-2026 Luis M. Minier. All rights reserved.
#
# Patent Pending: USPTO Application 19/169,399
# "Hybrid Computational Framework for Quantum and Resonance Simulation"
#
# CANONICAL RFT DEFINITION:
#   Phi[n,k] = (1/sqrt(N)) exp(j 2pi frac((k+1)*phi) * n)
#   Phi_tilde = Phi (Phi^H Phi)^{-1/2}   (Gram / Loewdin normalization)
#   Forward:  X = Phi_tilde^H x     Inverse:  x = Phi_tilde X
#
# UNITARITY NOTE (for independent reproducers):
#   Machine-precision unitarity (||Phi_tilde^H Phi_tilde - I||_F < 1e-12) is
#   achieved by computing G^{-1/2} via the SPECTRAL (eigh) form:
#
#       G = Phi^H Phi
#       eigvals, eigvecs = np.linalg.eigh(G)      # Hermitian-aware, backward-stable
#       G_inv_sqrt = eigvecs @ diag(1/sqrt(L)) @ eigvecs^H
#
#   This is NOT equivalent to scipy.linalg.sqrtm(G) + np.linalg.inv(), which
#   is ill-conditioned for the phi-grid Gram matrix and will NOT reproduce
#   these error levels.  See docs/reproducibility.md for a full explanation,
#   numerical table, and copy-paste verification snippet.
#
# PERMITTED: View for peer review and academic verification only.
# NOT PERMITTED: Copy, modify, redistribute, or use commercially.
# Commercial licensing: luisminier79@gmail.com
# =============================================================================

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import hashlib
from functools import lru_cache
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHI: float = (1.0 + np.sqrt(5.0)) / 2.0
PHI_INV: float = 1.0 / PHI


# ---------------------------------------------------------------------------
# Basis construction
# ---------------------------------------------------------------------------

def rft_frequency(k: int) -> float:
    return float(np.modf((k + 1) * PHI)[0])


def rft_phase(k: int, n: int, N: int) -> complex:
    return complex(np.exp(2j * np.pi * rft_frequency(k) * n))


def rft_basis_function(k: int, N: int) -> NDArray[np.complex128]:
    n = np.arange(N, dtype=np.float64)
    return (np.exp(2j * np.pi * rft_frequency(k) * n) / np.sqrt(N)).astype(np.complex128)


@lru_cache(maxsize=128)
def _rft_basis_matrix_cached(
    N: int,
    M: int,
    use_gram_normalization: bool,
) -> NDArray[np.complex128]:
    """Build and cache the N x M RFT basis matrix.

    When use_gram_normalization=True, returns the Loewdin-orthogonalized
    (Gram-normalized) basis Phi_tilde = Phi G^{-1/2} where G = Phi^H Phi.

    IMPORTANT — implementation choice for machine-precision unitarity:

    G^{-1/2} is computed via the SPECTRAL DECOMPOSITION of G:

        G = V Lambda V^H          (np.linalg.eigh — Hermitian-aware LAPACK)
        G^{-1/2} = V diag(1/sqrt(lambda_k)) V^H

    Why not scipy.linalg.sqrtm + np.linalg.inv?
    - sqrtm uses a Schur decomposition that is not backward-stable for
      ill-conditioned matrices.
    - The phi-grid Gram matrix G has a condition number kappa(G) that grows
      with N (see test_conditioning_gram_matrix_scaling).  Inverting its
      square root amplifies floating-point error by kappa(G), producing
      unitarity errors > 0.01 at N=128 and above.
    - np.linalg.eigh exploits Hermitian structure, returns orthonormal
      eigenvectors V with ||V^H V - I||_F ~ 1e-16, and applies the inverse
      square root as a scalar operation per eigenvalue — no ill-conditioned
      matrix inversion.

    Result: ||Phi_tilde^H Phi_tilde - I||_F < 1e-12 for N up to 1024.
    Cross-check: same matrix produced independently by scipy.linalg.polar
    in transform_theorems.canonical_unitary_basis, agreement < 1e-13.

    See docs/reproducibility.md for the full numerical table and a
    copy-paste verification snippet.
    """
    n = np.arange(N, dtype=np.float64)
    freqs = np.array([rft_frequency(k) for k in range(M)], dtype=np.float64)
    Phi = np.exp(2j * np.pi * np.outer(n, freqs)) / np.sqrt(N)
    Phi = Phi.astype(np.complex128)
    if not use_gram_normalization:
        return Phi

    # --- Gram normalization: spectral form of G^{-1/2} ---
    # Step 1: form Gram matrix G = Phi^H Phi  (M x M, Hermitian)
    G = Phi.conj().T @ Phi

    # Step 2: eigendecompose using eigh (Hermitian-aware, backward-stable)
    #   Returns real non-negative eigenvalues and orthonormal eigenvectors.
    eigvals, eigvecs = np.linalg.eigh(G)

    # Step 3: clamp any negative numerical noise (should be ~1e-16 at most)
    eigvals = np.maximum(eigvals, 0.0)

    # Step 4: assemble G^{-1/2} = V diag(1/sqrt(lambda + eps)) V^H
    #   + 1e-300 floor prevents divide-by-zero without disturbing real eigenvalues
    G_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + 1e-300)) @ eigvecs.conj().T

    # Step 5: Phi_tilde = Phi G^{-1/2}  =>  Phi_tilde^H Phi_tilde = I
    return (Phi @ G_inv_sqrt).astype(np.complex128)


def rft_basis_matrix(
    N: int,
    M: Optional[int] = None,
    use_gram_normalization: bool = True,
) -> NDArray[np.complex128]:
    """Return the N x M RFT basis matrix, cached per (N, M, gram) tuple."""
    if M is None:
        M = N
    return _rft_basis_matrix_cached(N, M, use_gram_normalization)


# ---------------------------------------------------------------------------
# Frame-based transforms
# ---------------------------------------------------------------------------

def rft_forward_frame(
    x: NDArray[np.complex128],
    Phi_tilde: NDArray[np.complex128],
    dual_frame: bool = False,
) -> NDArray[np.complex128]:
    """
    Apply the forward RFT frame operator.

    When dual_frame=False (default, unitary case):
        X = Phi_tilde^H x   (adjoint multiply — exact for Gram-normalised Phi_tilde)

    When dual_frame=True (overcomplete / unnormalised frame):
        X = (Phi^H Phi)^{-1} Phi^H x   (pseudo-inverse / dual-frame recovery)
        This is required when the caller passes a raw (non-Gram-normalised)
        frame matrix and expects exact coefficient recovery (Theorem 15).
    """
    x = np.asarray(x, dtype=np.complex128)
    Phi = np.asarray(Phi_tilde, dtype=np.complex128)
    if not dual_frame:
        return (Phi.conj().T @ x).astype(np.complex128)
    # Dual-frame path: X = (Phi^H Phi)^{-1} Phi^H x
    G = Phi.conj().T @ Phi          # M x M Gram matrix
    eigvals, eigvecs = np.linalg.eigh(G)
    eigvals = np.maximum(eigvals, 1e-300)
    G_inv = (eigvecs * (1.0 / eigvals)[np.newaxis, :]) @ eigvecs.conj().T
    return (G_inv @ (Phi.conj().T @ x)).astype(np.complex128)


def rft_inverse_frame(
    X: NDArray[np.complex128],
    Phi_tilde: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    return (np.asarray(Phi_tilde, dtype=np.complex128) @ np.asarray(X, dtype=np.complex128)).astype(np.complex128)


def rft_forward(x: NDArray, N: Optional[int] = None) -> NDArray[np.complex128]:
    x = np.asarray(x, dtype=np.complex128)
    M = len(x) if N is None else N
    Phi = rft_basis_matrix(len(x), M, use_gram_normalization=True)
    return rft_forward_frame(x, Phi)


def rft_inverse(X: NDArray, N: Optional[int] = None) -> NDArray[np.complex128]:
    X = np.asarray(X, dtype=np.complex128)
    n_out = len(X) if N is None else N
    Phi = rft_basis_matrix(n_out, len(X), use_gram_normalization=True)
    return rft_inverse_frame(X, Phi)


# ---------------------------------------------------------------------------
# Square (N x N) aliases
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Matrix-free operators for large N  (O(N) memory, O(N^2) time per call)
# ---------------------------------------------------------------------------

_MATRIX_FREE_THRESHOLD = 4096


def _phi_freq(k: int) -> float:
    """Fractional part of (k+1)*phi -- the canonical phi-grid frequency."""
    return float(np.modf((k + 1) * PHI)[0])


def _operator_adjoint(x_real: NDArray) -> NDArray[np.complex128]:
    """Phi^H x  computed element-wise -- no N*N matrix allocation."""
    N = len(x_real)
    inv_sqrt_n = 1.0 / np.sqrt(N)
    n = np.arange(N, dtype=np.float64)
    coeffs = np.empty(N, dtype=np.complex128)
    for k in range(N):
        f_k = _phi_freq(k)
        basis_conj = np.exp(-2j * np.pi * f_k * n) * inv_sqrt_n
        coeffs[k] = np.dot(basis_conj, x_real)
    return coeffs


def _operator_synthesis(coeffs: NDArray) -> NDArray[np.complex128]:
    """Phi @ coeffs  computed element-wise -- no N*N matrix allocation."""
    N = len(coeffs)
    inv_sqrt_n = 1.0 / np.sqrt(N)
    n = np.arange(N, dtype=np.float64)
    out = np.zeros(N, dtype=np.complex128)
    for k in range(N):
        f_k = _phi_freq(k)
        basis_k = np.exp(2j * np.pi * f_k * n) * inv_sqrt_n
        out += coeffs[k] * basis_k
    return out


def rft_forward_iht(
    x_real: NDArray,
    keep_k: int = 0,
    max_iter: int = 30,
    step_size: float = 0.85,
    tol: float = 1e-7,
) -> NDArray[np.complex128]:
    """Iterative Hard Thresholding forward using only Phi/Phi^H operators.

    Successive-approximation variant: if keep_k is small (RIP-safe zone),
    runs IHT to convergence, then optionally refines on the residual.

    Memory: O(N).  No Gram matrix.  No N*N allocation.
    """
    x_real = np.asarray(x_real, dtype=np.float64)
    N = len(x_real)
    if keep_k <= 0:
        keep_k = max(4, N // 16)

    coeffs = np.zeros(N, dtype=np.complex128)
    signal_norm = max(np.linalg.norm(x_real), 1e-30)

    for it in range(max_iter):
        synth = _operator_synthesis(coeffs).real
        residual = x_real - synth
        gradient = _operator_adjoint(residual)
        coeffs = coeffs + step_size * gradient

        # Hard threshold: keep top-k by magnitude
        if keep_k < N:
            mags = np.abs(coeffs)
            threshold = np.partition(mags, -keep_k)[-keep_k]
            coeffs[mags < threshold] = 0.0

        res_norm = np.linalg.norm(residual)
        if (res_norm / signal_norm) < tol:
            break

    return coeffs


# ---------------------------------------------------------------------------
# Square (N x N) aliases -- matrix-free for large N
# ---------------------------------------------------------------------------

def rft_forward_square(x: NDArray) -> NDArray[np.complex128]:
    """Forward RFT, M=N (square).

    For N > _MATRIX_FREE_THRESHOLD, uses operator-only adjoint (no N*N matrix)
    to avoid O(N^2) memory blowup.
    """
    x = np.asarray(x, dtype=np.complex128)
    N = len(x)
    if N > _MATRIX_FREE_THRESHOLD:
        return _operator_adjoint(x.real)
    return rft_forward(x)


def rft_inverse_square(X: NDArray) -> NDArray[np.complex128]:
    """Inverse RFT, N_out=M (square).

    For N > _MATRIX_FREE_THRESHOLD, uses operator-only synthesis (no N*N matrix).
    """
    X = np.asarray(X, dtype=np.complex128)
    N = len(X)
    if N > _MATRIX_FREE_THRESHOLD:
        return _operator_synthesis(X)
    return rft_inverse(X)


# ---------------------------------------------------------------------------
# Canonical aliases
# ---------------------------------------------------------------------------

def rft_matrix_canonical(N: int) -> NDArray[np.complex128]:
    """Return the N x N Gram-normalised canonical RFT basis matrix."""
    return rft_basis_matrix(N, N, use_gram_normalization=True)


def rft_phase_vectors_canonical(N: int) -> NDArray[np.complex128]:
    """Return the raw (non-normalised) phi-phase vectors, shape (N, N)."""
    return rft_basis_matrix(N, N, use_gram_normalization=False)


def rft_unitary_error_canonical(N: int) -> float:
    """Return ||U^H U - I||_F for the canonical RFT unitary."""
    U = rft_matrix_canonical(N)
    return float(np.linalg.norm(U.conj().T @ U - np.eye(N), ord='fro'))


# Convenience aliases
rft = rft_forward
irft = rft_inverse


# ---------------------------------------------------------------------------
# ResonantFourierTransform class
# ---------------------------------------------------------------------------

class ResonantFourierTransform:
    def __init__(self, N: int, M: Optional[int] = None, gram: bool = True):
        self.N = N
        self.M = M if M is not None else N
        self.gram = gram

    @property
    def basis(self) -> NDArray[np.complex128]:
        return rft_basis_matrix(self.N, self.M, self.gram)

    def forward(self, x: NDArray) -> NDArray[np.complex128]:
        return rft_forward_frame(np.asarray(x, dtype=np.complex128), self.basis)

    def inverse(self, X: NDArray) -> NDArray[np.complex128]:
        return rft_inverse_frame(np.asarray(X, dtype=np.complex128), self.basis)

    def __repr__(self) -> str:
        return f"ResonantFourierTransform(N={self.N}, M={self.M}, gram={self.gram})"


# ---------------------------------------------------------------------------
# BinaryRFT  -  encodes/decodes integer values via wave carriers
# ---------------------------------------------------------------------------

class BinaryRFT:
    """
    Binary wave computer: integer -> waveform -> integer.
    Carriers: f_k = (k+1)*phi  (raw, not folded) as per the original invention.
    """

    def __init__(self, N: int):
        self.N = N
        self.frequencies = np.array([(k + 1) * PHI for k in range(N)])
        self._t = np.arange(N, dtype=np.float64) / N
        self._carriers = np.sin(
            2.0 * np.pi * self.frequencies[np.newaxis, :] * self._t[:, np.newaxis]
        )  # (N, N)

    def encode(self, value: int) -> NDArray[np.float64]:
        bits = np.array([(value >> k) & 1 for k in range(self.N)], dtype=np.float64)
        return self._carriers @ bits

    def decode(self, wave: NDArray[np.float64]) -> int:
        correlations = self._carriers.T @ wave
        threshold = self.N / 4.0
        bits = (correlations > threshold).astype(int)
        result = 0
        for k, b in enumerate(bits):
            result |= (int(b) << k)
        return result & ((1 << self.N) - 1)

    def wave_xor(self, w1: NDArray[np.float64], w2: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.encode(self.decode(w1) ^ self.decode(w2))

    def wave_and(self, w1: NDArray[np.float64], w2: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.encode(self.decode(w1) & self.decode(w2))

    def wave_or(self, w1: NDArray[np.float64], w2: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.encode(self.decode(w1) | self.decode(w2))

    def wave_not(self, w: NDArray[np.float64]) -> NDArray[np.float64]:
        mask = (1 << self.N) - 1
        return self.encode((~self.decode(w)) & mask)


# ---------------------------------------------------------------------------
# RFTSISHash
# ---------------------------------------------------------------------------

class RFTSISHash:
    def __init__(self, N: int = 64, bits: int = 256):
        self.N = N
        self.bits = bits
        self._rft = ResonantFourierTransform(N)

    def hash(self, data: bytes) -> bytes:
        arr = np.frombuffer(
            data.ljust(self.N * 8, b'\x00')[:self.N * 8], dtype=np.float64
        ).copy()
        X = self._rft.forward(arr.astype(np.complex128))
        bits_arr = (X.real > 0).astype(np.uint8)
        packed = np.packbits(bits_arr).tobytes()
        h = hashlib.sha3_256(packed) if self.bits == 256 else hashlib.sha3_512(packed)
        return h.digest()

    def verify(self, data: bytes, digest: bytes) -> bool:
        return self.hash(data) == digest


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    'PHI', 'PHI_INV',
    'rft_frequency', 'rft_phase', 'rft_basis_function', 'rft_basis_matrix',
    'rft_forward', 'rft_inverse',
    'rft_forward_square', 'rft_inverse_square',
    'rft_matrix_canonical', 'rft_phase_vectors_canonical', 'rft_unitary_error_canonical',
    'rft', 'irft',
    'rft_forward_frame', 'rft_inverse_frame',
    'ResonantFourierTransform', 'BinaryRFT', 'RFTSISHash',
]
