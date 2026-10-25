# =============================================================================
# SPDX-License-Identifier: LicenseRef-SPX-Proprietary
# Copyright (c) 2024-2026 Luis M. Minier. All rights reserved.
#
# Patent Pending: USPTO Application 19/169,399
# "Hybrid Computational Framework for Quantum and Resonance Simulation"
#
# Fast Canonical RFT — NUFFT-Accelerated O(N log N) Implementation
#
# The canonical RFT is:
#   Φ_{n,k} = (1/√N) exp(j 2π frac((k+1)φ) n)
#   Φ̃ = Φ (Φ†Φ)^{-1/2}        [Löwdin orthogonalization]
#   Forward:  X = Φ̃† x          [analysis]
#   Inverse:  x̂ = Φ̃ X          [synthesis]
#
# Dense implementation is O(N²). This module accelerates it using:
# 1. NUFFT (Type-3) for all Φ and Φ† operations — O(N log N)
# 2. Lanczos-based f(G)v for matrix-free G^{-1/2} — O(m·N log N)
#
# PERMITTED: View for peer review and academic verification only.
# NOT PERMITTED: Copy, modify, redistribute, or use commercially.
# =============================================================================

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Literal

from ..core.resonant_fourier_transform import PHI, rft_basis_matrix

try:
    import finufft
    _HAS_FINUFFT = True
except ImportError:
    _HAS_FINUFFT = False


# ---------------------------------------------------------------------------
# φ-grid frequency construction
# ---------------------------------------------------------------------------

def _phi_frequencies(M: int) -> NDArray[np.float64]:
    """φ-grid frequencies: f_k = frac((k+1)·φ) for k = 0..M-1."""
    return np.mod(np.arange(1, M + 1, dtype=np.float64) * PHI, 1.0)


# ---------------------------------------------------------------------------
# NUFFT-based Φ operations (standalone, no pre-planning)
# ---------------------------------------------------------------------------

def _nufft_phi_adjoint(
    x: NDArray, freqs: NDArray, N: int, eps: float = 1e-12,
) -> NDArray[np.complex128]:
    """
    Compute Φ† x via NUFFT Type-3.

    (Φ†x)[k] = (1/√N) Σ_{n=0}^{N-1} x[n] exp(-j 2π f_k n)

    Uses finufft.nufft1d3 with source points at sample indices n
    and target points at -2π f_k.
    """
    x = np.asarray(x, dtype=np.complex128)
    n_arr = np.arange(N, dtype=np.float64)
    targets = -2 * np.pi * freqs
    return finufft.nufft1d3(n_arr, x / np.sqrt(N), targets, eps=eps)


def _nufft_phi_forward(
    c: NDArray, freqs: NDArray, N: int, eps: float = 1e-12,
) -> NDArray[np.complex128]:
    """
    Compute Φ c via NUFFT Type-3.

    (Φc)[n] = (1/√N) Σ_{k=0}^{M-1} c[k] exp(j 2π f_k n)

    Uses finufft.nufft1d3 with source points at 2π f_k
    and target points at sample indices n.
    """
    c = np.asarray(c, dtype=np.complex128)
    sources = 2 * np.pi * freqs
    targets = np.arange(N, dtype=np.float64)
    return finufft.nufft1d3(sources, c / np.sqrt(N), targets, eps=eps)


# ---------------------------------------------------------------------------
# Pre-planned NUFFT Φ engine — amortises setup cost across many calls
# ---------------------------------------------------------------------------

class _NufftPhiEngine:
    """
    Pre-planned NUFFT engine for repeated Φ and Φ† applications.

    Pre-plans two Type-3 NUFFTs at construction, reducing per-call
    overhead from ~10ms to ~0.1ms on large problems.
    """

    def __init__(self, N: int, freqs: NDArray[np.float64], eps: float = 1e-12):
        self.N = N
        self.M = len(freqs)
        self._inv_sqrt_N = 1.0 / np.sqrt(N)

        # Pre-plan the adjoint: Φ† x
        #   source = sample indices n, target = -2π f_k
        self._plan_adj = finufft.Plan(3, (1,), eps=eps, dtype='complex128')
        self._plan_adj.setpts(
            np.arange(N, dtype=np.float64),
            s=-2 * np.pi * freqs,
        )

        # Pre-plan the forward: Φ c
        #   source = 2π f_k, target = sample indices n
        self._plan_fwd = finufft.Plan(3, (1,), eps=eps, dtype='complex128')
        self._plan_fwd.setpts(
            2 * np.pi * freqs,
            s=np.arange(N, dtype=np.float64),
        )

    def phi_adjoint(self, x: NDArray) -> NDArray[np.complex128]:
        """Φ† x via pre-planned NUFFT — O(N log N)."""
        x = np.asarray(x, dtype=np.complex128)
        return self._plan_adj.execute(x * self._inv_sqrt_N)

    def phi_forward(self, c: NDArray) -> NDArray[np.complex128]:
        """Φ c via pre-planned NUFFT — O(N log N)."""
        c = np.asarray(c, dtype=np.complex128)
        return self._plan_fwd.execute(c * self._inv_sqrt_N)


# ---------------------------------------------------------------------------
# Lanczos f(A)b — matrix-free function of a Hermitian matrix applied to vector
# ---------------------------------------------------------------------------

def lanczos_func_vec(
    matvec,
    b: NDArray,
    func,
    m: int = 60,
) -> NDArray[np.complex128]:
    """
    Compute f(A) b via Lanczos approximation for Hermitian PSD A.

    Runs m steps of Lanczos iteration to build a tridiagonal
    approximation T_m of A projected into the Krylov subspace
    K_m(A, b) = span{b, Ab, A²b, ...}.

    Returns: ||b|| · Q_m f(T_m) e_1

    Parameters
    ----------
    matvec : callable — computes A @ v for arbitrary v
    b      : NDArray, shape (n,) — input vector
    func   : callable — scalar function applied to eigenvalues
    m      : int — number of Lanczos steps (capped at n)
    """
    n = len(b)
    m = min(m, n)

    beta_0 = np.linalg.norm(b)
    if beta_0 < 1e-300:
        return np.zeros(n, dtype=np.complex128)

    Q = np.zeros((n, m + 1), dtype=np.complex128)
    Q[:, 0] = b / beta_0

    alpha = np.zeros(m, dtype=np.float64)
    beta = np.zeros(m, dtype=np.float64)

    actual_m = m
    for j in range(m):
        w = matvec(Q[:, j])
        if j > 0:
            w -= beta[j - 1] * Q[:, j - 1]
        alpha[j] = np.real(np.vdot(Q[:, j], w))
        w -= alpha[j] * Q[:, j]

        # Full reorthogonalization for numerical stability
        for i in range(j + 1):
            coeff = np.vdot(Q[:, i], w)
            w -= coeff * Q[:, i]

        beta_j = np.linalg.norm(w)
        if beta_j < 1e-14 * beta_0:
            actual_m = j + 1
            break
        beta[j] = beta_j
        Q[:, j + 1] = w / beta_j

    # Eigendecompose the small tridiagonal matrix T_m (real symmetric)
    T = np.diag(alpha[:actual_m])
    if actual_m > 1:
        off = beta[: actual_m - 1]
        T += np.diag(off, 1) + np.diag(off, -1)
    eigvals, eigvecs = np.linalg.eigh(T)

    # Clamp any negative eigenvalues (numerical noise for PSD matrices)
    eigvals = np.maximum(eigvals, 1e-300)

    # Apply f to the projected eigenvalues and reconstruct
    f_diag = func(eigvals)
    f_T = eigvecs @ np.diag(f_diag) @ eigvecs.T

    e1 = np.zeros(actual_m)
    e1[0] = 1.0

    return (beta_0 * (Q[:, :actual_m] @ (f_T @ e1))).astype(np.complex128)


# ---------------------------------------------------------------------------
# FastCanonicalRFT class
# ---------------------------------------------------------------------------

class FastCanonicalRFT:
    """
    O(N log N) Canonical RFT using NUFFT + Lanczos G^{-1/2}.

    The canonical RFT definition (USPTO 19/169,399):
        Φ_{n,k} = (1/√N) exp(j 2π frac((k+1)φ) n)
        Φ̃ = Φ (Φ†Φ)^{-1/2}       [Löwdin orthogonalization]
        Forward:  X = Φ̃† x
        Inverse:  x̂ = Φ̃ X

    Two modes for the Gram correction G^{-1/2}:

    "lanczos" (default):
        Fully matrix-free. Each forward/inverse uses Lanczos to
        approximate G^{-1/2} v in the Krylov subspace K_m(G, v).
        All G matvecs use NUFFT → O(m · N log N) per call.
        Best for: large N, single-shot transforms.

    "precomputed":
        Builds G^{-1/2} at init via eigendecomposition (O(N²) one-time).
        Runtime applies dense correction (O(N²)) + NUFFT for Φ parts.
        Best for: repeated transforms at moderate N.

    Parameters
    ----------
    N         : int — transform size (number of input samples)
    M         : int — number of basis vectors (default = N)
    mode      : "lanczos" | "precomputed"
    eps       : float — NUFFT tolerance (default 1e-12)
    lanczos_m : int — Lanczos iteration count for matrix-free mode
    """

    def __init__(
        self,
        N: int,
        M: Optional[int] = None,
        mode: Literal["lanczos", "precomputed"] = "lanczos",
        eps: float = 1e-12,
        lanczos_m: int = 60,
    ):
        if not _HAS_FINUFFT:
            raise ImportError(
                "finufft is required for FastCanonicalRFT. "
                "Install with: pip install finufft"
            )

        self.N = N
        self.M = M if M is not None else N
        self.mode = mode
        self.eps = eps
        self.lanczos_m = lanczos_m

        # φ-grid frequencies
        self._freqs = _phi_frequencies(self.M)

        # Pre-planned NUFFT engine (amortises setup across calls)
        self._engine = _NufftPhiEngine(N, self._freqs, eps=eps)

        # Precomputed G^{-1/2} (only in precomputed mode)
        self._G_inv_sqrt: Optional[NDArray[np.complex128]] = None

        if mode == "precomputed":
            self._precompute_gram_correction()

    @property
    def phi(self) -> float:
        return PHI

    # ---- internal helpers ----

    def _precompute_gram_correction(self) -> None:
        """Build G^{-1/2} via eigendecomposition (one-time O(N²) cost)."""
        Phi = rft_basis_matrix(self.N, self.M, use_gram_normalization=False)
        G = Phi.conj().T @ Phi

        eigvals, eigvecs = np.linalg.eigh(G)
        eigvals = np.maximum(eigvals, 1e-300)
        inv_sqrt_diag = 1.0 / np.sqrt(eigvals)

        self._G_inv_sqrt = (
            eigvecs @ np.diag(inv_sqrt_diag) @ eigvecs.conj().T
        ).astype(np.complex128)

    def _apply_phi_adjoint(self, x: NDArray) -> NDArray[np.complex128]:
        """Φ† x via pre-planned NUFFT Type-3 — O(N log N)."""
        return self._engine.phi_adjoint(x)

    def _apply_phi(self, c: NDArray) -> NDArray[np.complex128]:
        """Φ c via pre-planned NUFFT Type-3 — O(N log N)."""
        return self._engine.phi_forward(c)

    def _apply_gram(self, v: NDArray) -> NDArray[np.complex128]:
        """G v = Φ†(Φ v) via 2 NUFFTs — O(N log N)."""
        return self._apply_phi_adjoint(self._apply_phi(v))

    def _apply_gram_inv_sqrt(self, v: NDArray) -> NDArray[np.complex128]:
        """Apply G^{-1/2} to vector v."""
        v = np.asarray(v, dtype=np.complex128)

        if self.mode == "precomputed":
            return (self._G_inv_sqrt @ v).astype(np.complex128)
        else:
            return lanczos_func_vec(
                self._apply_gram, v,
                func=lambda x: x ** (-0.5),
                m=self.lanczos_m,
            )

    # ---- public API ----

    def forward(self, x: NDArray) -> NDArray[np.complex128]:
        """
        Canonical forward RFT: X = Φ̃† x = G^{-1/2} Φ† x

        Complexity:
            lanczos mode:     O(m · N log N)
            precomputed mode: O(N² + N log N)
        """
        x = np.asarray(x, dtype=np.complex128)
        c = self._apply_phi_adjoint(x)       # Φ† x   [O(N log N)]
        return self._apply_gram_inv_sqrt(c)   # G^{-1/2} c

    def inverse(self, X: NDArray) -> NDArray[np.complex128]:
        """
        Canonical inverse RFT: x̂ = Φ̃ X = Φ G^{-1/2} X

        Complexity:
            lanczos mode:     O(m · N log N)
            precomputed mode: O(N² + N log N)
        """
        X = np.asarray(X, dtype=np.complex128)
        c = self._apply_gram_inv_sqrt(X)      # G^{-1/2} X
        return self._apply_phi(c)             # Φ c    [O(N log N)]

    def unitarity_error(self) -> float:
        """
        Estimate ‖Φ̃†Φ̃ - I‖_F via random probing.

        Runs k round-trips on random unit vectors and reports
        the mean reconstruction residual.
        """
        k = min(20, self.M)
        errs = []
        rng = np.random.RandomState(42)
        for _ in range(k):
            v = rng.randn(self.N) + 1j * rng.randn(self.N)
            v = v.astype(np.complex128)
            v /= np.linalg.norm(v)
            X = self.forward(v)
            v_hat = self.inverse(X)
            errs.append(np.linalg.norm(v - v_hat) / np.linalg.norm(v))
        return float(np.mean(errs))

    def reconstruction_error(self, x: NDArray) -> float:
        """Relative reconstruction error ‖x̂ - x‖ / ‖x‖ for a round-trip."""
        x = np.asarray(x, dtype=np.complex128)
        x_hat = self.inverse(self.forward(x))
        return float(np.linalg.norm(x_hat - x) / (np.linalg.norm(x) + 1e-300))

    @property
    def basis(self) -> NDArray[np.complex128]:
        """Return the dense Gram-normalised basis Φ̃ (for validation only)."""
        return rft_basis_matrix(self.N, self.M, use_gram_normalization=True)

    def __repr__(self) -> str:
        return (
            f"FastCanonicalRFT(N={self.N}, M={self.M}, "
            f"mode='{self.mode}', eps={self.eps:.0e})"
        )


__all__ = ["FastCanonicalRFT", "lanczos_func_vec"]
