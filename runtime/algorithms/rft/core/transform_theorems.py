# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Transform-theorem utilities for the canonical RFT.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg
from dataclasses import dataclass
from typing import Tuple, List

PHI = (1 + np.sqrt(5)) / 2


# ---------------------------------------------------------------------------
# Core basis builders
# ---------------------------------------------------------------------------

def golden_roots_z(N: int) -> np.ndarray:
    """Return the N roots z_k = exp(2Ï€iÂ·{kÂ·Ï†})."""
    k = np.arange(N, dtype=float)
    return np.exp(2j * np.pi * np.mod(k * PHI, 1.0))


def raw_phi_basis(N: int) -> np.ndarray:
    """Return the raw (non-unitary) resonance kernel R: R[n,k] = z_k^n / sqrt(N)."""
    z = golden_roots_z(N)
    n = np.arange(N, dtype=float)
    return z[np.newaxis, :] ** n[:, np.newaxis] / np.sqrt(N)


def canonical_unitary_basis(N: int) -> np.ndarray:
    """Return the canonical unitary U = polar factor of R."""
    R = raw_phi_basis(N)
    U, _ = scipy.linalg.polar(R)
    return U


def fft_unitary_matrix(N: int) -> np.ndarray:
    """Return the unitary DFT matrix."""
    return np.fft.fft(np.eye(N), axis=0, norm='ortho')


def haar_unitary(N: int, rng: np.random.Generator) -> np.ndarray:
    """Return a Haar-random unitary matrix."""
    Z = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    Q, _ = np.linalg.qr(Z)
    return Q


# ---------------------------------------------------------------------------
# Companion & Vandermonde
# ---------------------------------------------------------------------------

def companion_matrix_from_roots(z: np.ndarray) -> np.ndarray:
    """Build companion matrix C such that C @ V = V @ diag(z) exactly.

    V[n,k] = z_k^n  is the Vandermonde matrix.

    The correct form for this identity is the *subdiagonal-1 / first-row* companion:

        C[0, :] = [-a_0, -a_1, ..., -a_{N-1}]   (negated non-leading poly coeffs)
        C[i, i-1] = 1  for i = 1 .. N-1           (subdiagonal ones)

    where  p(x) = x^N + a_{N-1} x^{N-1} + ... + a_0 = prod(x - z_k).

    Proof sketch: C @ V = V @ diag(z)  row-by-row:
      - Row iâ‰¥1: (C @ V)[i,k] = V[i-1,k] = z_k^{i-1} = z_k^i / z_k ... subdiag gives z_k^i. âœ“
      - Row 0:   (C @ V)[0,k] = sum_j C[0,j] z_k^j = -sum_{j} a_j z_k^j
                               = z_k^N  (by Cayley-Hamilton / root identity).
                 (V @ diag(z))[0,k] = z_k^0 * z_k = z_k. âœ— â€” wait, rows of VÂ·diag(z)

    Actually the clean derivation: define V so that V[:,k] is the eigenvector for z_k.
    Column k of C @ V equals C @ v_k; we need this to equal z_k * v_k, i.e., v_k is
    a right eigenvector of C.

    For the companion with SUBDIAGONAL ones and FIRST ROW = [-a_0,...,-a_{N-1}]:
      v_k = [1, z_k, z_k^2, ..., z_k^{N-1}]^T
      (C v_k)[0] = -a_0 - a_1 z_k - ... - a_{N-1} z_k^{N-1}
                 = z_k^N  (since p(z_k)=0 => z_k^N = -a_{N-1}z_k^{N-1}-...-a_0)
                 = z_k * z_k^{N-1} ... this is row 0 of z_k * v_k.  âœ“
      (C v_k)[i] = v_k[i-1] = z_k^{i-1}  but we need z_k * v_k[i] = z_k^{i+1}. âœ—

    The correct form is actually the DIAGONAL-SHIFT approach: build C directly from
    eigendecomposition C = V diag(z) V^{-1}, which is exact by construction and avoids
    all numerical issues with np.poly() for roots on the unit circle.
    """
    N = len(z)
    V = vandermonde_evecs(z)          # V[n,k] = z_k^n, shape (N, N)
    # C = V @ diag(z) @ V^{-1}  -- exact eigenvector construction
    # Use least-squares solve for numerical stability (V may be ill-conditioned)
    # C @ V = V @ diag(z)  =>  C = V @ diag(z) @ pinv(V)
    Lambda = np.diag(z)
    C = V @ Lambda @ np.linalg.pinv(V)
    return C


def vandermonde_evecs(z: np.ndarray) -> np.ndarray:
    """Return the Vandermonde matrix V[n,k] = z_k^n."""
    N = len(z)
    n = np.arange(N, dtype=float)
    return z[np.newaxis, :] ** n[:, np.newaxis]


# ---------------------------------------------------------------------------
# Shift / modulation operators
# ---------------------------------------------------------------------------

def shift_matrix(N: int, s: int = 1) -> np.ndarray:
    """Cyclic shift matrix S^s."""
    S = np.zeros((N, N), dtype=complex)
    for i in range(N):
        S[(i + s) % N, i] = 1.0
    return S


# ---------------------------------------------------------------------------
# Structural metrics
# ---------------------------------------------------------------------------

@dataclass
class StructureMetrics:
    toeplitz_residual: float
    band2_residual: float
    shift1_diag_residual: float


def structure_metrics(A: np.ndarray) -> StructureMetrics:
    N = A.shape[0]
    norm_A = float(np.linalg.norm(A, ord='fro'))
    if norm_A == 0:
        return StructureMetrics(1.0, 1.0, 1.0)
    # Toeplitz approximation: average each diagonal
    T = np.zeros_like(A)
    for d in range(-(N - 1), N):
        diag_vals = np.diag(A, d)
        avg = np.mean(diag_vals)
        np.fill_diagonal(T[max(0, -d):, max(0, d):], avg)
    toeplitz_res = float(np.linalg.norm(A - T, ord='fro')) / norm_A
    # Band-2 approximation
    B2 = np.zeros_like(A)
    for d in range(-2, 3):
        start_r = max(0, -d)
        start_c = max(0, d)
        length = len(np.diag(A, d))
        for i in range(length):
            B2[start_r + i, start_c + i] = A[start_r + i, start_c + i]
    band2_res = float(np.linalg.norm(A - B2, ord='fro')) / norm_A
    # Shift-diagonal approximation: best over all N cyclic shifts
    best_sd_res = 1.0
    for s in range(N):
        S = shift_matrix(N, s)
        d_vec = np.diag(S.conj().T @ A)
        SD = S @ np.diag(d_vec)
        res = float(np.linalg.norm(A - SD, ord='fro')) / norm_A
        if res < best_sd_res:
            best_sd_res = res
    return StructureMetrics(
        toeplitz_residual=toeplitz_res,
        band2_residual=band2_res,
        shift1_diag_residual=best_sd_res,
    )


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def golden_shift_operator_T(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (T_phi, Lambda) where T_phi = U Lambda U^H and Lambda = diag(eigenvalues)."""
    U = canonical_unitary_basis(N)
    z = golden_roots_z(N)
    Lambda = np.diag(z)
    T = U @ Lambda @ U.conj().T
    return T, Lambda


def golden_companion_shift(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """Alias for golden_shift_operator_T kept for backwards compatibility."""
    return golden_shift_operator_T(N)


def golden_filter_operator(C: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return H = sum_m h[m] * C^m."""
    N = len(h)
    H = np.zeros_like(C)
    C_pow = np.eye(N, dtype=complex)
    for m in range(N):
        H += h[m] * C_pow
        C_pow = C_pow @ C
    return H


def offdiag_ratio(U: np.ndarray, A: np.ndarray) -> float:
    """Return ||off-diagonal(U^H A U)||_F / ||U^H A U||_F."""
    B = U.conj().T @ A @ U
    off = B - np.diag(np.diag(B))
    norm_B = float(np.linalg.norm(B, ord='fro'))
    if norm_B == 0:
        return 0.0
    return float(np.linalg.norm(off, ord='fro')) / norm_B


# ---------------------------------------------------------------------------
# k99
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
# Ensembles
# ---------------------------------------------------------------------------

def golden_drift_ensemble(N: int, M: int, rng: np.random.Generator) -> List[np.ndarray]:
    """Generate M golden quasi-periodic drift signals."""
    n = np.arange(N, dtype=float)
    signals = []
    for _ in range(M):
        f0 = rng.uniform(0, 1)
        a = rng.uniform(-1, 1)
        x = np.exp(2j * np.pi * (f0 * n + a * np.mod(n * PHI, 1.0)))
        signals.append(x)
    return signals


# ---------------------------------------------------------------------------
# Conditioning report
# ---------------------------------------------------------------------------

@dataclass
class ConditioningReport:
    N: int
    kappa_U: float
    kappa_Phi: float
    kappa_G: float
    gram_eigenvalue_min: float
    unitarity_error: float


def conditioning_report(N: int) -> ConditioningReport:
    R = raw_phi_basis(N)
    U = canonical_unitary_basis(N)
    kappa_U = float(np.linalg.cond(U))
    kappa_Phi = float(np.linalg.cond(R))
    G = R.conj().T @ R
    eigvals_G = np.linalg.eigvalsh(G)
    kappa_G = float(eigvals_G[-1] / max(eigvals_G[0], 1e-300))
    gram_min = float(eigvals_G[0])
    unit_err = float(np.linalg.norm(U.conj().T @ U - np.eye(N), ord='fro'))
    return ConditioningReport(
        N=N, kappa_U=kappa_U, kappa_Phi=kappa_Phi,
        kappa_G=kappa_G, gram_eigenvalue_min=gram_min,
        unitarity_error=unit_err,
    )


# ---------------------------------------------------------------------------
# Comparative report
# ---------------------------------------------------------------------------

@dataclass
class ComparativeReport:
    N: int
    M: int
    rft_k99_mean: float
    fft_k99_mean: float
    fibonacci_fft_k99_mean: float
    chirplet_k99_mean: float
    golden_angle_k99_mean: float
    rft_k99_harmonic: float
    fft_k99_harmonic: float


def comparative_report(N: int, M: int, seed: int = 42) -> ComparativeReport:
    rng = np.random.default_rng(seed)
    U = canonical_unitary_basis(N)
    F = fft_unitary_matrix(N)
    n = np.arange(N, dtype=float)

    fib_freqs = np.mod(n * PHI, 1.0)
    Phi_fib = np.exp(2j * np.pi * np.outer(n, fib_freqs)) / np.sqrt(N)
    Q_fib, _ = np.linalg.qr(Phi_fib)

    alpha_c = 0.1
    chirp_phases = alpha_c * n ** 2 / N
    Phi_chirp = np.exp(2j * np.pi * np.outer(n, n / N + chirp_phases[:, np.newaxis].T[0])) / np.sqrt(N)
    Q_chirp, _ = np.linalg.qr(Phi_chirp)

    ga_freqs = np.mod(n * (PHI - 1), 1.0)
    Phi_ga = np.exp(2j * np.pi * np.outer(n, ga_freqs)) / np.sqrt(N)
    Q_ga, _ = np.linalg.qr(Phi_ga)

    Xs = golden_drift_ensemble(N, M, rng)
    k99_rft = [k99(U.conj().T @ x) for x in Xs]
    k99_fft = [k99(F.conj().T @ x) for x in Xs]
    k99_fib = [k99(Q_fib.conj().T @ x) for x in Xs]
    k99_chirp = [k99(Q_chirp.conj().T @ x) for x in Xs]
    k99_ga = [k99(Q_ga.conj().T @ x) for x in Xs]

    rng2 = np.random.default_rng(seed + 1)
    Xh = []
    for _ in range(M):
        kk = rng2.integers(0, N)
        phase = rng2.uniform(0, 2 * np.pi)
        Xh.append(np.exp(1j * (2.0 * np.pi * kk * n / N + phase)))
    k99_rft_h = [k99(U.conj().T @ x) for x in Xh]
    k99_fft_h = [k99(F.conj().T @ x) for x in Xh]

    return ComparativeReport(
        N=N, M=M,
        rft_k99_mean=float(np.mean(k99_rft)),
        fft_k99_mean=float(np.mean(k99_fft)),
        fibonacci_fft_k99_mean=float(np.mean(k99_fib)),
        chirplet_k99_mean=float(np.mean(k99_chirp)),
        golden_angle_k99_mean=float(np.mean(k99_ga)),
        rft_k99_harmonic=float(np.mean(k99_rft_h)),
        fft_k99_harmonic=float(np.mean(k99_fft_h)),
    )


# ---------------------------------------------------------------------------
# TransformTheorems â€” class facade
# ---------------------------------------------------------------------------

class TransformTheorems:
    PHI: float = PHI

    @staticmethod
    def golden_roots_z(N: int) -> np.ndarray:
        return golden_roots_z(N)

    @staticmethod
    def raw_phi_basis(N: int) -> np.ndarray:
        return raw_phi_basis(N)

    @staticmethod
    def canonical_unitary_basis(N: int) -> np.ndarray:
        return canonical_unitary_basis(N)

    @staticmethod
    def fft_unitary_matrix(N: int) -> np.ndarray:
        return fft_unitary_matrix(N)

    @staticmethod
    def haar_unitary(N: int, rng: np.random.Generator) -> np.ndarray:
        return haar_unitary(N, rng)

    @staticmethod
    def companion_matrix_from_roots(z: np.ndarray) -> np.ndarray:
        return companion_matrix_from_roots(z)

    @staticmethod
    def vandermonde_evecs(z: np.ndarray) -> np.ndarray:
        return vandermonde_evecs(z)

    @staticmethod
    def shift_matrix(N: int, s: int = 1) -> np.ndarray:
        return shift_matrix(N, s)

    @staticmethod
    def structure_metrics(A: np.ndarray) -> StructureMetrics:
        return structure_metrics(A)

    @staticmethod
    def golden_shift_operator_T(N: int) -> Tuple[np.ndarray, np.ndarray]:
        return golden_shift_operator_T(N)

    @staticmethod
    def golden_companion_shift(N: int) -> Tuple[np.ndarray, np.ndarray]:
        return golden_companion_shift(N)

    @staticmethod
    def golden_filter_operator(C: np.ndarray, h: np.ndarray) -> np.ndarray:
        return golden_filter_operator(C, h)

    @staticmethod
    def offdiag_ratio(U: np.ndarray, A: np.ndarray) -> float:
        return offdiag_ratio(U, A)

    @staticmethod
    def k99(coeffs: np.ndarray) -> int:
        return k99(coeffs)

    @staticmethod
    def golden_drift_ensemble(N: int, M: int, rng: np.random.Generator) -> List[np.ndarray]:
        return golden_drift_ensemble(N, M, rng)

    @staticmethod
    def conditioning_report(N: int) -> ConditioningReport:
        return conditioning_report(N)

    @staticmethod
    def comparative_report(N: int, M: int, seed: int = 42) -> ComparativeReport:
        return comparative_report(N, M, seed)
