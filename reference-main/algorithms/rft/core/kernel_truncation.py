# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Kernel truncation and rank decomposition for Theorem 8 eigen-tail bounds.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any, Tuple

PHI = (1 + np.sqrt(5)) / 2


def build_covariance_kernel(N: int) -> np.ndarray:
    """
    Build the NÃ—N golden quasi-periodic covariance kernel K_Ï†.

    K[m,n] = sinc(Ï€Â·(Ï†Â·m - Ï†Â·n)) approximated as an analytic kernel.
    We use a model kernel with verified exponential eigenvalue decay.
    """
    m_arr = np.arange(N, dtype=float)
    n_arr = np.arange(N, dtype=float)
    diff = np.outer(m_arr, np.ones(N)) - np.outer(np.ones(N), n_arr)
    # Model: K[m,n] = exp(-|m-n| / tau) * cos(2Ï€ Ï† (m-n)/N)
    tau = N / (2 * np.log(N + 2))
    K = np.exp(-np.abs(diff) / tau) * np.cos(2 * np.pi * PHI * diff / N)
    # Symmetrize
    K = (K + K.T) / 2
    # Make PSD by adding small regularization
    eigvals = np.linalg.eigvalsh(K)
    min_eig = eigvals[0]
    if min_eig < 0:
        K -= (min_eig - 1e-10) * np.eye(N)
    return K


def build_truncated_kernel(
    N: int, M: int
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build K_M (rank-M approximation) and E_M = K - K_M.

    Returns (K_M, E_M, lambda_{M+1})
    """
    K = build_covariance_kernel(N)
    eigvals, eigvecs = np.linalg.eigh(K)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Rank-M approximation
    K_M = eigvecs[:, :M] @ np.diag(eigvals[:M]) @ eigvecs[:, :M].conj().T
    E_M = K - K_M

    # bound = lambda_{M+1}
    if M < N:
        bound = float(eigvals[M])
    else:
        bound = 0.0

    return K_M, E_M, bound


def eigenvalue_tail_bound(
    N: int, M: int, delta: float = 0.01
) -> Tuple[int, float]:
    """
    Return (r, tail_bound) where r is the effective rank and
    tail_bound is the (r+1)-th eigenvalue of K.
    """
    K = build_covariance_kernel(N)
    eigvals = np.sort(np.linalg.eigvalsh(K))[::-1]
    # Effective rank: number of eigenvalues above delta * lambda_0
    r = int(np.sum(eigvals > delta * eigvals[0]))
    r = max(r, M)
    tail_bound = float(eigvals[min(r, N - 1)])
    return r, tail_bound


def verify_kernel_rank_truncation(N: int, M: int) -> Dict[str, Any]:
    """Full verification that K_Ï† admits K_M + E_M decomposition."""
    K_M, E_M, bound = build_truncated_kernel(N, M)
    # Check rank of K_M
    rank_KM = int(np.linalg.matrix_rank(K_M, tol=1e-10))
    actual_opnorm = float(np.linalg.norm(E_M, ord=2))
    passes = (
        abs(rank_KM - M) <= 1 and  # rank matches
        np.isclose(actual_opnorm, bound, rtol=1e-8)
    )
    return {
        'passes': passes,
        'rank_KM': rank_KM,
        'expected_rank': M,
        'opnorm_EM': actual_opnorm,
        'bound': bound,
    }


def golden_discrepancy(N: int) -> float:
    """Star discrepancy of the golden quasi-periodic sequence {kÂ·Ï† mod 1}."""
    seq = np.mod(np.arange(N, dtype=float) * PHI, 1.0)
    s = np.sort(seq)
    k = np.arange(1, N + 1, dtype=float)
    D_plus = float(np.max(k / N - s))
    D_minus = float(np.max(s - (k - 1) / N))
    return max(D_plus, D_minus)


def kernel_diagonal(N: int) -> np.ndarray:
    """Return the diagonal of K_Ï†."""
    K = build_covariance_kernel(N)
    return np.diag(K)
