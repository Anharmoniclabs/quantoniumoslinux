# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
# MIGRATED FROM: algorithms/rft/core/absolute_novelty.py
# RENAMED TO:    dft_distance_certificate.py
#
# REASON FOR RENAME:
#   The original filename 'absolute_novelty.py' is not appropriate for
#   production or research code. The actual content is a well-defined
#   mathematical tool: certified lower-bound distance from RFT to DFT
#   under diagonal-unitary and permutation symmetries.
#   The new name reflects the mathematical function.
#
# IMPORTANT: This file contains VALID, CITABLE mathematics.
#   certified_abs_novelty_lower_bound_to_dft() gives a deterministic
#   lower bound for ||U - D1 P F D2||_F / sqrt(N), valid for all
#   allowed rephasings and row permutations. This result supports the
#   claim that the RFT is mathematically distinct from DFT.
#   Suitable for citation in Section 3 of the paper.

"""Certified distance-to-DFT certificate for the RFT basis.

Formal definition (see THEOREMS_RFT_IRONCLAD.md):

    N_abs(U; C) = inf_{V in C} inf_{D1,D2,P} ||U - D1 P V D2||_F / sqrt(N)

where D1,D2 are diagonal unitaries and P is a permutation matrix (row
permutation).

This module provides two practical ingredients:

1) A *certified lower bound* vs the unitary DFT family (C={F}):
   For any complex matrices A,B, we have ||A-B||_F >= || |A|-|B| ||_F.
   For any D1,D2,P, the matrix B = D1 P F D2 has constant entry magnitude
   |B_ij| = 1/sqrt(N). Therefore the magnitude mismatch yields a deterministic
   lower bound that holds for all allowed rephasings/permutations.

2) A bounded permutation search with a diagonal-phase minimization routine for
   each candidate permutation. This is not guaranteed to reach the true infimum;
   it is included as a practical upper-bound sanity check and for reporting.

CITATION NOTE:
   The certified lower bound is mathematically rigorous and can be cited
   directly in the paper as evidence that RFT != DFT under all rephasing
   and permutation symmetries.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def _as_c128(A: np.ndarray) -> np.ndarray:
    return np.asarray(A, dtype=np.complex128)


def _fro(A: np.ndarray) -> float:
    return float(np.linalg.norm(A, ord="fro"))


def certified_dft_distance_lower_bound(U: np.ndarray) -> float:
    """Certified lower bound for distance from U to the DFT family under {D1,D2,P}.

    Formerly: certified_abs_novelty_lower_bound_to_dft()

    Returns:
        Lower bound on inf_{D1,D2,P} ||U - D1 P F D2||_F / sqrt(N).

    Notes:
        The bound uses only |U| and the fact that the unitary DFT has constant
        entry magnitude 1/sqrt(N) (invariant under diagonal unitaries and row
        permutations).
    """
    U = _as_c128(U)
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("U must be square (N x N).")
    N = int(U.shape[0])
    target_modulus = 1.0 / np.sqrt(float(N))
    return _fro(np.abs(U) - target_modulus) / np.sqrt(float(N))


# Keep old name as alias for backward compatibility
certified_abs_novelty_lower_bound_to_dft = certified_dft_distance_lower_bound


@dataclass(frozen=True)
class AlignmentResult:
    distance: float
    perm: np.ndarray
    d1: np.ndarray
    d2: np.ndarray


def _structured_row_permutations(
    N: int,
    *,
    rng: np.random.Generator,
    num_random: int,
) -> list[np.ndarray]:
    """Small, deterministic candidate set of row permutations."""
    base = np.arange(int(N), dtype=np.int64)
    perms: list[np.ndarray] = [base.copy(), base[::-1].copy()]
    for s in (1, 2, 3, N // 2):
        if 0 < s < N:
            perms.append(np.roll(base, int(s)).copy())
    if N > 1 and (N & (N - 1)) == 0:
        bits = int(np.log2(N))
        rev = np.array([int(f"{i:0{bits}b}"[::-1], 2) for i in range(N)], dtype=np.int64)
        perms.append(rev)
    for _ in range(int(num_random)):
        perms.append(rng.permutation(N).astype(np.int64, copy=False))
    unique: list[np.ndarray] = []
    seen: set[bytes] = set()
    for p in perms:
        key = p.tobytes()
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def best_diagonal_phase_alignment(
    U: np.ndarray,
    V: np.ndarray,
    perm: np.ndarray,
    *,
    iters: int = 25,
    eps: float = 1e-16,
) -> AlignmentResult:
    """Minimize ||U - D1 P V D2||_F over diagonal phases for a fixed P."""
    U = _as_c128(U)
    V = _as_c128(V)
    if U.shape != V.shape or U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("U and V must be the same square shape (N x N).")
    N = int(U.shape[0])
    perm = np.asarray(perm, dtype=np.int64)
    A = V[perm, :]
    d1 = np.ones(N, dtype=np.complex128)
    d2 = np.ones(N, dtype=np.complex128)
    for _ in range(int(iters)):
        AD2 = A * d2[None, :]
        c = np.einsum("ij,ij->i", U, np.conj(AD2))
        d1 = c / np.maximum(np.abs(c), eps)
        D1A = d1[:, None] * A
        c2 = np.einsum("ij,ij->j", U, np.conj(D1A))
        d2 = c2 / np.maximum(np.abs(c2), eps)
    aligned = d1[:, None] * A * d2[None, :]
    dist = _fro(U - aligned) / np.sqrt(float(N))
    return AlignmentResult(distance=float(dist), perm=perm, d1=d1, d2=d2)


# Keep old name as alias
best_diagonal_phase_alignment_for_fixed_row_perm = best_diagonal_phase_alignment


def heuristic_dft_distance_upper_bound(
    U: np.ndarray,
    V: np.ndarray,
    *,
    num_random_perms: int = 8,
    phase_iters: int = 25,
    seed: int = 0x5EED,
) -> AlignmentResult:
    """Bounded search upper bound on distance to V under rephasing/permutation."""
    U = _as_c128(U)
    V = _as_c128(V)
    if U.shape != V.shape or U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("U and V must be the same square shape (N x N).")
    N = int(U.shape[0])
    rng = np.random.default_rng(int(seed))
    perms = _structured_row_permutations(N, rng=rng, num_random=int(num_random_perms))
    best: AlignmentResult | None = None
    for p in perms:
        r = best_diagonal_phase_alignment(U, V, p, iters=int(phase_iters))
        if best is None or r.distance < best.distance:
            best = r
    assert best is not None
    return best


# Keep old name as alias
heuristic_abs_novelty_upper_bound = heuristic_dft_distance_upper_bound
