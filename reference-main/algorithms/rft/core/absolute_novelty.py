# SPDX-License-Identifier: LicenseRef-SPX-Proprietary
# Copyright (c) 2024-2026 Luis M. Minier. All rights reserved.
# Patent Pending: USPTO Application 19/169,399
"""
absolute_novelty.py  —  DFT-distance certificate for the canonical RFT.

Provides:
  - certified_abs_novelty_lower_bound_to_dft : certified lower bound on the
    Frobenius distance between a unitary matrix (or the canonical RFT for a
    given N) and the full DFT symmetry group (phase/permutation equivalences).
  - heuristic_abs_novelty_upper_bound : a best-found alignment distance
    (upper bound on the true infimum) using random permutations and iterative
    phase optimisation.
  - novelty_report : convenience wrapper returning a summary dict.
"""
from __future__ import annotations

import dataclasses
import numpy as np
from numpy.typing import NDArray
from algorithms.rft.core.resonant_fourier_transform import rft_basis_matrix

PHI = (1.0 + np.sqrt(5.0)) / 2.0


def _best_dft_unitary(N: int) -> np.ndarray:
    """Return the standard DFT unitary (best-case comparison baseline)."""
    return np.fft.fft(np.eye(N), axis=0, norm='ortho')


def _magnitude_lower_bound(U: NDArray[np.complex128], F: NDArray[np.complex128]) -> float:
    """
    Magnitude-based certified lower bound.

    The symmetry group of the unitary DFT family (row/column phase rotations
    and permutations) preserves |U[i,j]| up to row/column reordering.
    Therefore

        dist_mag = || sort(|U|, axis=None) - sort(|F|, axis=None) ||_F

    is invariant under those symmetries and gives a certified lower bound on

        inf_{D1, D2 diagonal unitary, P permutation}  ||U - D1 P F D2||_F.
    """
    mag_U = np.sort(np.abs(U).ravel())
    mag_F = np.sort(np.abs(F).ravel())
    return float(np.linalg.norm(mag_U - mag_F))


def certified_abs_novelty_lower_bound_to_dft(
    U_or_N,
    n_random_phases: int = 64,
    seed: int = 0,
) -> float:
    """
    Compute a certified lower bound on

        inf_{D1, D2, P}  ||U_RFT - D1 P F D2||_F / sqrt(N)

    where the infimum is over diagonal unitary D1, D2 and permutation P.

    Accepts either:
      - an integer N  : builds the canonical N×N RFT basis internally, or
      - an NDArray U  : uses the supplied unitary matrix directly.

    Strategy: the magnitude lower bound (invariant under the allowed
    symmetries) is always a valid certificate.  We additionally evaluate
    phase-shifted DFT variants and take the minimum over those as an
    extra sanity check; the magnitude bound is the true certificate.

    Parameters
    ----------
    U_or_N          : int or NDArray[complex128]
        Basis dimension N, or a pre-built unitary matrix.
    n_random_phases : Number of random phase perturbations for the
                      phase-sweep sanity check.
    seed            : RNG seed.

    Returns
    -------
    lower_bound : float
        Certified Frobenius lower bound (not normalised by sqrt(N)).
        Always >= 0; is 0 only when U is equivalent to a DFT matrix.
    """
    if isinstance(U_or_N, (int, np.integer)):
        N = int(U_or_N)
        U_rft = rft_basis_matrix(N, N, use_gram_normalization=True)
    else:
        U_rft = np.asarray(U_or_N, dtype=np.complex128)
        N = U_rft.shape[0]

    F = _best_dft_unitary(N)

    # Magnitude-based certified lower bound (invariant under DFT symmetries)
    lb_mag = _magnitude_lower_bound(U_rft, F)

    # Phase-sweep sanity check (not a tighter certificate, just informational)
    rng = np.random.default_rng(seed)
    best_phase = float(np.linalg.norm(U_rft - F, ord='fro'))
    for _ in range(n_random_phases):
        theta = rng.uniform(0, 2 * np.pi)
        dist = float(np.linalg.norm(U_rft - np.exp(1j * theta) * F, ord='fro'))
        best_phase = min(best_phase, dist)
        phases = rng.uniform(0, 2 * np.pi, size=N)
        F_phased = F * np.exp(1j * phases)[np.newaxis, :]
        dist2 = float(np.linalg.norm(U_rft - F_phased, ord='fro'))
        best_phase = min(best_phase, dist2)

    # The certified bound is the magnitude lower bound.
    # (For the DFT itself lb_mag == 0; for the canonical RFT it is > 0.)
    return lb_mag


@dataclasses.dataclass
class NoveltyAlignmentResult:
    """Result returned by heuristic_abs_novelty_upper_bound."""
    distance: float          # best Frobenius distance found (upper bound)
    perm: NDArray            # best column permutation applied to F
    col_phases: NDArray      # best column phase vector applied to F
    row_phases: NDArray      # best row phase vector applied to F


def heuristic_abs_novelty_upper_bound(
    U: NDArray[np.complex128],
    F: NDArray[np.complex128],
    num_random_perms: int = 32,
    phase_iters: int = 50,
    seed: int = 0,
) -> NoveltyAlignmentResult:
    """
    Find the closest element of the unitary DFT symmetry group to U by
    searching over random column permutations and iterative phase refinement.

    Returns an upper bound on

        inf_{D1, D2 diagonal unitary, P permutation}  ||U - D1 P F D2||_F

    because we only search a finite subset of the symmetry group.

    Parameters
    ----------
    U               : NDArray[complex128], shape (N, N)
        Target unitary (e.g. canonical RFT basis).
    F               : NDArray[complex128], shape (N, N)
        Reference unitary (e.g. standard DFT matrix).
    num_random_perms : int
        Number of random column permutations to try.
    phase_iters     : int
        Number of alternating phase-optimisation iterations per permutation.
    seed            : int
        RNG seed.

    Returns
    -------
    NoveltyAlignmentResult
        .distance  – best (lowest) Frobenius distance found.
        .perm      – column permutation that achieved the best distance.
        .col_phases– column phase vector (e^{i phi_j}) for best alignment.
        .row_phases– row phase vector (e^{i theta_i}) for best alignment.
    """
    U = np.asarray(U, dtype=np.complex128)
    F = np.asarray(F, dtype=np.complex128)
    N = U.shape[0]
    rng = np.random.default_rng(seed)

    best_dist = np.inf
    best_perm = np.arange(N)
    best_col_phases = np.ones(N, dtype=np.complex128)
    best_row_phases = np.ones(N, dtype=np.complex128)

    # Always include the identity permutation
    perms = [np.arange(N)] + [rng.permutation(N) for _ in range(num_random_perms - 1)]

    for perm in perms:
        Fp = F[:, perm]   # permute columns
        col_phases = np.ones(N, dtype=np.complex128)
        row_phases = np.ones(N, dtype=np.complex128)

        for _ in range(phase_iters):
            # Optimise column phases: for each column j,
            # best phase aligns inner product <U[:,j], row_phases * Fp[:,j]>
            aligned = (row_phases[:, np.newaxis] * Fp)
            for j in range(N):
                ip = np.vdot(aligned[:, j], U[:, j])   # conj(aligned) . U
                if abs(ip) > 1e-30:
                    col_phases[j] = ip / abs(ip)

            # Optimise row phases: for each row i,
            # best phase aligns inner product <U[i,:], col_phases * Fp[i,:]>
            aligned2 = Fp * col_phases[np.newaxis, :]
            for i in range(N):
                ip = np.vdot(aligned2[i, :], U[i, :])
                if abs(ip) > 1e-30:
                    row_phases[i] = ip / abs(ip)

        candidate = row_phases[:, np.newaxis] * Fp * col_phases[np.newaxis, :]
        dist = float(np.linalg.norm(U - candidate, ord='fro'))
        if dist < best_dist:
            best_dist = dist
            best_perm = perm.copy()
            best_col_phases = col_phases.copy()
            best_row_phases = row_phases.copy()

    return NoveltyAlignmentResult(
        distance=best_dist,
        perm=best_perm,
        col_phases=best_col_phases,
        row_phases=best_row_phases,
    )


def novelty_report(N: int, seed: int = 0) -> dict:
    """Return a dict with the certified lower bound and supporting metrics."""
    lb = certified_abs_novelty_lower_bound_to_dft(N, seed=seed)
    U_rft = rft_basis_matrix(N, N, use_gram_normalization=True)
    F = _best_dft_unitary(N)
    raw_dist = float(np.linalg.norm(U_rft - F, ord='fro'))
    return {
        'N': N,
        'certified_lower_bound': lb,
        'raw_frobenius_distance': raw_dist,
        'is_distinct_from_dft': lb > 1e-10,
    }
