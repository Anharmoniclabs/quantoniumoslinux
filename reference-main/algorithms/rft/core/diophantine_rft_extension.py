# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Diophantine Irrational RFT Extension (Theorem 8 Universality).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import scipy.stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PHI = (1 + np.sqrt(5)) / 2
SQRT2 = np.sqrt(2)
SQRT3 = np.sqrt(3)
SQRT5 = np.sqrt(5)
SILVER_RATIO = 1 + np.sqrt(2)

DIOPHANTINE_CONSTANTS = {
    'phi': PHI,
    'sqrt2': SQRT2,
    'sqrt3': SQRT3,
    'silver': SILVER_RATIO,
}

# ---------------------------------------------------------------------------
# Continued fractions
# ---------------------------------------------------------------------------

def continued_fraction(alpha: float, max_terms: int = 20) -> List[int]:
    """Return the continued-fraction coefficients [a0; a1, a2, ...] of alpha."""
    result = []
    for _ in range(max_terms):
        a = int(alpha)
        result.append(a)
        frac = alpha - a
        if frac < 1e-12:
            break
        alpha = 1.0 / frac
    return result


def convergents(cf: List[int]) -> List[Tuple[int, int]]:
    """Return the convergents (p/q) from a continued-fraction list."""
    convs: List[Tuple[int, int]] = []
    p_prev, p_curr = 1, cf[0]
    q_prev, q_curr = 0, 1
    convs.append((p_curr, q_curr))
    for a in cf[1:]:
        p_next = a * p_curr + p_prev
        q_next = a * q_curr + q_prev
        convs.append((p_next, q_next))
        p_prev, p_curr = p_curr, p_next
        q_prev, q_curr = q_curr, q_next
    return convs


def diophantine_constant(alpha: float, num_tests: int = 200) -> float:
    """Empirically estimate the Diophantine approximation constant c for alpha."""
    values = []
    for q in range(1, num_tests + 1):
        best = min(abs(q * alpha - round(q * alpha)), 1e-15)
        if best > 0:
            values.append(best * q)
    return float(np.median(values)) if values else 0.0


# ---------------------------------------------------------------------------
# Basis construction
# ---------------------------------------------------------------------------

def diophantine_frequency_grid(N: int, alpha: float) -> np.ndarray:
    """Generate the phi-grid frequencies {frac(k*alpha)} for k=0..N-1."""
    return np.mod(np.arange(N, dtype=float) * alpha, 1.0)


def diophantine_basis_matrix(
    N: int, alpha: float, use_gram_normalization: bool = False
) -> np.ndarray:
    """Build the NÃ—N Vandermonde-like RFT basis matrix on the alpha grid."""
    freqs = diophantine_frequency_grid(N, alpha)
    n = np.arange(N, dtype=float)
    U = np.exp(2j * np.pi * np.outer(n, freqs)) / np.sqrt(N)
    if use_gram_normalization:
        # Gram-Schmidt orthonormalization via QR
        Q, _ = np.linalg.qr(U)
        return Q
    return U


# ---------------------------------------------------------------------------
# Discrepancy
# ---------------------------------------------------------------------------

@dataclass
class DiscrepancyResult:
    N: int
    alpha: float
    star_discrepancy: float
    expected_bound: float


def star_discrepancy(seq: np.ndarray) -> float:
    """Compute the star discrepancy D* of a sequence in [0,1)."""
    n = len(seq)
    s = np.sort(seq)
    k = np.arange(1, n + 1, dtype=float)
    D_plus = np.max(k / n - s)
    D_minus = np.max(s - (k - 1) / n)
    return float(max(D_plus, D_minus))


def analyze_equidistribution(N: int, alpha: float) -> DiscrepancyResult:
    seq = np.mod(np.arange(N) * alpha, 1.0)
    D = star_discrepancy(seq)
    # Theoretical Weyl-type bound C log(N)/N
    bound = np.log(N + 1) / N * 2.0
    return DiscrepancyResult(N=N, alpha=alpha, star_discrepancy=D, expected_bound=bound)


# ---------------------------------------------------------------------------
# Davis-Kahan
# ---------------------------------------------------------------------------

@dataclass
class DavisKahanResult:
    N: int
    alpha: float
    perturbation_norm: float
    minimal_gap: float
    dk_bound: float


def companion_matrix_alpha(N: int, alpha: float) -> np.ndarray:
    """Build the companion matrix whose eigenvalues are exp(2pi i k alpha)."""
    z = np.exp(2j * np.pi * np.arange(N) * alpha)
    C = np.zeros((N, N), dtype=complex)
    for j in range(N - 1):
        C[j, j + 1] = 1.0
    # Characteristic polynomial coefficients
    from numpy.polynomial import polynomial as P
    coeffs = np.poly(z)[::-1]
    C[N - 1, :] = -coeffs[:N]
    return C


def minimal_eigenvalue_gap(eigenvalues: np.ndarray) -> float:
    z = np.asarray(eigenvalues)
    N = len(z)
    diffs = []
    for i in range(N):
        for j in range(i + 1, N):
            diffs.append(abs(z[i] - z[j]))
    return float(min(diffs)) if diffs else 0.0


def davis_kahan_analysis(N: int, alpha: float) -> DavisKahanResult:
    U = diophantine_basis_matrix(N, alpha, use_gram_normalization=True)
    # Perturbation: difference between raw and Gram-normalized
    U_raw = diophantine_basis_matrix(N, alpha, use_gram_normalization=False)
    E = U - U_raw
    pert_norm = float(np.linalg.norm(E, ord=2))
    z = np.exp(2j * np.pi * np.arange(N) * alpha)
    gap = minimal_eigenvalue_gap(z)
    dk_bound = pert_norm / max(gap, 1e-15)
    return DavisKahanResult(N=N, alpha=alpha, perturbation_norm=pert_norm, minimal_gap=gap, dk_bound=dk_bound)


# ---------------------------------------------------------------------------
# K99
# ---------------------------------------------------------------------------

def k99(coeffs: np.ndarray) -> int:
    """Return the smallest K such that the top-K coefficients contain >=99% energy."""
    energy = np.abs(coeffs) ** 2
    total = energy.sum()
    if total == 0:
        return 1
    idx = np.argsort(energy)[::-1]
    cumsum = np.cumsum(energy[idx])
    k = int(np.searchsorted(cumsum, 0.99 * total) + 1)
    return min(k, len(coeffs))


# ---------------------------------------------------------------------------
# Ensemble analysis
# ---------------------------------------------------------------------------

def diophantine_drift_ensemble(
    N: int, M: int, alpha: float, rng: np.random.Generator
) -> np.ndarray:
    n = np.arange(N, dtype=float)
    out = np.empty((M, N), dtype=complex)
    for i in range(M):
        f0 = rng.uniform(0, 1)
        a = rng.uniform(-1, 1)
        out[i] = np.exp(2j * np.pi * (f0 * n + a * np.mod(n * alpha, 1.0)))
    return out


@dataclass
class DiophantineK99Result:
    N: int
    M: int
    alpha: float
    alpha_name: str
    mean_k99_rft: float
    mean_k99_dft: float


def compare_k99_diophantine(
    N: int, M: int, alpha: float, seed: int = 42
) -> DiophantineK99Result:
    rng = np.random.default_rng(seed)
    U = diophantine_basis_matrix(N, alpha, use_gram_normalization=True)
    F = np.fft.fft(np.eye(N), axis=0, norm='ortho')
    signals = diophantine_drift_ensemble(N, M, alpha, rng)
    k99_rft = [k99(U.conj().T @ x) for x in signals]
    k99_dft = [k99(F.conj().T @ x) for x in signals]
    return DiophantineK99Result(
        N=N, M=M, alpha=alpha, alpha_name=f'alpha={alpha:.4f}',
        mean_k99_rft=float(np.mean(k99_rft)),
        mean_k99_dft=float(np.mean(k99_dft)),
    )


# ---------------------------------------------------------------------------
# Scaling law / universality
# ---------------------------------------------------------------------------

@dataclass
class ScalingLawEntryResult:
    N: int
    alpha: float
    alpha_name: str
    mean_k99_rft: float
    mean_k99_dft: float
    ci_delta_low: float
    ci_delta_high: float
    p_value: float
    ci_includes_zero: bool


@dataclass
class ScalingLawResult:
    results: List[ScalingLawEntryResult]


# Backward compat alias
UniversalityResult = ScalingLawResult


_ALPHAS = [
    (PHI, 'golden ratio Ï†'),
    (SQRT2, 'âˆš2'),
    (SQRT3, 'âˆš3'),
    (SILVER_RATIO, 'silver ratio'),
]


def verify_scaling_law(N: int, M: int, seed: int = 42) -> ScalingLawResult:
    rng = np.random.default_rng(seed)
    entries = []
    F = np.fft.fft(np.eye(N), axis=0, norm='ortho')
    n = np.arange(N, dtype=float)
    for alpha, name in _ALPHAS:
        U = diophantine_basis_matrix(N, alpha, use_gram_normalization=True)
        k99_rft_list = []
        k99_dft_list = []
        deltas = []
        for _ in range(M):
            f0 = rng.uniform(0, 1)
            a = rng.uniform(-1, 1)
            x = np.exp(2j * np.pi * (f0 * n + a * np.mod(n * alpha, 1.0)))
            kr = k99(U.conj().T @ x)
            kd = k99(F.conj().T @ x)
            k99_rft_list.append(kr)
            k99_dft_list.append(kd)
            deltas.append(kd - kr)
        deltas = np.array(deltas, dtype=float)
        mean_delta = float(np.mean(deltas))
        se = float(np.std(deltas, ddof=1) / np.sqrt(M))
        ci_low = mean_delta - 1.96 * se
        ci_high = mean_delta + 1.96 * se
        # t-test p-value
        t_stat = mean_delta / max(se, 1e-15)
        p_val = float(2 * (1 - scipy.stats.t.cdf(abs(t_stat), df=M - 1)))
        entries.append(ScalingLawEntryResult(
            N=N, alpha=alpha, alpha_name=name,
            mean_k99_rft=float(np.mean(k99_rft_list)),
            mean_k99_dft=float(np.mean(k99_dft_list)),
            ci_delta_low=ci_low, ci_delta_high=ci_high,
            p_value=p_val,
            ci_includes_zero=(ci_low <= 0 <= ci_high),
        ))
    return ScalingLawResult(results=entries)


# Backward compat alias
def verify_universality(N: int, M: int, seed: int = 42) -> ScalingLawResult:
    return verify_scaling_law(N=N, M=M, seed=seed)


# ---------------------------------------------------------------------------
# Sharp log(N) bounds
# ---------------------------------------------------------------------------

@dataclass
class SharpBoundResult:
    alpha: float
    N_values: List[int]
    k99_means: List[float]
    fitted_slope: float
    r_squared: float


def verify_sharp_logn_bound(
    alpha: float,
    N_values: List[int],
    M: int,
    seed: int = 42,
) -> SharpBoundResult:
    rng = np.random.default_rng(seed)
    k99_means = []
    for N in N_values:
        U = diophantine_basis_matrix(N, alpha, use_gram_normalization=True)
        n = np.arange(N, dtype=float)
        ks = []
        for _ in range(M):
            f0 = rng.uniform(0, 1)
            a = rng.uniform(-1, 1)
            x = np.exp(2j * np.pi * (f0 * n + a * np.mod(n * alpha, 1.0)))
            ks.append(k99(U.conj().T @ x))
        k99_means.append(float(np.mean(ks)))
    log_N = np.log(N_values)
    slope, intercept = np.polyfit(log_N, k99_means, 1)
    predicted = slope * log_N + intercept
    ss_res = np.sum((np.array(k99_means) - predicted) ** 2)
    ss_tot = np.sum((np.array(k99_means) - np.mean(k99_means)) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-15)
    return SharpBoundResult(
        alpha=alpha, N_values=list(N_values),
        k99_means=k99_means,
        fitted_slope=float(slope),
        r_squared=float(r2),
    )
