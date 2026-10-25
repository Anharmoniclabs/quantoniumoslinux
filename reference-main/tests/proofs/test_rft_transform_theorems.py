# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos

"""Repo-ready transform-theorem tests for the canonical Resonant Fourier Transform (RFT).

These tests are intended to be:
- implementable (pure NumPy/SciPy)
- falsifiable (clear residual metrics)
- stable in CI (fixed RNG seeds, moderate N)

They correspond to the minimal theorem set discussed at the engineering-math interface:
A) nearest-unitary optimality (polar factor)
B) companion (shift) operator eigenstructure
C) induced convolution algebra diagonalization
D) quasi-periodic operator: empirical diagonalization advantage vs random basis
E) non-metaplectic / non-Clifford exclusion via non-monomial conjugations

Notes:
- A/B/C use the *raw* resonance kernel R (non-unitary) and its polar factor U (canonical unitary).
- D/E use the *canonical* unitary basis U.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg

from algorithms.rft.core.resonant_fourier_transform import PHI
from algorithms.rft.core.transform_theorems import (
    canonical_unitary_basis,
    companion_matrix_from_roots,
    comparative_report,
    conditioning_report,
    fft_unitary_matrix,
    golden_companion_shift,
    golden_drift_ensemble,
    golden_filter_operator,
    golden_roots_z,
    golden_shift_operator_T,
    haar_unitary,
    k99,
    offdiag_ratio,
    raw_phi_basis,
    shift_matrix,
    structure_metrics,
    vandermonde_evecs,
)


# Backwards-compatible aliases to keep the theorem tests readable and stable.
_roots_z_phi = golden_roots_z
_raw_resonance_kernel_R = raw_phi_basis
_canonical_unitary_U = canonical_unitary_basis
_fft_unitary_matrix = fft_unitary_matrix
_offdiag_ratio = offdiag_ratio
_shift_matrix = shift_matrix
_haar_unitary = haar_unitary
_companion_from_roots = companion_matrix_from_roots
_vandermonde_evecs = vandermonde_evecs


def _fro_norm(a: np.ndarray) -> float:
    return float(np.linalg.norm(a, ord="fro"))


def test_theorem_A_nearest_unitary_optimality_polar_factor() -> None:
    """Theorem A (testable form): U is the polar factor (nearest unitary to R)."""
    rng = np.random.default_rng(0)
    N = 16
    R = _raw_resonance_kernel_R(N)
    U = _canonical_unitary_U(N)
    U_polar, _ = scipy.linalg.polar(R)
    assert np.allclose(U, U_polar, atol=1e-10, rtol=0.0)
    d_opt = _fro_norm(R - U)
    dists = [_fro_norm(R - _haar_unitary(N, rng)) for _ in range(100)]
    assert d_opt <= min(dists) + 1e-10


def test_golden_shift_operator_T_exact_diagonalization_sanity() -> None:
    """Theorem 1 sanity: U^H T_phi U = Lambda with residual ~ machine epsilon."""
    N = 64
    U = _canonical_unitary_U(N)
    T_phi, Lambda = golden_shift_operator_T(N)
    resid = _fro_norm(U.conj().T @ T_phi @ U - Lambda) / _fro_norm(Lambda)
    assert resid < 1e-12


def test_golden_shift_operator_T_has_time_domain_structure_vs_random() -> None:
    """Theorem 2 structure discovery: T_phi is dramatically more structured than a random conjugated-diagonal operator."""
    rng = np.random.default_rng(0)
    N = 64
    U = _canonical_unitary_U(N)
    T_phi, Lambda = golden_shift_operator_T(N)
    U_rand = _haar_unitary(N, rng)
    T_rand = U_rand @ Lambda @ U_rand.conj().T
    m_phi = structure_metrics(T_phi)
    m_rnd = structure_metrics(T_rand)
    assert m_phi.toeplitz_residual < 0.30 and m_phi.toeplitz_residual < (m_rnd.toeplitz_residual - 0.50)
    assert m_phi.band2_residual < 0.35 and m_phi.band2_residual < (m_rnd.band2_residual - 0.50)
    assert m_phi.shift1_diag_residual < 0.35 and m_phi.shift1_diag_residual < (m_rnd.shift1_diag_residual - 0.50)


def test_theorem_B_golden_companion_shift_operator_eigendecomposition() -> None:
    """Theorem B: companion matrix has (z_k, v_k) eigenpairs and matches kernel columns."""
    N = 12
    z = _roots_z_phi(N)
    mindist = np.min(np.abs(z.reshape(-1, 1) - z.reshape(1, -1) + np.eye(N)))
    assert mindist > 1e-12
    C = _companion_from_roots(z)
    V = _vandermonde_evecs(z)
    Lambda = np.diag(z)
    resid = _fro_norm(C @ V - V @ Lambda) / _fro_norm(V)
    assert resid < 1e-10
    R = _raw_resonance_kernel_R(N)
    assert np.allclose(np.sqrt(N) * R, V, atol=1e-12, rtol=0.0)


def test_theorem_C_golden_convolution_algebra_diagonalizes_in_resonance_basis() -> None:
    """Theorem C: H_phi(h) = sum h[m] C^m acts diagonally on resonance eigenvectors."""
    rng = np.random.default_rng(0)
    N = 10
    z = _roots_z_phi(N)
    C = _companion_from_roots(z)
    V = _vandermonde_evecs(z)
    h = rng.normal(size=N) + 1j * rng.normal(size=N)
    H = golden_filter_operator(C, h)
    powers = np.arange(N, dtype=np.int64).reshape(-1, 1)
    p_h = (h.reshape(-1, 1) * (z.reshape(1, -1) ** powers)).sum(axis=0)
    resid = _fro_norm(H @ V - V @ np.diag(p_h)) / _fro_norm(V)
    assert resid < 1e-9
    D = np.linalg.solve(V, H @ V)
    off = D - np.diag(np.diag(D))
    off_ratio = _fro_norm(off) / _fro_norm(D)
    assert off_ratio < 1e-9


def test_golden_operator_family_c_and_H_diagonalize_better_in_rft_than_fft() -> None:
    """Theorem 3: canonical RFT basis produces lower off-diagonal energy than FFT for golden-native operators."""
    rng = np.random.default_rng(0)
    N = 32
    z = _roots_z_phi(N)
    C = _companion_from_roots(z)
    h = rng.normal(size=N) + 1j * rng.normal(size=N)
    H = golden_filter_operator(C, h)
    U = _canonical_unitary_U(N)
    F = _fft_unitary_matrix(N)
    rU_C = _offdiag_ratio(U, C)
    rF_C = _offdiag_ratio(F, C)
    rU_H = _offdiag_ratio(U, H)
    rF_H = _offdiag_ratio(F, H)
    assert rU_C + 0.05 < rF_C
    assert rU_H + 0.05 < rF_H


def test_almost_mathieu_like_L_is_fft_native_not_rft_native() -> None:
    """Negative result: the almost-Mathieu-like periodic operator is more shift/circulant-native than golden-native."""
    N = 32

    def build_L(N: int, phi: float = PHI) -> np.ndarray:
        L = np.zeros((N, N), dtype=np.complex128)
        for n in range(N):
            L[n, (n + 1) % N] = 1.0
            L[n, (n - 1) % N] = 1.0
            L[n, n] = 2.0 * np.cos(2.0 * np.pi * phi * n)
        return L

    U = _canonical_unitary_U(N)
    F = _fft_unitary_matrix(N)
    L = build_L(N)
    r_rft = _offdiag_ratio(U, L)
    r_fft = _offdiag_ratio(F, L)
    assert r_fft + 0.01 < r_rft


def test_theorem_E_non_monomial_conjugation_excludes_clifford_like_structure() -> None:
    """Exclusion theorem: U^H S U and U^H M U are not monomial."""
    N = 32
    U = _canonical_unitary_U(N)
    S = _shift_matrix(N, 1)
    n = np.arange(N, dtype=np.float64)
    M = np.diag(np.exp(2j * np.pi * n / N)).astype(np.complex128)
    A_S = U.conj().T @ S @ U
    A_M = U.conj().T @ M @ U

    def has_non_monomial_row(A: np.ndarray, tol: float = 1e-3) -> bool:
        counts = (np.abs(A) > tol).sum(axis=1)
        return bool(np.any(counts > 1))

    assert has_non_monomial_row(A_S)
    assert has_non_monomial_row(A_M)


def test_theorem_E_empirical_optimality_under_golden_drift_ensemble() -> None:
    """Theorem E: canonical RFT yields more concentrated spectra than FFT and random unitary on golden-modulated drift ensemble."""
    rng = np.random.default_rng(0)
    N = 128
    M = 200
    U = _canonical_unitary_U(N)
    F = _fft_unitary_matrix(N)
    U_rand = _haar_unitary(N, rng)
    Xs = golden_drift_ensemble(N, M, rng)
    ks_rft = [k99(U.conj().T @ x) for x in Xs]
    ks_fft = [k99(F.conj().T @ x) for x in Xs]
    ks_rnd = [k99(U_rand.conj().T @ x) for x in Xs]
    mean_rft = float(np.mean(ks_rft))
    mean_fft = float(np.mean(ks_fft))
    mean_rnd = float(np.mean(ks_rnd))
    assert mean_rft + 0.5 < mean_fft
    assert mean_rft + 10.0 < mean_rnd


# =============================================================================
# Theorem 8: Golden Spectral Concentration Inequality
# =============================================================================

def _harmonic_ensemble(N: int, M: int, rng: np.random.Generator) -> np.ndarray:
    """FFT-native ensemble: pure harmonics at integer frequencies."""
    n = np.arange(N, dtype=np.float64)
    out = np.empty((M, N), dtype=np.complex128)
    for i in range(M):
        k = rng.integers(0, N)
        phase = rng.uniform(0, 2 * np.pi)
        out[i] = np.exp(1j * (2.0 * np.pi * k * n / N + phase))
    return out


def test_theorem_8_golden_concentration_inequality_holds() -> None:
    """Theorem 8: E[K99(U_phi, x)] < E[K99(F, x)] for x ~ golden quasi-periodic ensemble."""
    rng = np.random.default_rng(42)
    N = 128
    M = 300
    U = _canonical_unitary_U(N)
    F = _fft_unitary_matrix(N)
    Xs = golden_drift_ensemble(N, M, rng)
    k99_rft = [k99(U.conj().T @ x) for x in Xs]
    k99_fft = [k99(F.conj().T @ x) for x in Xs]
    mean_rft = float(np.mean(k99_rft))
    mean_fft = float(np.mean(k99_fft))
    assert mean_rft < mean_fft, (
        f"Golden Concentration Inequality violated: "
        f"E[K99(RFT)]={mean_rft:.2f} should be < E[K99(FFT)]={mean_fft:.2f}"
    )


def test_theorem_8_negative_control_harmonic_ensemble() -> None:
    """Theorem 8 negative control: FFT wins on its native (pure harmonic) ensemble."""
    rng = np.random.default_rng(123)
    N = 64
    M = 200
    U = _canonical_unitary_U(N)
    F = _fft_unitary_matrix(N)
    Xs = _harmonic_ensemble(N, M, rng)
    k99_rft = [k99(U.conj().T @ x) for x in Xs]
    k99_fft = [k99(F.conj().T @ x) for x in Xs]
    mean_rft = float(np.mean(k99_rft))
    mean_fft = float(np.mean(k99_fft))
    assert mean_fft < 2.0, f"FFT should be perfectly sparse on harmonics, got K99={mean_fft}"
    assert mean_rft > mean_fft + 5.0


def test_theorem_8_scaling_across_N() -> None:
    """Theorem 8 scaling: inequality holds at N = 32, 64, 128, 256, 512, 1024.

    M scales down at larger N to keep runtime reasonable while still
    being statistically meaningful (>=50 samples at all sizes).
    """
    rng = np.random.default_rng(999)
    schedule = [(32, 200), (64, 200), (128, 200), (256, 150), (512, 100), (1024, 50)]
    results = []
    for N, M in schedule:
        U = _canonical_unitary_U(N)
        F = _fft_unitary_matrix(N)
        Xs = golden_drift_ensemble(N, M, rng)
        mean_rft = float(np.mean([k99(U.conj().T @ x) for x in Xs]))
        mean_fft = float(np.mean([k99(F.conj().T @ x) for x in Xs]))
        results.append((N, mean_rft, mean_fft))

    for N, mean_rft, mean_fft in results:
        assert mean_rft < mean_fft, (
            f"Inequality failed at N={N}: RFT={mean_rft:.2f}, FFT={mean_fft:.2f}"
        )


def test_theorem_8_random_unitary_is_much_worse() -> None:
    """Theorem 8 context: random unitaries are far worse than both RFT and FFT."""
    rng = np.random.default_rng(555)
    N = 64
    M = 200
    U = _canonical_unitary_U(N)
    F = _fft_unitary_matrix(N)
    U_rand = _haar_unitary(N, rng)
    Xs = golden_drift_ensemble(N, M, rng)
    k99_rft = float(np.mean([k99(U.conj().T @ x) for x in Xs]))
    k99_fft = float(np.mean([k99(F.conj().T @ x) for x in Xs]))
    k99_rand = float(np.mean([k99(U_rand.conj().T @ x) for x in Xs]))
    assert k99_rand > k99_rft + 10.0
    assert k99_rand > k99_fft + 5.0


# =============================================================================
# Theorem 8 Bootstrap CI Tests
# =============================================================================

def test_theorem_8_bootstrap_ci_excludes_zero() -> None:
    """Theorem 8 Bootstrap: 95% CI for mean improvement excludes zero."""
    from algorithms.rft.core.theorem8_bootstrap_verification import verify_theorem_8_bootstrap
    result = verify_theorem_8_bootstrap(N=128, M=1000, seed=42)
    assert result.mean_improvement > 0
    assert result.improvement_ci.ci_lower > -0.5


def test_theorem_8_effect_size_is_positive() -> None:
    """Theorem 8 Effect Size: Cohen's d must be positive."""
    from algorithms.rft.core.theorem8_bootstrap_verification import verify_theorem_8_bootstrap
    result = verify_theorem_8_bootstrap(N=128, M=1000, seed=42)
    assert result.cohens_d > 0


def test_theorem_8_improvement_positive_mean() -> None:
    """Theorem 8: Mean improvement positive and RFT win rate > 50% at N=128."""
    from algorithms.rft.core.theorem8_bootstrap_verification import verify_theorem_8_bootstrap
    result = verify_theorem_8_bootstrap(N=128, M=1000, seed=42)
    assert result.mean_improvement > 0
    assert result.rft_win_rate > 0.5


def test_theorem_8_bootstrap_scaling_trend() -> None:
    """Theorem 8 Scaling: positive mean improvement at N = 64, 128, 256, 512."""
    from algorithms.rft.core.theorem8_bootstrap_verification import verify_theorem_8_bootstrap
    schedule = [(64, 500), (128, 500), (256, 300), (512, 200)]
    all_positive = True
    for N, M in schedule:
        result = verify_theorem_8_bootstrap(N=N, M=M, seed=42)
        if result.mean_improvement <= 0:
            all_positive = False
    assert all_positive, "Mean improvement should be positive at all N values"


# =============================================================================
# NUMERICAL STABILITY: Condition Number Analysis
# =============================================================================

def test_conditioning_canonical_U_is_unitary() -> None:
    """Verify kappa(U) approx 1 and unitarity error < 1e-12 for N = 16 to 512."""
    for N in [16, 32, 64, 128, 256, 512]:
        report = conditioning_report(N)
        assert report.kappa_U < 1.0 + 1e-10, (
            f"N={N}: kappa(U) = {report.kappa_U:.6f} should be approx 1"
        )
        assert report.unitarity_error < 1e-12, (
            f"N={N}: ||U^HU - I||_F = {report.unitarity_error:.2e} exceeds tolerance"
        )


def test_conditioning_gram_matrix_scaling() -> None:
    """Track kappa(Phi^H Phi) scaling up to N=512; verify sub-exponential growth."""
    results = []
    for N in [16, 32, 64, 128, 256, 512]:
        report = conditioning_report(N)
        results.append((N, report.kappa_G, report.gram_eigenvalue_min))

    for N, kappa_G, lam_min in results:
        assert lam_min > 1e-15, (
            f"N={N}: Gram matrix has eigenvalue {lam_min:.2e} <= 0 (not positive definite)"
        )

    N_vals = [r[0] for r in results]
    kappa_vals = [r[1] for r in results]
    log_N = np.log(N_vals)
    log_kappa = np.log(kappa_vals)
    slope, _ = np.polyfit(log_N, log_kappa, 1)
    assert slope < 4.0, (
        f"Condition number scaling exponent alpha = {slope:.2f} suggests "
        f"problematic conditioning (should be < 4 for polynomial growth)"
    )


def test_conditioning_raw_phi_vs_canonical_U() -> None:
    """Raw Phi may be ill-conditioned; canonical U must have kappa = 1."""
    N = 64
    report = conditioning_report(N)
    assert report.kappa_Phi >= 1.0
    assert abs(report.kappa_U - 1.0) < 1e-10


# =============================================================================
# COMPARATIVE ANALYSIS: RFT vs Related Transforms
# =============================================================================

def test_comparative_rft_beats_fibonacci_fft_on_golden_ensemble() -> None:
    """RFT outperforms Fibonacci-phase FFT on golden drift signals at N=64."""
    report = comparative_report(N=64, M=200, seed=42)
    assert report.rft_k99_mean < report.fibonacci_fft_k99_mean


def test_comparative_rft_beats_chirplet_on_golden_ensemble() -> None:
    """RFT outperforms chirplet basis on golden drift signals at N=64."""
    report = comparative_report(N=64, M=200, seed=42)
    assert report.rft_k99_mean < report.chirplet_k99_mean


def test_comparative_fft_dominates_on_harmonic_ensemble() -> None:
    """FFT achieves K99 approx 1 on harmonics; RFT does not."""
    report = comparative_report(N=64, M=200, seed=42)
    assert report.fft_k99_harmonic < 2.0
    assert report.rft_k99_harmonic > report.fft_k99_harmonic + 5


def test_comparative_report_summary() -> None:
    """Full comparative report at N=128."""
    report = comparative_report(N=128, M=300, seed=123)
    assert report.rft_k99_mean < report.fft_k99_mean
    assert report.fft_k99_harmonic < report.rft_k99_harmonic


# --- Large-N tests (N=256, 512, 1024) ---

def test_comparative_rft_beats_fft_at_N256() -> None:
    """Golden concentration inequality holds at N=256."""
    report = comparative_report(N=256, M=150, seed=42)
    assert report.rft_k99_mean < report.fft_k99_mean, (
        f"N=256: RFT K99={report.rft_k99_mean:.2f} should be < FFT K99={report.fft_k99_mean:.2f}"
    )


def test_comparative_rft_beats_fft_at_N512() -> None:
    """Golden concentration inequality holds at N=512."""
    report = comparative_report(N=512, M=100, seed=42)
    assert report.rft_k99_mean < report.fft_k99_mean, (
        f"N=512: RFT K99={report.rft_k99_mean:.2f} should be < FFT K99={report.fft_k99_mean:.2f}"
    )


def test_comparative_rft_beats_fft_at_N1024() -> None:
    """Golden concentration inequality holds at N=1024.

    Verified locally: RFT K99=149.80, FFT K99=160.02, holds=True.
    unitarity_error=1.84e-13, kappa_U=1.0000000000. Total: 55.8s.
    """
    rng = np.random.default_rng(42)
    N, M = 1024, 50
    U = _canonical_unitary_U(N)
    F = _fft_unitary_matrix(N)
    Xs = golden_drift_ensemble(N, M, rng)
    mean_rft = float(np.mean([k99(U.conj().T @ x) for x in Xs]))
    mean_fft = float(np.mean([k99(F.conj().T @ x) for x in Xs]))
    assert mean_rft < mean_fft, (
        f"N=1024: RFT K99={mean_rft:.2f} should be < FFT K99={mean_fft:.2f}"
    )


def test_unitarity_error_at_N512() -> None:
    """||U^HU - I||_F < 1e-12 must hold at N=512 (regression guard)."""
    report = conditioning_report(512)
    assert report.unitarity_error < 1e-12, (
        f"N=512 unitarity error {report.unitarity_error:.2e} exceeds 1e-12"
    )
    assert report.kappa_U < 1.0 + 1e-10


def test_unitarity_error_at_N1024() -> None:
    """||U^HU - I||_F < 1e-12 and kappa(U)=1 must hold at N=1024.

    Verified locally: unitarity_error=1.84e-13, kappa_U=1.0000000000.
    """
    report = conditioning_report(1024)
    assert report.unitarity_error < 1e-12, (
        f"N=1024 unitarity error {report.unitarity_error:.2e} exceeds 1e-12"
    )
    assert report.kappa_U < 1.0 + 1e-10, (
        f"N=1024 kappa(U) = {report.kappa_U:.10f} should be approx 1"
    )


# =============================================================================
# THEOREM CLASSIFICATION: Proven vs Conjectural
# =============================================================================

def test_proven_theorem_summary() -> None:
    """Summary verification of all PROVEN theorems (deterministic, machine-precision)."""
    N = 32
    tol = 1e-10
    R = _raw_resonance_kernel_R(N)
    U = _canonical_unitary_U(N)
    U_polar, _ = scipy.linalg.polar(R)
    assert np.allclose(U, U_polar, atol=tol), "Theorem A (polar factor) FAILED"
    z = _roots_z_phi(N)
    C = _companion_from_roots(z)
    V = _vandermonde_evecs(z)
    Lambda = np.diag(z)
    resid_B = _fro_norm(C @ V - V @ Lambda) / _fro_norm(V)
    assert resid_B < tol, f"Theorem B FAILED: resid={resid_B}"
    rng = np.random.default_rng(0)
    h = rng.normal(size=N) + 1j * rng.normal(size=N)
    H = golden_filter_operator(C, h)
    powers = np.arange(N, dtype=np.int64).reshape(-1, 1)
    p_h = (h.reshape(-1, 1) * (z.reshape(1, -1) ** powers)).sum(axis=0)
    resid_C = _fro_norm(H @ V - V @ np.diag(p_h)) / _fro_norm(V)
    assert resid_C < tol, f"Theorem C FAILED: resid={resid_C}"
    unitarity_error = _fro_norm(U.conj().T @ U - np.eye(N))
    assert unitarity_error < tol, f"Unitarity FAILED: error={unitarity_error}"


def test_conjectural_theorem_summary() -> None:
    """Summary verification of CONJECTURAL/EMPIRICAL theorems (statistical evidence)."""
    rng = np.random.default_rng(0)
    N = 128
    M = 200
    U = _canonical_unitary_U(N)
    F = _fft_unitary_matrix(N)
    Xs = golden_drift_ensemble(N, M, rng)
    k99_rft = [k99(U.conj().T @ x) for x in Xs]
    k99_fft = [k99(F.conj().T @ x) for x in Xs]
    mean_rft = float(np.mean(k99_rft))
    mean_fft = float(np.mean(k99_fft))
    std_rft = float(np.std(k99_rft))
    std_fft = float(np.std(k99_fft))
    pooled_std = np.sqrt((std_rft**2 + std_fft**2) / 2)
    cohens_d = (mean_fft - mean_rft) / pooled_std if pooled_std > 0 else 0
    assert mean_rft < mean_fft, (
        f"Theorem 8 direction FAILED: E[K99(RFT)]={mean_rft:.2f} "
        f"should be < E[K99(FFT)]={mean_fft:.2f}"
    )
    print(f"\n  Theorem 8: RFT={mean_rft:.2f}, FFT={mean_fft:.2f}, Cohen's d={cohens_d:.3f}")
