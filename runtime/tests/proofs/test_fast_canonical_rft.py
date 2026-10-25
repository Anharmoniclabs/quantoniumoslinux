# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos

"""
Tests for the Fast Canonical RFT (NUFFT-accelerated).

Validates:
  1. NUFFT-based Φ† and Φ match dense implementations
  2. NUFFT-based Gram matvec G·v = Φ†Φ·v matches dense
  3. Lanczos f(G)b matches dense G^{-1/2} b
  4. Full forward/inverse match CanonicalTrueRFT
  5. Round-trip reconstruction
  6. Energy preservation (unitarity)
  7. Both "precomputed" and "lanczos" modes
  8. Scaling across multiple N
  9. Timing benchmark vs dense
"""

from __future__ import annotations

import time
import numpy as np
import pytest

# Skip entire module if finufft not available
finufft = pytest.importorskip("finufft")

from algorithms.rft.core.resonant_fourier_transform import (
    PHI, rft_basis_matrix,
)
from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT
from algorithms.rft.fast.fast_canonical_rft import (
    FastCanonicalRFT,
    lanczos_func_vec,
    _nufft_phi_adjoint,
    _nufft_phi_forward,
    _phi_frequencies,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_dense_phi(N: int, M: int = None) -> np.ndarray:
    """Build the raw (non-Gram-normalised) Φ matrix."""
    M = M or N
    return rft_basis_matrix(N, M, use_gram_normalization=False)


def _build_dense_phi_tilde(N: int, M: int = None) -> np.ndarray:
    """Build the Gram-normalised Φ̃ matrix."""
    M = M or N
    return rft_basis_matrix(N, M, use_gram_normalization=True)


# ---------------------------------------------------------------------------
# Test: NUFFT Φ† matches dense
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", [32, 64, 128])
def test_nufft_phi_adjoint_matches_dense(N: int) -> None:
    """Φ† x via NUFFT should match dense Φ.conj().T @ x."""
    rng = np.random.default_rng(42 + N)
    Phi = _build_dense_phi(N)
    freqs = _phi_frequencies(N)

    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    x = x.astype(np.complex128)

    dense_result = Phi.conj().T @ x
    nufft_result = _nufft_phi_adjoint(x, freqs, N, eps=1e-12)

    np.testing.assert_allclose(
        nufft_result, dense_result, atol=1e-8, rtol=1e-8,
        err_msg=f"Φ† x mismatch at N={N}",
    )


# ---------------------------------------------------------------------------
# Test: NUFFT Φ matches dense
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", [32, 64, 128])
def test_nufft_phi_forward_matches_dense(N: int) -> None:
    """Φ c via NUFFT should match dense Φ @ c."""
    rng = np.random.default_rng(123 + N)
    Phi = _build_dense_phi(N)
    freqs = _phi_frequencies(N)

    c = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    c = c.astype(np.complex128)

    dense_result = Phi @ c
    nufft_result = _nufft_phi_forward(c, freqs, N, eps=1e-12)

    np.testing.assert_allclose(
        nufft_result, dense_result, atol=1e-8, rtol=1e-8,
        err_msg=f"Φ c mismatch at N={N}",
    )


# ---------------------------------------------------------------------------
# Test: NUFFT-based Gram matvec
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", [32, 64, 128])
def test_gram_matvec_via_nufft(N: int) -> None:
    """G·v via 2 NUFFTs should match dense Φ†Φ·v."""
    rng = np.random.default_rng(999 + N)
    Phi = _build_dense_phi(N)
    freqs = _phi_frequencies(N)

    v = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    v = v.astype(np.complex128)

    dense_Gv = Phi.conj().T @ (Phi @ v)

    # Via NUFFT chain
    Phi_v = _nufft_phi_forward(v, freqs, N, eps=1e-12)
    nufft_Gv = _nufft_phi_adjoint(Phi_v, freqs, N, eps=1e-12)

    np.testing.assert_allclose(
        nufft_Gv, dense_Gv, atol=1e-7, rtol=1e-7,
        err_msg=f"Gram matvec mismatch at N={N}",
    )


# ---------------------------------------------------------------------------
# Test: Lanczos f(G)b for G^{-1/2}
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", [32, 64])
def test_lanczos_gram_inv_sqrt(N: int) -> None:
    """Lanczos G^{-1/2} b should match dense eigendecomposition result."""
    rng = np.random.default_rng(777 + N)
    Phi = _build_dense_phi(N)
    G = Phi.conj().T @ Phi

    # Dense G^{-1/2}
    eigvals, eigvecs = np.linalg.eigh(G)
    G_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.conj().T

    b = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    b = b.astype(np.complex128)

    dense_result = G_inv_sqrt @ b

    # Lanczos with NUFFT-based G matvec
    freqs = _phi_frequencies(N)

    def gram_matvec(v):
        Phi_v = _nufft_phi_forward(v, freqs, N, eps=1e-13)
        return _nufft_phi_adjoint(Phi_v, freqs, N, eps=1e-13)

    lanczos_result = lanczos_func_vec(
        gram_matvec, b,
        func=lambda x: x ** (-0.5),
        m=min(N, 60),
    )

    rel_err = np.linalg.norm(lanczos_result - dense_result) / np.linalg.norm(dense_result)
    assert rel_err < 1e-4, (
        f"Lanczos G^{{-1/2}} b: rel_err={rel_err:.2e} at N={N}"
    )


# ---------------------------------------------------------------------------
# Test: Forward matches CanonicalTrueRFT — precomputed mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", [32, 64, 128])
def test_forward_matches_canonical_precomputed(N: int) -> None:
    """Fast forward (precomputed mode) should match CanonicalTrueRFT.forward."""
    rng = np.random.default_rng(314 + N)
    x = rng.standard_normal(N).astype(np.complex128)

    canonical = CanonicalTrueRFT(N)
    fast = FastCanonicalRFT(N, mode="precomputed", eps=1e-13)

    X_canonical = canonical.forward(x)
    X_fast = fast.forward(x)

    rel_err = np.linalg.norm(X_fast - X_canonical) / np.linalg.norm(X_canonical)
    assert rel_err < 1e-8, (
        f"Forward mismatch (precomputed): rel_err={rel_err:.2e} at N={N}"
    )


# ---------------------------------------------------------------------------
# Test: Forward matches CanonicalTrueRFT — lanczos mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", [32, 64])
def test_forward_matches_canonical_lanczos(N: int) -> None:
    """Fast forward (lanczos mode) should match CanonicalTrueRFT.forward."""
    rng = np.random.default_rng(271 + N)
    x = rng.standard_normal(N).astype(np.complex128)

    canonical = CanonicalTrueRFT(N)
    fast = FastCanonicalRFT(N, mode="lanczos", eps=1e-13, lanczos_m=N)

    X_canonical = canonical.forward(x)
    X_fast = fast.forward(x)

    rel_err = np.linalg.norm(X_fast - X_canonical) / np.linalg.norm(X_canonical)
    assert rel_err < 1e-3, (
        f"Forward mismatch (lanczos): rel_err={rel_err:.2e} at N={N}"
    )


# ---------------------------------------------------------------------------
# Test: Inverse matches CanonicalTrueRFT — precomputed mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", [32, 64, 128])
def test_inverse_matches_canonical_precomputed(N: int) -> None:
    """Fast inverse (precomputed mode) should match CanonicalTrueRFT.inverse."""
    rng = np.random.default_rng(628 + N)
    X = rng.standard_normal(N).astype(np.complex128)

    canonical = CanonicalTrueRFT(N)
    fast = FastCanonicalRFT(N, mode="precomputed", eps=1e-13)

    x_canonical = canonical.inverse(X)
    x_fast = fast.inverse(X)

    rel_err = np.linalg.norm(x_fast - x_canonical) / np.linalg.norm(x_canonical)
    assert rel_err < 1e-8, (
        f"Inverse mismatch (precomputed): rel_err={rel_err:.2e} at N={N}"
    )


# ---------------------------------------------------------------------------
# Test: Round-trip reconstruction — precomputed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", [32, 64, 128, 256])
def test_roundtrip_precomputed(N: int) -> None:
    """inverse(forward(x)) ≈ x in precomputed mode."""
    rng = np.random.default_rng(1000 + N)
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    x = x.astype(np.complex128)

    fast = FastCanonicalRFT(N, mode="precomputed", eps=1e-13)
    x_hat = fast.inverse(fast.forward(x))

    rel_err = np.linalg.norm(x_hat - x) / np.linalg.norm(x)
    assert rel_err < 1e-8, (
        f"Round-trip error (precomputed): {rel_err:.2e} at N={N}"
    )


# ---------------------------------------------------------------------------
# Test: Round-trip reconstruction — lanczos
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", [32, 64])
def test_roundtrip_lanczos(N: int) -> None:
    """inverse(forward(x)) ≈ x in lanczos mode."""
    rng = np.random.default_rng(2000 + N)
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    x = x.astype(np.complex128)

    fast = FastCanonicalRFT(N, mode="lanczos", eps=1e-13, lanczos_m=N)
    x_hat = fast.inverse(fast.forward(x))

    rel_err = np.linalg.norm(x_hat - x) / np.linalg.norm(x)
    assert rel_err < 1e-3, (
        f"Round-trip error (lanczos): {rel_err:.2e} at N={N}"
    )


# ---------------------------------------------------------------------------
# Test: Energy preservation (Parseval)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", [32, 64, 128])
def test_energy_preservation_precomputed(N: int) -> None:
    """‖forward(x)‖ ≈ ‖x‖ (Parseval identity)."""
    rng = np.random.default_rng(3000 + N)
    x = rng.standard_normal(N).astype(np.complex128)

    fast = FastCanonicalRFT(N, mode="precomputed", eps=1e-13)
    X = fast.forward(x)

    energy_in = np.linalg.norm(x)
    energy_out = np.linalg.norm(X)

    np.testing.assert_allclose(
        energy_out, energy_in, rtol=1e-6,
        err_msg=f"Energy not preserved at N={N}",
    )


# ---------------------------------------------------------------------------
# Test: Inner product preservation (unitarity)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", [32, 64, 128])
def test_inner_product_preservation(N: int) -> None:
    """<Φ̃†x, Φ̃†y> ≈ <x, y> (unitarity of Φ̃)."""
    rng = np.random.default_rng(4000 + N)
    x = rng.standard_normal(N).astype(np.complex128)
    y = rng.standard_normal(N).astype(np.complex128)
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)

    fast = FastCanonicalRFT(N, mode="precomputed", eps=1e-13)
    Fx = fast.forward(x)
    Fy = fast.forward(y)

    ip_in = np.vdot(x, y)
    ip_out = np.vdot(Fx, Fy)

    np.testing.assert_allclose(
        ip_out, ip_in, atol=1e-6, rtol=1e-6,
        err_msg=f"Inner product not preserved at N={N}",
    )


# ---------------------------------------------------------------------------
# Test: unitarity_error() method
# ---------------------------------------------------------------------------

def test_unitarity_error_precomputed() -> None:
    """unitarity_error() should report near-zero for precomputed mode."""
    fast = FastCanonicalRFT(64, mode="precomputed", eps=1e-13)
    err = fast.unitarity_error()
    assert err < 1e-7, f"unitarity_error too large: {err:.2e}"


# ---------------------------------------------------------------------------
# Test: reconstruction_error() method
# ---------------------------------------------------------------------------

def test_reconstruction_error_precomputed() -> None:
    """reconstruction_error() should be near machine precision."""
    fast = FastCanonicalRFT(64, mode="precomputed", eps=1e-13)
    rng = np.random.default_rng(5555)
    x = rng.standard_normal(64).astype(np.complex128)
    err = fast.reconstruction_error(x)
    assert err < 1e-8, f"reconstruction_error too large: {err:.2e}"


# ---------------------------------------------------------------------------
# Test: repr
# ---------------------------------------------------------------------------

def test_repr() -> None:
    fast = FastCanonicalRFT(128, mode="precomputed")
    r = repr(fast)
    assert "FastCanonicalRFT" in r
    assert "128" in r
    assert "precomputed" in r


# ---------------------------------------------------------------------------
# Benchmark: timing comparison
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", [64, 256, 1024, 4096])
def test_benchmark_forward(N: int) -> None:
    """
    Benchmark: compare fast (precomputed) vs dense canonical forward.

    This test always passes — it just prints timing results.
    NUFFT Φ operation wins at N ≥ 2048; full forward (with dense G^{-1/2})
    approaches parity around N ≈ 4096.
    """
    rng = np.random.default_rng(9999 + N)
    x = rng.standard_normal(N).astype(np.complex128)

    # Dense canonical
    canonical = CanonicalTrueRFT(N)
    # Warmup
    _ = canonical.forward(x)
    t0 = time.perf_counter()
    n_trials = 20
    for _ in range(n_trials):
        _ = canonical.forward(x)
    t_dense = (time.perf_counter() - t0) / n_trials

    # Fast precomputed
    fast = FastCanonicalRFT(N, mode="precomputed", eps=1e-12)
    _ = fast.forward(x)
    t0 = time.perf_counter()
    for _ in range(n_trials):
        _ = fast.forward(x)
    t_fast = (time.perf_counter() - t0) / n_trials

    # Accuracy check
    X_dense = canonical.forward(x)
    X_fast = fast.forward(x)
    rel_err = np.linalg.norm(X_fast - X_dense) / np.linalg.norm(X_dense)

    ratio = t_dense / max(t_fast, 1e-12)

    print(
        f"\n  N={N:4d} | dense={t_dense*1e3:7.3f}ms | "
        f"fast={t_fast*1e3:7.3f}ms | ratio={ratio:.2f}x | "
        f"rel_err={rel_err:.2e}"
    )

    # The test always passes — the benchmark is informational
    assert rel_err < 1e-6, f"Accuracy regression at N={N}: {rel_err:.2e}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
