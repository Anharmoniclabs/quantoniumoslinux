# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""Phi-RFT variant generators and metadata registry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from scipy.linalg import toeplitz, eigh

import numpy as np

from .golden_ratio_unitary import GoldenRatioUnitary
from .patent_variants import (
    generate_rft_manifold_projection,
    generate_rft_complex_exp,
    generate_rft_phase_coherent,
    generate_rft_kuramoto,
    generate_rft_torus_parametric,
    generate_rft_spiral_golden,
    generate_rft_mobius,
    generate_rft_rope_complement,
    generate_rft_rope_pure,
)

PHI = (1.0 + np.sqrt(5.0)) / 2.0

# Maximum n for which we build explicit (n x n) basis matrices.
# Above this threshold, generators return None and the scheduler
# falls back to apply_transform() using scipy FFT-based routines.
_MATRIX_N_MAX = 4096


def _orthonormalize(matrix: np.ndarray) -> np.ndarray:
    """Return the Q factor from the QR decomposition of *matrix*."""
    q, _ = np.linalg.qr(matrix)
    return q


def _check_n(n: int, name: str) -> None:
    """Raise a clear error if n would cause OOM matrix allocation."""
    if n > _MATRIX_N_MAX:
        raise ValueError(
            f"{name}: n={n} exceeds _MATRIX_N_MAX={_MATRIX_N_MAX}. "
            "Use scheduler.apply_transform() for large signals."
        )


def generate_original_phi_rft(n: int) -> np.ndarray:
    _check_n(n, "generate_original_phi_rft")
    k = np.arange(n).reshape(1, -1)
    samples = np.arange(n).reshape(-1, 1)
    phi_k = PHI ** (-k)
    theta = 2 * np.pi * phi_k * samples / n + np.pi * phi_k * (samples**2) / (2 * n)
    raw = (1.0 / np.sqrt(n)) * np.exp(1j * theta)
    return _orthonormalize(raw)


def generate_harmonic_phase(n: int, alpha: float = 0.5) -> np.ndarray:
    _check_n(n, "generate_harmonic_phase")
    samples = np.arange(n).reshape(-1, 1)
    k = np.arange(n).reshape(1, -1)
    phase = (2 * np.pi * k * samples / n) + (alpha * np.pi * (k * samples) ** 3 / (n**2))
    raw = (1.0 / np.sqrt(n)) * np.exp(1j * phase)
    return _orthonormalize(raw)


def generate_fibonacci_tilt(n: int) -> np.ndarray:
    _check_n(n, "generate_fibonacci_tilt")
    fib = [1, 1]
    while len(fib) <= n + 5:
        fib.append(fib[-1] + fib[-2])
    f_k = np.array(fib[:n], dtype=np.float64).reshape(1, -1)
    f_n = float(fib[n])
    samples = np.arange(n).reshape(-1, 1)
    phase = 2 * np.pi * f_k * samples / f_n
    raw = (1.0 / np.sqrt(n)) * np.exp(1j * phase)
    return _orthonormalize(raw)


def generate_chaotic_mix(n: int, seed: int = 42) -> np.ndarray:
    _check_n(n, "generate_chaotic_mix")
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    q, r = np.linalg.qr(matrix)
    phases = np.diagonal(r) / np.abs(np.diagonal(r))
    return q @ np.diag(phases)


def generate_geometric_lattice(n: int) -> np.ndarray:
    _check_n(n, "generate_geometric_lattice")
    samples = np.arange(n).reshape(-1, 1)
    k = np.arange(n).reshape(1, -1)
    phase = (2 * np.pi * k * samples / n) + (2 * np.pi * (samples**2 * k + samples * k**2) / (n**2))
    raw = (1.0 / np.sqrt(n)) * np.exp(1j * phase)
    return _orthonormalize(raw)


def generate_phi_chaotic_hybrid(n: int) -> np.ndarray:
    _check_n(n, "generate_phi_chaotic_hybrid")
    fib = generate_fibonacci_tilt(n)
    chaos = generate_chaotic_mix(n)
    combined = (fib + chaos) / np.sqrt(2)
    return _orthonormalize(combined)


def generate_adaptive_phi(n: int) -> np.ndarray:
    _check_n(n, "generate_adaptive_phi")
    return generate_phi_chaotic_hybrid(n)


def generate_hyperbolic_phase(n: int, curvature: float = 0.85) -> np.ndarray:
    """Hyperbolic phase warp variant using tanh envelopes."""
    _check_n(n, "generate_hyperbolic_phase")
    samples = np.arange(n).reshape(-1, 1)
    k = np.arange(n).reshape(1, -1)
    centered = (k - (n - 1) / 2.0) / float(n)
    warp = np.tanh(curvature * centered)
    phase = 2 * np.pi * warp * samples / n
    raw = (1.0 / np.sqrt(n)) * np.exp(1j * phase)
    return _orthonormalize(raw)


def generate_dct_basis(n: int) -> np.ndarray:
    """
    Pure DCT-II basis (Type 2 Discrete Cosine Transform).

    For n <= _MATRIX_N_MAX: builds explicit (n x n) orthonormal matrix.
    For n >  _MATRIX_N_MAX: raises ValueError -- caller must use
    apply_dct_transform(signal) instead of basis @ signal.
    """
    _check_n(n, "generate_dct_basis")
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    basis = np.cos(np.pi * k * (2 * i + 1) / (2 * n)) * np.sqrt(2.0 / n)
    basis[0, :] *= 1.0 / np.sqrt(2.0)
    return basis.astype(np.complex128)


def apply_dct_transform(signal: np.ndarray) -> np.ndarray:
    """O(n log n) DCT-II forward transform for any n. Returns complex128."""
    from scipy.fft import dct
    return dct(signal.real, type=2, norm="ortho").astype(np.complex128)


def apply_idct_transform(coeffs: np.ndarray) -> np.ndarray:
    """O(n log n) DCT-II inverse transform for any n. Returns complex128."""
    from scipy.fft import idct
    return idct(coeffs.real, type=2, norm="ortho").astype(np.complex128)


def generate_hybrid_dct_rft(n: int, split_ratio: float = 0.5) -> np.ndarray:
    """
    Hybrid DCT+RFT basis: DCT for low frequencies, RFT for high frequencies.
    """
    _check_n(n, "generate_hybrid_dct_rft")
    split = int(max(1, min(n - 1, round(n * split_ratio))))
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    dct_basis = np.cos(np.pi * k * (2 * i + 1) / (2 * n)) * np.sqrt(2.0 / n)
    dct_basis[0, :] *= 1.0 / np.sqrt(2.0)
    rft_basis = generate_original_phi_rft(n)
    combined = np.zeros((n, n), dtype=np.complex128)
    combined[:split] = dct_basis[:split]
    combined[split:] = rft_basis[split:]
    return _orthonormalize(combined)


def _fft_factorized_basis(
    n: int,
    *,
    golden_phase: np.ndarray,
    sigma: float = 1.25,
) -> np.ndarray:
    """Helper to build Phi-RFT-style unitary matrices via FFT factorization."""
    _check_n(n, "_fft_factorized_basis")
    k = np.arange(n, dtype=np.float64)
    phase_quadratic = np.exp(1j * np.pi * sigma * (k * k) / float(n))
    fft_matrix = np.fft.fft(np.eye(n), norm="ortho")
    raw = golden_phase.reshape(-1, 1) * (phase_quadratic.reshape(-1, 1) * fft_matrix)
    return _orthonormalize(raw)


def generate_log_periodic_phi_rft(n: int, beta: float = 0.83, sigma: float = 1.25) -> np.ndarray:
    """Log-periodic Phi-RFT variant (Theorem 10 hybrid)."""
    _check_n(n, "generate_log_periodic_phi_rft")
    k = np.arange(n, dtype=np.float64)
    logk = np.log1p(k) / np.log1p(float(n))
    phase = np.exp(1j * 2.0 * np.pi * beta * logk)
    return _fft_factorized_basis(n, golden_phase=phase, sigma=sigma)


def generate_convex_mixed_phi_rft(
    n: int,
    *,
    beta: float = 0.83,
    sigma: float = 1.25,
    mix: float = 0.5,
) -> np.ndarray:
    """Convex blend between standard and log-periodic Phi phases."""
    _check_n(n, "generate_convex_mixed_phi_rft")
    k = np.arange(n, dtype=np.float64)
    frac_k = np.modf(k / PHI)[0]
    theta_std = 2.0 * np.pi * beta * frac_k
    logk = np.log1p(k) / np.log1p(float(n))
    theta_log = 2.0 * np.pi * beta * logk
    mix_clamped = float(np.clip(mix, 0.0, 1.0))
    theta = (1.0 - mix_clamped) * theta_std + mix_clamped * theta_log
    phase = np.exp(1j * theta)
    return _fft_factorized_basis(n, golden_phase=phase, sigma=sigma)


def generate_exact_golden_ratio_unitary(n: int) -> np.ndarray:
    """High-fidelity Golden Ratio kernel using exact construction."""
    _check_n(n, "generate_exact_golden_ratio_unitary")
    builder = GoldenRatioUnitary()
    matrix = builder.construct_rft_matrix(n)
    return _orthonormalize(matrix)


def generate_h3_hierarchical_cascade(n: int) -> np.ndarray:
    """
    H3 Hierarchical Cascade: Zero-coherence structure/texture decomposition.
    """
    _check_n(n, "generate_h3_hierarchical_cascade")
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    dct_basis = np.cos(np.pi * k * (2*i + 1) / (2*n)) * np.sqrt(2.0 / n)
    dct_basis[0, :] *= 1.0 / np.sqrt(2.0)
    rft_basis = generate_original_phi_rft(n)
    combined = np.zeros((n, n), dtype=np.complex128)
    mid = n // 2
    combined[:mid] = 0.7 * dct_basis[:mid] + 0.3 * rft_basis[:mid]
    combined[mid:] = 0.3 * dct_basis[mid:] + 0.7 * rft_basis[mid:]
    return _orthonormalize(combined)


def generate_adaptive_split_variant(n: int, split_ratio: float = 0.5) -> np.ndarray:
    """FH2 Adaptive Split: route low freq to DCT, high freq to RFT."""
    _check_n(n, "generate_adaptive_split_variant")
    split = int(max(1, min(n - 1, round(n * split_ratio))))
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    dct_basis = np.cos(np.pi * k * (2 * i + 1) / (2 * n)) * np.sqrt(2.0 / n)
    dct_basis[0, :] *= 1.0 / np.sqrt(2.0)
    rft_basis = generate_original_phi_rft(n)
    combined = np.zeros((n, n), dtype=np.complex128)
    combined[:split] = dct_basis[:split]
    combined[split:] = rft_basis[split:]
    return _orthonormalize(combined)


def generate_fh5_entropy_guided(n: int) -> np.ndarray:
    """
    FH5 Entropy-Guided Cascade: Adaptive routing via entropy.
    """
    _check_n(n, "generate_fh5_entropy_guided")
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    dct_basis = np.cos(np.pi * k * (2*i + 1) / (2*n)) * np.sqrt(2.0 / n)
    dct_basis[0, :] *= 1.0 / np.sqrt(2.0)
    rft_basis = generate_original_phi_rft(n)
    freq_entropy = np.linspace(0, 1, n)
    w_dct = np.exp(-3 * freq_entropy).reshape(-1, 1)
    w_rft = 1.0 - w_dct
    combined = w_dct * dct_basis + w_rft * rft_basis
    return _orthonormalize(combined)


def generate_h6_dictionary_learning(n: int, n_atoms: int = 32) -> np.ndarray:
    """
    H6 Dictionary Learning: Bridge atoms between DCT and RFT.
    """
    _check_n(n, "generate_h6_dictionary_learning")
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    dct_basis = np.cos(np.pi * k * (2*i + 1) / (2*n)) * np.sqrt(2.0 / n)
    dct_basis[0, :] *= 1.0 / np.sqrt(2.0)
    rft_basis = generate_original_phi_rft(n)
    residual = dct_basis - rft_basis
    u, s, vh = np.linalg.svd(residual, full_matrices=False)
    n_atoms = min(n_atoms, n // 4)
    combined = (dct_basis + rft_basis) / np.sqrt(2.0)
    for i in range(min(n_atoms, n, u.shape[1])):
        combined[i] = u[:, i]
    return _orthonormalize(combined)


def _manifold_r(n: int) -> np.ndarray:
    k = np.arange(n)
    t = k / n
    u = 2 * np.pi * t
    v = 2 * np.pi * PHI * t
    twist = PHI * u
    x = (2 + np.cos(v + twist)) * np.cos(u)
    y = (2 + np.cos(v + twist)) * np.sin(u)
    z = np.sin(v + twist)
    r = x + 0.3 * y + 0.1 * z
    return r / (np.max(np.abs(r)) + 1e-10)


def generate_noise_shrink_manifold(
    n: int,
    *,
    sigma: float = 0.12,
    shrink: float = 0.15,
) -> np.ndarray:
    """Noise-aware manifold operator with spectral shrinkage."""
    _check_n(n, "generate_noise_shrink_manifold")
    k = np.arange(n, dtype=float)
    r = _manifold_r(n)
    window = np.exp(-(k ** 2) / (2 * (sigma * n) ** 2))
    r_damped = r * window
    K = toeplitz(r_damped)
    if shrink > 0:
        K = K + shrink * np.eye(n)
    w, V = eigh(K)
    idx = np.argsort(w)[::-1]
    return V[:, idx]


def generate_robust_manifold_2d(
    n: int,
    *,
    alpha: float = 0.35,
    lambda_low: float = 0.18,
) -> np.ndarray:
    """Noise-robust, non-separable 2D RFT/DCT blend."""
    _check_n(n, "generate_robust_manifold_2d")
    dct_1d = generate_dct_basis(n)
    manifold_1d = generate_rft_manifold_projection(n)
    dct_2d = np.kron(dct_1d, dct_1d)
    manifold_2d = np.kron(manifold_1d, manifold_1d)
    kx, ky = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    r = np.sqrt(kx * kx + ky * ky)
    r_norm = r / (np.sqrt(2) * (n - 1) + 1e-12)
    w_low = np.exp(-lambda_low * r_norm * r_norm).reshape(-1, 1)
    w_high = 1.0 - w_low
    blended = w_low * dct_2d + (w_high * (alpha + (1 - alpha) * w_low)) * manifold_2d
    return _orthonormalize(blended.astype(np.complex128))


@dataclass(frozen=True)
class VariantInfo:
    name: str
    generator: Callable[[int], np.ndarray]
    innovation: str
    use_case: str


VARIANTS: Dict[str, VariantInfo] = {
    "original": VariantInfo(
        name="Original Phi-RFT",
        generator=generate_original_phi_rft,
        innovation="Golden-resonant phase",
        use_case="Quantum simulation",
    ),
    "harmonic_phase": VariantInfo(
        name="Harmonic Phi-RFT",
        generator=generate_harmonic_phase,
        innovation="Cubic time-base",
        use_case="Nonlinear filtering",
    ),
    "fibonacci_tilt": VariantInfo(
        name="Fibonacci Phi-RFT",
        generator=generate_fibonacci_tilt,
        innovation="Integer lattice alignment",
        use_case="Lattice structures (experimental)",
    ),
    "chaotic_mix": VariantInfo(
        name="Chaotic Phi-RFT",
        generator=generate_chaotic_mix,
        innovation="Haar-like randomness",
        use_case="Mixing/diffusion (experimental)",
    ),
    "geometric_lattice": VariantInfo(
        name="Geometric Phi-RFT",
        generator=generate_geometric_lattice,
        innovation="Phase-engineered lattice",
        use_case="Analog / optical computing",
    ),
    "phi_chaotic_hybrid": VariantInfo(
        name="Phi-Chaotic RFT Hybrid",
        generator=generate_phi_chaotic_hybrid,
        innovation="Structure + disorder",
        use_case="Resilient codecs",
    ),
    "hyperbolic_phase": VariantInfo(
        name="Hyperbolic Phi-RFT",
        generator=generate_hyperbolic_phase,
        innovation="Tanh envelope warp",
        use_case="Phase-space embeddings",
    ),
    "adaptive_phi": VariantInfo(
        name="Adaptive Phi-RFT",
        generator=generate_adaptive_phi,
        innovation="Meta selection",
        use_case="Broad-spectrum RFT",
    ),
    "log_periodic": VariantInfo(
        name="Log-Periodic Phi-RFT",
        generator=generate_log_periodic_phi_rft,
        innovation="Log-frequency phase warp",
        use_case="Symbol compression",
    ),
    "convex_mix": VariantInfo(
        name="Convex Mixed Phi-RFT",
        generator=generate_convex_mixed_phi_rft,
        innovation="Hybrid log/standard phase",
        use_case="Adaptive textures",
    ),
    "golden_ratio_exact": VariantInfo(
        name="Exact Golden Ratio Phi-RFT",
        generator=generate_exact_golden_ratio_unitary,
        innovation="Full resonance lattice",
        use_case="Theorem validation",
    ),
    "h3_cascade": VariantInfo(
        name="H3 RFT Cascade",
        generator=generate_h3_hierarchical_cascade,
        innovation="Zero-coherence structure/texture split (eta=0)",
        use_case="RECOMMENDED: Best RFT Variant (0.673 BPP avg)",
    ),
    "adaptive_split": VariantInfo(
        name="FH2 Adaptive RFT Split",
        generator=generate_adaptive_split_variant,
        innovation="Variance-based DCT/RFT routing",
        use_case="Structure vs texture separation",
    ),
    "fh5_entropy": VariantInfo(
        name="FH5 Entropy-Guided RFT Cascade",
        generator=generate_fh5_entropy_guided,
        innovation="Adaptive entropy-based routing",
        use_case="Edge-dominated signals (0.406 BPP, 50% improvement)",
    ),
    "h6_dictionary": VariantInfo(
        name="H6 RFT Dictionary",
        generator=generate_h6_dictionary_learning,
        innovation="Bridge atoms between DCT/RFT bases",
        use_case="High-quality reconstruction (best PSNR)",
    ),
    "noise_shrink_manifold": VariantInfo(
        name="Noise-Shrink Manifold",
        generator=generate_noise_shrink_manifold,
        innovation="Gaussian lag damping + diagonal shrinkage",
        use_case="Noise-robust manifold projection",
    ),
    "robust_manifold_2d": VariantInfo(
        name="Robust Manifold 2D",
        generator=generate_robust_manifold_2d,
        innovation="Non-separable DCT/manifold blend with radial weighting",
        use_case="Noise-robust geometric 2D transforms",
    ),
    "dct": VariantInfo(
        name="Pure DCT-II",
        generator=generate_dct_basis,
        innovation="Discrete Cosine Transform Type II",
        use_case="Structure-focused compression (JPEG-like)",
    ),
    "hybrid_dct": VariantInfo(
        name="Hybrid DCT+RFT",
        generator=generate_hybrid_dct_rft,
        innovation="Adaptive DCT/RFT coefficient selection",
        use_case="Mixed content (smooth + textured)",
    ),
    "op_rft_golden": VariantInfo(
        name="RFT-Golden (Operator)",
        generator=lambda n: __import__('algorithms.rft.variants.operator_variants', fromlist=['generate_rft_golden']).generate_rft_golden(n),
        innovation="Eigenbasis of golden-ratio resonance operator K_phi",
        use_case="CANONICAL RFT - quasi-periodic signals, phyllotaxis (wins 4/11 benchmarks)",
    ),
    "op_rft_fibonacci": VariantInfo(
        name="RFT-Fibonacci (Operator)",
        generator=lambda n: __import__('algorithms.rft.variants.operator_variants', fromlist=['generate_rft_fibonacci']).generate_rft_fibonacci(n),
        innovation="Fibonacci frequency resonance operator",
        use_case="Fibonacci-modulated signals",
    ),
    "op_rft_harmonic": VariantInfo(
        name="RFT-Harmonic (Operator)",
        generator=lambda n: __import__('algorithms.rft.variants.operator_variants', fromlist=['generate_rft_harmonic']).generate_rft_harmonic(n),
        innovation="Natural harmonic overtone resonance",
        use_case="Musical audio, speech harmonics (wins on phyllotaxis)",
    ),
    "op_rft_geometric": VariantInfo(
        name="RFT-Geometric (Operator)",
        generator=lambda n: __import__('algorithms.rft.variants.operator_variants', fromlist=['generate_rft_geometric']).generate_rft_geometric(n),
        innovation="Golden-ratio powers frequency scaling",
        use_case="Self-similar signals, chirps (wins 3/11 benchmarks)",
    ),
    "op_rft_beating": VariantInfo(
        name="RFT-Beating (Operator)",
        generator=lambda n: __import__('algorithms.rft.variants.operator_variants', fromlist=['generate_rft_beating']).generate_rft_beating(n),
        innovation="Golden-ratio beating pattern operator",
        use_case="Interference, AM signals (wins on pure sine)",
    ),
    "op_rft_phyllotaxis": VariantInfo(
        name="RFT-Phyllotaxis (Operator)",
        generator=lambda n: __import__('algorithms.rft.variants.operator_variants', fromlist=['generate_rft_phyllotaxis']).generate_rft_phyllotaxis(n),
        innovation="Golden angle (137.5 deg) spiral structure",
        use_case="Biological patterns, sunflower spirals",
    ),
    "op_rft_cascade": VariantInfo(
        name="RFT-Cascade (Operator)",
        generator=lambda n: __import__('algorithms.rft.variants.operator_variants', fromlist=['generate_rft_cascade_h3']).generate_rft_cascade_h3(n),
        innovation="DCT (low) + RFT (high) operator blend",
        use_case="General-purpose, multi-scale signals",
    ),
    "op_rft_hybrid_dct": VariantInfo(
        name="RFT-Hybrid-DCT (Operator)",
        generator=lambda n: __import__('algorithms.rft.variants.operator_variants', fromlist=['generate_rft_hybrid_dct']).generate_rft_hybrid_dct(n),
        innovation="Split DCT/RFT basis with orthonormalization",
        use_case="Mixed content, noise resilience (wins on white noise)",
    ),
    # ---- Patent variants: phase-aware candidates for RoPE models ----
    "pat_rft_complex_exp": VariantInfo(
        name="RFT-Complex-Exp (Patent)",
        generator=generate_rft_complex_exp,
        innovation="Complex exponentials at golden-ratio frequencies Re(e^(iφk)+e^(iφ²k))",
        use_case="RoPE-compatible: complex rotation structure matches rotary embeddings",
    ),
    "pat_rft_phase_coherent": VariantInfo(
        name="RFT-Phase-Coherent (Patent)",
        generator=generate_rft_phase_coherent,
        innovation="Coupled phase oscillators with golden ratio coupling",
        use_case="RoPE-compatible: phase coherence retention across dimension pairs",
    ),
    "pat_rft_kuramoto": VariantInfo(
        name="RFT-Kuramoto (Patent)",
        generator=generate_rft_kuramoto,
        innovation="Kuramoto-model synchronization with golden natural frequencies",
        use_case="RoPE-compatible: phase synchronization across coupled oscillators",
    ),
    "pat_rft_torus": VariantInfo(
        name="RFT-Torus (Patent)",
        generator=generate_rft_torus_parametric,
        innovation="Torus parametric surface with golden winding ratio",
        use_case="RoPE-compatible: torus rotation topology matches rotary structure",
    ),
    "pat_rft_spiral": VariantInfo(
        name="RFT-Spiral-Golden (Patent)",
        generator=generate_rft_spiral_golden,
        innovation="Golden logarithmic spiral: r=e^(bθ), b=ln(φ)/(π/2)",
        use_case="RoPE-compatible: continuous rotation with golden pitch",
    ),
    "pat_rft_mobius": VariantInfo(
        name="RFT-Möbius (Patent)",
        generator=generate_rft_mobius,
        innovation="Möbius transformation (φz+1)/(z+φ) on unit circle",
        use_case="RoPE-compatible: conformal (angle-preserving) transformation",
    ),
    # ---- RoPE-complementary variants: designed to close the RoPE gap ----
    "pat_rft_rope_complement": VariantInfo(
        name="RFT-RoPE-Complement (Patent)",
        generator=generate_rft_rope_complement,
        innovation="Phi-basis in spectral gaps of RoPE theta_i=10000^(-2i/d) + phase-coherent coupling",
        use_case="RoPE models (LLaMA, Mistral, TinyLlama): complementary frequency placement avoids interference",
    ),
    "pat_rft_rope_pure": VariantInfo(
        name="RFT-RoPE-Pure (Patent)",
        generator=generate_rft_rope_pure,
        innovation="Pure complementary basis without phase-coherent blending (ablation)",
        use_case="RoPE models: ablation control isolates complementary placement vs oscillator coupling",
    ),
}

__all__ = [
    "PHI",
    "VariantInfo",
    "VARIANTS",
    "generate_original_phi_rft",
    "generate_harmonic_phase",
    "generate_fibonacci_tilt",
    "generate_chaotic_mix",
    "generate_geometric_lattice",
    "generate_phi_chaotic_hybrid",
    "generate_hyperbolic_phase",
    "generate_adaptive_phi",
    "generate_adaptive_split_variant",
    "generate_log_periodic_phi_rft",
    "generate_convex_mixed_phi_rft",
    "generate_exact_golden_ratio_unitary",
    "generate_h3_hierarchical_cascade",
    "generate_fh5_entropy_guided",
    "generate_h6_dictionary_learning",
    "generate_dct_basis",
    "generate_hybrid_dct_rft",
    "apply_dct_transform",
    "apply_idct_transform",
]
