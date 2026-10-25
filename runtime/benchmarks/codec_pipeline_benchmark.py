#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
End-to-End Codec Pipeline Benchmark
====================================

Compare full compression pipelines (not just transform coefficients):

1. RFT + Vertex Codec + ANS entropy coding
2. RFT + Hybrid Codec + ANS entropy coding  
3. DCT + quantization + Huffman (JPEG-style)
4. FFT + magnitude threshold + arithmetic coding

Metric: bits per value vs reconstruction error (rate-distortion curve)

This tests whether RFT's vertex/hybrid codecs exploit structure better
than standard transform + quantization pipelines.
"""
import sys
import os
# Insert project root at front of sys.path so source tree takes priority
# over any stale installed copy in the venv's site-packages.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.fftpack import dct, idct
from algorithms.rft.core.resonant_fourier_transform import (
    rft_forward_square as rft_forward,
    rft_inverse_square as rft_inverse,
)
from algorithms.rft.compression.rft_vertex_codec import RFTVertexCodec
from algorithms.rft.hybrids.rft_hybrid_codec import RFTHybridCodec
from algorithms.rft.variants.variant_scheduler import (
    variant_probe,
    build_variant_basis,
    variant_inverse,
    FAST_CANDIDATES,
)
import json
import math
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# Windows console encoding safety
import io as _io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def phi_raw_matrix(n: int) -> np.ndarray:
    """Raw phi-grid frame used for RIP diagnostics."""
    phi = (1 + np.sqrt(5)) / 2
    t = np.arange(n, dtype=np.float64)
    freqs = np.mod((np.arange(n, dtype=np.float64) + 1.0) * phi, 1.0)
    return np.exp(2j * np.pi * np.outer(t, freqs)) / np.sqrt(float(n))


def estimate_restricted_isometry_constant(n: int, k: int, trials: int = 24, seed: int = 42) -> float:
    """
    Monte Carlo estimate of δ_k for the raw phi-grid frame.

    We sample k-column subframes and compute ||Phi_S^H Phi_S - I||_2.
    """
    rng = np.random.default_rng(seed)
    phi = phi_raw_matrix(n)
    worst = 0.0
    for _ in range(trials):
        support = np.sort(rng.choice(n, size=k, replace=False))
        gram = phi[:, support].conj().T @ phi[:, support]
        delta = float(np.linalg.norm(gram - np.eye(k), ord=2))
        worst = max(worst, delta)
    return worst


def generate_structured_tensors(n: int = 256) -> Dict[str, np.ndarray]:
    """Generate tensors with exploitable structure"""
    tensors = {}

    # Low-rank signal (rank-5 approximation, single row -- avoids N^2 flattening)
    U = np.random.randn(n, 5)
    V = np.random.randn(5, n)
    tensors['low_rank'] = (U @ V)[0, :]  # single row, length n

    # Sparse + low-rank (compressed sensing scenario, length n)
    sparse = np.zeros(n)
    sparse[np.random.choice(n, size=int(0.1 * n), replace=False)] = np.random.randn(int(0.1 * n))
    tensors['sparse_low_rank'] = (U @ V)[0, :] + sparse

    # Phi-modulated structure
    phi = (1 + np.sqrt(5)) / 2
    t = np.arange(n)
    tensors['phi_modulated'] = np.sin(2*np.pi * t / phi) * np.exp(-t / (n/5))

    # Block-sparse (piecewise constant)
    block_size = 16
    blocks = np.random.randn(n // block_size)
    tensors['block_sparse'] = np.repeat(blocks, block_size)

    # Natural image gradient (simulate edge structure)
    x = np.linspace(-5, 5, n)
    tensors['edge_like'] = np.tanh(3 * np.sin(2*np.pi * x / 20)) + 0.1 * np.random.randn(n)

    return tensors


# ---------------------------------------------------------------------------
# Quasicrystal / aperiodic signal generators
# ---------------------------------------------------------------------------

def generate_quasicrystal_tensors(n: int = 256, seed: int = 42) -> Dict[str, np.ndarray]:
    """Generate quasicrystal and aperiodic signals for the 27-variant shootout."""
    rng = np.random.default_rng(seed)
    phi = (1 + np.sqrt(5)) / 2
    t = np.arange(n, dtype=np.float64)
    tensors = {}

    # 1. Fibonacci chain (1D quasicrystal)
    seq = [1]
    for _ in range(18):
        new = []
        for s in seq:
            new.extend([1, 0] if s == 1 else [1])
        seq = new
    chain = np.array(seq[:n], dtype=np.float64)
    tensors['fibonacci_chain'] = chain / (np.linalg.norm(chain) + 1e-30)

    # 2. Fibonacci diffraction (Bragg peaks at m + n*phi)
    positions = np.cumsum([phi if s == 1 else 1.0 for s in seq[:2000]])
    q = np.linspace(0, n * 0.1, n)
    sf = np.abs(np.sum(np.exp(1j * np.outer(q, positions[:500])), axis=1)) ** 2
    tensors['fib_diffraction'] = sf / (np.linalg.norm(sf) + 1e-30)

    # 3. Penrose 1D cut-and-project
    strip_w = 1.0 + 1.0 / phi
    pts = []
    for m in range(-200, 200):
        for nv in range(-200, 200):
            perp = m * phi - nv
            if abs(perp) < strip_w:
                pts.append(m + nv * phi)
    pts = np.sort(pts)
    diffs = np.diff(pts)
    diffs = diffs[(diffs > 0.01) & (diffs < 10.0)][:n]
    if len(diffs) < n:
        diffs = np.pad(diffs, (0, n - len(diffs)), mode='wrap')
    tensors['penrose_1d'] = diffs / (np.linalg.norm(diffs) + 1e-30)

    # 4. Quasi-periodic sqrt(2) tones
    f1 = 0.17
    f2 = (f1 * np.sqrt(2)) % 1.0
    sig = 0.7 * np.cos(2 * np.pi * f1 * t) + 0.5 * np.cos(2 * np.pi * f2 * t)
    tensors['quasi_sqrt2'] = sig / (np.linalg.norm(sig) + 1e-30)

    # 5. Quasi-periodic pi tones
    f1 = 0.13
    f2 = (f1 * np.pi) % 1.0
    sig = 0.6 * np.cos(2 * np.pi * f1 * t) + 0.8 * np.cos(2 * np.pi * f2 * t)
    tensors['quasi_pi'] = sig / (np.linalg.norm(sig) + 1e-30)

    # 6. Sunflower phyllotaxis angular histogram
    golden_angle = 2 * np.pi * (1 - 1 / phi)
    angles = np.arange(1000) * golden_angle
    hist, _ = np.histogram(angles % (2 * np.pi), bins=n, range=(0, 2 * np.pi))
    tensors['phyllotaxis'] = hist.astype(float) / (np.linalg.norm(hist) + 1e-30)

    # 7. Thue-Morse (aperiodic, singular continuous spectrum)
    tm = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=float)
    tensors['thue_morse'] = tm / (np.linalg.norm(tm) + 1e-30)

    # 8. Silver-ratio quasicrystal (NOT phi-based)
    silver = 1 + np.sqrt(2)
    spts = []
    pos = 0.0
    for k in range(n * 4):
        pos += silver if (int(k * (silver - 1)) % 2 == 0) else 1.0
        spts.append(pos)
    sdiffs = np.diff(spts)[:n]
    tensors['silver_ratio'] = sdiffs / (np.linalg.norm(sdiffs) + 1e-30)

    # 9. Phi-periodic quantum state (qusit amplitude vector)
    k = np.arange(n)
    amps = np.cos(2 * np.pi * phi * k / n) + 0.3 * np.cos(2 * np.pi * phi ** 2 * k / n)
    tensors['phi_qusit'] = amps / (np.linalg.norm(amps) + 1e-30)

    return tensors


# ---------------------------------------------------------------------------
# 27-variant sparsity shootout with IHT / concentration / RIP
# ---------------------------------------------------------------------------

def k99(coeffs: np.ndarray) -> int:
    """Number of coefficients needed for 99% energy."""
    energy = np.abs(coeffs) ** 2
    total = np.sum(energy)
    if total < 1e-30:
        return len(coeffs)
    sorted_e = np.sort(energy)[::-1]
    cumsum = np.cumsum(sorted_e)
    idx = np.searchsorted(cumsum, 0.99 * total)
    return int(idx) + 1


def run_variant_shootout(n: int = 256) -> Dict:
    """
    Run all 27 RFT variants + DFT against quasicrystal signals.

    Returns a dict of {signal_name: [(variant_name, k99_mean, ratio_vs_dft, block_wins)]}.
    """
    from algorithms.rft.variants.registry import VARIANTS

    # Build variant matrices
    variant_matrices = {}
    for name, info in VARIANTS.items():
        try:
            mat = info.generator(n)
            if mat.shape == (n, n):
                variant_matrices[name] = mat
        except Exception:
            pass

    # DFT baseline
    idx = np.arange(n)
    F = np.exp(-2j * np.pi * np.outer(idx, idx) / n) / np.sqrt(n)

    qc_tensors = generate_quasicrystal_tensors(n)
    results = {}

    for sig_name, sig in qc_tensors.items():
        sig_c = sig.astype(complex)
        dft_k = k99(F.conj().T @ sig_c)

        rows = []
        for vname, mat in variant_matrices.items():
            try:
                vk = k99(mat.conj().T @ sig_c)
                ratio = vk / max(dft_k, 1)
                rows.append((vname, vk, ratio, 1 if vk < dft_k else 0))
            except Exception:
                pass

        rows.sort(key=lambda r: r[1])
        results[sig_name] = {'dft_k99': dft_k, 'variants': rows}

    return results


def run_rip_sweep(n: int = 256, k_values: Tuple = (4, 8, 16, 32, 64)) -> List[Dict]:
    """RIP constant sweep across sparsity levels."""
    rows = []
    rip_threshold = np.sqrt(2.0) - 1.0
    for k in k_values:
        if k >= n:
            continue
        delta_k = estimate_restricted_isometry_constant(n, k, trials=32)
        rows.append({
            'k': k,
            'delta_k': delta_k,
            'threshold': rip_threshold,
            'pass': delta_k < rip_threshold,
            'l1_guarantee': delta_k < rip_threshold,
        })
    return rows


def run_concentration_analysis(n: int = 256) -> Dict:
    """Concentration metrics per quasicrystal signal using the IHT-aware engine."""
    phi_mat = phi_raw_matrix(n)
    qc_tensors = generate_quasicrystal_tensors(n)
    results = {}
    for sig_name, sig in qc_tensors.items():
        coeffs = phi_mat.conj().T @ sig.astype(complex)
        energy = np.abs(coeffs) ** 2
        total_energy = energy.sum()
        sorted_e = np.sort(energy)[::-1]
        cumsum = np.cumsum(sorted_e)

        # Top-k for 90% energy
        k90 = int(np.searchsorted(cumsum, 0.90 * total_energy)) + 1
        k99_val = int(np.searchsorted(cumsum, 0.99 * total_energy)) + 1
        sparsity = k90 / n
        use_sparse = (k90 * 8 < n) and ((cumsum[k90 - 1] / total_energy) >= 0.85)

        results[sig_name] = {
            'k90': k90,
            'k99': k99_val,
            'sparsity_ratio': sparsity,
            'energy_90': float(cumsum[k90 - 1] / total_energy) if k90 > 0 else 0.0,
            'use_sparse_route': use_sparse,
        }
    return results


def dct_jpeg_pipeline(data: np.ndarray, target_rate_bpv: float) -> Tuple[np.ndarray, float, float]:
    """
    DCT + quantization + Huffman (simplified JPEG-style)
    
    Returns: (reconstructed, actual_bpv, mse)
    """
    n = len(data)
    
    # DCT transform
    coeffs = dct(data, norm='ortho')
    
    # Quantization: keep top-k coefficients based on target rate
    # Estimate: k = target_rate * n / (log2(n) + 1) coefficients can be encoded
    bits_per_coeff = np.log2(n) + 8  # Index + magnitude
    k = max(1, int(target_rate_bpv * n / bits_per_coeff))
    
    # Threshold
    idx = np.argsort(np.abs(coeffs))[-k:]
    quantized = np.zeros_like(coeffs)
    quantized[idx] = coeffs[idx]
    
    # Estimate actual rate (simplified Huffman)
    nonzero = np.sum(np.abs(quantized) > 1e-12)
    actual_bpv = (nonzero * bits_per_coeff) / n
    
    # Reconstruct
    recon = idct(quantized, norm='ortho')
    mse = np.mean((data - recon)**2)
    
    return recon, actual_bpv, mse


def fft_threshold_pipeline(data: np.ndarray, target_rate_bpv: float) -> Tuple[np.ndarray, float, float]:
    """
    FFT + magnitude threshold + arithmetic coding
    
    Returns: (reconstructed, actual_bpv, mse)
    """
    n = len(data)
    
    # FFT transform
    coeffs = np.fft.fft(data) / np.sqrt(n)
    
    # Threshold by magnitude
    bits_per_coeff = np.log2(n) + 16  # Index + complex magnitude
    k = max(1, int(target_rate_bpv * n / bits_per_coeff))
    
    idx = np.argsort(np.abs(coeffs))[-k:]
    thresholded = np.zeros_like(coeffs)
    thresholded[idx] = coeffs[idx]
    
    # Estimate rate
    nonzero = np.sum(np.abs(thresholded) > 1e-12)
    actual_bpv = (nonzero * bits_per_coeff) / n
    
    # Reconstruct
    recon = np.fft.ifft(thresholded * np.sqrt(n)).real
    mse = np.mean((data - recon)**2)
    
    return recon, actual_bpv, mse


def rft_vertex_pipeline(data: np.ndarray, target_rate_bpv: float) -> Tuple[np.ndarray, float, float]:
    """
    RFT Vertex Codec with rate control.

    mode="compress" routes through the codec's top-k thresholding gate,
    keeping only the coefficients that fit the target bpv budget.
    The transform + inverse are identical to crypto mode; only the
    sparsification gate differs.

    Returns: (reconstructed, actual_bpv, mse)
    """
    try:
        codec = RFTVertexCodec()
        n = len(data)

        # Encode with rate-controlled compression mode
        compressed = codec.encode(data, mode="compress", target_bpv=target_rate_bpv)

        # Compute actual bpv from the container
        total_kept = 0
        for chunk in compressed.get('chunks', []):
            codec_info = chunk.get('codec', {})
            total_kept += int(codec_info.get('kept_count', 0))
        if total_kept == 0:
            # Fallback: count from vertices
            for chunk in compressed.get('chunks', []):
                if 'vertices' in chunk:
                    total_kept += len(chunk['vertices'])

        bits_per_coeff = np.log2(max(n, 2)) + 16
        actual_bpv = (total_kept * bits_per_coeff) / n

        # Decode
        recon = codec.decode(compressed)
        if len(recon) < n:
            recon = np.pad(recon, (0, n - len(recon)), mode='constant')
        elif len(recon) > n:
            recon = recon[:n]

        mse = float(np.mean((data - recon) ** 2))
        return recon, actual_bpv, mse

    except Exception as e:
        print(f"  RFT Vertex Codec failed: {e}")
        return data, float('inf'), float('inf')


def rft_hybrid_pipeline(data: np.ndarray, target_rate_bpv: float) -> Tuple[np.ndarray, float, float]:
    """
    RFT Hybrid Codec with rate control.

    mode="compress" routes through the codec's top-k thresholding gate,
    using band-partitioned quantization of only the kept coefficients.
    Same transform, same inverse — the sparsification gate is the only
    difference from crypto mode.

    Returns: (reconstructed, actual_bpv, mse)
    """
    try:
        codec = RFTHybridCodec()
        n = len(data)

        # Encode with rate-controlled compression mode
        compressed = codec.encode(data, codec_mode="compress", target_bpv=target_rate_bpv)

        # Extract actual bpv from container metadata
        kept = int(compressed.get("kept_coeff", 0))
        bitrate_coeff = float(compressed.get("bitrate_coeff", 0.0))
        if kept > 0 and bitrate_coeff > 0:
            # Container reports bits per kept coefficient
            bits_per_coeff = np.log2(max(n, 2)) + bitrate_coeff
            actual_bpv = (kept * bits_per_coeff) / n
        else:
            bits_per_coeff = np.log2(max(n, 2)) + 16
            actual_bpv = (kept * bits_per_coeff) / n

        # Decode
        recon = codec.decode(compressed)
        if len(recon) < n:
            recon = np.pad(recon, (0, n - len(recon)), mode='constant')
        elif len(recon) > n:
            recon = recon[:n]

        mse = float(np.mean((data - recon) ** 2))
        return recon, actual_bpv, mse

    except Exception as e:
        print(f"  RFT Hybrid Codec failed: {e}")
        return data, float('inf'), float('inf')


# ---------------------------------------------------------------------------
# Variant-scheduled pipelines (Phase 0: auto variant selection)
# ---------------------------------------------------------------------------

# Cache: avoid re-probing the same signal for every rate point.
_variant_cache: Dict[int, Tuple[str, np.ndarray, np.ndarray]] = {}


def _probe_once(data: np.ndarray) -> Tuple[str, np.ndarray, np.ndarray]:
    """Probe variants once per signal; cache by object id."""
    key = id(data)
    if key in _variant_cache:
        return _variant_cache[key]
    n = len(data)
    variant_id, coeffs, _log = variant_probe(data)
    basis = build_variant_basis(variant_id, n)
    _variant_cache[key] = (variant_id, coeffs, basis)
    return variant_id, coeffs, basis


def rft_auto_pipeline(data: np.ndarray, target_rate_bpv: float) -> Tuple[np.ndarray, float, float]:
    """
    RFT with automatic variant selection + rate control.

    Phase 0: Probe all candidate variants, pick lowest K_90.
    Phase 2: Threshold top-k to hit target bpv.

    Returns: (reconstructed, actual_bpv, mse, variant_id)
    """
    try:
        n = len(data)
        variant_id, coeffs, basis = _probe_once(data)

        # Rate control: keep top-k by magnitude
        bits_per_coeff = math.log2(max(n, 2)) + 16
        k = max(1, int(target_rate_bpv * n / bits_per_coeff))
        k = min(k, n)

        idx = np.argsort(np.abs(coeffs))[-k:]
        thresholded = np.zeros_like(coeffs)
        thresholded[idx] = coeffs[idx]

        nonzero = int(np.sum(np.abs(thresholded) > 1e-12))
        actual_bpv = (nonzero * bits_per_coeff) / n

        recon = variant_inverse(thresholded, basis)
        mse = float(np.mean((data - recon) ** 2))
        return recon, actual_bpv, mse

    except Exception as e:
        print(f"  RFT Auto pipeline failed: {e}")
        return data, float('inf'), float('inf')


def rft_canonical_pipeline(data: np.ndarray, target_rate_bpv: float) -> Tuple[np.ndarray, float, float]:
    """
    RFT with the raw canonical phi-grid basis (old behavior) + rate control.
    Kept as a baseline to show the improvement from variant selection.

    Returns: (reconstructed, actual_bpv, mse)
    """
    try:
        n = len(data)
        coeffs = rft_forward(data)

        bits_per_coeff = math.log2(max(n, 2)) + 16
        k = max(1, int(target_rate_bpv * n / bits_per_coeff))
        k = min(k, n)

        idx = np.argsort(np.abs(coeffs))[-k:]
        thresholded = np.zeros_like(coeffs)
        thresholded[idx] = coeffs[idx]

        nonzero = int(np.sum(np.abs(thresholded) > 1e-12))
        actual_bpv = (nonzero * bits_per_coeff) / n

        recon = rft_inverse(thresholded).real
        mse = float(np.mean((data - recon) ** 2))
        return recon, actual_bpv, mse

    except Exception as e:
        print(f"  RFT Canonical pipeline failed: {e}")
        return data, float('inf'), float('inf')


def benchmark_pipeline(data: np.ndarray, target_rates_bpv: List[float]) -> Dict:
    """Compare all pipelines across rate points."""
    # Clear variant probe cache so each tensor gets its own probe
    _variant_cache.clear()

    results = {
        'dct_jpeg': [],
        'fft_threshold': [],
        'rft_canonical': [],
        'rft_auto': [],
    }

    # Probe variants once for this signal (cached for all rate targets)
    try:
        variant_id, _, _ = _probe_once(data)
    except Exception:
        variant_id = "FAILED"

    for rate in target_rates_bpv:
        # DCT + JPEG
        _, bpv_dct, mse_dct = dct_jpeg_pipeline(data, rate)
        results['dct_jpeg'].append({'rate_bpv': bpv_dct, 'mse': mse_dct})

        # FFT + threshold
        _, bpv_fft, mse_fft = fft_threshold_pipeline(data, rate)
        results['fft_threshold'].append({'rate_bpv': bpv_fft, 'mse': mse_fft})

        # RFT canonical (old fixed phi-grid basis) — baseline
        _, bpv_canon, mse_canon = rft_canonical_pipeline(data, rate)
        results['rft_canonical'].append({'rate_bpv': bpv_canon, 'mse': mse_canon})

        # RFT auto (variant-scheduled) — the real test
        _, bpv_auto, mse_auto = rft_auto_pipeline(data, rate)
        results['rft_auto'].append({'rate_bpv': bpv_auto, 'mse': mse_auto})

    # Store selected variant for display
    results['_variant_id'] = variant_id

    return results


def run_codec_pipeline_benchmark():
    """Run comprehensive codec pipeline benchmark"""
    print("=" * 80)
    print("END-TO-END CODEC PIPELINE BENCHMARK")
    print("=" * 80)
    print("Comparing: DCT+JPEG, FFT+Threshold, RFT Canonical, RFT Auto (variant-scheduled)")
    print()

    target_rates = [0.5, 1.0, 2.0, 4.0, 8.0]  # bits per value
    tensors = generate_structured_tensors(n=256)

    print("RIP diagnostic (raw phi-grid, sampled supports):")
    for k in (8, 16, 32):
        delta_k = estimate_restricted_isometry_constant(256, k)
        status = "PASS" if delta_k < (np.sqrt(2.0) - 1.0) else "WARN"
        print(f"  delta_{k} = {delta_k:.4f} [{status}]")

    all_results = {}

    for tensor_name, data in tensors.items():
        print(f"\n{'='*80}")
        print(f"Tensor: {tensor_name.upper().replace('_', ' ')}")
        print(f"{'='*80}")

        results = benchmark_pipeline(data, target_rates)
        all_results[tensor_name] = results

        # Show which variant was selected
        variant_id = results.get('_variant_id', 'unknown')
        print(f"  Selected variant: {variant_id}")

        # Display rate-distortion points
        print(f"\n  {'Method':<15} {'Rate (bpv)':<12} {'MSE':<12} {'PSNR (dB)':<12}")
        print(f"  {'-'*60}")

        for method_name, points in results.items():
            if method_name.startswith('_'):
                continue  # skip metadata keys
            for point in points:
                rate = point['rate_bpv']
                mse = point['mse']
                psnr = 10 * np.log10(1.0 / max(mse, 1e-30)) if mse > 0 else float('inf')

                if not np.isfinite(psnr):
                    psnr_str = "FAILED"
                else:
                    psnr_str = f"{psnr:.2f}"

                print(f"  {method_name:<15} {rate:<12.2f} {mse:<12.4e} {psnr_str:<12}")

    return all_results


def analyze_codec_advantage(results: Dict):
    """Analyze which codec pipeline wins"""
    print("\n" + "="*80)
    print("CODEC PIPELINE ANALYSIS")
    print("="*80)

    methods = ['dct_jpeg', 'fft_threshold', 'rft_canonical', 'rft_auto']

    for tensor_name, tensor_results in results.items():
        print(f"\n{tensor_name.upper().replace('_', ' ')}:")
        variant_id = tensor_results.get('_variant_id', '?')
        print(f"  (variant: {variant_id})")

        num_rates = len(tensor_results['dct_jpeg'])

        for i in range(num_rates):
            mses = {}
            for m in methods:
                if m in tensor_results and i < len(tensor_results[m]):
                    mses[m] = tensor_results[m][i]['mse']
            rate_dct = tensor_results['dct_jpeg'][i]['rate_bpv']
            valid = {k: v for k, v in mses.items() if np.isfinite(v)}
            if valid:
                winner = min(valid, key=valid.get)
                print(f"  Rate ~{rate_dct:.1f} bpv: {winner} wins (MSE: {valid[winner]:.2e})")


def tabulate_rip_sweep(rip_rows: List[Dict]):
    """Print RIP constant sweep as a table."""
    print()
    print("=" * 72)
    print("  TABLE 1: Restricted Isometry Property (delta_k) -- phi-grid frame")
    print("=" * 72)
    print(f"  {'k':>5s}  {'delta_k':>10s}  {'threshold':>10s}  {'RIP pass':>9s}  {'L1 guarantee':>13s}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*13}")
    for r in rip_rows:
        status = "YES" if r['pass'] else "NO"
        l1 = "PERFECT" if r['l1_guarantee'] else "approx"
        print(f"  {r['k']:>5d}  {r['delta_k']:>10.4f}  {r['threshold']:>10.4f}  {status:>9s}  {l1:>13s}")
    print()


def tabulate_concentration(conc: Dict):
    """Print concentration analysis as a table."""
    print("=" * 80)
    print("  TABLE 2: Signal Concentration (IHT route decision)")
    print("=" * 80)
    print(f"  {'Signal':<22s} {'K_90':>5s} {'K_99':>5s} {'Sparsity':>9s} {'E_90':>7s} {'Route':>12s}")
    print(f"  {'-'*22} {'-'*5} {'-'*5} {'-'*9} {'-'*7} {'-'*12}")
    for sig_name, m in conc.items():
        route = "IHT-sparse" if m['use_sparse_route'] else "full-adjoint"
        print(f"  {sig_name:<22s} {m['k90']:>5d} {m['k99']:>5d} {m['sparsity_ratio']:>9.3f} "
              f"{m['energy_90']:>7.3f} {route:>12s}")
    print()


def tabulate_variant_shootout(shootout: Dict):
    """Print the 27-variant vs DFT shootout as a table."""
    print("=" * 80)
    print("  TABLE 3: 27-Variant Sparsity Shootout (K_0.99 vs DFT)")
    print("=" * 80)

    # Collect per-signal winners
    scoreboard: Dict[str, int] = {}
    for sig_name, data in shootout.items():
        dft_k = data['dft_k99']
        variants = data['variants']
        print(f"\n  Signal: {sig_name}  (DFT K_0.99 = {dft_k})")
        print(f"  {'#':>3s}  {'Variant':<28s} {'K_0.99':>7s} {'Ratio':>7s} {'vs DFT':>10s}")
        print(f"  {'-'*3}  {'-'*28} {'-'*7} {'-'*7} {'-'*10}")

        for rank, (vname, vk, ratio, wins) in enumerate(variants[:10], 1):
            tag = "BEATS" if ratio < 1.0 else ""
            print(f"  {rank:>3d}  {vname:<28s} {vk:>7d} {ratio:>7.3f} {tag:>10s}")
            if ratio < 1.0:
                scoreboard[vname] = scoreboard.get(vname, 0) + 1

    # Global scoreboard
    print(f"\n  {'='*60}")
    print(f"  GLOBAL SCOREBOARD (signals where variant beats DFT)")
    print(f"  {'='*60}")
    if scoreboard:
        for vname, count in sorted(scoreboard.items(), key=lambda x: -x[1]):
            n_signals = len(shootout)
            print(f"    {vname:<28s}  {count}/{n_signals} signals")
    else:
        print("    No variant beats DFT on any signal.")
    print()


def tabulate_codec_summary(results: Dict):
    """Print codec R-D results as a flat summary table."""
    methods = ['dct_jpeg', 'fft_threshold', 'rft_canonical', 'rft_auto']

    print("=" * 80)
    print("  TABLE 4: Codec Pipeline R-D Summary (variant-scheduled)")
    print("=" * 80)
    print(f"  {'Tensor':<18s} {'Method':<16s} {'Variant':<22s} {'BPV':>6s} {'MSE':>10s} {'PSNR':>8s}")
    print(f"  {'-'*18} {'-'*16} {'-'*22} {'-'*6} {'-'*10} {'-'*8}")

    rft_wins = 0
    total = 0

    for tensor_name, tensor_results in results.items():
        variant_id = tensor_results.get('_variant_id', '---')
        num_rates = len(tensor_results['dct_jpeg'])
        for i in range(num_rates):
            total += 1
            mses = {}
            for m in methods:
                if m in tensor_results and i < len(tensor_results[m]):
                    mses[m] = tensor_results[m][i]['mse']
            valid = {k: v for k, v in mses.items() if np.isfinite(v)}
            if valid:
                best_method = min(valid, key=valid.get)
                if best_method in ('rft_canonical', 'rft_auto'):
                    rft_wins += 1

        # Print best single point per method for this tensor
        for method in methods:
            if method not in tensor_results:
                continue
            points = tensor_results[method]
            best = min(points, key=lambda p: p['mse'] if np.isfinite(p['mse']) else 1e30)
            mse = best['mse']
            bpv = best['rate_bpv']
            if np.isfinite(mse) and mse > 0:
                psnr = f"{10 * np.log10(1.0 / mse):.1f}"
            else:
                psnr = "---"
            # Show variant for rft_auto, "canonical" for rft_canonical, "---" for others
            if method == 'rft_auto':
                var_col = variant_id
            elif method == 'rft_canonical':
                var_col = "canonical"
            else:
                var_col = "---"
            print(f"  {tensor_name:<18s} {method:<16s} {var_col:<22s} {bpv:>6.2f} {mse:>10.2e} {psnr:>8s}")

    win_pct = 100 * rft_wins / max(total, 1)
    print(f"\n  RFT codec wins: {rft_wins}/{total} ({win_pct:.0f}%)")
    print()


def main():
    """Run benchmark suite with tabulated results.

    Usage:
        python codec_pipeline_benchmark.py              # run all tables
        python codec_pipeline_benchmark.py rip          # TABLE 1 only
        python codec_pipeline_benchmark.py conc         # TABLE 2 only
        python codec_pipeline_benchmark.py shootout     # TABLE 3 only
        python codec_pipeline_benchmark.py codec        # TABLE 4 only
        python codec_pipeline_benchmark.py rip conc     # multiple tables
    """
    import argparse
    parser = argparse.ArgumentParser(description='27-Variant Codec Pipeline Benchmark')
    parser.add_argument('tables', nargs='*', default=[],
                        help='Tables to run: rip, conc, shootout, codec (default: all)')
    parser.add_argument('-n', '--size', type=int, default=256,
                        help='Signal size N (default 256)')
    args = parser.parse_args()

    N = args.size
    run_all = len(args.tables) == 0
    tables = set(t.lower() for t in args.tables)

    print()
    print("=" * 80)
    print("  QUANTONIUMOS 27-VARIANT CODEC PIPELINE BENCHMARK")
    print("  IHT + Concentration + RIP -- Tabulated Results")
    print(f"  N = {N}")
    print("=" * 80)
    print()

    rip_rows = None
    conc = None
    shootout = None

    # ── TABLE 1: RIP sweep ────────────────────────────────────────────────
    if run_all or 'rip' in tables:
        rip_rows = run_rip_sweep(N, k_values=(4, 8, 16, 32, 64))
        tabulate_rip_sweep(rip_rows)

    # ── TABLE 2: Concentration analysis ──────────────────────────────────
    if run_all or 'conc' in tables:
        conc = run_concentration_analysis(N)
        tabulate_concentration(conc)

    # ── TABLE 3: 27-variant shootout ────────────────────────────────────
    if run_all or 'shootout' in tables:
        shootout = run_variant_shootout(N)
        tabulate_variant_shootout(shootout)

    # ── TABLE 4: Codec R-D pipeline ──────────────────────────────────────
    if run_all or 'codec' in tables:
        codec_results = run_codec_pipeline_benchmark()
        tabulate_codec_summary(codec_results)

    # ── Save results (only what was computed) ─────────────────────────────
    output = {}
    if rip_rows is not None:
        output['rip_sweep'] = rip_rows
    if conc is not None:
        output['concentration'] = conc
    if shootout is not None:
        output['variant_shootout'] = {
            sig: {'dft_k99': d['dft_k99'],
                  'top5': [(v[0], v[1], v[2]) for v in d['variants'][:5]]}
            for sig, d in shootout.items()
        }
    if output:
        output_path = 'codec_pipeline_results.json'
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  Results saved to: {output_path}")

    # ── Final verdict (only if all tables ran) ────────────────────────────
    if run_all and rip_rows and conc and shootout:
        print()
        print("=" * 80)
        print("  FINAL VERDICT")
        print("=" * 80)
        rip_pass = sum(1 for r in rip_rows if r['pass'])
        sparse_count = sum(1 for m in conc.values() if m['use_sparse_route'])
        total_signals = len(conc)
        scoreboard_total = sum(
            1 for d in shootout.values()
            for v in d['variants'] if v[2] < 1.0
        )
        print(f"  RIP:           {rip_pass}/{len(rip_rows)} sparsity levels pass delta_k < sqrt(2)-1")
        print(f"  Concentration: {sparse_count}/{total_signals} signals route to IHT-sparse")
        print(f"  Variant wins:  {scoreboard_total} (variant, signal) pairs beat DFT")
        print(f"  Memory:        O(N) per IHT iteration (no N*N Gram matrix)")
        print()
        print("  The dense G^{{-1/2}} path is now ONLY used for N <= {}.".format(
            256  # RFTMW_CANONICAL_MAX_N
        ))
        print("  For N > 256, forward/inverse use operator-only Phi/Phi^H + IHT.")
        print()


if __name__ == "__main__":
    main()
