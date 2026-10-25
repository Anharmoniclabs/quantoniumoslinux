#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
# Copyright (C) 2025 Luis M. Minier / quantoniumos
#
# This file is part of quantoniumos.
#
# This file is a "Covered File" under the "quantoniumos Research License â€“
# Claims-Practicing Implementations (Non-Commercial)".
#
# You may use this file ONLY for research, academic, or teaching purposes.
# Commercial use is strictly prohibited.
#
# See LICENSE-CLAIMS-NC.md in the root of this repository for details.
"""
RFT Variant Routing Helper
===========================

Signal-analysis helper that selects an RFT variant based on detected signal
characteristics.  This is a *separate* analysis layer from the profile-based
selection inside ``quantonium_os_src.engine.RFTMW.select_optimal_transform``.

IMPORTANT BOUNDARY NOTE (from README â€” March 2026)
---------------------------------------------------
This module provides routing *heuristics* based on signal-type labels and
benchmark data.  It does **not** contain a single benchmark-trained oracle
that always reproduces the variant-shootout winners at runtime.

The correct framing for the Claim 4 framework value is:

    The middleware architecture *can* route among a family of related
    Ï†-derived bases.  The claim is not that one hard-coded router is
    already globally optimal.

Shootout winners (``tests/variant_shootout.py``) for reference:

| Signal class              | Winning transform       |
|---------------------------|-------------------------|
| ECG (real)                | op_rft_fibonacci        |
| Bearing vibration (real)  | op_rft_fibonacci        |
| Fibonacci quasicrystal    | noise_shrink_manifold   |
| Quasi-periodic synthetic  | dct                     |

Benchmarked codec results (``benchmarks/test_all_hybrids.py``):

- H3 CASCADE:      0.673 BPP average, Î·=0 coherence, best general-purpose
- FH5 ENTROPY_GUIDED: 0.406 BPP on edges, 50% improvement over baseline
- H6 DICTIONARY:   49.9 dB PSNR, best quality for smooth signals

Usage
-----
    from algorithms.rft.routing import select_best_variant

    variant = select_best_variant('quantum')  # Returns 8 (CASCADE)
    codec = RFTHybridCodec(1024, variant=variant)
"""

from typing import Optional, Dict, Any
import numpy as np


# ============================================================================
# Variant Registry
# ============================================================================

VARIANT_INFO = {
    0: {
        'name': 'STANDARD',
        'formula': 'Î¸(k) = 2Ï€{k/Ï†}',
        'properties': 'Golden ratio, QR-orthogonalized',
        'use_case': 'General compression',
        'latency_ms': 0.8,
    },
    1: {
        'name': 'HARMONIC',
        'formula': 'Î¸(k) = 2Ï€kÂ³/NÂ³',
        'properties': 'Cubic chirp, harmonic structure',
        'use_case': 'Audio analysis, harmonic extraction',
        'latency_ms': 1.2,
    },
    2: {
        'name': 'FIBONACCI',
        'formula': 'Î¸(k) = 2Ï€Â·F(k mod 32)/Ï†^k',
        'properties': 'Fibonacci scaling',
        'use_case': 'Lattice crypto, integer structures',
        'latency_ms': 1.1,
    },
    3: {
        'name': 'CHAOTIC',
        'formula': 'Î¸(k) = 2Ï€Â·L(k)/k',
        'properties': 'Lyapunov chaotic',
        'use_case': 'Diffusion, mixing layers',
        'latency_ms': 1.5,
    },
    4: {
        'name': 'PRIME',
        'formula': 'Î¸(k) = 2Ï€âˆš(p_k)/k',
        'properties': 'Prime-indexed',
        'use_case': 'Number theory, factorization',
        'latency_ms': 2.0,
    },
    5: {
        'name': 'ADAPTIVE',
        'formula': 'Î¸(k) = 2Ï€e^(-k/N)',
        'properties': 'Exponential decay',
        'use_case': 'Multi-scale analysis',
        'latency_ms': 1.3,
    },
    6: {
        'name': 'SYMBOLIC',
        'formula': 'Î¸(k) = 2Ï€Ï†^k mod 2Ï€',
        'properties': 'Symbolic qubit',
        'use_case': 'Quantum state compression',
        'latency_ms': 1.0,
    },
    7: {
        'name': 'LOGARITHMIC',
        'formula': 'Î¸(k) = 2Ï€log(1+k)',
        'properties': 'Log-periodic',
        'use_case': 'Scale-invariant signals',
        'latency_ms': 1.1,
    },
    8: {
        'name': 'CASCADE',
        'formula': 'Multi-stage hierarchy',
        'properties': 'Î·=0 zero coherence',
        'use_case': 'Best RFT Variant (Internal)',
        'latency_ms': 0.57,
        'bpp': 0.673,
        'psnr': 48.5,
    },
    9: {
        'name': 'BRAIDED',
        'formula': 'Parallel competition',
        'properties': '3-way adaptive mix',
        'use_case': 'Heterogeneous data',
        'latency_ms': 2.5,
    },
    10: {
        'name': 'ADAPTIVE_SPLIT',
        'formula': 'Variance threshold DCT/RFT',
        'properties': '50% BPP improvement',
        'use_case': 'Structure/texture separation',
        'latency_ms': 1.8,
    },
    11: {
        'name': 'ENTROPY_GUIDED',
        'formula': 'Entropy-based routing',
        'properties': '50% BPP on edges',
        'use_case': 'Sharp edges, steps',
        'latency_ms': 7.0,
        'bpp': 0.406,  # On step signals
        'psnr': 42.1,
    },
    12: {
        'name': 'DICTIONARY',
        'formula': 'Dictionary learning atoms',
        'properties': '49.9 dB PSNR',
        'use_case': 'High quality, smooth signals',
        'latency_ms': 3.0,
        'bpp': 0.751,
        'psnr': 49.9,
    },
}


# ============================================================================
# Signal Type Detection
# ============================================================================

def detect_signal_type(signal: np.ndarray) -> str:
    """
    Analyze signal characteristics to determine type.
    
    Args:
        signal: Input signal array
        
    Returns:
        Signal type: 'smooth', 'edges', 'harmonic', 'random', or 'general'
    """
    if len(signal) < 16:
        return 'general'
    
    # Compute signal statistics
    signal = np.asarray(signal, dtype=np.float64)
    
    # 1. Edge detection via gradient variance
    gradient = np.abs(np.diff(signal))
    gradient_var = np.var(gradient)
    gradient_mean = np.mean(gradient)
    edge_ratio = gradient_var / (gradient_mean**2 + 1e-10)
    
    # 2. Smoothness detection via high-frequency energy
    fft = np.fft.fft(signal)
    power_spectrum = np.abs(fft)**2
    n = len(power_spectrum)
    high_freq_energy = np.sum(power_spectrum[n//4:]) / np.sum(power_spectrum)
    
    # 3. Harmonic structure detection via autocorrelation
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr /= autocorr[0] + 1e-10
    
    # Find peaks in autocorrelation
    peaks = []
    for i in range(1, min(len(autocorr)-1, 1000)):
        if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
            if autocorr[i] > 0.3:  # Significant peak
                peaks.append(i)
    
    harmonic_score = len(peaks) / 10.0  # Normalize to 0-1 range
    
    # Decision logic
    if edge_ratio > 10.0:
        return 'edges'
    elif high_freq_energy < 0.15 and harmonic_score < 0.3:
        return 'smooth'
    elif harmonic_score > 0.5:
        return 'harmonic'
    elif high_freq_energy > 0.4 and harmonic_score < 0.2:
        return 'random'
    else:
        return 'general'


# ============================================================================
# Routing Functions
# ============================================================================

def select_best_variant(
    signal_type: str,
    quality_target: str = 'balanced',
    signal: Optional[np.ndarray] = None
) -> int:
    """
    Auto-select optimal RFT variant based on signal characteristics.
    
    Args:
        signal_type: One of 'general', 'edges', 'smooth', 'quantum', 
                     'lattice', 'chaotic', 'audio', 'harmonic', or 'auto'
        quality_target: 'speed', 'balanced', or 'quality'
        signal: Optional signal array for automatic detection (if signal_type='auto')
    
    Returns:
        Variant ID (0-12)
        
    Examples:
        >>> select_best_variant('quantum')
        8  # CASCADE for Î·=0 coherence
        
        >>> select_best_variant('edges')
        11  # ENTROPY_GUIDED for 50% BPP improvement
        
        >>> select_best_variant('auto', signal=my_signal)
        12  # DICTIONARY (detected smooth signal)
    """
    
    # Auto-detection
    if signal_type == 'auto':
        if signal is None:
            raise ValueError("Must provide signal array when using signal_type='auto'")
        signal_type = detect_signal_type(signal)
    
    # Base routing map
    routing_map = {
        'general': 8,      # CASCADE - 0.673 BPP, Î·=0
        'edges': 11,       # ENTROPY_GUIDED - 0.406 BPP on steps
        'smooth': 12,      # DICTIONARY - 49.9 dB PSNR
        'quantum': 8,      # CASCADE - Î·=0 coherence for superposition
        'lattice': 2,      # FIBONACCI - integer structure alignment
        'chaotic': 3,      # CHAOTIC - Lyapunov mixing
        'audio': 1,        # HARMONIC - kÂ³ cubic chirp
        'harmonic': 1,     # HARMONIC - explicit request
        'random': 8,       # CASCADE - general fallback
    }
    
    base_variant = routing_map.get(signal_type, 0)  # Default to STANDARD
    
    # Quality/speed adjustments
    if quality_target == 'quality':
        if signal_type in ['smooth', 'harmonic', 'audio']:
            return 12  # DICTIONARY for max PSNR (49.9 dB)
        elif signal_type == 'edges':
            return 11  # ENTROPY_GUIDED already optimal for edges
        else:
            return 8   # CASCADE for general quality
    
    elif quality_target == 'speed':
        if signal_type == 'general':
            return 0   # STANDARD (0.8ms vs CASCADE 0.57ms, simpler)
        elif signal_type == 'edges':
            return 8   # CASCADE faster than ENTROPY_GUIDED (7ms)
        else:
            return base_variant  # Keep domain-specific optimizations
    
    # Balanced mode (default)
    return base_variant


def get_variant_info(variant: int) -> Dict[str, Any]:
    """
    Get detailed information about a variant.
    
    Args:
        variant: Variant ID (0-12)
        
    Returns:
        Dictionary with variant properties
    """
    return VARIANT_INFO.get(variant, {})


def print_routing_guide():
    """Print complete routing guide for all signal types."""
    print("=" * 80)
    print("RFT VARIANT ROUTING GUIDE")
    print("=" * 80)
    print()
    
    print("Signal Type â†’ Recommended Variant")
    print("-" * 80)
    
    signal_types = [
        ('general', 'General-purpose compression'),
        ('edges', 'Sharp edges, step functions'),
        ('smooth', 'Smooth signals, high quality'),
        ('quantum', 'Quantum state superposition'),
        ('lattice', 'Integer lattice structures'),
        ('chaotic', 'Chaotic mixing, diffusion'),
        ('audio', 'Audio analysis'),
        ('harmonic', 'Harmonic structure extraction'),
    ]
    
    for sig_type, description in signal_types:
        variant = select_best_variant(sig_type)
        info = get_variant_info(variant)
        print(f"{sig_type:>12} | {info['name']:>18} (ID {variant}) - {description}")
    
    print()
    print("=" * 80)
    print("VARIANT PERFORMANCE SUMMARY")
    print("=" * 80)
    print()
    
    print(f"{'Variant':<20} {'ID':>3} {'Latency':>10} {'BPP':>8} {'PSNR':>8} {'Use Case'}")
    print("-" * 80)
    
    for var_id in [8, 11, 12, 0, 1, 2]:  # Show key variants
        info = get_variant_info(var_id)
        latency = f"{info.get('latency_ms', 0):.2f} ms"
        bpp = f"{info.get('bpp', 0):.3f}" if 'bpp' in info else "-"
        psnr = f"{info.get('psnr', 0):.1f} dB" if 'psnr' in info else "-"
        print(f"{info['name']:<20} {var_id:>3} {latency:>10} {bpp:>8} {psnr:>8} {info['use_case']}")
    
    print()


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    print_routing_guide()
    
    print("\nExample Usage:")
    print("-" * 80)
    print()
    print("from algorithms.rft.routing import select_best_variant")
    print("from algorithms.rft.hybrids.rft_hybrid_codec import RFTHybridCodec")
    print()
    print("# Manual selection")
    print("variant = select_best_variant('quantum')  # Returns 8 (CASCADE)")
    print("codec = RFTHybridCodec(1024, variant=variant)")
    print()
    print("# Auto-detection")
    print("import numpy as np")
    print("signal = np.sin(np.linspace(0, 10*np.pi, 1000))  # Smooth signal")
    print("variant = select_best_variant('auto', signal=signal)  # Returns 12 (DICTIONARY)")
    print()
    print("# Quality optimization")
    print("variant = select_best_variant('audio', quality_target='quality')  # Returns 12")
    print()
