# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Three-Distance Spectral Router
================================

DECISION 3 in the RFTMW compression pipeline.

The Steinhaus-Sos Three-Distance Theorem (1957) guarantees that N points
of the form {frac((k+1)*phi)} on the unit circle have gaps taking exactly
2 or 3 distinct values.  This module exploits that structure for KV-cache
and general spectral compression by:

1. Computing the phi-grid gap structure for any dimension N
2. Partitioning coefficients into spectral zones defined by the gap hierarchy
3. Allocating retention budget proportional to per-zone energy
4. Providing a compress/decompress API that wraps any basis with
   three-distance-aware coefficient allocation

Pipeline position::

    DECISION 1   Backend selection     (unified_transform_scheduler.py)
    DECISION 2   Codec variant         (routing.py)
  > DECISION 3   Spectral allocation   (three_distance_router.py)   <-- THIS
    R-D CHECK    RFT vs INT8+zlib      (entropy.py)
    CODEC        Encode                (rft_hybrid_codec.py)

The three-distance router does NOT change which basis is used -- it changes
HOW coefficients are retained after the forward transform.  It replaces
flat top-k with zone-aware allocation that preserves spectral locality.

Theory
------
For phi = (1+sqrt(5))/2 and N points {frac(k*phi)} sorted on [0,1):

- Gaps take exactly 2 or 3 values (Three-Distance Theorem)
- Gap sizes are determined by Fibonacci representation of N (Zeckendorf)
- Dense zones (small gaps) contain correlated frequency content
- Sparse zones (large gaps) contain isolated spectral peaks

Retaining contiguous spectral zones preserves inter-coefficient
correlations that flat top-k destroys.  This is the same geometric
property that gives the 47-80% cross-window leakage reduction in
positional encoding (benchmarks/kv_cache_boundary_error.py).
"""

from __future__ import annotations

import numpy as np
from functools import lru_cache
from typing import Optional, Tuple

PHI = (1.0 + np.sqrt(5.0)) / 2.0


# ── Gap structure analysis ──────────────────────────────────────────────────

@lru_cache(maxsize=64)
def analyze_gap_structure(n: int) -> dict:
    """Analyze the three-distance gap structure for dimension N.

    Returns a dict with:
        freqs:          sorted phi-grid frequencies
        sort_order:     indices mapping original -> sorted order
        inv_order:      indices mapping sorted -> original order
        gap_sizes:      the 2-3 distinct gap values
        zone_boundaries: indices where large gaps occur (zone separators)
        zones:          list of index arrays, one per spectral zone
        density:        per-coefficient density score (inverse of local gap)
    """
    freqs = np.mod(np.arange(1, n + 1, dtype=np.float64) * PHI, 1.0)
    sort_order = np.argsort(freqs)
    sorted_freqs = freqs[sort_order]

    # Inverse mapping
    inv_order = np.empty_like(sort_order)
    inv_order[sort_order] = np.arange(n)

    # Compute gaps (including wraparound)
    gaps = np.diff(sorted_freqs)
    gap_wrap = 1.0 + sorted_freqs[0] - sorted_freqs[-1]
    all_gaps = np.concatenate([gaps, [gap_wrap]])

    # Identify distinct gap sizes (cluster with tolerance)
    sorted_gaps = np.sort(np.unique(np.round(all_gaps, 10)))

    # Per-coefficient local density (smaller gap = denser)
    left_gaps = np.empty(n)
    left_gaps[0] = gap_wrap
    left_gaps[1:] = gaps
    right_gaps = np.empty(n)
    right_gaps[-1] = gap_wrap
    right_gaps[:-1] = gaps

    local_gap = (left_gaps + right_gaps) / 2.0
    # Map to original order
    density_sorted = 1.0 / np.maximum(local_gap, 1e-15)
    density = np.empty(n)
    density[sort_order] = density_sorted

    # Zone boundaries: split at the largest gap size
    if len(sorted_gaps) > 1:
        # Threshold between largest and second-largest gap
        threshold = (sorted_gaps[-1] + sorted_gaps[-2]) / 2.0
    else:
        threshold = sorted_gaps[0] * 1.5

    zone_boundaries = np.where(all_gaps > threshold)[0]

    # Build raw zones (contiguous runs between large gaps)
    raw_zones = []
    prev = 0
    for zb in zone_boundaries:
        if zb + 1 > prev:
            raw_zones.append(sort_order[prev:zb + 1])
        prev = zb + 1
    if prev < n:
        raw_zones.append(sort_order[prev:])
    raw_zones = [z for z in raw_zones if len(z) > 0]

    # Merge small adjacent zones to get O(log_phi(N)) coarse zones.
    # Target: each zone should have at least n / (2*log_phi(n)) elements
    # so that energy-proportional allocation has room to differentiate.
    import math
    log_phi_n = max(3, int(math.log(max(n, 2)) / math.log(PHI)))
    min_zone_size = max(2, n // (2 * log_phi_n))

    zones = []
    accum = np.array([], dtype=int)
    for rz in raw_zones:
        accum = np.concatenate([accum, rz]) if len(accum) else rz
        if len(accum) >= min_zone_size:
            zones.append(accum)
            accum = np.array([], dtype=int)
    # Merge any leftover into the last zone
    if len(accum) > 0:
        if len(zones) > 0:
            zones[-1] = np.concatenate([zones[-1], accum])
        else:
            zones.append(accum)

    return {
        'freqs': freqs,
        'sort_order': sort_order,
        'inv_order': inv_order,
        'gap_sizes': sorted_gaps,
        'zone_boundaries': zone_boundaries,
        'zones': zones,
        'density': density,
        'n_zones': len(zones),
    }


# ── Zone-aware coefficient allocation ───────────────────────────────────────

def allocate_budget(coeffs: np.ndarray, retain_frac: float,
                    gap_info: Optional[dict] = None) -> np.ndarray:
    """Allocate retention budget across three-distance spectral zones.

    Instead of flat top-k (which scatters retained coefficients randomly
    across the spectrum), this allocates k coefficients proportional to
    per-zone energy density.  Zones with concentrated energy keep more
    coefficients; low-energy zones are pruned more aggressively.

    Parameters
    ----------
    coeffs      : Complex coefficient array from forward transform
    retain_frac : Fraction of coefficients to retain (0.0 to 1.0)
    gap_info    : Pre-computed gap structure from analyze_gap_structure().
                  If None, computed from len(coeffs).

    Returns
    -------
    Masked coefficient array (same shape, zeros where pruned)
    """
    n = len(coeffs)
    total_k = max(1, int(n * retain_frac))

    if gap_info is None:
        gap_info = analyze_gap_structure(n)

    zones = gap_info['zones']
    density = gap_info['density']

    if len(zones) <= 1:
        # Single zone -- fall back to flat top-k
        return _top_k(coeffs, total_k)

    # Per-zone energy weighted by density
    zone_scores = np.array([
        np.sum(np.abs(coeffs[z]) ** 2 * density[z])
        for z in zones
    ])
    total_score = zone_scores.sum()

    if total_score < 1e-30:
        return _top_k(coeffs, total_k)

    # Proportional allocation with minimum 1 per zone
    raw_alloc = zone_scores / total_score * total_k
    zone_k = np.maximum(1, np.round(raw_alloc).astype(int))

    # Clamp to zone size
    for i, z in enumerate(zones):
        zone_k[i] = min(zone_k[i], len(z))

    # Balance to hit exact total
    _balance_allocation(zone_k, zones, zone_scores, total_k)

    # Top-k within each zone
    out = np.zeros_like(coeffs)
    for i, z in enumerate(zones):
        k_z = int(zone_k[i])
        if k_z > 0:
            mags = np.abs(coeffs[z])
            top_idx = np.argsort(mags)[::-1][:k_z]
            out[z[top_idx]] = coeffs[z[top_idx]]

    return out


def _top_k(coeffs: np.ndarray, k: int) -> np.ndarray:
    """Standard top-k by magnitude."""
    idx = np.argsort(np.abs(coeffs))[::-1][:k]
    out = np.zeros_like(coeffs)
    out[idx] = coeffs[idx]
    return out


def _balance_allocation(zone_k, zones, zone_scores, total_k):
    """Adjust zone_k to sum exactly to total_k."""
    while zone_k.sum() > total_k:
        # Trim from zone with lowest energy per allocated coefficient
        per_coeff = zone_scores / np.maximum(zone_k, 1)
        trim_idx = int(np.argmin(per_coeff))
        if zone_k[trim_idx] > 1:
            zone_k[trim_idx] -= 1
        else:
            break
    while zone_k.sum() < total_k:
        # Boost zone with highest energy per allocated coefficient
        per_coeff = zone_scores / np.maximum(zone_k, 1)
        boost_idx = int(np.argmax(per_coeff))
        if zone_k[boost_idx] < len(zones[boost_idx]):
            zone_k[boost_idx] += 1
        else:
            break


# ── Compress / Decompress wrappers ──────────────────────────────────────────

def three_distance_compress(
    signal: np.ndarray,
    basis: np.ndarray,
    retain_frac: float,
    gap_info: Optional[dict] = None,
) -> Tuple[np.ndarray, dict]:
    """Compress a signal using three-distance-aware coefficient allocation.

    Parameters
    ----------
    signal      : Real-valued input signal (1D)
    basis       : Unitary basis matrix (N x N, complex)
    retain_frac : Fraction of coefficients to retain
    gap_info    : Pre-computed from analyze_gap_structure(). If None, computed.

    Returns
    -------
    (reconstructed_signal, metadata)
    """
    n = len(signal)
    if gap_info is None:
        gap_info = analyze_gap_structure(n)

    # Forward transform
    coeffs = basis.conj().T @ signal.astype(np.complex128)

    # Three-distance allocation
    masked_coeffs = allocate_budget(coeffs, retain_frac, gap_info)

    # Count retained
    n_retained = int(np.count_nonzero(masked_coeffs))

    # Inverse transform
    reconstructed = np.real(basis @ masked_coeffs)

    metadata = {
        'n_zones': gap_info['n_zones'],
        'n_retained': n_retained,
        'retain_frac_actual': n_retained / n,
        'gap_sizes': gap_info['gap_sizes'].tolist(),
        'mse': float(np.mean((signal - reconstructed) ** 2)),
    }

    return reconstructed, metadata


# ── Diagnostic ──────────────────────────────────────────────────────────────

def diagnose(n: int) -> str:
    """Print diagnostic for the three-distance structure at dimension N."""
    info = analyze_gap_structure(n)
    lines = [
        f"Three-Distance Analysis for N={n}",
        f"  Phi-grid frequencies: frac((k+1)*phi) for k=0..{n-1}",
        f"  Distinct gap sizes: {len(info['gap_sizes'])} ({', '.join(f'{g:.6f}' for g in info['gap_sizes'])})",
        f"  Number of spectral zones: {info['n_zones']}",
        f"  Zone sizes: {[len(z) for z in info['zones']]}",
        f"  Density range: [{info['density'].min():.2f}, {info['density'].max():.2f}]",
    ]
    return '\n'.join(lines)
