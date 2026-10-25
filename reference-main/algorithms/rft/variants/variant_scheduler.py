# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Variant Scheduler â€” Phase 0 of the RFT codec pipeline.

Probes candidate RFT variants via concentration analysis (K_90) and selects
the variant that concentrates the most signal energy into the fewest
coefficients.  The selected variant's basis is then passed to Phase 1
(N-based dispatch) and Phase 2 (mode gate: crypto vs compress).

Cost: one adjoint application per candidate variant.  At N=256 with 27
variants this is ~1.8M operations â€” negligible compared to IHT iterations.
For N > 4096, the candidate set is reduced to the 6 operator-family winners
to keep probe cost under control.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .registry import VARIANTS, VariantInfo

# Operator-family variants that won 6/9 in the K_99 shootout.
# Used as the fast candidate set for large N.
FAST_CANDIDATES = [
    "op_rft_cascade",
    "op_rft_geometric",
    "op_rft_fibonacci",
    "op_rft_harmonic",
    "op_rft_beating",
    "op_rft_golden",
]

# Maximum N for which we probe all 27 variants.
# Above this, only FAST_CANDIDATES are tested.
_FULL_PROBE_N_MAX = 4096


def build_variant_basis(variant_id: str, n: int) -> np.ndarray:
    """Build the (n x n) orthonormal basis matrix for a named variant.

    Returns
    -------
    np.ndarray of shape (n, n), dtype complex128.

    Raises
    ------
    KeyError   if variant_id is not in the registry.
    ValueError if the generator fails or returns wrong shape.
    """
    if variant_id not in VARIANTS:
        raise KeyError(f"Unknown variant '{variant_id}'. Available: {sorted(VARIANTS)}")
    info: VariantInfo = VARIANTS[variant_id]
    basis = info.generator(n)
    if basis.shape != (n, n):
        raise ValueError(
            f"Variant '{variant_id}' returned shape {basis.shape}, expected ({n}, {n})"
        )
    return np.asarray(basis, dtype=np.complex128)


def variant_probe(
    signal: np.ndarray,
    candidates: Optional[List[str]] = None,
    energy_threshold: float = 0.90,
) -> Tuple[str, np.ndarray, List[Dict]]:
    """Probe candidate RFT variants and select the best one for *signal*.

    "Best" = lowest K_{energy_threshold} (fewest coefficients to capture
    the given fraction of signal energy).

    Parameters
    ----------
    signal : 1-D real or complex array of length N.
    candidates : variant IDs to test, or None for auto (all if N <= 4096,
                 FAST_CANDIDATES otherwise).
    energy_threshold : energy fraction for the K metric (default 0.90 = K_90).

    Returns
    -------
    best_variant_id : str
    best_coeffs     : np.ndarray (length N, complex128) â€” cached from the
                      winning variant's adjoint application.
    probe_log       : list of per-variant result dicts for diagnostics.
    """
    signal = np.asarray(signal, dtype=np.complex128).ravel()
    n = len(signal)

    if candidates is None:
        if n > _FULL_PROBE_N_MAX:
            candidates = FAST_CANDIDATES
        else:
            candidates = list(VARIANTS.keys())

    probe_log: List[Dict] = []

    for vid in candidates:
        try:
            basis = build_variant_basis(vid, n)
        except Exception as exc:
            probe_log.append({
                "variant_id": vid,
                "k_threshold": n,
                "energy_captured": 0.0,
                "sparsity_ratio": 1.0,
                "error": str(exc),
            })
            continue

        # Adjoint: coefficients = basis^H @ signal
        coeffs = basis.conj().T @ signal

        mag_sq = np.abs(coeffs) ** 2
        total_energy = mag_sq.sum()

        if total_energy < 1e-30:
            probe_log.append({
                "variant_id": vid,
                "k_threshold": n,
                "energy_captured": 0.0,
                "sparsity_ratio": 1.0,
            })
            continue

        # K_threshold: fewest coefficients to reach energy_threshold
        sorted_energy = np.sort(mag_sq)[::-1]
        cumulative = np.cumsum(sorted_energy) / total_energy
        k_threshold = int(np.searchsorted(cumulative, energy_threshold) + 1)

        probe_log.append({
            "variant_id": vid,
            "k_threshold": k_threshold,
            "energy_captured": float(cumulative[min(k_threshold - 1, n - 1)]),
            "sparsity_ratio": k_threshold / n,
            "coeffs": coeffs,
        })

    # Pick the variant with the lowest K_threshold
    valid = [r for r in probe_log if "error" not in r and "coeffs" in r]
    if not valid:
        raise RuntimeError(
            "No variants produced valid results. "
            f"Tried: {candidates}. Errors: {[r.get('error') for r in probe_log]}"
        )

    best = min(valid, key=lambda r: r["k_threshold"])
    return best["variant_id"], best["coeffs"], probe_log


def variant_forward(
    signal: np.ndarray,
    variant_id: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute forward transform for a specific named variant.

    Returns (coeffs, basis_matrix) so the caller can reconstruct later.
    """
    signal = np.asarray(signal, dtype=np.complex128).ravel()
    n = len(signal)
    basis = build_variant_basis(variant_id, n)
    coeffs = basis.conj().T @ signal
    return coeffs, basis


def variant_inverse(
    coeffs: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """Reconstruct signal from coefficients and basis matrix.

    Returns real-valued reconstruction.
    """
    return np.real(basis @ coeffs)


__all__ = [
    "FAST_CANDIDATES",
    "build_variant_basis",
    "variant_probe",
    "variant_forward",
    "variant_inverse",
]
