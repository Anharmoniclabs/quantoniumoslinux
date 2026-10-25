# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
Entropy estimation and quantization utilities for Rate-Distortion analysis.
"""
import numpy as np
from collections import Counter


def uniform_quantizer(coeffs: np.ndarray, step_size: float) -> np.ndarray:
    """
    Uniform scalar quantization with a dead-zone at zero.

    Behavior:
    - Uses truncation toward zero: q = sign(x) * floor(|x| / step_size)
    - No saturation/clipping is applied
    - Complex inputs are quantized per real/imag component
    """
    if step_size <= 0:
        return coeffs

    if np.iscomplexobj(coeffs):
        q_real = np.sign(coeffs.real) * np.floor(np.abs(coeffs.real) / step_size)
        q_imag = np.sign(coeffs.imag) * np.floor(np.abs(coeffs.imag) / step_size)
        return q_real + 1j * q_imag
    else:
        return np.sign(coeffs) * np.floor(np.abs(coeffs) / step_size)


def estimate_bitrate(coeffs: np.ndarray) -> float:
    """
    Estimate the bitrate (total bits) required to encode the coefficients.

    Model:
    1. Sparsity Map: binary-entropy cost of the zero/non-zero mask.
    2. Values: 0th-order entropy of non-zero quantised bins.

    Returns total bits (not BPP — divide by N at the call-site).
    """
    N = coeffs.size
    if N == 0:
        return 0.0

    non_zeros = coeffs[coeffs != 0]
    K = non_zeros.size

    if K == 0:
        return 0.0

    # 1. Position cost
    if K == N:
        pos_bits = 0.0
    else:
        p = K / N
        h_p = -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
        pos_bits = N * h_p

    # 2. Value cost
    if np.iscomplexobj(non_zeros):
        symbols = np.concatenate([non_zeros.real, non_zeros.imag])
        symbols = symbols[symbols != 0]
    else:
        symbols = non_zeros

    if symbols.size == 0:
        val_bits = 0.0
    else:
        counts = Counter(symbols)
        total_syms = len(symbols)
        probs = np.array(list(counts.values()), dtype=np.float64) / total_syms
        entropy = -np.sum(probs * np.log2(probs))
        val_bits = total_syms * entropy

    return float(pos_bits + val_bits)


def calculate_rd_point(signal, transform_func, inverse_func, step_size,
                       precomputed_coeffs=None):
    """
    Calculate a single (Rate, Distortion) point for a given transform and
    quantisation step size.

    If *precomputed_coeffs* is supplied the forward transform is skipped,
    saving an expensive O(n²) / O(n³) matrix build.

    Returns
    -------
    bpp        : float  -- bits per sample at this operating point
    mse        : float  -- mean-squared reconstruction error
    coeffs     : ndarray -- the *raw* (unquantised) transform coefficients
                           computed here so the caller can reuse them without
                           a second forward-transform call.
    recon_coeffs: ndarray -- dequantised coefficients (q_coeffs * step_size)
                            ready to feed straight into the codec encoder so
                            the codec can skip its own forward pass entirely.
    """
    # 1. Forward transform  (ONE call — reused by caller)
    coeffs = precomputed_coeffs if precomputed_coeffs is not None else transform_func(signal)

    # 2. Quantise
    q_coeffs = uniform_quantizer(coeffs, step_size)

    # 3. Rate estimate
    total_bits = estimate_bitrate(q_coeffs)
    bpp = total_bits / signal.size

    # 4. Dequantise + inverse  (ONE inverse call — result also reused by caller)
    recon_coeffs = q_coeffs * step_size
    recon = inverse_func(recon_coeffs)

    # 5. Distortion
    if np.iscomplexobj(signal):
        mse = float(np.mean(np.abs(signal - recon) ** 2))
    else:
        mse = float(np.mean((signal - recon.real) ** 2))

    return bpp, mse, coeffs, recon_coeffs
