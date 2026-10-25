# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Resonant Fourier Transform (RFT) - Algorithms Package
======================================================

USPTO Patent 19/169,399: "Hybrid Computational Framework for Quantum and Resonance Simulation"

CANONICAL RFT DEFINITION:
    Φ_{n,k} = (1/√N) exp(j 2π frac((k+1)·φ) · n)
    Φ̃ = Φ (Φᴴ Φ)^{-1/2}    (Gram / Löwdin normalization)

    Forward:  X = Φ̃ᴴ x
    Inverse:  x = Φ̃  X

    where φ = (1+√5)/2 (golden ratio), frac(·) = fractional part.

⚠ The older waveform formulation (f_k = (k+1)×φ, θ_k = 2πk/φ, no Gram
normalization) is preserved for backward compatibility in rft_phi_legacy.py
but is NOT "the RFT."

See algorithms/rft/core/resonant_fourier_transform.py for the canonical implementation.
"""

import os as _os
import sys as _sys

# FIX: re-apply DLL directory injection at this level too.
# Required when this sub-package is imported directly without going
# through algorithms/__init__.py first.
if _sys.platform == 'win32':
    for _dll_dir in [
        _os.path.join(_os.environ.get('SystemRoot', r'C:\Windows'), 'System32'),
        _os.path.dirname(_sys.executable),
    ]:
        if hasattr(_os, 'add_dll_directory') and _os.path.isdir(_dll_dir):
            try:
                _os.add_dll_directory(_dll_dir)
            except OSError:
                pass

import numpy as np

from algorithms.rft.core.resonant_fourier_transform import (
    PHI,
    PHI_INV,
    rft_frequency,
    rft_phase,
    rft_basis_function,
    rft_basis_matrix,
    rft_forward,
    rft_inverse,
    rft,
    irft,
    rft_forward_frame,
    rft_inverse_frame,
    BinaryRFT,
)

# Import the full-featured RFT-SIS hash from crypto module (patent Claim 2),
# not the stub in core/resonant_fourier_transform.py
from algorithms.rft.crypto.rft_sis_hash import RFTSISHash


def rft_forward_canonical(x: np.ndarray) -> np.ndarray:
    """Canonical RFT forward transform using Gram-normalized φ-grid basis."""
    phi = rft_basis_matrix(len(x), len(x), use_gram_normalization=True)
    return rft_forward_frame(x, phi)


def rft_inverse_canonical(X: np.ndarray) -> np.ndarray:
    """Canonical RFT inverse transform using Gram-normalized φ-grid basis."""
    phi = rft_basis_matrix(len(X), len(X), use_gram_normalization=True)
    return rft_inverse_frame(X, phi)

__all__ = [
    'PHI',
    'PHI_INV',
    'rft_frequency',
    'rft_phase',
    'rft_basis_function',
    'rft_basis_matrix',
    'rft_forward',
    'rft_inverse',
    'rft',
    'irft',
    'BinaryRFT',
    'RFTSISHash',
]
