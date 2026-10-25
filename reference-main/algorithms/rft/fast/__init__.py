# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Fast RFT Module
===============

Provides fast basis construction via caching, approximation,
and (eventually) O(N log N) algorithms.
"""

from .cached_basis import (
    get_cached_basis,
    get_fast_basis,
    rft_forward_fast,
    rft_inverse_fast,
    precompute_all_bases,
    STANDARD_SIZES,
)
from .fast_canonical_rft import FastCanonicalRFT, lanczos_func_vec

__all__ = [
    'get_cached_basis',
    'get_fast_basis',
    'rft_forward_fast',
    'rft_inverse_fast',
    'precompute_all_bases',
    'STANDARD_SIZES',
    'FastCanonicalRFT',
    'lanczos_func_vec',
]
