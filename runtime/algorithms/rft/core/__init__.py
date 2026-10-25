# =============================================================================
# SPDX-License-Identifier: LicenseRef-SPX-Proprietary
# Copyright (c) 2024-2026 Luis M. Minier. All rights reserved.
#
# Patent Pending: USPTO Application 19/169,399
# "Hybrid Computational Framework for Quantum and Resonance Simulation"
#
# This file contains proprietary mathematical frameworks, algorithm
# specifications, and/or theorem proofs that are the exclusive intellectual
# property of Luis M. Minier.
#
# PERMITTED: View for peer review and academic verification only.
#            Run to reproduce published experimental results.
# NOT PERMITTED: Copy, modify, redistribute, fork, implement independently,
#                or use commercially without a written patent license.
#
# Commercial licensing: luisminier79@gmail.com
# See LICENSE-SPX-PROPRIETARY.md for full terms.
# =============================================================================

from .resonant_fourier_transform import ResonantFourierTransform
from .canonical_true_rft import CanonicalTrueRFT
from .gram_utils import gram_normalize
from .fast_rft import fast_rft
from .transform_theorems import TransformTheorems

__all__ = [
    'ResonantFourierTransform',
    'CanonicalTrueRFT',
    'gram_normalize',
    'fast_rft',
    'TransformTheorems',
]
