# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
quantoniumos Engine Package
============================

Public surface::

    from quantonium_os_src.engine import compress, decompress
    from quantonium_os_src.engine import RFTMWMemoryLayer
    from quantonium_os_src.engine import RFTMWInferenceEngine

All heavy imports are lazy â€” nothing is loaded until the symbol is first used.
"""

# Unified compression entry point (new in this commit)
from quantonium_os_src.engine.rft_compress import compress, decompress

# Existing engine exports
try:
    from quantonium_os_src.engine.rftmw_memory import RFTMWMemoryLayer
except ImportError:  # optional heavy deps (torch, transformers) may be absent
    RFTMWMemoryLayer = None  # type: ignore

try:
    from quantonium_os_src.engine.rftmw_inference import RFTMWInferenceEngine
except ImportError:
    RFTMWInferenceEngine = None  # type: ignore

__all__ = [
    'compress',
    'decompress',
    'RFTMWMemoryLayer',
    'RFTMWInferenceEngine',
]
