# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
"""Algorithm package root (RFT, compression, crypto)."""

import os as _os
import sys as _sys

# FIX: Windows Python 3.8+ no longer searches PATH for DLLs.
# Inject all known DLL locations before any C++ extension loads.
if _sys.platform == 'win32':
    _repo_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    for _dll_dir in [
        _os.path.join(_os.environ.get('SystemRoot', r'C:\Windows'), 'System32'),
        _os.path.dirname(_sys.executable),
        _os.path.join(_os.path.dirname(_sys.executable), 'Library', 'bin'),
        # rftmw_native companion DLLs built by CMake
        _os.path.join(_repo_root, 'src', 'rftmw_native', 'build'),
        _os.path.join(_repo_root, 'src', 'rftmw_native', 'build', 'Release'),
        _os.path.join(_repo_root, 'src', 'rftmw_native', 'build', 'Debug'),
    ]:
        if hasattr(_os, 'add_dll_directory') and _os.path.isdir(_dll_dir):
            try:
                _os.add_dll_directory(_dll_dir)
            except OSError:
                pass
