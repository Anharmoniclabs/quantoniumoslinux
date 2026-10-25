# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.

import os as _os
import sys as _sys

# FIX: Windows DLL search path - must run before rftmw_native.pyd loads.
if _sys.platform == 'win32':
    _repo_root = _os.path.dirname(
        _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    )
    for _dll_dir in [
        _os.path.join(_os.environ.get('SystemRoot', r'C:\Windows'), 'System32'),
        _os.path.dirname(_sys.executable),
        _os.path.join(_repo_root, 'src', 'rftmw_native', 'build'),
        _os.path.join(_repo_root, 'src', 'rftmw_native', 'build', 'Release'),
    ]:
        if hasattr(_os, 'add_dll_directory') and _os.path.isdir(_dll_dir):
            try:
                _os.add_dll_directory(_dll_dir)
            except OSError:
                pass
