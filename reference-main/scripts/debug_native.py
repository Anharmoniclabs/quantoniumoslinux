#!/usr/bin/env python3
"""Diagnose rftmw_native backend activation in RFTVertexCodec."""
import os
import sys

# Must add DLL directory BEFORE importing rftmw_native on Windows
if sys.platform == 'win32':
    system32 = os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32')
    if hasattr(os, 'add_dll_directory') and os.path.isdir(system32):
        os.add_dll_directory(system32)

import rftmw_native
import numpy as np
import time

print('=== rftmw_native diagnostics ===')
print(f'Version:       {rftmw_native.__version__}')
print(f'HAS_AVX2:      {rftmw_native.HAS_AVX2}')
print(f'HAS_AVX512:    {rftmw_native.HAS_AVX512}')
print(f'HAS_FMA:       {rftmw_native.HAS_FMA}')
print(f'HAS_ASM:       {rftmw_native.HAS_ASM_KERNELS}')
print(f'asm_available: {rftmw_native.asm_kernels_available()}')

for size in [8, 16, 32, 64, 128]:
    try:
        t0 = time.time()
        engine = rftmw_native.RFTMWEngine(size)
        data = np.random.randn(size).tolist()
        result = engine.forward(data)
        elapsed = round(time.time() - t0, 4)
        ok = result is not None and len(result) == size
        print(f'RFTMWEngine({size:<4}): OK={ok} | Time={elapsed}s | out_len={len(result) if result else None}')
    except Exception as e:
        print(f'RFTMWEngine({size:<4}): FAILED -> {e}')

try:
    err = rftmw_native.roundtrip_error(64)
    print(f'roundtrip_error(64): {err:.2e}')
except Exception as e:
    print(f'roundtrip_error: FAILED -> {e}')

print()
print('=== Codec integration test ===')
from algorithms.rft.compression.rft_vertex_codec import (
    ASSEMBLY_AVAILABLE, ASSEMBLY_ENABLED, enable_assembly_rft, encode_tensor
)
print(f'ASSEMBLY_AVAILABLE: {ASSEMBLY_AVAILABLE}')
print(f'ASSEMBLY_ENABLED:   {ASSEMBLY_ENABLED}')

enable_assembly_rft(True)
print(f'After enable:       {ASSEMBLY_ENABLED}')

data = np.random.randn(128).astype(np.float64)
t0 = time.time()
enc = encode_tensor(data, chunk_size=128)
print(f'encode_tensor(128): Backend={enc["backend"]} | Time={round(time.time()-t0,4)}s')
