#!/usr/bin/env python3
"""Benchmark optimal chunk_size for RFTVertexCodec assembly backend."""
import numpy as np
import time

# Import codec FIRST — this triggers the Windows DLL directory fix
# inside rft_vertex_codec.py before rftmw_native is loaded.
from algorithms.rft.compression.rft_vertex_codec import (
    encode_tensor, enable_assembly_rft, ASSEMBLY_AVAILABLE
)

if not ASSEMBLY_AVAILABLE:
    print('WARNING: rftmw_native not available, running Python backend only.')
else:
    enable_assembly_rft(True)

data = np.random.randn(1024, 3).astype(np.float64)

print(f'--- CHUNK SIZE BENCHMARK (assembly={ASSEMBLY_AVAILABLE}) ---')
for cs in [64, 128, 256, 512, 1024]:
    t0 = time.time()
    encoded = encode_tensor(data, chunk_size=cs)
    elapsed = round(time.time() - t0, 3)
    print(f'chunk={cs:<6} | Time: {elapsed:>7}s | Backend: {encoded["backend"]}')

print('--- DONE ---')
