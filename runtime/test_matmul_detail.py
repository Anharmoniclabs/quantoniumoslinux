"""Detailed matmul diagnosis: compare C++ path, NumPy path, and full dequant path."""
import sys, os, numpy as np
os.environ['RFTMW_NO_NATIVE'] = '1'  # force NumPy path initially

from gguf import GGUFReader
from gguf.quants import dequantize

GGUF = r'C:\Users\lansky\.ollama\models\blobs\sha256-4a188102020e9c9530b687fd6400f775c45e90a0d7baafe65bd0a36963fbb7ba'

reader = GGUFReader(GGUF)
for t in reader.tensors:
    if t.name == 'blk.0.attn_q.weight':
        q_tensor = t; break

ne0, ne1 = q_tensor.shape[0], q_tensor.shape[1]
print(f'ne = [{ne0}, {ne1}]')

# Get raw data
raw_data = np.array(q_tensor.data, dtype=np.uint8)
bpr = raw_data.shape[1]
print(f'Raw shape: {raw_data.shape}, bpr={bpr}')

# Independent reference: gguf library dequant
ref_all = dequantize(q_tensor.data, q_tensor.tensor_type)
print(f'GGUF lib dequant shape: {ref_all.shape}')

# Our dequant via dequant_all
from quantonium_os_src.engine.rftmw_numpy_engine import QuantWeight, _dequant_q4k_blocks

qw = QuantWeight(raw_data.copy(), (ne1, bpr))
our_all = qw.dequant_all()
print(f'Our dequant_all shape: {our_all.shape}')

# Compare full dequant matrices
dq_diff = np.abs(ref_all.astype(np.float32) - our_all)
print(f'\nFull dequant comparison:')
print(f'  max_diff: {dq_diff.max():.2e}')
print(f'  mean_diff: {dq_diff.mean():.2e}')

# Check row-by-row if they match
for row_idx in [0, 1, 100, 2047]:
    d = np.abs(ref_all[row_idx].astype(np.float32) - our_all[row_idx])
    print(f'  Row {row_idx}: max_diff={d.max():.2e}')

# Test vector
np.random.seed(42)
x = np.random.randn(ne0).astype(np.float32)
x = x / np.linalg.norm(x)

# Method 1: ref_full @ x
ref_result = ref_all.astype(np.float32) @ x
print(f'\nMethod 1 (gguf_lib dequant @ x): first 5 = {ref_result[:5]}')

# Method 2: our dequant_all @ x
our_dequant_result = our_all @ x
print(f'Method 2 (our dequant_all @ x):   first 5 = {our_dequant_result[:5]}')
print(f'  M1 vs M2 max_diff: {np.abs(ref_result - our_dequant_result).max():.2e}')

# Method 3: our chunked NumPy matmul (NO native)
numpy_result = qw.matmul(x)
print(f'Method 3 (NumPy matmul):           first 5 = {numpy_result[:5]}')
print(f'  M1 vs M3 max_diff: {np.abs(ref_result - numpy_result).max():.2e}')

# Method 4: C++ matmul
del os.environ['RFTMW_NO_NATIVE']
sys.path.insert(0, r'c:\quantoniumos\src\rftmw_native\build')
try:
    import rftmw_native as _ni
    print(f'\nC++ module loaded, AVX2={_ni.infer.HAS_AVX2}')
    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
    raw_flat = np.ascontiguousarray(qw.raw.ravel())
    print(f'  raw_flat: shape={raw_flat.shape}, dtype={raw_flat.dtype}')
    cpp_result = _ni.infer.q4k_matmul(raw_flat, x_f32, qw.out_features, qw.bytes_per_row)
    print(f'Method 4 (C++ matmul):             first 5 = {cpp_result[:5]}')
    print(f'  M1 vs M4 max_diff: {np.abs(ref_result - cpp_result).max():.2e}')
    print(f'  M3 vs M4 max_diff: {np.abs(numpy_result - cpp_result).max():.2e}')
    print(f'  M1 vs M4 corr: {np.corrcoef(ref_result, cpp_result)[0,1]:.8f}')
except Exception as e:
    print(f'C++ load error: {e}')

# Skip pack comparison for now
