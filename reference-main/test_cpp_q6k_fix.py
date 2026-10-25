"""Verify C++ Q6K dequant now matches gguf library after fix."""
import sys, os, numpy as np
sys.path.insert(0, r'c:\quantoniumos\src\rftmw_native\build')

from gguf import GGUFReader
from gguf.quants import dequantize

GGUF = r'C:\Users\lansky\.ollama\models\blobs\sha256-4a188102020e9c9530b687fd6400f775c45e90a0d7baafe65bd0a36963fbb7ba'

reader = GGUFReader(GGUF)
for t in reader.tensors:
    if t.name == 'token_embd.weight':
        emb_tensor = t; break

raw_data = np.array(emb_tensor.data, dtype=np.uint8)
ne0, ne1 = emb_tensor.shape  # [2048, 151936]
bpr = raw_data.shape[1]  # bytes per row
print(f'Embedding: ne=[{ne0},{ne1}], raw_shape={raw_data.shape}, bpr={bpr}')

# Reference dequant from gguf library
ref_all = dequantize(emb_tensor.data, emb_tensor.tensor_type)
print(f'Ref shape: {ref_all.shape}')

# Our Python dequant
from quantonium_os_src.engine.rftmw_numpy_engine import QuantWeight
qw = QuantWeight(raw_data.copy(), (ne1, bpr))
our_all = qw.dequant_all()
py_diff = np.abs(ref_all.astype(np.float32) - our_all)
print(f'Python dequant diff: max={py_diff.max():.2e}, mean={py_diff.mean():.2e}')

# C++ dequant via matmul
import rftmw_native
print(f'C++ loaded OK')

# Test matmul with a known vector
np.random.seed(42)
x = np.random.randn(ne0).astype(np.float32)
x = x / np.linalg.norm(x)

# Reference result
ref_result = ref_all.astype(np.float32) @ x

# Python chunked result
py_result = qw._matmul_vec(x)

# C++ result
raw_flat = np.ascontiguousarray(qw.raw.ravel())
x_f32 = np.ascontiguousarray(x, dtype=np.float32)
cpp_result = rftmw_native.infer.q6k_matmul(raw_flat, x_f32, qw.out_features, qw.bytes_per_row)

print(f'\nMatmul comparison (embedding @ random vec):')
print(f'  Ref vs Python: max_diff={np.abs(ref_result - py_result).max():.2e}, corr={np.corrcoef(ref_result, py_result)[0,1]:.8f}')
print(f'  Ref vs C++:    max_diff={np.abs(ref_result - cpp_result).max():.2e}, corr={np.corrcoef(ref_result, cpp_result)[0,1]:.8f}')
print(f'  Python vs C++: max_diff={np.abs(py_result - cpp_result).max():.2e}, corr={np.corrcoef(py_result, cpp_result)[0,1]:.8f}')
print(f'  Ref[:5]     = {ref_result[:5]}')
print(f'  Python[:5]  = {py_result[:5]}')
print(f'  C++[:5]     = {cpp_result[:5]}')

if np.abs(ref_result - cpp_result).max() < 0.01:
    print('\n*** C++ Q6K FIXED! Matches reference. ***')
else:
    print('\n*** C++ Q6K STILL BROKEN ***')
