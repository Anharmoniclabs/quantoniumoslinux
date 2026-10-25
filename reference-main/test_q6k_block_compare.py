"""Compare C++ vs Python Q6K dequant for a SINGLE block."""
import sys, os, numpy as np
sys.path.insert(0, r'c:\quantoniumos\src\rftmw_native\build')

from gguf import GGUFReader
GGUF = r'C:\Users\lansky\.ollama\models\blobs\sha256-4a188102020e9c9530b687fd6400f775c45e90a0d7baafe65bd0a36963fbb7ba'

reader = GGUFReader(GGUF)
for t in reader.tensors:
    if t.name == 'token_embd.weight':
        emb_tensor = t; break

raw_data = np.array(emb_tensor.data, dtype=np.uint8)

# Get a single block = one row, first block (210 bytes produces 256 values)
block = raw_data[0, :210].copy()  # row 0, block 0

# Python dequant
from quantonium_os_src.engine.rftmw_numpy_engine import _dequant_q6k_blocks
py_vals = _dequant_q6k_blocks(block.reshape(1, 210)).flatten()

# C++ dequant via a 1-row matmul with identity-like vector
# Actually let's use: result[0] = sum(W[0, :] * x[:])
# If x = e_j (unit vector), result[0] = W[0, j]
import rftmw_native
# Dequant the full row 0 (may have multiple blocks) via matmul trick
bpr = raw_data.shape[1]  # 1680 bytes per row
in_features = (bpr // 210) * 256  # 2048

# Use unit vectors to extract dequanted values one at a time?
# That's slow. Instead, let me construct a weight matrix with just 1 row
# and get dequanted values via matmul with basis vectors
single_row = raw_data[0].copy()  # (1680,)

# Get all 2048 values by doing 1-row matmul with each unit vector
# Actually smarter: use an identity matrix approach
# y = W @ I = W  -- but that's expensive

# Let me just try extracting values directly:
# Use matmul: y = W @ x where W is (1, 2048) and x is (2048,)
# Set x = [1, 0, 0, ...] to get the first element, etc.

# Actually, let me try a full single-row matmul comparison
x = np.ones(in_features, dtype=np.float32) / np.sqrt(in_features)  # normalized
raw_flat = np.ascontiguousarray(single_row)
x_f32 = np.ascontiguousarray(x, dtype=np.float32)

# C++ matmul: y = W_row @ x (single row)
cpp_y = rftmw_native.infer.q6k_matmul(raw_flat, x_f32, 1, bpr)

# Python matmul: same thing
from quantonium_os_src.engine.rftmw_numpy_engine import QuantWeight
qw = QuantWeight(single_row.copy(), (1, bpr))
py_y = qw._matmul_vec(x)

# Full dequant comparison
py_full = qw.dequant_all()  # (1, 2048)
py_ref_y = py_full @ x  # should match py_y

print(f'Single row matmul:')
print(f'  Python matmul: {py_y}')
print(f'  C++ matmul:    {cpp_y}')
print(f'  Python ref:    {py_ref_y}')
print(f'  diff py vs cpp: {abs(py_y[0] - cpp_y[0]):.6e}')

# Now extract individual dequanted values using unit vectors
print(f'\nExtracting dequanted values position by position...')
n_check = 40  # check first 40
for j in range(n_check):
    ej = np.zeros(in_features, dtype=np.float32)
    ej[j] = 1.0
    cpp_val = rftmw_native.infer.q6k_matmul(raw_flat, ej, 1, bpr)[0]
    py_val = py_full[0, j]
    marker = '  ***' if abs(cpp_val - py_val) > 1e-5 else ''
    if j < 20 or abs(cpp_val - py_val) > 1e-5:
        print(f'  [{j:3d}] py={py_val:10.6f}  cpp={cpp_val:10.6f}  diff={abs(cpp_val-py_val):.2e}{marker}')
