"""Isolate the C++ dequant bug by comparing block-by-block."""
import sys, os, time
import numpy as np

sys.path.insert(0, r"c:\quantoniumos")
sys.path.insert(0, r"c:\quantoniumos\src\rftmw_native\build")
os.environ["RFTMW_NO_NATIVE"] = ""

from quantonium_os_src.engine.rftmw_numpy_engine import (
    Qwen25Engine, _dequant_q4k_blocks, _dequant_q6k_blocks
)

print("Loading engine...")
engine = Qwen25Engine(r"c:\quantoniumos\qwen2.5-coder_3b.rftmwpk")
engine.load()

import rftmw_native
ni = rftmw_native.infer

layer0 = engine.layers[0]
qw = layer0.q_proj  # Q4K: (2048, 1152), 8 blocks/row × 144 bytes

# ===== Test: Compare dequant row 0 =====
print("\n=== Comparing dequant of entire row 0 ===")
row0_raw = qw.raw[0].copy()  # (1152,) = 8 Q4K blocks
nblocks = len(row0_raw) // 144

# Python dequant
blocks = row0_raw.reshape(nblocks, 144)
py_row = _dequant_q4k_blocks(blocks).flatten()  # (8*256,) = (2048,)
print(f"Python row0: shape={py_row.shape}, min={py_row.min():.6f}, max={py_row.max():.6f}")

# C++ dequant via matmul trick: use identity matrix columns
# For row 0 only: do matmul with x = all zeros except set one element at a time
# Actually simpler: matmul of just 1 row × full x
# Even simpler: dequant full and compare explicitly

# Full dequant in Python
full_W_py = qw.dequant_all()  # (2048, 2048)
print(f"Full Python dequant: shape={full_W_py.shape}")

# Compare: Python matmul (full W @ x) vs C++ matmul
np.random.seed(42)
x = np.random.randn(qw.in_features).astype(np.float32)

py_full_matmul = full_W_py @ x  # (2048,) - ground truth
py_chunked_matmul = qw._matmul_vec(x)  # (2048,) - chunked Python

# C++ matmul
raw_flat = np.ascontiguousarray(qw.raw.ravel())
x_f32 = np.ascontiguousarray(x, dtype=np.float32)
cpp_matmul = np.asarray(ni.q4k_matmul(raw_flat, x_f32, qw.out_features, qw.bytes_per_row))

# Compare all three
diff_full_vs_chunked = np.abs(py_full_matmul - py_chunked_matmul)
diff_full_vs_cpp = np.abs(py_full_matmul - cpp_matmul)
diff_chunked_vs_cpp = np.abs(py_chunked_matmul - cpp_matmul)

print(f"\nPython full vs Python chunked: max={diff_full_vs_chunked.max():.8f} mean={diff_full_vs_chunked.mean():.8f}")
print(f"Python full vs C++: max={diff_full_vs_cpp.max():.8f} mean={diff_full_vs_cpp.mean():.8f}")
print(f"Python chunked vs C++: max={diff_chunked_vs_cpp.max():.8f} mean={diff_chunked_vs_cpp.mean():.8f}")

# ===== Test: Check fp16 conversion =====
print("\n=== Testing fp16 conversion ===")
# Extract the d and dmin from first block of first row
block0 = qw.raw[0, :144].copy()
d_bytes = block0[0:2]
dmin_bytes = block0[2:4]

# Python (numpy) fp16 
d_py = np.frombuffer(d_bytes.tobytes(), dtype=np.float16).astype(np.float32)[0]
dmin_py = np.frombuffer(dmin_bytes.tobytes(), dtype=np.float16).astype(np.float32)[0]
print(f"Python fp16: d={d_py:.10f}, dmin={dmin_py:.10f}")
print(f"  d raw bytes: {d_bytes}")
print(f"  dmin raw bytes: {dmin_bytes}")

# C++ fp16 conversion (manual check)
d_u16 = int.from_bytes(d_bytes.tobytes(), 'little')
dmin_u16 = int.from_bytes(dmin_bytes.tobytes(), 'little')
print(f"  d_u16 = 0x{d_u16:04x}, dmin_u16 = 0x{dmin_u16:04x}")

# Manual fp16→f32 (same as C++ code)
def fp16_to_f32_manual(h):
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF
    if exp == 0:
        if mant == 0:
            return -0.0 if sign else 0.0
        import math
        val = math.ldexp(mant, -24)
        return -val if sign else val
    if exp == 31:
        return float('nan') if mant else (float('-inf') if sign else float('inf'))
    import math
    val = math.ldexp(mant + 1024, exp - 25)
    return -val if sign else val

d_manual = fp16_to_f32_manual(d_u16)
dmin_manual = fp16_to_f32_manual(dmin_u16)
print(f"Manual fp16: d={d_manual:.10f}, dmin={dmin_manual:.10f}")
print(f"  Match: d={d_py == d_manual}, dmin={dmin_py == dmin_manual}")

# ===== Test: Element-by-element dequant comparison =====
print("\n=== Element-by-element dequant of block 0, row 0 ===")
# Python: py_vals for block 0
py_vals = _dequant_q4k_blocks(block0.reshape(1, 144)).flatten()  # (256,)

# To get C++ dequant: use x = unit vectors
# Actually, let me dequant all row 0 blocks and compare with full dequant
py_row0_full = full_W_py[0, :]  # (2048,) from full dequant
py_row0_blockwise = py_row  # (2048,) from block-wise dequant above

diff_row0 = np.abs(py_row0_full - py_row0_blockwise)
print(f"Python full vs blockwise row0: max={diff_row0.max():.8f}")
# These should match exactly

# Now, to test C++ dequant: we need a way to extract just dequanted values
# Use: matmul of just row 0 with specific input
# Result[0] = sum(W[0,:] * x[:])
# With known W[0,:] from Python, we can check if C++ gives same dot product

# Test with x = ones (just sum the row)
x_ones = np.ones(qw.in_features, dtype=np.float32)
raw_flat = np.ascontiguousarray(qw.raw.ravel())
cpp_ones = np.asarray(ni.q4k_matmul(raw_flat, x_ones, qw.out_features, qw.bytes_per_row))
py_ones = full_W_py.sum(axis=1)  # row sums

diff_ones = np.abs(py_ones - cpp_ones)
print(f"\nRow sums (x=ones): max diff={diff_ones.max():.8f}, mean={diff_ones.mean():.8f}")
print(f"  Row 0: py={py_ones[0]:.6f} cpp={cpp_ones[0]:.6f} diff={diff_ones[0]:.8f}")
print(f"  Row 1: py={py_ones[1]:.6f} cpp={cpp_ones[1]:.6f} diff={diff_ones[1]:.8f}")

# Test with x = [1,0,0,...,0] to extract column 0 (= first dequanted value of each row)
x_e0 = np.zeros(qw.in_features, dtype=np.float32)
x_e0[0] = 1.0
cpp_col0 = np.asarray(ni.q4k_matmul(raw_flat, x_e0, qw.out_features, qw.bytes_per_row))
py_col0 = full_W_py[:, 0]

diff_col0 = np.abs(py_col0 - cpp_col0)
print(f"\nColumn 0 (x=e0): max diff={diff_col0.max():.8f}, mean={diff_col0.mean():.8f}")
for i in [0, 1, 2, 3, 100, 500, 1000, 2000]:
    if i < len(py_col0):
        print(f"  [{i}] py={py_col0[i]:.8f} cpp={cpp_col0[i]:.8f} diff={abs(py_col0[i]-cpp_col0[i]):.8f}")

# Also test column 256 (first value of block 1)
x_e256 = np.zeros(qw.in_features, dtype=np.float32)
x_e256[256] = 1.0
cpp_col256 = np.asarray(ni.q4k_matmul(raw_flat, x_e256, qw.out_features, qw.bytes_per_row))
py_col256 = full_W_py[:, 256]

diff_col256 = np.abs(py_col256 - cpp_col256)
print(f"\nColumn 256 (x=e256): max diff={diff_col256.max():.8f}, mean={diff_col256.mean():.8f}")
for i in [0, 1, 2]:
    if i < len(py_col256):
        print(f"  [{i}] py={py_col256[i]:.8f} cpp={cpp_col256[i]:.8f}")

print("\nDone!")
