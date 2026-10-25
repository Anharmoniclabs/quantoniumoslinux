"""Compare C++ vs NumPy dequantization and matmul on REAL weights.

This test:
1. Loads the engine (gets real Q4K/Q6K weight blocks)
2. Compares C++ dequant_q4k vs Python _dequant_q4k_blocks on 1 block
3. Compares C++ matmul vs Python matmul on a small weight
4. Reports max/mean absolute differences
"""
import sys, os, time
import numpy as np
from pathlib import Path

# Find project root
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
# Try multiple build folder names
for build_dir in ["build_linux", "build_asm", "build"]:
    build_path = REPO_ROOT / "src" / "rftmw_native" / build_dir
    if build_path.exists():
        sys.path.insert(0, str(build_path))

os.environ["RFTMW_NO_NATIVE"] = ""  # Allow native loading for this test

# Load engine
from quantonium_os_src.engine.rftmw_numpy_engine import (
    Qwen25Engine, _dequant_q4k_blocks, _dequant_q6k_blocks
)

print("Loading engine...")
t0 = time.time()
engine = Qwen25Engine(str(REPO_ROOT / "qwen2.5-coder_3b.rftmwpk"))
engine.load()
print(f"Loaded in {time.time()-t0:.1f}s")

# Load C++ module
import rftmw_native
ni = rftmw_native.infer
print(f"C++ infer module loaded, available={ni.available}")

# ===== Test 1: Q4K dequant on single block =====
print("\n=== Test 1: Q4K dequant (single block) ===")
layer0 = engine.layers[0]
qw = layer0.q_proj  # Q4K weight
print(f"q_proj: quant={qw.quant_type}, shape=({qw.out_features}, {qw.bytes_per_row})")
print(f"  blocks_per_row = {qw.bytes_per_row // qw.block_size}")

# Get first block of first row (144 bytes)
raw_block = qw.raw[0, :144].copy()  # one Q4K block
print(f"  raw_block[:16] = {raw_block[:16]}")

# Python dequant
py_vals = _dequant_q4k_blocks(raw_block.reshape(1, 144)).flatten()  # (256,)
print(f"  Python dequant: min={py_vals.min():.6f}, max={py_vals.max():.6f}, mean={py_vals.mean():.6f}")

# C++ dequant (via matmul with identity-like test)
# Since C++ doesn't expose bare dequant, test via matmul:
# Do matmul of first row with a unit vector e_0 = [1,0,0,...,0]
# This should give us the first element of the dequanted row
# Better: do full matmul comparison.

# ===== Test 2: Full matmul comparison (1 row) =====
print("\n=== Test 2: Q4K matmul comparison ===")
# Create random input vector
np.random.seed(42)
x = np.random.randn(qw.in_features).astype(np.float32)

# Python path: dequant + matmul
py_result = qw._matmul_vec(x)
print(f"  Python matmul done, shape={py_result.shape}")

# C++ path: fused dequant+matmul
raw_flat = np.ascontiguousarray(qw.raw.ravel())
x_f32 = np.ascontiguousarray(x, dtype=np.float32)
cpp_result = np.asarray(ni.q4k_matmul(raw_flat, x_f32, qw.out_features, qw.bytes_per_row))
print(f"  C++ matmul done, shape={cpp_result.shape}")

# Compare
diff = np.abs(py_result - cpp_result)
print(f"  Max diff: {diff.max():.8f}")
print(f"  Mean diff: {diff.mean():.8f}")
print(f"  Relative max: {(diff.max() / (np.abs(py_result).max() + 1e-10)):.8f}")
print(f"  Close (atol=1e-3): {np.allclose(py_result, cpp_result, atol=1e-3)}")
print(f"  Close (atol=1e-2): {np.allclose(py_result, cpp_result, atol=1e-2)}")

# Sample some values
for i in [0, 1, 2, 100, 500, 1000, 2000]:
    if i < len(py_result):
        print(f"  [{i}] py={py_result[i]:.6f}  cpp={cpp_result[i]:.6f}  diff={abs(py_result[i]-cpp_result[i]):.8f}")

# ===== Test 3: Check for Q6K weights =====
print("\n=== Test 3: Q6K matmul comparison ===")
# Find a Q6K weight
q6k_weight = None
for layer in engine.layers:
    for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
        w = getattr(layer, name)
        if w.quant_type == 'q6k':
            q6k_weight = w
            print(f"  Found Q6K weight: {name}, shape=({w.out_features}, {w.bytes_per_row})")
            break
    if q6k_weight:
        break

if q6k_weight:
    x6 = np.random.randn(q6k_weight.in_features).astype(np.float32)
    py_r6 = q6k_weight._matmul_vec(x6)
    raw_flat6 = np.ascontiguousarray(q6k_weight.raw.ravel())
    cpp_r6 = np.asarray(ni.q6k_matmul(raw_flat6, x6, q6k_weight.out_features, q6k_weight.bytes_per_row))
    diff6 = np.abs(py_r6 - cpp_r6)
    print(f"  Max diff: {diff6.max():.8f}")
    print(f"  Mean diff: {diff6.mean():.8f}")
    print(f"  Close (atol=1e-3): {np.allclose(py_r6, cpp_r6, atol=1e-3)}")
else:
    print("  No Q6K weight found")

# ===== Test 4: RMS norm =====
print("\n=== Test 4: RMS norm ===")
norm_w = layer0.attn_norm
h = np.random.randn(engine.config.hidden_size).astype(np.float32)
from quantonium_os_src.engine.rftmw_numpy_engine import _rms_norm
py_norm = _rms_norm(h, norm_w, engine.config.rms_norm_eps)
cpp_norm = np.asarray(ni.rms_norm(h, norm_w, engine.config.rms_norm_eps))
norm_diff = np.abs(py_norm - cpp_norm)
print(f"  Max diff: {norm_diff.max():.8f}")
print(f"  Close (atol=1e-6): {np.allclose(py_norm, cpp_norm, atol=1e-6)}")

# ===== Test 5: Embedding + one layer forward =====
print("\n=== Test 5: Single token forward (pure NumPy vs pure NumPy sanity) ===")
# Just check if logits look sane for a known token
test_prompt = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
engine._ensure_tokenizer()
ids = engine.encode(test_prompt)
print(f"  Encoded '{test_prompt[:30]}...' → {len(ids)} tokens")
print(f"  Token IDs: {ids[:10]}...")

# Check if special tokens are correct
im_start = engine.encode("<|im_start|>")
im_end = engine.encode("<|im_end|>")
print(f"  <|im_start|> → {im_start}")
print(f"  <|im_end|> → {im_end}")

# Check embedding for first token
h = engine._embed_token(ids[0])
print(f"  Embedding shape: {h.shape}, dtype: {h.dtype}")
print(f"  Embedding stats: min={h.min():.4f}, max={h.max():.4f}, mean={h.mean():.4f}, std={h.std():.4f}")
has_nan = np.any(np.isnan(h))
has_inf = np.any(np.isinf(h))
print(f"  NaN: {has_nan}, Inf: {has_inf}")

print("\nDone!")
