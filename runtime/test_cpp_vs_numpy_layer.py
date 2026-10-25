"""Compare C++ vs NumPy forward for a single layer to isolate buggy kernel."""
import sys, os, numpy as np
from pathlib import Path

# Find project root
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
# Try multiple build folder names
for build_dir in ["build_linux", "build_asm", "build"]:
    build_path = REPO_ROOT / "src" / "rftmw_native" / build_dir
    if build_path.exists():
        sys.path.insert(0, str(build_path))
os.environ.pop('RFTMW_NO_NATIVE', None)

from quantonium_os_src.engine.rftmw_numpy_engine import (
    Qwen25Engine, _rms_norm, _apply_rope, _softmax, _silu,
    _get_native_infer, _rope_freqs
)

print('Loading engine...')
e = Qwen25Engine(str(REPO_ROOT / "qwen2.5-coder_3b.rftmwpk"))
e.load()
cfg = e.config

_ni = _get_native_infer()
print(f'Native: {_ni is not None}')

# Get a hidden state to work with
h = e._embed_token(785)  # "The"

layer = e.layers[0]

# ============= Compare RMS norm =============
normed_cpp = _rms_norm(h, layer.attn_norm, cfg.rms_norm_eps)
# Force NumPy path
rms = np.sqrt(np.mean(h**2) + cfg.rms_norm_eps)
normed_np = (h / rms) * layer.attn_norm
print(f'\nRMS norm: max_diff={np.max(np.abs(normed_cpp - normed_np)):.2e}')

# ============= Compare RoPE =============
q_raw = layer.q_proj.matmul(normed_cpp)
k_raw = layer.k_proj.matmul(normed_cpp)
if layer.q_bias is not None:
    q_raw = q_raw + layer.q_bias
if layer.k_bias is not None:
    k_raw = k_raw + layer.k_bias

q = q_raw.reshape(cfg.num_heads, cfg.head_dim)
k = k_raw.reshape(cfg.num_kv_heads, cfg.head_dim)

# NumPy RoPE
q_np = _apply_rope(q.copy(), e.rope_freqs, 0)
k_np = _apply_rope(k.copy(), e.rope_freqs, 0)

# C++ RoPE
f = e.rope_freqs[0]
cos_f = np.ascontiguousarray(f[:, 0], dtype=np.float32)
sin_f = np.ascontiguousarray(f[:, 1], dtype=np.float32)
q_cpp = np.asarray(_ni.apply_rope(
    np.ascontiguousarray(q.copy(), dtype=np.float32),
    cos_f, sin_f, cfg.num_heads, cfg.head_dim))
k_cpp = np.asarray(_ni.apply_rope(
    np.ascontiguousarray(k.copy(), dtype=np.float32),
    cos_f, sin_f, cfg.num_kv_heads, cfg.head_dim))

print(f'RoPE Q: max_diff={np.max(np.abs(q_cpp - q_np)):.2e}')
print(f'  NumPy Q head0[:8]: {q_np[0,:8]}')
print(f'  C++   Q head0[:8]: {q_cpp[0,:8]}')
print(f'RoPE K: max_diff={np.max(np.abs(k_cpp - k_np)):.2e}')

# ============= Compare GQA Attention =============
# Set up cache
k_cache = np.zeros((100, cfg.num_kv_heads, cfg.head_dim), dtype=np.float32)
v_cache = np.zeros((100, cfg.num_kv_heads, cfg.head_dim), dtype=np.float32)

v_raw = layer.v_proj.matmul(normed_cpp)
if layer.v_bias is not None:
    v_raw = v_raw + layer.v_bias
v = v_raw.reshape(cfg.num_kv_heads, cfg.head_dim)

k_cache[0] = k_np  # use NumPy RoPE'd k
v_cache[0] = v
seq_len = 1

# NumPy GQA
heads_per_group = cfg.num_heads // cfg.num_kv_heads
attn_np = np.zeros((cfg.num_heads, cfg.head_dim), dtype=np.float32)
for kv_idx in range(cfg.num_kv_heads):
    k_seq = k_cache[:seq_len, kv_idx]
    v_seq = v_cache[:seq_len, kv_idx]
    for h_off in range(heads_per_group):
        h_idx = kv_idx * heads_per_group + h_off
        scores = q_np[h_idx] @ k_seq.T / np.sqrt(np.float32(cfg.head_dim))
        weights = _softmax(scores)
        attn_np[h_idx] = weights @ v_seq

# C++ GQA
attn_cpp = np.asarray(_ni.gqa_attention(
    np.ascontiguousarray(q_np, dtype=np.float32),  # use SAME q as NumPy
    np.ascontiguousarray(k_cache[:seq_len], dtype=np.float32),
    np.ascontiguousarray(v_cache[:seq_len], dtype=np.float32),
    cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, seq_len))

print(f'\nGQA attention: max_diff={np.max(np.abs(attn_cpp - attn_np)):.2e}')
print(f'  NumPy head0[:8]: {attn_np[0,:8]}')
print(f'  C++   head0[:8]: {attn_cpp[0,:8]}')

# ============= Compare SiLU*mul =============
gate = np.random.randn(cfg.intermediate_size).astype(np.float32)
up = np.random.randn(cfg.intermediate_size).astype(np.float32)

swiglu_np = (1.0 / (1.0 + np.exp(-gate))) * gate * up
swiglu_cpp = np.asarray(_ni.silu_mul(
    np.ascontiguousarray(gate), np.ascontiguousarray(up)))
print(f'\nSiLU*mul: max_diff={np.max(np.abs(swiglu_cpp - swiglu_np)):.2e}')

# ============= FULL LAYER comparison: C++ vs NumPy =============
print('\n=== Full layer 0: C++ vs NumPy ===')

# Reset caches
kc1 = np.zeros((100, cfg.num_kv_heads, cfg.head_dim), dtype=np.float32)
vc1 = np.zeros((100, cfg.num_kv_heads, cfg.head_dim), dtype=np.float32)
kc2 = np.zeros((100, cfg.num_kv_heads, cfg.head_dim), dtype=np.float32)
vc2 = np.zeros((100, cfg.num_kv_heads, cfg.head_dim), dtype=np.float32)

# C++ path (default)
h_cpp = layer.forward(h.copy(), 0, e.rope_freqs, kc1, vc1, 0)

# NumPy path — force it
import quantonium_os_src.engine.rftmw_numpy_engine as eng_mod
old_ni = eng_mod._native_infer
old_checked = eng_mod._native_checked
eng_mod._native_infer = None
eng_mod._native_checked = True

h_np_full = layer.forward(h.copy(), 0, e.rope_freqs, kc2, vc2, 0)

eng_mod._native_infer = old_ni
eng_mod._native_checked = old_checked

diff = np.max(np.abs(h_cpp - h_np_full))
print(f'Full layer output max_diff: {diff:.2e}')
print(f'  C++   first 8: {h_cpp[:8]}')
print(f'  NumPy first 8: {h_np_full[:8]}')
if diff > 0.01:
    print('  ** SIGNIFICANT MISMATCH — C++ kernel bug likely **')
else:
    print('  Kernels match — problem is elsewhere')
