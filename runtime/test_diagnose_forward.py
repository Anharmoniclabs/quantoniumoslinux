"""Diagnose where inference goes wrong — compare layer by layer."""
import sys, os, numpy as np, struct
sys.path.insert(0, r'c:\quantoniumos\src\rftmw_native\build')
os.environ.pop('RFTMW_NO_NATIVE', None)

from quantonium_os_src.engine.rftmw_numpy_engine import (
    Qwen25Engine, _rms_norm, _apply_rope, _rope_freqs,
    _dequant_q4k_blocks, _dequant_q6k_blocks, _parse_gguf_f32_index
)

print('Loading engine...')
e = Qwen25Engine(r'c:\quantoniumos\qwen2.5-coder_3b.rftmwpk')
e.load()

# Use a simple single-token test to avoid prefill complexity
token_id = 785  # "Hi"
print(f'\n=== Testing single token: id={token_id} text={repr(e.decode([token_id]))} ===')

# Step 1: Check embedding
h = e._embed_token(token_id)
print(f'\nEmbedding: shape={h.shape} min={h.min():.4f} max={h.max():.4f} '
      f'mean={h.mean():.4f} std={h.std():.4f} norm={np.linalg.norm(h):.4f}')
print(f'  First 8: {h[:8]}')

# Step 2: Check RMS norm of embedding
cfg = e.config
normed = _rms_norm(h, e.layers[0].attn_norm, cfg.rms_norm_eps)
print(f'\nAfter attn_norm: min={normed.min():.4f} max={normed.max():.4f} '
      f'mean={normed.mean():.4f} norm={np.linalg.norm(normed):.4f}')

# Step 3: Check Q/K/V projections
q = e.layers[0].q_proj.matmul(normed)
k = e.layers[0].k_proj.matmul(normed)
v = e.layers[0].v_proj.matmul(normed)
print(f'\nQ proj: shape={q.shape} min={q.min():.4f} max={q.max():.4f} mean={q.mean():.4f}')
print(f'K proj: shape={k.shape} min={k.min():.4f} max={k.max():.4f} mean={k.mean():.4f}')
print(f'V proj: shape={v.shape} min={v.min():.4f} max={v.max():.4f} mean={v.mean():.4f}')

# Step 4: Apply biases
if e.layers[0].q_bias is not None:
    q = q + e.layers[0].q_bias
    print(f'Q+bias: min={q.min():.4f} max={q.max():.4f}')
if e.layers[0].k_bias is not None:
    k = k + e.layers[0].k_bias
if e.layers[0].v_bias is not None:
    v = v + e.layers[0].v_bias

# Step 5: Check matmul weight layout
# The critical check: are rows/cols correct?
qw = e.layers[0].q_proj
print(f'\nQ weight: quant={qw.quant_type} out_features={qw.out_features} '
      f'in_features={qw.in_features} raw_shape={qw.raw.shape}')

# Verify: dequant all of q_proj and check shape
if qw.quant_type == 'q4k':
    blocks = qw.raw.reshape(-1, 144)
    full_w = _dequant_q4k_blocks(blocks)
    row_count = qw.raw.shape[0]
    cols = full_w.size // row_count
    full_w = full_w.reshape(row_count, -1)[:, :qw.in_features]
else:
    blocks = qw.raw.reshape(-1, 210)
    full_w = _dequant_q6k_blocks(blocks)
    row_count = qw.raw.shape[0]
    full_w = full_w.reshape(row_count, -1)[:, :qw.in_features]

print(f'Dequant W shape: {full_w.shape}')
print(f'W @ normed should be: ({row_count},) x ({normed.shape[0]},) -> ({row_count},)')
manual_q = full_w @ normed
print(f'Manual matmul: shape={manual_q.shape} first 5={manual_q[:5]}')
print(f'Engine matmul: shape={q.shape} first 5={q[:5]}')
print(f'Max diff manual vs engine: {np.max(np.abs(manual_q[:qw.out_features] - (q - e.layers[0].q_bias if e.layers[0].q_bias is not None else q))[:qw.out_features]):.6e}')

# Step 6: Full forward through layer 0
print('\n=== Full layer 0 forward ===')
k_cache = np.zeros((100, cfg.num_kv_heads, cfg.head_dim), dtype=np.float32)
v_cache = np.zeros((100, cfg.num_kv_heads, cfg.head_dim), dtype=np.float32)

h_out = e.layers[0].forward(h.copy(), 0, e.rope_freqs, k_cache, v_cache, 0)
print(f'Layer 0 output: min={h_out.min():.4f} max={h_out.max():.4f} '
      f'mean={h_out.mean():.4f} norm={np.linalg.norm(h_out):.4f}')
print(f'  First 8: {h_out[:8]}')

# Step 7: Run full model and check logits
print('\n=== Full model forward (single token) ===')
h_full = e._embed_token(token_id)
max_seq = 100
k_caches = [np.zeros((max_seq, cfg.num_kv_heads, cfg.head_dim), dtype=np.float32)
             for _ in range(cfg.num_layers)]
v_caches = [np.zeros((max_seq, cfg.num_kv_heads, cfg.head_dim), dtype=np.float32)
             for _ in range(cfg.num_layers)]

for li, layer in enumerate(e.layers):
    h_full = layer.forward(h_full, 0, e.rope_freqs, k_caches[li], v_caches[li], 0)
    if li < 3 or li == 35:
        print(f'  After layer {li}: norm={np.linalg.norm(h_full):.4f} '
              f'min={h_full.min():.4f} max={h_full.max():.4f}')

h_full = _rms_norm(h_full, e.output_norm, cfg.rms_norm_eps)
logits = e._lm_head(h_full)
print(f'\nLogits: shape={logits.shape} min={logits.min():.4f} max={logits.max():.4f}')

top10 = np.argsort(logits)[-10:][::-1]
print('Top 10 predictions after "Hi":')
for tid in top10:
    tok = e.decode([tid])
    print(f'  token={tid:6d}  logit={logits[tid]:8.3f}  text={repr(tok)}')
