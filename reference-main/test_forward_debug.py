"""Deep forward pass comparison to isolate the bug.

Since GGUF bytes are identical between pack and GGUF, and Ollama produces
correct output, the bug must be in our forward pass logic.

Strategy: test individual operations in isolation.
"""
import sys, os, time
import numpy as np

sys.path.insert(0, r"c:\quantoniumos")
os.environ["RFTMW_NO_NATIVE"] = "1"

from quantonium_os_src.engine.rftmw_numpy_engine import (
    Qwen25Engine, _rms_norm, _apply_rope, _softmax, _silu,
    _dequant_q4k_blocks, _dequant_q6k_blocks
)

print("Loading engine...")
engine = Qwen25Engine(r"c:\quantoniumos\qwen2.5-coder_3b.rftmwpk")
engine.load()
engine._ensure_tokenizer()

cfg = engine.config

# ===== Test 1: Embedding matrix orientation =====
print("\n=== Test 1: Embedding matrix orientation ===")
print(f"embed: quant={engine.embed.quant_type}, out_features={engine.embed.out_features}, "
      f"in_features={engine.embed.in_features}, bytes_per_row={engine.embed.bytes_per_row}")

# The embed is QuantWeight with:
# out_features = first dim of original_shape in pack
# in_features = computed from bytes_per_row
# For token_embd.weight: should be (vocab_size=151936, hidden_size=2048)
assert engine.embed.out_features == cfg.vocab_size, f"embed out_features={engine.embed.out_features} != vocab_size={cfg.vocab_size}"
assert engine.embed.in_features == cfg.hidden_size, f"embed in_features={engine.embed.in_features} != hidden_size={cfg.hidden_size}"
print(f"  Embedding shape: ({engine.embed.out_features}, {engine.embed.in_features}) ✓")

# Embedding lookup: _embed_token dequants one row
h0 = engine._embed_token(151644)  # <|im_start|>
print(f"  Token 151644 embedding: shape={h0.shape}, norm={np.linalg.norm(h0):.4f}")

# ===== Test 2: Layer weight shapes =====
print("\n=== Test 2: Layer 0 weight shapes ===")
layer0 = engine.layers[0]
print(f"  q_proj: out={layer0.q_proj.out_features}, in={layer0.q_proj.in_features} (expect {cfg.num_heads*cfg.head_dim}, {cfg.hidden_size})")
print(f"  k_proj: out={layer0.k_proj.out_features}, in={layer0.k_proj.in_features} (expect {cfg.num_kv_heads*cfg.head_dim}, {cfg.hidden_size})")
print(f"  v_proj: out={layer0.v_proj.out_features}, in={layer0.v_proj.in_features} (expect {cfg.num_kv_heads*cfg.head_dim}, {cfg.hidden_size})")
print(f"  o_proj: out={layer0.o_proj.out_features}, in={layer0.o_proj.in_features} (expect {cfg.hidden_size}, {cfg.num_heads*cfg.head_dim})")
print(f"  gate_proj: out={layer0.gate_proj.out_features}, in={layer0.gate_proj.in_features} (expect {cfg.intermediate_size}, {cfg.hidden_size})")
print(f"  up_proj: out={layer0.up_proj.out_features}, in={layer0.up_proj.in_features} (expect {cfg.intermediate_size}, {cfg.hidden_size})")
print(f"  down_proj: out={layer0.down_proj.out_features}, in={layer0.down_proj.in_features} (expect {cfg.hidden_size}, {cfg.intermediate_size})")

# Check Q bias
print(f"\n  q_bias: shape={layer0.q_bias.shape if layer0.q_bias is not None else None}")
print(f"  k_bias: shape={layer0.k_bias.shape if layer0.k_bias is not None else None}")
print(f"  v_bias: shape={layer0.v_bias.shape if layer0.v_bias is not None else None}")

if layer0.q_bias is not None:
    print(f"  q_bias stats: min={layer0.q_bias.min():.6f}, max={layer0.q_bias.max():.6f}, mean={layer0.q_bias.mean():.6f}")
if layer0.k_bias is not None:
    print(f"  k_bias stats: min={layer0.k_bias.min():.6f}, max={layer0.k_bias.max():.6f}, mean={layer0.k_bias.mean():.6f}")

# ===== Test 3: Manual forward of layer 0 with detailed output =====
print("\n=== Test 3: Manual layer 0 forward ===")
# Use token <|im_start|> (id=151644)
h = engine._embed_token(151644).astype(np.float32)
norm_w = layer0.attn_norm
print(f"  Input h: norm={np.linalg.norm(h):.4f}")

# RMS norm
normed = _rms_norm(h, norm_w, cfg.rms_norm_eps)
print(f"  After attn norm: norm={np.linalg.norm(normed):.4f}, mean={normed.mean():.6f}")

# Q/K/V projections
q = layer0.q_proj.matmul(normed)
k = layer0.k_proj.matmul(normed)
v = layer0.v_proj.matmul(normed)
print(f"  q: shape={q.shape}, norm={np.linalg.norm(q):.4f}, mean={q.mean():.6f}")
print(f"  k: shape={k.shape}, norm={np.linalg.norm(k):.4f}, mean={k.mean():.6f}")
print(f"  v: shape={v.shape}, norm={np.linalg.norm(v):.4f}, mean={v.mean():.6f}")

# Add biases
if layer0.q_bias is not None:
    q = q + layer0.q_bias
    print(f"  q+bias: norm={np.linalg.norm(q):.4f}, mean={q.mean():.6f}")
if layer0.k_bias is not None:
    k = k + layer0.k_bias
if layer0.v_bias is not None:
    v = v + layer0.v_bias

# Reshape for attention
q_heads = q.reshape(cfg.num_heads, cfg.head_dim)     # (16, 128)
k_heads = k.reshape(cfg.num_kv_heads, cfg.head_dim)  # (2, 128)
v_heads = v.reshape(cfg.num_kv_heads, cfg.head_dim)  # (2, 128)

# RoPE at position 0
q_roped = _apply_rope(q_heads, engine.rope_freqs, 0)
k_roped = _apply_rope(k_heads, engine.rope_freqs, 0)
print(f"  q_roped: norm={np.linalg.norm(q_roped):.4f}")
print(f"  k_roped: norm={np.linalg.norm(k_roped):.4f}")

# Self-attention (only 1 token → trivial)
# scores = q @ k^T / sqrt(d) → softmax → weighted sum of v
# With GQA: heads_per_group = 16/2 = 8
heads_per_group = cfg.num_heads // cfg.num_kv_heads
attn_out = np.zeros((cfg.num_heads, cfg.head_dim), dtype=np.float32)
for kv_idx in range(cfg.num_kv_heads):
    k_vec = k_roped[kv_idx]  # (128,)
    v_vec = v_heads[kv_idx]  # (128,)
    for h_off in range(heads_per_group):
        h_idx = kv_idx * heads_per_group + h_off
        score = q_roped[h_idx] @ k_vec / np.sqrt(np.float32(cfg.head_dim))
        # Only 1 token, so softmax([score]) = [1.0]
        attn_out[h_idx] = v_vec  # weight = 1.0

attn_flat = attn_out.reshape(-1)  # (2048,)
print(f"  attn_out: norm={np.linalg.norm(attn_flat):.4f}")

# O projection + residual
o = layer0.o_proj.matmul(attn_flat)
print(f"  o_proj: norm={np.linalg.norm(o):.4f}")
h_post_attn = h + o
print(f"  h + o: norm={np.linalg.norm(h_post_attn):.4f}")

# FFN
normed2 = _rms_norm(h_post_attn, layer0.ffn_norm, cfg.rms_norm_eps)
gate = layer0.gate_proj.matmul(normed2)
up = layer0.up_proj.matmul(normed2)
print(f"  gate: norm={np.linalg.norm(gate):.4f}, mean={gate.mean():.6f}")
print(f"  up: norm={np.linalg.norm(up):.4f}, mean={up.mean():.6f}")

# SwiGLU
swiglu = _silu(gate) * up
print(f"  swiglu: norm={np.linalg.norm(swiglu):.4f}")

# Down projection
ffn_out = layer0.down_proj.matmul(swiglu)
print(f"  down_proj: norm={np.linalg.norm(ffn_out):.4f}")

h_final = h_post_attn + ffn_out
print(f"  h_final (layer 0 output): norm={np.linalg.norm(h_final):.4f}")

# ===== Test 4: Check lm_head (tied embeddings) =====
print("\n=== Test 4: LM head orientation check ===")
# lm_head is: logits = embed.matmul(h_final_normed)
# This should be: logits[i] = embed_row[i] @ h_final_normed
# Where embed_row[i] is the embedding vector for token i

# Quick sanity: the logit for the input token should be high-ish
h_test = engine._embed_token(151644)  # Token we embedded
# If embeddings are correct:
# logits = embed.matmul(h_test) should give high logit for tokens
# "similar" to 151644

# First: check that embed.matmul works like h @ W_embed.T
# where W_embed has shape (vocab_size, hidden_size)
# embed.matmul(h) computes for row i: dot(W_embed[i], h)
# = (vocab_size,) where entry i is the dot product of embed[i] with h

# More importantly: the GGUF ne for token_embd is [2048, 151936]
# GGUF ne[0]=2048 is the fast dimension (columns in row-major)
# GGUF ne[1]=151936 is the slow dimension (rows)
# So in GGUF memory, it's stored as: 151936 rows of 2048 values

# And in Q6K format: each row is 2048 values → 8 blocks of 210 bytes = 1680 bytes per row

# Check: does the pack store it row-major (vocab as first dim)?
print(f"  embed raw shape: ({engine.embed.raw.shape})")
# Should be (151936, 1680) for Q6K with hidden_size=2048

# Now check the matmul direction:
# For tied embedding, lm_head should compute: hidden @ embed_weight^T
# Since embed.matmul(x) computes W @ x (each row dot x), we get:
# embed.matmul(h_normed) → (vocab_size,) = logits

# This is correct IF embed stores (vocab_size, hidden_size)
# And matmul computes y[i] = embed[i] @ h (transpose of h dot embed)

# Let's verify with the embedding we just looked up
# embed_token(151644) = dequant(embed.raw[151644]) → (2048,)
# If we do embed.matmul(h_test), logit[151644] should be ||h_test||^2
# (dot product of h_test with itself if the embedding is exactly h_test)

# But due to quantization, embed_token dequants block-by-block while
# matmul dequants differently (chunked). Let's check:
h_test_norm = _rms_norm(h_test, engine.output_norm, cfg.rms_norm_eps)
logits_test = engine._lm_head(h_test_norm)

print(f"  logit[151644] (self-token): {logits_test[151644]:.4f}")
print(f"  max logit: {logits_test.max():.4f} at token {logits_test.argmax()}")
print(f"  ||h_test||^2: {np.dot(h_test, h_test):.4f}")
print(f"  ||h_test_norm||^2: {np.dot(h_test_norm, h_test_norm):.4f}")

# If the embedding orientation is wrong, the self-token logit would be random
# If correct, self-token should have a relatively high logit (but not necessarily max)

print("\nDone!")
