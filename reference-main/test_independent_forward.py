"""Minimal independent forward pass using gguf library for dequant.
Compare against llama-cpp-python logits to find the bug."""
import sys, os, numpy as np
os.environ['RFTMW_NO_NATIVE'] = '1'

from gguf import GGUFReader
from gguf.quants import dequantize

GGUF = r'C:\Users\lansky\.ollama\models\blobs\sha256-4a188102020e9c9530b687fd6400f775c45e90a0d7baafe65bd0a36963fbb7ba'

print("Loading GGUF...", flush=True)
reader = GGUFReader(GGUF)

# Build tensor lookup
tensors = {}
for t in reader.tensors:
    tensors[t.name] = t

def get_f32(name):
    """Get a tensor as float32 numpy array."""
    t = tensors[name]
    if t.tensor_type == 0:  # F32
        return np.array(t.data, dtype=np.float32)
    elif t.tensor_type in (12, 14):  # Q4_K, Q6_K
        return dequantize(t.data, t.tensor_type).astype(np.float32)
    else:
        raise ValueError(f"Unknown type {t.tensor_type} for {name}")

# Load just layer 0 weights + embedding + output_norm
print("Dequantizing embedding...", flush=True)
embed_w = get_f32('token_embd.weight')  # (vocab, hidden)
print(f"  embed shape: {embed_w.shape}")

output_norm_w = get_f32('output_norm.weight')  # (hidden,)
print(f"  output_norm shape: {output_norm_w.shape}")

# Layer 0 weights
print("Dequantizing layer 0...", flush=True)
attn_norm_w = get_f32('blk.0.attn_norm.weight')
ffn_norm_w = get_f32('blk.0.ffn_norm.weight')

q_w = get_f32('blk.0.attn_q.weight')    # (out, in) = (2048, 2048)
k_w = get_f32('blk.0.attn_k.weight')    # (256, 2048)
v_w = get_f32('blk.0.attn_v.weight')    # (256, 2048)
o_w = get_f32('blk.0.attn_output.weight') # (2048, 2048)

q_b = get_f32('blk.0.attn_q.bias')
k_b = get_f32('blk.0.attn_k.bias')
v_b = get_f32('blk.0.attn_v.bias')

gate_w = get_f32('blk.0.ffn_gate.weight')  # (11008, 2048)
up_w = get_f32('blk.0.ffn_up.weight')      # (11008, 2048)
down_w = get_f32('blk.0.ffn_down.weight')  # (2048, 11008)

print(f"  q_w={q_w.shape} k_w={k_w.shape} v_w={v_w.shape} o_w={o_w.shape}")
print(f"  gate={gate_w.shape} up={up_w.shape} down={down_w.shape}")

# Config
hidden = 2048
n_heads = 16
n_kv_heads = 2
head_dim = 128
eps = 1e-6

def rms_norm(x, w):
    rms = np.sqrt(np.mean(x * x) + eps)
    return x / rms * w

def silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -88, 88)))

# ---- TOKEN EMBEDDING ----
# Use token 785 ("The")
token_id = 785
h = embed_w[token_id].copy()
print(f"\nEmbedding: norm={np.linalg.norm(h):.4f}, first3={h[:3]}")

# ---- LAYER 0 FORWARD (pos=0, seq_len=1) ----
# Attention
normed = rms_norm(h, attn_norm_w)
print(f"Normed: norm={np.linalg.norm(normed):.4f}")

q = q_w @ normed + q_b   # (2048,)
k = k_w @ normed + k_b   # (256,)
v = v_w @ normed + v_b   # (256,)
print(f"Q: norm={np.linalg.norm(q):.4f}, first3={q[:3]}")
print(f"K: norm={np.linalg.norm(k):.4f}")
print(f"V: norm={np.linalg.norm(v):.4f}")

# Reshape
q = q.reshape(n_heads, head_dim)     # (16, 128)
k = k.reshape(n_kv_heads, head_dim)  # (2, 128)
v = v.reshape(n_kv_heads, head_dim)  # (2, 128)

# RoPE at pos=0: cos=1, sin=0, so x is unchanged
# (Skipping since pos=0 is identity)

# GQA attention with seq_len=1: output = v repeated for each head group
heads_per_group = n_heads // n_kv_heads  # 8
attn_out = np.zeros((n_heads, head_dim), dtype=np.float32)
for kv_idx in range(n_kv_heads):
    for h_off in range(heads_per_group):
        h_idx = kv_idx * heads_per_group + h_off
        # scores = q[h_idx] @ k[kv_idx] / sqrt(128) --> single value
        score = np.dot(q[h_idx], k[kv_idx]) / np.sqrt(np.float32(head_dim))
        # softmax of single value = 1.0
        attn_out[h_idx] = v[kv_idx]  # weight=1.0

attn_flat = attn_out.reshape(-1)  # (2048,)
o = o_w @ attn_flat   # (2048,)
h = h + o             # residual

print(f"After attn residual: norm={np.linalg.norm(h):.4f}")

# FFN
normed = rms_norm(h, ffn_norm_w)
gate = gate_w @ normed    # (11008,)
up = up_w @ normed        # (11008,)
ffn_out = down_w @ (silu(gate) * up)  # (2048,)
h = h + ffn_out

print(f"After layer 0: norm={np.linalg.norm(h):.4f}")

# ---- Now compare with ENGINE's layer 0 ----
print("\n=== Comparing with engine layer 0 ===")
from quantonium_os_src.engine.rftmw_numpy_engine import Qwen25Engine, _rms_norm, _rope_freqs

eng = Qwen25Engine(r'c:\quantoniumos\qwen2.5-coder_3b.rftmwpk')
eng.load()

h_eng = eng._embed_token(token_id)
print(f"Engine embed: norm={np.linalg.norm(h_eng):.4f}")
print(f"Embed diff: max={np.abs(h_eng - embed_w[token_id]).max():.2e}")

# Reset h to original embedding for engine comparison
h_manual = embed_w[token_id].copy()

# Compare normalization
normed_manual = rms_norm(h_manual, attn_norm_w)
normed_eng = _rms_norm(h_eng, eng.layers[0].attn_norm, eps)
print(f"Norm diff: max={np.abs(normed_manual - normed_eng).max():.2e}")

# Compare Q projection
q_manual = q_w @ normed_manual + q_b
q_eng = eng.layers[0].q_proj.matmul(normed_eng) + eng.layers[0].q_bias
print(f"Q diff: max={np.abs(q_manual - q_eng).max():.2e}")

# Compare full layer 0
kc = np.zeros((100, n_kv_heads, head_dim), dtype=np.float32)
vc = np.zeros((100, n_kv_heads, head_dim), dtype=np.float32)
rope = _rope_freqs(head_dim, 32768, 1e6)
h_eng_l0 = eng.layers[0].forward(h_eng, 0, rope, kc, vc, 0)

# My manual layer 0 result is 'h' from above
print(f"\nManual layer 0: norm={np.linalg.norm(h):.4f}, first3={h[:3]}")
print(f"Engine layer 0: norm={np.linalg.norm(h_eng_l0):.4f}, first3={h_eng_l0[:3]}")
l0_diff = np.abs(h - h_eng_l0)
print(f"Layer 0 diff: max={l0_diff.max():.2e}, mean={l0_diff.mean():.2e}")
print(f"Layer 0 corr: {np.corrcoef(h, h_eng_l0)[0,1]:.8f}")

# ---- Run ALL 36 layers manually ----
print("\n=== Running all 36 layers manually with gguf-dequanted weights ===")
h_full = embed_w[token_id].copy()

for layer_idx in range(36):
    if layer_idx % 5 == 0:
        print(f"  Layer {layer_idx}: norm={np.linalg.norm(h_full):.4f}", flush=True)
    
    an = get_f32(f'blk.{layer_idx}.attn_norm.weight')
    fn = get_f32(f'blk.{layer_idx}.ffn_norm.weight')
    
    qw = get_f32(f'blk.{layer_idx}.attn_q.weight')
    kw = get_f32(f'blk.{layer_idx}.attn_k.weight')
    vw = get_f32(f'blk.{layer_idx}.attn_v.weight')
    ow = get_f32(f'blk.{layer_idx}.attn_output.weight')
    
    qb = get_f32(f'blk.{layer_idx}.attn_q.bias')
    kb = get_f32(f'blk.{layer_idx}.attn_k.bias')
    vb = get_f32(f'blk.{layer_idx}.attn_v.bias')
    
    gw = get_f32(f'blk.{layer_idx}.ffn_gate.weight')
    uw = get_f32(f'blk.{layer_idx}.ffn_up.weight')
    dw = get_f32(f'blk.{layer_idx}.ffn_down.weight')
    
    # Attention
    normed = rms_norm(h_full, an)
    q = qw @ normed + qb
    k = kw @ normed + kb
    v = vw @ normed + vb
    
    q = q.reshape(n_heads, head_dim)
    k = k.reshape(n_kv_heads, head_dim)
    v = v.reshape(n_kv_heads, head_dim)
    
    # GQA with seq_len=1 at pos=0
    attn_out = np.zeros((n_heads, head_dim), dtype=np.float32)
    for kv_i in range(n_kv_heads):
        for h_off in range(heads_per_group):
            attn_out[kv_i * heads_per_group + h_off] = v[kv_i]
    
    o = ow @ attn_out.reshape(-1)
    h_full = h_full + o
    
    # FFN
    normed = rms_norm(h_full, fn)
    gate = gw @ normed
    up = uw @ normed
    ffn_out = dw @ (silu(gate) * up)
    h_full = h_full + ffn_out

print(f"  Final: norm={np.linalg.norm(h_full):.4f}")

# Final norm + logits
h_normed = rms_norm(h_full, output_norm_w)
logits = embed_w @ h_normed  # (vocab,) = (vocab, hidden) @ (hidden,)

top_10 = np.argsort(logits)[-10:][::-1]
print(f"\nManual forward pass top-10 logits:")
from tokenizers import Tokenizer
tok = Tokenizer.from_file(r'c:\quantoniumos\tokenizer.json')
for idx in top_10:
    print(f"  {idx}: {tok.decode([int(idx)])!r} = {logits[idx]:.4f}")

# Compare against llama-cpp-python
print("\n=== Comparing against llama-cpp-python ===")
from llama_cpp import Llama
ref = Llama(model_path=GGUF, n_ctx=256, n_gpu_layers=0, verbose=False, logits_all=True)
ref.reset()
ref.eval([785])
ref_logits = np.array(ref.scores[0])

print(f"Ref logits: min={ref_logits.min():.4f}, max={ref_logits.max():.4f}")
print(f"Our logits: min={logits.min():.4f}, max={logits.max():.4f}")

corr = np.corrcoef(logits, ref_logits)[0,1]
print(f"Correlation: {corr:.6f}")
max_diff = np.abs(logits - ref_logits).max()
print(f"Max diff: {max_diff:.4f}")

ref_top10 = np.argsort(ref_logits)[-10:][::-1]
print(f"\nRef top-10:")
for idx in ref_top10:
    print(f"  {idx}: {tok.decode([int(idx)])!r} = {ref_logits[idx]:.4f}")

our_set = set(top_10.tolist())
ref_set = set(ref_top10.tolist())
print(f"\nTop-10 overlap: {len(our_set & ref_set)}")
