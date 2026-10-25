"""Compute layer 0 forward pass using ONLY data from the GGUF file,
and compare against our engine. This isolates whether the divergence
is in the pack data or the forward pass logic."""
import sys, os, struct, numpy as np
sys.path.insert(0, r'c:\quantoniumos\src\rftmw_native\build')
os.environ.pop('RFTMW_NO_NATIVE', None)

GGUF = r'C:\Users\lansky\.ollama\models\blobs\sha256-4a188102020e9c9530b687fd6400f775c45e90a0d7baafe65bd0a36963fbb7ba'

# Import our dequant functions
from quantonium_os_src.engine.rftmw_numpy_engine import (
    Qwen25Engine, _dequant_q4k_blocks, _dequant_q6k_blocks,
    _rms_norm, _apply_rope, _softmax, _silu, _rope_freqs
)

# Parse GGUF header
print('=== Parsing GGUF ===')
with open(GGUF, 'rb') as f:
    magic = f.read(4)
    ver = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]
    
    def read_string():
        slen = struct.unpack('<Q', f.read(8))[0]
        return f.read(slen).decode('utf-8')
    def skip_value(vtype):
        if vtype == 0: f.read(1)
        elif vtype == 1: f.read(1)
        elif vtype == 2: f.read(2)
        elif vtype == 3: f.read(2)
        elif vtype == 4: f.read(4)
        elif vtype == 5: f.read(4)
        elif vtype == 6: f.read(4)
        elif vtype == 7: f.read(1)
        elif vtype == 8: read_string()
        elif vtype == 9:
            arr_type = struct.unpack('<I', f.read(4))[0]
            arr_len = struct.unpack('<Q', f.read(8))[0]
            for _ in range(arr_len): skip_value(arr_type)
        elif vtype == 10: f.read(8)
        elif vtype == 11: f.read(8)
        elif vtype == 12: f.read(8)
    
    for _ in range(n_kv):
        read_string()
        vtype = struct.unpack('<I', f.read(4))[0]
        skip_value(vtype)
    
    tinfo = {}
    for _ in range(n_tensors):
        name = read_string()
        n_dims = struct.unpack('<I', f.read(4))[0]
        ne = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        ttype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        tinfo[name] = {'ne': ne, 'type': ttype, 'offset': offset}
    data_start = (f.tell() + 31) & ~31

def read_gguf_f32(name):
    """Read an F32 tensor from the GGUF."""
    t = tinfo[name]
    n = 1
    for d in t['ne']: n *= d
    with open(GGUF, 'rb') as f:
        f.seek(data_start + t['offset'])
        return np.frombuffer(f.read(n * 4), dtype=np.float32).copy()

def read_gguf_quant_row(name, row_idx):
    """Read a single row of a quantized weight from GGUF."""
    t = tinfo[name]
    ne0 = t['ne'][0]  # elements per row
    tp = t['type']
    if tp == 12:  # Q4_K
        bpr = (ne0 // 256) * 144
    elif tp == 14:  # Q6_K
        bpr = (ne0 // 256) * 210
    else:
        raise ValueError(f'Unsupported type {tp}')
    
    with open(GGUF, 'rb') as f:
        f.seek(data_start + t['offset'] + row_idx * bpr)
        return np.frombuffer(f.read(bpr), dtype=np.uint8).copy()

def gguf_matmul_vec(name, x):
    """Compute weight @ x reading directly from GGUF."""
    t = tinfo[name]
    ne0, ne1 = t['ne'][0], t['ne'][1]  # ne0=in_features, ne1=out_features
    tp = t['type']
    
    if tp == 12:  # Q4_K
        block_size = 144
        dequant = _dequant_q4k_blocks
    elif tp == 14:  # Q6_K
        block_size = 210
        dequant = _dequant_q6k_blocks
    else:
        raise ValueError(f'Unsupported type {tp}')
    
    bpr = (ne0 // 256) * block_size
    result = np.empty(ne1, dtype=np.float32)
    
    CHUNK = 256
    with open(GGUF, 'rb') as f:
        for start in range(0, ne1, CHUNK):
            end = min(start + CHUNK, ne1)
            n_rows = end - start
            f.seek(data_start + t['offset'] + start * bpr)
            raw = np.frombuffer(f.read(n_rows * bpr), dtype=np.uint8).copy()
            blocks = raw.reshape(-1, block_size)
            vals = dequant(blocks).reshape(n_rows, ne0)
            result[start:end] = vals @ x
    
    return result

# Load our engine for comparison
print('=== Loading engine ===')
e = Qwen25Engine(r'c:\quantoniumos\qwen2.5-coder_3b.rftmwpk')
e.load()
cfg = e.config

# Step 1: Embedding (already verified identical)
tid = 785
h = e._embed_token(tid)
print(f'Embedding norm: {np.linalg.norm(h):.6f}')

# Step 2: Layer 0 forward — using GGUF data directly
print('\n=== Layer 0 forward (GGUF-direct vs engine) ===')

# RMS norm
attn_norm_gguf = read_gguf_f32('blk.0.attn_norm.weight')
normed_gguf = _rms_norm(h, attn_norm_gguf, cfg.rms_norm_eps)
normed_eng = _rms_norm(h, e.layers[0].attn_norm, cfg.rms_norm_eps)
print(f'attn_norm weights diff: {np.max(np.abs(attn_norm_gguf - e.layers[0].attn_norm)):.2e}')
print(f'normed diff: {np.max(np.abs(normed_gguf - normed_eng)):.2e}')

# Q projection
q_gguf = gguf_matmul_vec('blk.0.attn_q.weight', normed_gguf)
q_eng = e.layers[0].q_proj.matmul(normed_gguf)
print(f'Q proj diff: {np.max(np.abs(q_gguf - q_eng)):.2e}  (gguf first 3: {q_gguf[:3]}  eng: {q_eng[:3]})')

# K projection
k_gguf = gguf_matmul_vec('blk.0.attn_k.weight', normed_gguf)
k_eng = e.layers[0].k_proj.matmul(normed_gguf)
print(f'K proj diff: {np.max(np.abs(k_gguf - k_eng)):.2e}')

# V projection
v_gguf = gguf_matmul_vec('blk.0.attn_v.weight', normed_gguf)
v_eng = e.layers[0].v_proj.matmul(normed_gguf)
print(f'V proj diff: {np.max(np.abs(v_gguf - v_eng)):.2e}')

# Add biases
q_bias_gguf = read_gguf_f32('blk.0.attn_q.bias')
k_bias_gguf = read_gguf_f32('blk.0.attn_k.bias')
v_bias_gguf = read_gguf_f32('blk.0.attn_v.bias')
print(f'Q bias diff: {np.max(np.abs(q_bias_gguf - e.layers[0].q_bias)):.2e}')
q_gguf += q_bias_gguf
k_gguf += k_bias_gguf
v_gguf += v_bias_gguf

# Reshape
q_reshaped = q_gguf.reshape(cfg.num_heads, cfg.head_dim)
k_reshaped = k_gguf.reshape(cfg.num_kv_heads, cfg.head_dim)
v_reshaped = v_gguf.reshape(cfg.num_kv_heads, cfg.head_dim)

# RoPE
rope = _rope_freqs(cfg.head_dim, cfg.max_position, cfg.rope_theta)
q_roped = _apply_rope(q_reshaped, rope, 0)
k_roped = _apply_rope(k_reshaped, rope, 0)

# Attention (single token -> attention output = V for each KV group)
heads_per_group = cfg.num_heads // cfg.num_kv_heads
attn = np.zeros((cfg.num_heads, cfg.head_dim), dtype=np.float32)
for kv_idx in range(cfg.num_kv_heads):
    for h_off in range(heads_per_group):
        h_idx = kv_idx * heads_per_group + h_off
        # Single token: score = q @ k / sqrt(d), softmax([score]) = 1.0
        attn[h_idx] = v_reshaped[kv_idx]  # weight = 1.0

# O projection
attn_flat = attn.reshape(-1)
o_gguf = gguf_matmul_vec('blk.0.attn_output.weight', attn_flat)
o_eng = e.layers[0].o_proj.matmul(attn_flat)
print(f'O proj diff: {np.max(np.abs(o_gguf - o_eng)):.2e}')

# Residual
h_after_attn_gguf = h + o_gguf

# FFN
ffn_norm_gguf = read_gguf_f32('blk.0.ffn_norm.weight')
ffn_normed = _rms_norm(h_after_attn_gguf, ffn_norm_gguf, cfg.rms_norm_eps)

gate_gguf = gguf_matmul_vec('blk.0.ffn_gate.weight', ffn_normed)
up_gguf = gguf_matmul_vec('blk.0.ffn_up.weight', ffn_normed)
down_input = _silu(gate_gguf) * up_gguf
down_gguf = gguf_matmul_vec('blk.0.ffn_down.weight', down_input)

h_after_ffn_gguf = h_after_attn_gguf + down_gguf

# Compare full layer 0 output
k_cache = np.zeros((10, cfg.num_kv_heads, cfg.head_dim), dtype=np.float32)
v_cache = np.zeros((10, cfg.num_kv_heads, cfg.head_dim), dtype=np.float32)
h_after_layer0_eng = e.layers[0].forward(h.copy(), 0, rope, k_cache, v_cache, 0)

diff = np.abs(h_after_ffn_gguf - h_after_layer0_eng)
print(f'\n=== Layer 0 full output ===')
print(f'GGUF-direct: norm={np.linalg.norm(h_after_ffn_gguf):.4f} first 4={h_after_ffn_gguf[:4]}')
print(f'Engine     : norm={np.linalg.norm(h_after_layer0_eng):.4f} first 4={h_after_layer0_eng[:4]}')
print(f'Max diff: {diff.max():.2e}  Mean diff: {diff.mean():.2e}')
corr = np.corrcoef(h_after_ffn_gguf, h_after_layer0_eng)[0, 1]
print(f'Correlation: {corr:.6f}')
