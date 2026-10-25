"""Step-by-step comparison: engine vs independent forward pass."""""
import sys, os, numpy as np
os.environ['RFTMW_NO_NATIVE'] = '1'

from gguf import GGUFReader
from gguf.quants import dequantize

GGUF = r'C:\Users\lansky\.ollama\models\blobs\sha256-4a188102020e9c9530b687fd6400f775c45e90a0d7baafe65bd0a36963fbb7ba'
reader = GGUFReader(GGUF)
tensors = {t.name: t for t in reader.tensors}

def get_f32(name):
    t = tensors[name]
    if t.tensor_type == 0:
        return np.array(t.data, dtype=np.float32)
    return dequantize(t.data, t.tensor_type).astype(np.float32)

# Load engine
sys.path.insert(0, 'src/rftmw_native/build')
from quantonium_os_src.engine.rftmw_numpy_engine import Qwen25Engine, _rms_norm, _rope_freqs

eng = Qwen25Engine(r'c:\quantoniumos\qwen2.5-coder_3b.rftmwpk')
eng.load()

cfg = eng.config
eps = cfg.rms_norm_eps
token_id = 785

# ---- EMBEDDING ----
h_eng = eng._embed_token(token_id)

embed_w = get_f32('token_embd.weight')
h_ref = embed_w[token_id].copy()

print(f"Embed: eng_norm={np.linalg.norm(h_eng):.6f} ref_norm={np.linalg.norm(h_ref):.6f}")
print(f"  max_diff={np.abs(h_eng - h_ref).max():.6e} corr={np.corrcoef(h_eng, h_ref)[0,1]:.8f}")

# ---- LAYER 0 STEP BY STEP ----
print("\n=== Layer 0 step by step ===")

# Attn norm
an_w = get_f32('blk.0.attn_norm.weight')
normed_ref = h_ref / np.sqrt(np.mean(h_ref**2) + eps) * an_w
normed_eng = _rms_norm(h_eng, eng.layers[0].attn_norm, eps)
print(f"Norm: ref_first3={normed_ref[:3]} eng_first3={normed_eng[:3]}")
print(f"  max_diff={np.abs(normed_ref - normed_eng).max():.6e}")

# Check if the norm weights match
print(f"  attn_norm weights diff: {np.abs(an_w - eng.layers[0].attn_norm).max():.6e}")

# Q projection  
q_w = get_f32('blk.0.attn_q.weight')
q_b = get_f32('blk.0.attn_q.bias')
q_ref = q_w @ normed_ref + q_b
q_eng = eng.layers[0].q_proj.matmul(normed_eng) + eng.layers[0].q_bias
print(f"\nQ: ref_first3={q_ref[:3]}")
print(f"   eng_first3={q_eng[:3]}")
print(f"  max_diff={np.abs(q_ref - q_eng).max():.6e}")

# Also check: matmul the ref normed through engine's q_proj
q_cross = eng.layers[0].q_proj.matmul(normed_ref) + eng.layers[0].q_bias
print(f"  cross_check (eng_weight @ ref_normed): max_diff from ref={np.abs(q_ref - q_cross).max():.6e}")

# K, V
k_w = get_f32('blk.0.attn_k.weight')
k_b = get_f32('blk.0.attn_k.bias')
k_ref = k_w @ normed_ref + k_b
k_eng = eng.layers[0].k_proj.matmul(normed_eng) + eng.layers[0].k_bias
print(f"\nK: max_diff={np.abs(k_ref - k_eng).max():.6e}")

v_w = get_f32('blk.0.attn_v.weight')
v_b = get_f32('blk.0.attn_v.bias')
v_ref = v_w @ normed_ref + v_b
v_eng = eng.layers[0].v_proj.matmul(normed_eng) + eng.layers[0].v_bias
print(f"V: max_diff={np.abs(v_ref - v_eng).max():.6e}")

# GQA attention at pos=0, seq_len=1 (just passthrough v)
n_heads = cfg.num_heads
n_kv = cfg.num_kv_heads
hd = cfg.head_dim
hpg = n_heads // n_kv

v_ref_r = v_ref.reshape(n_kv, hd)
attn_ref = np.zeros((n_heads, hd), dtype=np.float32)
for kv_i in range(n_kv):
    for ho in range(hpg):
        attn_ref[kv_i * hpg + ho] = v_ref_r[kv_i]

# Engine attention
kc = np.zeros((10, n_kv, hd), dtype=np.float32)
vc = np.zeros((10, n_kv, hd), dtype=np.float32)
rope = eng.rope_freqs

# Full engine layer 0
h_eng_l0 = eng.layers[0].forward(h_eng.copy(), 0, rope, kc, vc, 0)

# Manual full layer 0 using ref weights
o_w = get_f32('blk.0.attn_output.weight')
o_ref = o_w @ attn_ref.reshape(-1)
h_ref_after_attn = h_ref + o_ref

fn_w = get_f32('blk.0.ffn_norm.weight')
normed_ffn_ref = h_ref_after_attn / np.sqrt(np.mean(h_ref_after_attn**2) + eps) * fn_w
gw = get_f32('blk.0.ffn_gate.weight')
uw = get_f32('blk.0.ffn_up.weight')
dw = get_f32('blk.0.ffn_down.weight')
gate_ref = gw @ normed_ffn_ref
up_ref = uw @ normed_ffn_ref
silu_ref = gate_ref / (1.0 + np.exp(-np.clip(gate_ref, -88, 88)))
ffn_ref = dw @ (silu_ref * up_ref)
h_ref_l0 = h_ref_after_attn + ffn_ref

print(f"\nFull layer 0:")
print(f"  ref_norm={np.linalg.norm(h_ref_l0):.4f} eng_norm={np.linalg.norm(h_eng_l0):.4f}")
print(f"  max_diff={np.abs(h_ref_l0 - h_eng_l0).max():.6e}")
print(f"  corr={np.corrcoef(h_ref_l0, h_eng_l0)[0,1]:.8f}")

# If layer 0 matches, let's see where the trouble starts
# Run ref and engine through multiple layers
h_ref_cur = h_ref.copy()
h_eng_cur = h_eng.copy()

# Engine KV caches
kcs = [np.zeros((10, n_kv, hd), dtype=np.float32) for _ in range(cfg.num_layers)]
vcs = [np.zeros((10, n_kv, hd), dtype=np.float32) for _ in range(cfg.num_layers)]

for layer_idx in range(min(5, cfg.num_layers)):
    # Engine forward
    h_eng_cur = eng.layers[layer_idx].forward(h_eng_cur, 0, rope, kcs[layer_idx], vcs[layer_idx], 0)
    
    # Manual forward with float32 dequanted weights
    an = get_f32(f'blk.{layer_idx}.attn_norm.weight')
    qw = get_f32(f'blk.{layer_idx}.attn_q.weight')
    kw = get_f32(f'blk.{layer_idx}.attn_k.weight')
    vw = get_f32(f'blk.{layer_idx}.attn_v.weight')
    ow = get_f32(f'blk.{layer_idx}.attn_output.weight')
    qb = get_f32(f'blk.{layer_idx}.attn_q.bias')
    kb = get_f32(f'blk.{layer_idx}.attn_k.bias')
    vb = get_f32(f'blk.{layer_idx}.attn_v.bias')
    fn = get_f32(f'blk.{layer_idx}.ffn_norm.weight')    
    gw = get_f32(f'blk.{layer_idx}.ffn_gate.weight')
    uw = get_f32(f'blk.{layer_idx}.ffn_up.weight')
    dw = get_f32(f'blk.{layer_idx}.ffn_down.weight')
    
    normed = h_ref_cur / np.sqrt(np.mean(h_ref_cur**2) + eps) * an
    q = qw @ normed + qb
    k = kw @ normed + kb
    v = vw @ normed + vb
    v_r = v.reshape(n_kv, hd)
    attn = np.zeros((n_heads, hd), dtype=np.float32)
    for ki in range(n_kv):
        for ho in range(hpg):
            attn[ki * hpg + ho] = v_r[ki]
    o = ow @ attn.reshape(-1)
    h_ref_cur = h_ref_cur + o
    
    normed = h_ref_cur / np.sqrt(np.mean(h_ref_cur**2) + eps) * fn
    gate = gw @ normed
    up = uw @ normed
    ffn = dw @ (gate / (1 + np.exp(-np.clip(gate, -88, 88))) * up)
    h_ref_cur = h_ref_cur + ffn
    
    d = np.abs(h_ref_cur - h_eng_cur)
    c = np.corrcoef(h_ref_cur, h_eng_cur)[0, 1]
    print(f"Layer {layer_idx}: ref_norm={np.linalg.norm(h_ref_cur):.4f} eng_norm={np.linalg.norm(h_eng_cur):.4f} max_diff={d.max():.4e} corr={c:.8f}")
