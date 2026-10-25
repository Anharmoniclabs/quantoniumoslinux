"""Compare ALL 1D tensor values (norms, biases) between GGUF and pack.

2D weight matrices match exactly (verified). But 1D tensors use INT8_ZLIB
or NONE compression — let's check if they're accurate.
"""
import struct, json, io, zlib, numpy as np, sys, os

sys.path.insert(0, r"c:\quantoniumos")
os.environ["RFTMW_NO_NATIVE"] = "1"

GGUF_PATH = r"C:\Users\lansky\.ollama\models\blobs\sha256-4a188102020e9c9530b687fd6400f775c45e90a0d7baafe65bd0a36963fbb7ba"
PACK_PATH = r"c:\quantoniumos\qwen2.5-coder_3b.rftmwpk"

# ===== Parse GGUF =====
from test_gguf_vs_pack import read_gguf_header
kv, tensors, data_start = read_gguf_header(GGUF_PATH)

# ===== Load pack engine =====
from quantonium_os_src.engine.rftmw_numpy_engine import Qwen25Engine
engine = Qwen25Engine(PACK_PATH)
engine.load()

# ===== Compare 1D tensors =====
def read_gguf_f32_tensor(name):
    """Read an F32 tensor from GGUF."""
    t = tensors[name]
    assert t['type'] == 0 or t['type_name'] == 'F32', f"{name} is {t['type_name']}, not F32"
    n_elements = 1
    for d in t['ne']:
        n_elements *= d
    with open(GGUF_PATH, 'rb') as f:
        f.seek(data_start + t['offset'])
        data = np.frombuffer(f.read(n_elements * 4), dtype=np.float32).copy()
    return data

# Compare output_norm
print("=== output_norm.weight ===")
gguf_onorm = read_gguf_f32_tensor('output_norm.weight')
pack_onorm = engine.output_norm
diff = np.abs(gguf_onorm - pack_onorm)
print(f"  GGUF: range=[{gguf_onorm.min():.6f}, {gguf_onorm.max():.6f}], mean={gguf_onorm.mean():.6f}")
print(f"  Pack: range=[{pack_onorm.min():.6f}, {pack_onorm.max():.6f}], mean={pack_onorm.mean():.6f}")
print(f"  Max diff: {diff.max():.8f}, Mean diff: {diff.mean():.8f}")
print(f"  Relative max diff: {(diff / (np.abs(gguf_onorm) + 1e-10)).max():.6f}")

# Compare layer 0 norms and biases
for i in [0, 1, 17, 35]:
    print(f"\n=== Layer {i} ===")
    
    # attn_norm
    name = f'blk.{i}.attn_norm.weight'
    gguf_v = read_gguf_f32_tensor(name)
    pack_v = engine.layers[i].attn_norm
    d = np.abs(gguf_v - pack_v)
    print(f"  attn_norm: gguf_range=[{gguf_v.min():.4f},{gguf_v.max():.4f}] max_diff={d.max():.6f} rel_max={d.max()/(np.abs(gguf_v).max()+1e-10):.6f}")
    
    # ffn_norm
    name = f'blk.{i}.ffn_norm.weight'
    gguf_v = read_gguf_f32_tensor(name)
    pack_v = engine.layers[i].ffn_norm
    d = np.abs(gguf_v - pack_v)
    print(f"  ffn_norm:  gguf_range=[{gguf_v.min():.4f},{gguf_v.max():.4f}] max_diff={d.max():.6f} rel_max={d.max()/(np.abs(gguf_v).max()+1e-10):.6f}")
    
    # q_bias (INT8_ZLIB)
    name = f'blk.{i}.attn_q.bias'
    gguf_v = read_gguf_f32_tensor(name)
    pack_v = engine.layers[i].q_bias
    d = np.abs(gguf_v - pack_v)
    print(f"  q_bias:    gguf_range=[{gguf_v.min():.4f},{gguf_v.max():.4f}] pack_range=[{pack_v.min():.4f},{pack_v.max():.4f}]")
    print(f"             max_diff={d.max():.6f} mean_diff={d.mean():.6f} rel_max={d.max()/(np.abs(gguf_v).max()+1e-10):.6f}")
    
    # k_bias (NONE)
    name = f'blk.{i}.attn_k.bias'
    gguf_v = read_gguf_f32_tensor(name)
    pack_v = engine.layers[i].k_bias
    d = np.abs(gguf_v - pack_v)
    print(f"  k_bias:    gguf_range=[{gguf_v.min():.4f},{gguf_v.max():.4f}] max_diff={d.max():.8f}")
    
    # v_bias (NONE)
    name = f'blk.{i}.attn_v.bias'
    gguf_v = read_gguf_f32_tensor(name) 
    pack_v = engine.layers[i].v_bias
    d = np.abs(gguf_v - pack_v)
    print(f"  v_bias:    gguf_range=[{gguf_v.min():.4f},{gguf_v.max():.4f}] max_diff={d.max():.8f}")

# ===== Summary of all norms/biases =====
print("\n=== Summary: Worst-case errors across all 36 layers ===")
worst_norm = 0
worst_bias = 0
for i in range(36):
    for attr_name, gguf_name in [('attn_norm', f'blk.{i}.attn_norm.weight'), 
                                   ('ffn_norm', f'blk.{i}.ffn_norm.weight')]:
        gguf_v = read_gguf_f32_tensor(gguf_name)
        pack_v = getattr(engine.layers[i], attr_name)
        d = np.abs(gguf_v - pack_v).max()
        worst_norm = max(worst_norm, d)
    
    gguf_v = read_gguf_f32_tensor(f'blk.{i}.attn_q.bias')
    pack_v = engine.layers[i].q_bias
    d = np.abs(gguf_v - pack_v).max()
    worst_bias = max(worst_bias, d)

print(f"  Worst norm error: {worst_norm:.8f}")
print(f"  Worst q_bias error: {worst_bias:.8f}")

# Check output_norm
d = np.abs(read_gguf_f32_tensor('output_norm.weight') - engine.output_norm).max()
print(f"  Output norm error: {d:.8f}")

print("\nDone!")
