"""DEFINITIVE TEST: Run full 36-layer forward pass using weights loaded
directly from the GGUF file, bypassing the pack entirely.

If this produces correct output, the pack data is the problem.
If it still produces garbage, the forward pass logic is wrong."""
import sys, os, struct, time, numpy as np
sys.path.insert(0, r'c:\quantoniumos\src\rftmw_native\build')
os.environ['RFTMW_NO_NATIVE'] = '1'  # pure NumPy to eliminate C++ variables

GGUF = r'C:\Users\lansky\.ollama\models\blobs\sha256-4a188102020e9c9530b687fd6400f775c45e90a0d7baafe65bd0a36963fbb7ba'

from quantonium_os_src.engine.rftmw_numpy_engine import (
    _dequant_q4k_blocks, _dequant_q6k_blocks, _rms_norm,
    _apply_rope, _softmax, _silu, _rope_freqs, 
    QuantWeight, Qwen25Config, _parse_gguf_f32_index
)

# Parse GGUF header
print('Parsing GGUF header...')
with open(GGUF, 'rb') as f:
    f.read(4); struct.unpack('<I', f.read(4))
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]
    def rs():
        slen = struct.unpack('<Q', f.read(8))[0]
        return f.read(slen).decode('utf-8')
    def sv(vt):
        if vt in (0,1,7): f.read(1)
        elif vt in (2,3): f.read(2)
        elif vt in (4,5,6): f.read(4)
        elif vt == 8: rs()
        elif vt == 9:
            at = struct.unpack('<I', f.read(4))[0]
            al = struct.unpack('<Q', f.read(8))[0]
            for _ in range(al): sv(at)
        elif vt in (10,11,12): f.read(8)
    for _ in range(n_kv):
        rs(); sv(struct.unpack('<I', f.read(4))[0])
    ti = {}
    for _ in range(n_tensors):
        nm = rs()
        nd = struct.unpack('<I', f.read(4))[0]
        ne = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
        tt = struct.unpack('<I', f.read(4))[0]
        off = struct.unpack('<Q', f.read(8))[0]
        ti[nm] = {'ne': ne, 'type': tt, 'offset': off}
    DS = (f.tell() + 31) & ~31

def load_f32(name):
    t = ti[name]; n = 1
    for d in t['ne']: n *= d
    with open(GGUF, 'rb') as f:
        f.seek(DS + t['offset'])
        return np.frombuffer(f.read(n*4), dtype=np.float32).copy()

def load_quant(name):
    """Load a quantized weight directly from GGUF as QuantWeight."""
    t = ti[name]
    ne0, ne1 = t['ne']
    tt = t['type']
    if tt == 12:  bs = 144
    elif tt == 14: bs = 210
    else: raise ValueError(f'Unknown type {tt}')
    bpr = (ne0 // 256) * bs
    total = ne1 * bpr
    with open(GGUF, 'rb') as f:
        f.seek(DS + t['offset'])
        raw = np.frombuffer(f.read(total), dtype=np.uint8).copy()
    return QuantWeight(raw, (ne1, bpr))

# Load ALL weights from GGUF
cfg = Qwen25Config()
print('Loading embedding from GGUF...')
embed = load_quant('token_embd.weight')
output_norm = load_f32('output_norm.weight')

rope = _rope_freqs(cfg.head_dim, cfg.max_position, cfg.rope_theta)

layers = []
for i in range(cfg.num_layers):
    print(f'Loading layer {i}/36 from GGUF...', end='\r')
    L = {
        'attn_norm': load_f32(f'blk.{i}.attn_norm.weight'),
        'ffn_norm': load_f32(f'blk.{i}.ffn_norm.weight'),
        'q_proj': load_quant(f'blk.{i}.attn_q.weight'),
        'k_proj': load_quant(f'blk.{i}.attn_k.weight'),
        'v_proj': load_quant(f'blk.{i}.attn_v.weight'),
        'o_proj': load_quant(f'blk.{i}.attn_output.weight'),
        'gate_proj': load_quant(f'blk.{i}.ffn_gate.weight'),
        'up_proj': load_quant(f'blk.{i}.ffn_up.weight'),
        'down_proj': load_quant(f'blk.{i}.ffn_down.weight'),
        'q_bias': load_f32(f'blk.{i}.attn_q.bias'),
        'k_bias': load_f32(f'blk.{i}.attn_k.bias'),
        'v_bias': load_f32(f'blk.{i}.attn_v.bias'),
    }
    layers.append(L)
print(f'All 36 layers loaded from GGUF directly.')

# Manual forward pass
def forward_layer(h, layer, pos, rope_freqs, k_cache, v_cache, cache_len):
    L = layer
    normed = _rms_norm(h, L['attn_norm'], cfg.rms_norm_eps)
    
    q = L['q_proj'].matmul(normed) + L['q_bias']
    k = L['k_proj'].matmul(normed) + L['k_bias']
    v = L['v_proj'].matmul(normed) + L['v_bias']
    
    q = q.reshape(cfg.num_heads, cfg.head_dim)
    k = k.reshape(cfg.num_kv_heads, cfg.head_dim)
    v = v.reshape(cfg.num_kv_heads, cfg.head_dim)
    
    q = _apply_rope(q, rope_freqs, pos)
    k = _apply_rope(k, rope_freqs, pos)
    
    k_cache[cache_len] = k
    v_cache[cache_len] = v
    seq_len = cache_len + 1
    
    hpg = cfg.num_heads // cfg.num_kv_heads
    attn = np.zeros((cfg.num_heads, cfg.head_dim), dtype=np.float32)
    for kvi in range(cfg.num_kv_heads):
        ks = k_cache[:seq_len, kvi]
        vs = v_cache[:seq_len, kvi]
        for ho in range(hpg):
            hi = kvi * hpg + ho
            scores = q[hi] @ ks.T / np.sqrt(np.float32(cfg.head_dim))
            w = _softmax(scores)
            attn[hi] = w @ vs
    
    o = L['o_proj'].matmul(attn.reshape(-1))
    h = h + o
    
    normed = _rms_norm(h, L['ffn_norm'], cfg.rms_norm_eps)
    gate = L['gate_proj'].matmul(normed)
    up = L['up_proj'].matmul(normed)
    ffn_out = L['down_proj'].matmul(_silu(gate) * up)
    h = h + ffn_out
    
    return h

# Token
tid = 785  # "The"
# Embedding lookup
row = embed.raw[tid].reshape(1, -1)
if embed.quant_type == 'q6k':
    blocks = row.reshape(-1, 210)
    h = _dequant_q6k_blocks(blocks).flatten()[:cfg.hidden_size]
else:
    blocks = row.reshape(-1, 144)
    h = _dequant_q4k_blocks(blocks).flatten()[:cfg.hidden_size]

print(f'\nEmbedding: norm={np.linalg.norm(h):.6f}')

# Forward through all layers
max_seq = 10
kcaches = [np.zeros((max_seq, cfg.num_kv_heads, cfg.head_dim), dtype=np.float32) for _ in range(36)]
vcaches = [np.zeros((max_seq, cfg.num_kv_heads, cfg.head_dim), dtype=np.float32) for _ in range(36)]

t0 = time.time()
for li in range(36):
    h = forward_layer(h, layers[li], 0, rope, kcaches[li], vcaches[li], 0)
    if li < 4 or li == 35:
        print(f'  Layer {li}: norm={np.linalg.norm(h):.4f}')

h_normed = _rms_norm(h, output_norm, cfg.rms_norm_eps)
logits = embed.matmul(h_normed)
elapsed = time.time() - t0
print(f'\nForward done in {elapsed:.1f}s')
print(f'Logits: min={logits.min():.4f} max={logits.max():.4f}')

top10 = np.argsort(logits)[-10:][::-1]
# Use llama-cpp-python tokenizer for decoding
from llama_cpp import Llama
llm = Llama(model_path=GGUF, n_ctx=8, verbose=False)
print('\nTop 10 predictions (GGUF-direct forward):')
for t in top10:
    tok = llm.detokenize([t]).decode('utf-8', errors='replace')
    print(f'  token={t:6d}  logit={logits[t]:8.3f}  text={repr(tok)}')
