"""Trace divergence: compare our dequant against manual Q6K reference."""
import sys, os, struct, numpy as np

sys.path.insert(0, r'c:\quantoniumos\src\rftmw_native\build')
os.environ.pop('RFTMW_NO_NATIVE', None)

GGUF = r'C:\Users\lansky\.ollama\models\blobs\sha256-4a188102020e9c9530b687fd6400f775c45e90a0d7baafe65bd0a36963fbb7ba'

# Parse GGUF to find token_embd.weight
print('=== Reading GGUF tensor info ===')
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
    
    tensors = {}
    for _ in range(n_tensors):
        name = read_string()
        n_dims = struct.unpack('<I', f.read(4))[0]
        ne = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        ttype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        tensors[name] = {'ne': ne, 'type': ttype, 'offset': offset}
    
    data_start = (f.tell() + 31) & ~31

# Read embedding raw bytes for token 785 directly from GGUF
emb_info = tensors['token_embd.weight']
print(f'token_embd: ne={emb_info["ne"]} type={emb_info["type"]} offset={emb_info["offset"]}')

# ne=[2048, 151936] type=14 (Q6_K)
# Q6K: 210 bytes per block, 256 values per block
# Each row has ne[0]=2048 values = 8 Q6K blocks = 8*210 = 1680 bytes
bytes_per_row = 1680
token_id = 785

with open(GGUF, 'rb') as f:
    f.seek(data_start + emb_info['offset'] + token_id * bytes_per_row)
    raw_bytes = np.frombuffer(f.read(bytes_per_row), dtype=np.uint8)

print(f'Raw bytes for token {token_id}: shape={raw_bytes.shape} first 10={raw_bytes[:10]}')

# Manually dequant Q6K following llama.cpp's exact formula
def reference_dequant_q6k(raw_210):
    """Dequant a single Q6_K block (210 bytes) → 256 float32 values.
    
    Follows llama.cpp dequantize_row_q6_K exactly.
    """
    ql = raw_210[0:128]    # lower quant bits
    qh = raw_210[128:192]  # upper quant bits
    sc = raw_210[192:208].view(np.int8)  # 16 signed scales
    d = np.frombuffer(raw_210[208:210].tobytes(), dtype=np.float16).astype(np.float32)[0]
    
    y = np.empty(256, dtype=np.float32)
    
    for half in range(2):
        ql_off = half * 64
        qh_off = half * 32
        sc_off = half * 8
        base = half * 128
        
        for l in range(32):
            is_val = (half * 128) // 16  # = 0 for half=0, 8 for half=1
            
            q1 = int(ql[ql_off + l] & 0xF) | (int(qh[qh_off + l] & 0x03) << 4)
            q2 = int(ql[ql_off + 32 + l] & 0xF) | (int((qh[qh_off + l] >> 2) & 0x03) << 4)
            q3 = int(ql[ql_off + l] >> 4) | (int((qh[qh_off + l] >> 4) & 0x03) << 4)
            q4 = int(ql[ql_off + 32 + l] >> 4) | (int((qh[qh_off + l] >> 6) & 0x03) << 4)
            
            q1 -= 32; q2 -= 32; q3 -= 32; q4 -= 32
            
            y[base + l + 0]  = d * float(sc[is_val + 0]) * q1
            y[base + l + 32] = d * float(sc[is_val + 2]) * q2
            y[base + l + 64] = d * float(sc[is_val + 4]) * q3
            y[base + l + 96] = d * float(sc[is_val + 6]) * q4
    
    return y

# Dequant all 8 blocks
ref_embedding = np.empty(2048, dtype=np.float32)
for blk in range(8):
    block_raw = raw_bytes[blk * 210 : (blk + 1) * 210].copy()
    ref_embedding[blk * 256 : (blk + 1) * 256] = reference_dequant_q6k(block_raw)

print(f'\nReference embedding (manual dequant):')
print(f'  first 8: {ref_embedding[:8]}')
print(f'  norm: {np.linalg.norm(ref_embedding):.6f}')

# Now compare with our engine
from quantonium_os_src.engine.rftmw_numpy_engine import Qwen25Engine, _dequant_q6k_blocks

e = Qwen25Engine(r'c:\quantoniumos\qwen2.5-coder_3b.rftmwpk')
e.load()

our_embedding = e._embed_token(token_id)
print(f'\nOur engine embedding:')
print(f'  first 8: {our_embedding[:8]}')
print(f'  norm: {np.linalg.norm(our_embedding):.6f}')

diff = np.abs(ref_embedding - our_embedding)
print(f'\nEmbedding diff: max={diff.max():.2e} mean={diff.mean():.2e}')

# Also check: raw bytes from pack vs GGUF for this specific token
pack_raw = e.embed.raw[token_id]  # (1680,) uint8
print(f'\nPack raw first 10: {pack_raw[:10]}')
print(f'GGUF raw first 10: {raw_bytes[:10]}')
byte_diff = np.sum(pack_raw != raw_bytes)
print(f'Byte differences: {byte_diff} / {len(raw_bytes)}')

if byte_diff > 0:
    mismatches = np.where(pack_raw != raw_bytes)[0]
    print(f'First 5 mismatch positions: {mismatches[:5]}')
    for pos in mismatches[:5]:
        print(f'  pos={pos}: pack={pack_raw[pos]} gguf={raw_bytes[pos]}')
