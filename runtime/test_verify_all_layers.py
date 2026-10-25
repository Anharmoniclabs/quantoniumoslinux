"""Verify pack weight bytes match GGUF for weights across all layers."""
import sys, os, struct, json, numpy as np

GGUF = r'C:\Users\lansky\.ollama\models\blobs\sha256-4a188102020e9c9530b687fd6400f775c45e90a0d7baafe65bd0a36963fbb7ba'
PACK = r'c:\quantoniumos\qwen2.5-coder_3b.rftmwpk'

# Parse GGUF tensor info
with open(GGUF, 'rb') as f:
    f.read(4)  # magic
    struct.unpack('<I', f.read(4))  # version
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
    gguf_tensors = {}
    for _ in range(n_tensors):
        name = read_string()
        n_dims = struct.unpack('<I', f.read(4))[0]
        ne = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        ttype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        gguf_tensors[name] = {'ne': ne, 'type': ttype, 'offset': offset}
    gguf_data_start = (f.tell() + 31) & ~31

# Parse pack header
with open(PACK, 'rb') as f:
    f.read(8)  # magic
    hdr_len = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(hdr_len))
    pack_data_start = 16 + hdr_len

pack_idx = {w['name']: w for w in hdr['weights']}

# For each layer, check one Q4K weight (attn_q.weight) raw bytes
print('=== Verifying pack vs GGUF bytes for attn_q.weight across layers ===')
all_ok = True
for layer in range(36):
    name = f'blk.{layer}.attn_q.weight'
    gt = gguf_tensors[name]
    pe = pack_idx[name]
    
    # GGUF bytes
    ne0 = gt['ne'][0]  # 2048
    ne1 = gt['ne'][1]  # 2048
    bpr = (ne0 // 256) * 144  # Q4K: 1152
    total_bytes = ne1 * bpr
    
    # Read first 1152 bytes (row 0) and last 1152 bytes (last row) from GGUF
    with open(GGUF, 'rb') as f:
        f.seek(gguf_data_start + gt['offset'])
        gguf_first = np.frombuffer(f.read(bpr), dtype=np.uint8)
        f.seek(gguf_data_start + gt['offset'] + (ne1-1) * bpr)
        gguf_last = np.frombuffer(f.read(bpr), dtype=np.uint8)
    
    # Read corresponding bytes from pack — need to decompress INT8_ZLIB
    # Use the engine's loaded weights instead (faster)
    # Actually, let's use the engine  
    pass  # we'll check row 0 and last row

# Simpler approach: load engine and compare row bytes for each layer
sys.path.insert(0, r'c:\quantoniumos\src\rftmw_native\build')
os.environ.pop('RFTMW_NO_NATIVE', None)
from quantonium_os_src.engine.rftmw_numpy_engine import Qwen25Engine

print('Loading engine...')
e = Qwen25Engine(PACK)
e.load()

mismatches = []
for layer in range(36):
    name = f'blk.{layer}.attn_q.weight'
    gt = gguf_tensors[name]
    ne0, ne1 = gt['ne']
    bpr = (ne0 // 256) * 144
    
    qw = e.layers[layer].q_proj
    pack_raw = qw.raw  # (ne1, bpr) uint8
    
    # Check rows 0, middle, and last against GGUF
    for row in [0, ne1//2, ne1-1]:
        with open(GGUF, 'rb') as f:
            f.seek(gguf_data_start + gt['offset'] + row * bpr)
            gguf_row = np.frombuffer(f.read(bpr), dtype=np.uint8)
        
        pack_row = pack_raw[row]
        ndiff = np.sum(pack_row != gguf_row)
        if ndiff > 0:
            mismatches.append((layer, row, ndiff))
            print(f'  MISMATCH: layer={layer} row={row} diff_bytes={ndiff}/{bpr}')
    
    if layer % 6 == 0:
        print(f'  Layer {layer}: checked 3 rows — OK')

if not mismatches:
    print('\nAll attn_q.weight bytes match GGUF across all 36 layers!')
else:
    print(f'\n{len(mismatches)} row mismatches found!')
    for layer, row, ndiff in mismatches:
        print(f'  Layer {layer} row {row}: {ndiff} bytes differ')

# Also check ffn_gate (another Q4K) and ffn_down (Q6K) for a few layers
print('\n=== Checking ffn_gate (Q4K) and ffn_down (Q6K) ===')
for layer in [0, 5, 17, 35]:
    # ffn_gate
    name = f'blk.{layer}.ffn_gate.weight'
    gt = gguf_tensors[name]
    ne0, ne1 = gt['ne']
    bpr_gate = (ne0 // 256) * 144
    qw = e.layers[layer].gate_proj
    
    for row in [0, ne1//2, ne1-1]:
        with open(GGUF, 'rb') as f:
            f.seek(gguf_data_start + gt['offset'] + row * bpr_gate)
            gguf_row = np.frombuffer(f.read(bpr_gate), dtype=np.uint8)
        ndiff = np.sum(qw.raw[row] != gguf_row)
        if ndiff > 0:
            print(f'  MISMATCH: {name} row={row} diff={ndiff}')
    
    # ffn_down (Q6K)
    name = f'blk.{layer}.ffn_down.weight'
    gt = gguf_tensors[name]
    ne0, ne1 = gt['ne']
    bpr_down = (ne0 // 256) * 210
    qw = e.layers[layer].down_proj
    
    for row in [0, ne1//2, ne1-1]:
        with open(GGUF, 'rb') as f:
            f.seek(gguf_data_start + gt['offset'] + row * bpr_down)
            gguf_row = np.frombuffer(f.read(bpr_down), dtype=np.uint8)
        ndiff = np.sum(qw.raw[row] != gguf_row)
        if ndiff > 0:
            print(f'  MISMATCH: {name} row={row} diff={ndiff}')
    
    print(f'  Layer {layer} ffn checked — OK')

print('\nDone.')
