"""Check if matmul is transposed — compare W@x vs W.T@x."""
import sys, os, numpy as np
sys.path.insert(0, r'c:\quantoniumos\src\rftmw_native\build')
os.environ.pop('RFTMW_NO_NATIVE', None)

from quantonium_os_src.engine.rftmw_numpy_engine import (
    Qwen25Engine, _rms_norm, _dequant_q4k_blocks
)

print('Loading engine...')
e = Qwen25Engine(r'c:\quantoniumos\qwen2.5-coder_3b.rftmwpk')
e.load()
cfg = e.config

# Check: does Q4K matmul compute W@x or W.T@x?
# In GGUF, weight shapes for attn_q are [out_features, in_features]
# llama.cpp does: output = weight_matrix * input  (row-by-row dot products)
#
# Our QuantWeight.matmul does: raw[row] dequant → dot with x → output[row]
# That means output[i] = W[i,:] @ x  which is standard W@x
#
# GGUF stores ne=[out, in] for q_proj, so row i = the i-th output neuron's weights
# This should be correct.

# But let's verify by checking the GGUF tensor shapes
import struct
gguf_path = r'C:\Users\lansky\.ollama\models\blobs\sha256-4a188102020e9c9530b687fd6400f775c45e90a0d7baafe65bd0a36963fbb7ba'
with open(gguf_path, 'rb') as f:
    magic = f.read(4)
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]
    
    def read_string():
        slen = struct.unpack('<Q', f.read(8))[0]
        return f.read(slen).decode('utf-8')
    
    def read_value(vtype):
        if vtype == 0: return struct.unpack('<B', f.read(1))[0]
        elif vtype == 1: return struct.unpack('<b', f.read(1))[0]
        elif vtype == 2: return struct.unpack('<H', f.read(2))[0]
        elif vtype == 3: return struct.unpack('<h', f.read(2))[0]
        elif vtype == 4: return struct.unpack('<I', f.read(4))[0]
        elif vtype == 5: return struct.unpack('<i', f.read(4))[0]
        elif vtype == 6: return struct.unpack('<f', f.read(4))[0]
        elif vtype == 7: return struct.unpack('<B', f.read(1))[0] != 0
        elif vtype == 8: return read_string()
        elif vtype == 9:
            arr_type = struct.unpack('<I', f.read(4))[0]
            arr_len = struct.unpack('<Q', f.read(8))[0]
            return [read_value(arr_type) for _ in range(arr_len)]
        elif vtype == 10: return struct.unpack('<Q', f.read(8))[0]
        elif vtype == 11: return struct.unpack('<q', f.read(8))[0]
        elif vtype == 12: return struct.unpack('<d', f.read(8))[0]
        else: raise ValueError(f"Unknown type: {vtype}")
    
    for _ in range(n_kv):
        read_string()
        vtype = struct.unpack('<I', f.read(4))[0]
        read_value(vtype)
    
    for _ in range(n_tensors):
        name = read_string()
        n_dims = struct.unpack('<I', f.read(4))[0]
        ne = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        ttype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        
        TYPES = {0:'F32', 2:'Q4_0', 12:'Q4_K', 13:'Q5_K', 14:'Q6_K'}
        tn = TYPES.get(ttype, f'T{ttype}')
        
        if name in ['blk.0.attn_q.weight', 'blk.0.attn_k.weight', 'blk.0.attn_v.weight',
                     'blk.0.attn_output.weight', 'blk.0.ffn_gate.weight', 'blk.0.ffn_up.weight',
                     'blk.0.ffn_down.weight', 'token_embd.weight', 'output.weight']:
            print(f'GGUF {name:40s}  ne={ne}  type={tn}')

# Now check our pack's weight index
print()
import json
with open(r'c:\quantoniumos\qwen2.5-coder_3b.rftmwpk', 'rb') as f:
    magic = f.read(8)
    hdr_len = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(hdr_len))

wi = {w['name']: w for w in hdr['weights']}
for name in ['blk.0.attn_q.weight', 'blk.0.attn_k.weight', 'blk.0.attn_v.weight',
             'blk.0.attn_output.weight', 'blk.0.ffn_gate.weight', 'blk.0.ffn_up.weight',
             'blk.0.ffn_down.weight', 'token_embd.weight']:
    if name in wi:
        w = wi[name]
        print(f'Pack {name:40s}  shape={w["original_shape"]}  gguf_type={w.get("gguf_type","")}')

# Check: our engine's QuantWeight dimensions
qw = e.layers[0].q_proj
print(f'\nEngine q_proj: out_features={qw.out_features} in_features={qw.in_features} raw_shape={qw.raw.shape}')
print(f'  out_features should = num_heads * head_dim = {cfg.num_heads * cfg.head_dim}')
print(f'  in_features should  = hidden_size = {cfg.hidden_size}')

# CRITICAL CHECK: are rows = output neurons or input neurons?
# If GGUF ne=[2048, 2048] for q_proj, that means:
#   ne[0]=2048 (fastest-varying dim = columns = in_features)
#   ne[1]=2048 (rows = out_features)
# So GGUF stores row-major with [out, in] but ne reports [in, out]!
# That means our raw[i] is actually the i-th OUTPUT neuron — correct for W@x

# But wait — does this depend on the quantization format?
# In Q4_K, each block of 256 values stores a row of 256 columns.
# If the weight has ne=[2048, 2048], each row has 2048 values = 2048/256 = 8 blocks
print(f'\nQ4K blocks per row: {qw.bytes_per_row // 144}')
print(f'Q4K values per row: {(qw.bytes_per_row // 144) * 256}')
