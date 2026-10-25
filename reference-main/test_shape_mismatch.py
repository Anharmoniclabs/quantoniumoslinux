"""Check pack shapes vs GGUF shapes for ALL weights to find mismatches."""
import struct, json, numpy as np

GGUF = r'C:\Users\lansky\.ollama\models\blobs\sha256-4a188102020e9c9530b687fd6400f775c45e90a0d7baafe65bd0a36963fbb7ba'
PACK = r'c:\quantoniumos\qwen2.5-coder_3b.rftmwpk'

# Parse GGUF
with open(GGUF, 'rb') as f:
    f.read(4)
    struct.unpack('<I', f.read(4))
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
        gguf_tensors[name] = {'ne': ne, 'type': ttype}

# Parse pack
with open(PACK, 'rb') as f:
    f.read(8)
    hdr_len = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(hdr_len))
pack_idx = {w['name']: w for w in hdr['weights']}

TYPES = {0:'F32', 12:'Q4_K', 14:'Q6_K'}

# Compare shapes for all 2D weights
print('=== Shape comparison: Pack vs GGUF ===')
print(f'{"Name":45s} {"GGUF ne":20s} {"Type":6s} {"GGUF bpr":>8s} {"Pack shape":20s} {"Pack bpr":>8s} {"Match":>6s}')
print('-' * 125)

mismatches = []
for name in sorted(gguf_tensors.keys()):
    gt = gguf_tensors[name]
    if gt['type'] not in (12, 14):  # Only check quantized (2D) tensors
        continue
    if name not in pack_idx:
        continue
    
    pe = pack_idx[name]
    ne = gt['ne']
    ttype = gt['type']
    tn = TYPES.get(ttype, f'?{ttype}')
    
    # Expected bytes per row
    if ttype == 12:  # Q4K
        expected_bpr = (ne[0] // 256) * 144
    elif ttype == 14:  # Q6K
        expected_bpr = (ne[0] // 256) * 210
    
    pack_shape = pe['original_shape']
    pack_bpr = pack_shape[1] if len(pack_shape) == 2 else 0
    
    match = 'OK' if pack_bpr == expected_bpr else '**MISMATCH**'
    
    if pack_bpr != expected_bpr:
        mismatches.append(name)
    
    # Only print first occurrence per pattern + all mismatches
    if name.startswith('blk.0.') or name == 'token_embd.weight' or pack_bpr != expected_bpr:
        print(f'{name:45s} {str(ne):20s} {tn:6s} {expected_bpr:8d} {str(pack_shape):20s} {pack_bpr:8d} {match:>6s}')

if mismatches:
    print(f'\n*** {len(mismatches)} MISMATCHES FOUND! ***')
    for name in mismatches:
        gt = gguf_tensors[name]
        pe = pack_idx[name]
        ne = gt['ne']
        ttype = gt['type']
        tn = TYPES.get(ttype, f'?{ttype}')
        if ttype == 12:
            expected_bpr = (ne[0] // 256) * 144
        else:
            expected_bpr = (ne[0] // 256) * 210
        pack_bpr = pe['original_shape'][1]
        
        # What quant type does the engine detect for this?
        if pack_bpr % 144 == 0 and pack_bpr % 210 != 0:
            detected = 'Q4K'
        elif pack_bpr % 210 == 0:
            detected = 'Q6K'
        else:
            detected = '???'
        
        print(f'  {name}: GGUF={tn}({expected_bpr} bpr) Pack=({pack_bpr} bpr, detected={detected})')
else:
    print('\nAll shapes match!')
