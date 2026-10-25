"""List ALL tensor names in the GGUF to check for output weight."""
import struct

gguf_path = r'C:\Users\lansky\.ollama\models\blobs\sha256-4a188102020e9c9530b687fd6400f775c45e90a0d7baafe65bd0a36963fbb7ba'
TYPES = {0:'F32', 1:'F16', 2:'Q4_0', 3:'Q4_1', 6:'Q5_0', 7:'Q5_1',
         8:'Q8_0', 9:'Q8_1', 12:'Q4_K', 13:'Q5_K', 14:'Q6_K', 15:'Q8_K'}

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
    
    # Print relevant KV pairs
    for _ in range(n_kv):
        key = read_string()
        vtype = struct.unpack('<I', f.read(4))[0]
        val = read_value(vtype)
        # Print important config keys
        if any(k in key for k in ['head_count', 'embed', 'layer', 'rope', 'context', 'vocab']):
            print(f'KV: {key} = {val}')
    
    print(f'\n=== ALL {n_tensors} tensors ===')
    non_blk = []
    blk0 = []
    for _ in range(n_tensors):
        name = read_string()
        n_dims = struct.unpack('<I', f.read(4))[0]
        ne = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        ttype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        tn = TYPES.get(ttype, f'T{ttype}')
        
        if not name.startswith('blk.'):
            non_blk.append(f'{name:40s}  ne={ne}  type={tn}')
        elif name.startswith('blk.0.'):
            blk0.append(f'{name:40s}  ne={ne}  type={tn}')
    
    print('\nNon-block tensors:')
    for t in non_blk:
        print(f'  {t}')
    
    print(f'\nBlock 0 tensors (pattern repeats for blk.1-35):')
    for t in blk0:
        print(f'  {t}')
