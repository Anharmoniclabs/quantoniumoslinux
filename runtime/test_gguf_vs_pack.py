"""Compare pack's INT8_ZLIB reconstructed bytes against original GGUF bytes.

This verifies if the pack compression damaged the GGUF data.
"""
import struct, json, numpy as np, sys, os

sys.path.insert(0, r"c:\quantoniumos")

GGUF_PATH = r"C:\Users\lansky\.ollama\models\blobs\sha256-4a188102020e9c9530b687fd6400f775c45e90a0d7baafe65bd0a36963fbb7ba"
PACK_PATH = r"c:\quantoniumos\qwen2.5-coder_3b.rftmwpk"

# ===== Step 1: Parse the GGUF to find tensor data offsets =====
print("=== Parsing GGUF ===")

def read_gguf_header(path):
    """Parse GGUF v3 header to get tensor info."""
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == b'GGUF', f"Bad magic: {magic}"
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        print(f"  GGUF v{version}, {n_tensors} tensors, {n_kv} KV entries")
        
        # Skip KV metadata
        def read_string():
            slen = struct.unpack('<Q', f.read(8))[0]
            return f.read(slen).decode('utf-8')
        
        def read_value(vtype):
            """Read a GGUF value by type."""
            if vtype == 0:   # UINT8
                return struct.unpack('<B', f.read(1))[0]
            elif vtype == 1: # INT8
                return struct.unpack('<b', f.read(1))[0]
            elif vtype == 2: # UINT16
                return struct.unpack('<H', f.read(2))[0]
            elif vtype == 3: # INT16
                return struct.unpack('<h', f.read(2))[0]
            elif vtype == 4: # UINT32
                return struct.unpack('<I', f.read(4))[0]
            elif vtype == 5: # INT32
                return struct.unpack('<i', f.read(4))[0]
            elif vtype == 6: # FLOAT32
                return struct.unpack('<f', f.read(4))[0]
            elif vtype == 7: # BOOL
                return struct.unpack('<B', f.read(1))[0] != 0
            elif vtype == 8: # STRING
                return read_string()
            elif vtype == 9: # ARRAY
                arr_type = struct.unpack('<I', f.read(4))[0]
                arr_len = struct.unpack('<Q', f.read(8))[0]
                return [read_value(arr_type) for _ in range(arr_len)]
            elif vtype == 10: # UINT64
                return struct.unpack('<Q', f.read(8))[0]
            elif vtype == 11: # INT64
                return struct.unpack('<q', f.read(8))[0]
            elif vtype == 12: # FLOAT64
                return struct.unpack('<d', f.read(8))[0]
            else:
                raise ValueError(f"Unknown GGUF value type: {vtype}")
        
        # Read KV pairs
        kv = {}
        for _ in range(n_kv):
            key = read_string()
            vtype = struct.unpack('<I', f.read(4))[0]
            val = read_value(vtype)
            kv[key] = val
        
        # Read tensor info
        GGUF_TYPES = {
            0: ('F32', 4, 1), 1: ('F16', 2, 1), 2: ('Q4_0', 18, 32),
            3: ('Q4_1', 20, 32), 6: ('Q5_0', 22, 32), 7: ('Q5_1', 24, 32),
            8: ('Q8_0', 34, 32), 9: ('Q8_1', 40, 32),
            12: ('Q4_K', 144, 256), 13: ('Q5_K', 176, 256),
            14: ('Q6_K', 210, 256), 15: ('Q8_K', 292, 256),
            # Add more as needed
        }
        
        tensors = {}
        for _ in range(n_tensors):
            name = read_string()
            n_dims = struct.unpack('<I', f.read(4))[0]
            ne = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            
            type_info = GGUF_TYPES.get(ttype, (f'UNK{ttype}', 0, 0))
            tensors[name] = {
                'ne': ne, 'type': ttype, 'type_name': type_info[0],
                'block_bytes': type_info[1], 'block_vals': type_info[2],
                'offset': offset
            }
        
        data_offset = f.tell()
        # Align to 32 bytes
        data_offset = (data_offset + 31) & ~31
        
        return kv, tensors, data_offset

kv, tensors, data_start = read_gguf_header(GGUF_PATH)

# Print some key tensors
for name in ['token_embd.weight', 'blk.0.attn_q.weight', 'blk.0.attn_q.bias',
             'blk.0.attn_k.weight', 'blk.0.ffn_gate.weight', 'output_norm.weight']:
    if name in tensors:
        t = tensors[name]
        print(f"  {name:40s}  ne={t['ne']}  type={t['type_name']}  offset={t['offset']}")

# ===== Step 2: Read raw bytes from GGUF for a specific tensor =====
print("\n=== Comparing blk.0.attn_q.weight bytes ===")

tensor_name = 'blk.0.attn_q.weight'
t = tensors[tensor_name]
ne = t['ne']
print(f"  GGUF ne={ne}, type={t['type_name']}, block_bytes={t['block_bytes']}, block_vals={t['block_vals']}")

# Calculate size
n_elements = 1
for d in ne:
    n_elements *= d
n_blocks = n_elements // t['block_vals']
byte_size = n_blocks * t['block_bytes']
print(f"  n_elements={n_elements}, n_blocks={n_blocks}, byte_size={byte_size}")

# Read raw bytes from GGUF
with open(GGUF_PATH, 'rb') as f:
    f.seek(data_start + t['offset'])
    gguf_bytes = np.frombuffer(f.read(byte_size), dtype=np.uint8).copy()
print(f"  Read {len(gguf_bytes)} bytes from GGUF")

# ===== Step 3: Load the same tensor from the pack =====
print("\n=== Loading same tensor from pack ===")
os.environ["RFTMW_NO_NATIVE"] = "1"
from quantonium_os_src.engine.rftmw_numpy_engine import Qwen25Engine

engine = Qwen25Engine(PACK_PATH)
engine.load()

# Get the reconstructed bytes
pack_bytes = engine.layers[0].q_proj.raw.ravel()
print(f"  Pack raw bytes: {len(pack_bytes)} ({engine.layers[0].q_proj.out_features} x {engine.layers[0].q_proj.bytes_per_row})")

# ===== Step 4: Compare byte-by-byte =====
print("\n=== Byte comparison ===")

# The GGUF stores as (ne[0], ne[1]) = (2048, 2048) elements
# Pack stores as (out_features=2048, bytes_per_row=1152) raw GGUF bytes
# With Q4K: 1152 bytes / 144 bytes_per_block = 8 blocks
# 8 blocks * 256 vals = 2048 input features

# Check dimensions match
gguf_n_rows = ne[1] if len(ne) > 1 else 1  # ne[1] = number of rows in GGUF
gguf_n_cols = ne[0]  # ne[0] = columns (inner dimension)
print(f"  GGUF: {gguf_n_rows} rows × {gguf_n_cols} columns")
print(f"  Pack: {engine.layers[0].q_proj.out_features} rows × {engine.layers[0].q_proj.in_features} cols")

# For Q4K, each row is blocks of 144 bytes covering 256 values
n_blocks_per_row = gguf_n_cols // t['block_vals']
bytes_per_row = n_blocks_per_row * t['block_bytes']
print(f"  Blocks per row: {n_blocks_per_row}, bytes per row: {bytes_per_row}")

# Compare first row
gguf_row0 = gguf_bytes[:bytes_per_row]
pack_row0 = pack_bytes[:bytes_per_row]

diff_count = np.sum(gguf_row0 != pack_row0)
diff_positions = np.where(gguf_row0 != pack_row0)[0]
total = len(gguf_row0)
print(f"\n  Row 0: {diff_count}/{total} bytes differ ({100*diff_count/total:.1f}%)")

if diff_count > 0:
    print(f"  First 20 diff positions: {diff_positions[:20]}")
    for pos in diff_positions[:10]:
        print(f"    byte[{pos}]: GGUF={gguf_row0[pos]} pack={pack_row0[pos]} diff={int(gguf_row0[pos]) - int(pack_row0[pos])}")

# Compare all rows stats
gguf_2d = gguf_bytes.reshape(gguf_n_rows, bytes_per_row)
pack_2d = pack_bytes.reshape(engine.layers[0].q_proj.out_features, engine.layers[0].q_proj.bytes_per_row)

if gguf_2d.shape == pack_2d.shape:
    total_diff = np.sum(gguf_2d != pack_2d)
    total_bytes = gguf_2d.size
    print(f"\n  All rows: {total_diff}/{total_bytes} bytes differ ({100*total_diff/total_bytes:.1f}%)")
    
    # Check how many differ by more than 1
    abs_diff = np.abs(gguf_2d.astype(np.int16) - pack_2d.astype(np.int16))
    diff_by_1 = np.sum(abs_diff == 1)
    diff_by_more = np.sum(abs_diff > 1)
    print(f"  Bytes off by exactly 1: {diff_by_1}")
    print(f"  Bytes off by >1: {diff_by_more}")
    print(f"  Max abs diff: {abs_diff.max()}")
    
    # Check scale bytes specifically (first 4 bytes of each 144-byte block)
    n_blocks_total = gguf_2d.shape[0] * n_blocks_per_row
    gguf_blocks = gguf_2d.reshape(-1, t['block_bytes'])  # (n_blocks, 144)
    pack_blocks = pack_2d.reshape(-1, t['block_bytes'])
    
    scale_diff = np.sum(gguf_blocks[:, :4] != pack_blocks[:, :4])
    subsc_diff = np.sum(gguf_blocks[:, 4:16] != pack_blocks[:, 4:16])
    data_diff = np.sum(gguf_blocks[:, 16:] != pack_blocks[:, 16:])
    
    print(f"\n  Scale bytes (0-3): {scale_diff}/{n_blocks_total*4} differ")
    print(f"  Sub-scale bytes (4-15): {subsc_diff}/{n_blocks_total*12} differ")
    print(f"  Data bytes (16-143): {data_diff}/{n_blocks_total*128} differ")
else:
    print(f"\n  Shape mismatch: GGUF={gguf_2d.shape} pack={pack_2d.shape}")

# ===== Step 5: Dequant comparison =====
print("\n=== Dequantized value comparison (first block) ===")
from quantonium_os_src.engine.rftmw_numpy_engine import _dequant_q4k_blocks

gguf_dequant = _dequant_q4k_blocks(gguf_row0[:144].reshape(1, 144)).flatten()
pack_dequant = _dequant_q4k_blocks(pack_row0[:144].reshape(1, 144)).flatten()

dq_diff = np.abs(gguf_dequant - pack_dequant)
print(f"  Max dequant diff: {dq_diff.max():.8f}")
print(f"  Mean dequant diff: {dq_diff.mean():.8f}")
print(f"  GGUF range: [{gguf_dequant.min():.6f}, {gguf_dequant.max():.6f}]")
print(f"  Pack range: [{pack_dequant.min():.6f}, {pack_dequant.max():.6f}]")

# Check the fp16 scales
d_gguf = np.frombuffer(gguf_row0[:2].tobytes(), dtype=np.float16).astype(np.float32)[0]
d_pack = np.frombuffer(pack_row0[:2].tobytes(), dtype=np.float16).astype(np.float32)[0]
dmin_gguf = np.frombuffer(gguf_row0[2:4].tobytes(), dtype=np.float16).astype(np.float32)[0]
dmin_pack = np.frombuffer(pack_row0[2:4].tobytes(), dtype=np.float16).astype(np.float32)[0]
print(f"\n  d:    GGUF={d_gguf:.10f}  pack={d_pack:.10f}")
print(f"  dmin: GGUF={dmin_gguf:.10f}  pack={dmin_pack:.10f}")

print("\nDone!")
