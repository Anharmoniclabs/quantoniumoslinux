"""Inspect the pack file weight index to find missing weights."""
import json, struct
pack = r"c:\quantoniumos\qwen2.5-coder_3b.rftmwpk"
with open(pack, 'rb') as f:
    magic = f.read(8)
    hdr_len = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(hdr_len))

weights = hdr['weights']
names = sorted([w['name'] for w in weights])
print(f"Total weights: {len(names)}")
print(f"\nAll weight names:")
for n in names:
    w = next(w for w in weights if w['name'] == n)
    print(f"  {n:40s}  shape={w['original_shape']}  method={w['method']}")

# Check for biases
print(f"\n=== Bias check ===")
for key in ['attn_q.bias', 'attn_k.bias', 'attn_v.bias', 'attn_output.bias']:
    found = [n for n in names if key in n]
    print(f"  {key}: {len(found)} found {'✓' if found else '✗ MISSING'}")

# Check layer 0 specifically
print(f"\n=== Layer 0 weights ===")
for n in names:
    if 'blk.0.' in n:
        w = next(w for w in weights if w['name'] == n)
        print(f"  {n:40s}  shape={w['original_shape']}  method={w['method']}")
