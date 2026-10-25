"""Check pack weight index — compare with GGUF."""
import struct, json

with open(r'c:\quantoniumos\qwen2.5-coder_3b.rftmwpk', 'rb') as f:
    magic = f.read(8)
    hdr_len = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(hdr_len))

weights = hdr['weights']
print(f'Pack has {len(weights)} weight entries')

# Check for any missing or extra compared to expected 434
non_blk = [w for w in weights if not w['name'].startswith('blk.')]
print(f'\nNon-block weights ({len(non_blk)}):')
for w in non_blk:
    print(f"  {w['name']:40s}  shape={w['original_shape']}  method={w['method']}")

# Check block 0
blk0 = [w for w in weights if w['name'].startswith('blk.0.')]
print(f'\nBlock 0 weights ({len(blk0)}):')
for w in blk0:
    print(f"  {w['name']:40s}  shape={w['original_shape']}  method={w['method']}")

# Check a few specific things  
print(f'\n=== Weight method distribution ===')
methods = {}
for w in weights:
    m = w['method']
    methods[m] = methods.get(m, 0) + 1
for m, c in sorted(methods.items()):
    print(f'  {m}: {c}')
