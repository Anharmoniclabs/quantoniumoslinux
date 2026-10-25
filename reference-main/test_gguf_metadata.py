"""Read GGUF metadata and verify model config parameters."""
import numpy as np
from gguf import GGUFReader

GGUF = r'C:\Users\lansky\.ollama\models\blobs\sha256-4a188102020e9c9530b687fd6400f775c45e90a0d7baafe65bd0a36963fbb7ba'

reader = GGUFReader(GGUF)

print("=== GGUF Metadata ===")
for field in reader.fields.values():
    name = field.name
    # Skip large/binary fields
    if len(field.data) > 100:
        continue
    try:
        # Get the actual value(s)
        parts = field.parts
        if len(parts) >= 2:
            val = parts[-1]
            if hasattr(val, 'tolist'):
                val = val.tolist()
            print(f"  {name} = {val}")
        else:
            print(f"  {name} = (parts={len(parts)})")
    except Exception as e:
        print(f"  {name} = ERROR: {e}")

# Specifically look for key config values
print("\n=== Key Config Values ===")
key_fields = [
    'qwen2.embedding_length',
    'qwen2.attention.head_count', 
    'qwen2.attention.head_count_kv',
    'qwen2.feed_forward_length',
    'qwen2.block_count',
    'qwen2.rope.freq_base',
    'qwen2.attention.layer_norm_rms_epsilon',
    'qwen2.context_length',
    'general.architecture',
    'general.name',
]

for key in key_fields:
    if key in reader.fields:
        field = reader.fields[key]
        parts = field.parts
        if len(parts) >= 2:
            val = parts[-1]
            if hasattr(val, 'tolist'):
                val = val.tolist()
            print(f"  {key} = {val}")
    else:
        # Try finding it with partial match
        for fname in reader.fields:
            if key.split('.')[-1] in fname:
                field = reader.fields[fname]
                parts = field.parts
                if len(parts) >= 2:
                    val = parts[-1]
                    if hasattr(val, 'tolist'):
                        val = val.tolist()
                    print(f"  {fname} = {val}")

# Compare against our hardcoded config
print("\n=== Our Hardcoded Config ===")
print("  hidden_size = 2048")
print("  num_heads = 16")
print("  num_kv_heads = 2")
print("  head_dim = 128")
print("  intermediate_size = 11008")
print("  num_layers = 36")
print("  vocab_size = 151936")
print("  rope_theta = 1000000.0")
print("  rms_norm_eps = 1e-6")
