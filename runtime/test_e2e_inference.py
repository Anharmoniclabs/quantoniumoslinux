"""End-to-end inference test: generate with C++ and compare first token logits."""
import sys, os, time
import numpy as np

sys.path.insert(0, r"c:\quantoniumos")
os.environ["RFTMW_NO_NATIVE"] = "1"  # Force NumPy-only

from quantonium_os_src.engine.rftmw_numpy_engine import (
    Qwen25Engine, _rms_norm, _apply_rope, _softmax, _silu,
    _dequant_q4k_blocks, _dequant_q6k_blocks, _get_native_infer
)

print("=== Loading engine (NumPy-only mode) ===")
engine = Qwen25Engine(r"c:\quantoniumos\qwen2.5-coder_3b.rftmwpk")
engine.load()
engine._ensure_tokenizer()

prompt = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
ids = engine.encode(prompt)
print(f"Prompt: {prompt!r}")
print(f"Tokens ({len(ids)}): {ids}")

cfg = engine.config
print(f"Config: hidden={cfg.hidden_size}, heads={cfg.num_heads}, kv_heads={cfg.num_kv_heads}, head_dim={cfg.head_dim}")
print(f"  layers={cfg.num_layers}, vocab={cfg.vocab_size}, intermediate={cfg.intermediate_size}")

# Allocate KV caches
max_seq = len(ids) + 10
k_caches = [np.zeros((max_seq, cfg.num_kv_heads, cfg.head_dim), dtype=np.float32)
             for _ in range(cfg.num_layers)]
v_caches = [np.zeros((max_seq, cfg.num_kv_heads, cfg.head_dim), dtype=np.float32)
             for _ in range(cfg.num_layers)]

# Prefill all prompt tokens
print(f"\n=== Prefill ({len(ids)} tokens) ===")
cache_len = 0
for pos_idx, tid in enumerate(ids):
    h = engine._embed_token(tid)
    if pos_idx == 0:
        print(f"  Token 0 (id={tid}): embed stats: min={h.min():.4f} max={h.max():.4f} mean={h.mean():.6f}")
    for li, layer in enumerate(engine.layers):
        h = layer.forward(h, pos_idx, engine.rope_freqs,
                          k_caches[li], v_caches[li], cache_len)
    if pos_idx == 0:
        print(f"  After all layers: h stats: min={h.min():.4f} max={h.max():.4f} mean={h.mean():.6f}")
    h = _rms_norm(h, engine.output_norm, cfg.rms_norm_eps)
    if pos_idx == 0:
        print(f"  After output norm: h stats: min={h.min():.4f} max={h.max():.4f} mean={h.mean():.6f}")
    cache_len += 1
    if pos_idx % 3 == 0:
        print(f"  Prefilled token {pos_idx}/{len(ids)}")

# After prefill, compute logits
logits = engine._lm_head(h)
print(f"\n=== Logits after prefill ===")
print(f"  Shape: {logits.shape}, dtype: {logits.dtype}")
print(f"  Stats: min={logits.min():.4f} max={logits.max():.4f} mean={logits.mean():.6f}")
print(f"  NaN: {np.any(np.isnan(logits))}, Inf: {np.any(np.isinf(logits))}")

# Top-10 tokens
top10_idx = np.argsort(logits)[-10:][::-1]
print(f"\n  Top-10 logits:")
for idx in top10_idx:
    tok_text = engine.decode([int(idx)])
    print(f"    [{idx:6d}] logit={logits[idx]:8.3f}  text={tok_text!r}")

# Sample
next_id = engine._sample(logits, temperature=0.3, top_k=40)
sampled_text = engine.decode([next_id])
print(f"\n  Sampled: id={next_id}, text={sampled_text!r}")

# Generate a few more tokens
print(f"\n=== Generating 5 tokens ===")
generated = []
for step in range(5):
    pos = cache_len
    h = engine._embed_token(next_id)
    for li, layer in enumerate(engine.layers):
        h = layer.forward(h, pos, engine.rope_freqs,
                          k_caches[li], v_caches[li], cache_len)
    h = _rms_norm(h, engine.output_norm, cfg.rms_norm_eps)
    cache_len += 1
    
    logits = engine._lm_head(h)
    next_id = engine._sample(logits, temperature=0.3, top_k=40)
    tok_text = engine.decode([next_id])
    generated.append(tok_text)
    
    # Check for stop tokens
    if next_id in [151643, 151645]:
        print(f"  Step {step}: STOP token {next_id}")
        break
    
    top3_idx = np.argsort(logits)[-3:][::-1]
    top3_text = [f"{engine.decode([int(i)])}({logits[i]:.2f})" for i in top3_idx]
    print(f"  Step {step}: id={next_id} text={tok_text!r} top3={top3_text}")

print(f"\n  Full generated: {''.join(generated)!r}")
print("\nDone!")
