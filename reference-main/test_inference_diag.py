"""Diagnostic: run a single forward pass and inspect logits/top tokens."""
import sys, os, numpy as np
sys.path.insert(0, '.')
os.environ['RFTMW_NO_NATIVE'] = '1'  # pure NumPy for debugging

from quantonium_os_src.engine.rftmw_numpy_engine import (
    Qwen25Engine, _rms_norm, _dequant_q4k_blocks, _dequant_q6k_blocks
)

e = Qwen25Engine(r'c:\quantoniumos\qwen2.5-coder_3b.rftmwpk')
e.load()
e._ensure_tokenizer()

# Simple prompt
prompt = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
input_ids = e.encode(prompt)
print(f"Prompt tokens ({len(input_ids)}): {input_ids}")
print(f"Decoded back: {e.decode(input_ids)}")

cfg = e.config

# Prefill: process all prompt tokens
max_seq = len(input_ids) + 10
k_caches = [np.zeros((max_seq, cfg.num_kv_heads, cfg.head_dim), dtype=np.float32)
             for _ in range(cfg.num_layers)]
v_caches = [np.zeros((max_seq, cfg.num_kv_heads, cfg.head_dim), dtype=np.float32)
             for _ in range(cfg.num_layers)]

cache_len = 0
for pos_idx, tid in enumerate(input_ids):
    h = e._embed_token(tid)
    if pos_idx == 0:
        print(f"\nEmbed token 0 (id={tid}): norm={np.linalg.norm(h):.4f}, "
              f"min={h.min():.4f}, max={h.max():.4f}, mean={h.mean():.6f}")
    for li, layer in enumerate(e.layers):
        h = layer.forward(h, pos_idx, e.rope_freqs,
                          k_caches[li], v_caches[li], cache_len)
        if pos_idx == len(input_ids) - 1 and li in [0, 17, 35]:
            print(f"  After layer {li}: norm={np.linalg.norm(h):.4f}, "
                  f"min={h.min():.4f}, max={h.max():.4f}")
    h = _rms_norm(h, e.output_norm, cfg.rms_norm_eps)
    if pos_idx == len(input_ids) - 1:
        print(f"  After output norm: norm={np.linalg.norm(h):.4f}, "
              f"min={h.min():.6f}, max={h.max():.6f}")
    cache_len += 1

# Compute logits from final hidden state
logits = e._lm_head(h)
print(f"\nLogits: shape={logits.shape}, min={logits.min():.4f}, "
      f"max={logits.max():.4f}, mean={logits.mean():.6f}")

# Top 10 most likely tokens
top_idx = np.argsort(logits)[-10:][::-1]
print("\nTop 10 predictions:")
for i, idx in enumerate(top_idx):
    tok_str = e.decode([idx])
    print(f"  {i+1}. token_id={idx:6d}  logit={logits[idx]:8.3f}  text={tok_str!r}")

# Check if '4' is in top 100
token_4_ids = e.encode("4")
print(f"\n'4' encodes to: {token_4_ids}")
for t4 in token_4_ids:
    rank = int((logits >= logits[t4]).sum())
    print(f"  token_id={t4} logit={logits[t4]:.4f} rank={rank}/{len(logits)}")

# Temperature=0 greedy
greedy_id = int(np.argmax(logits))
print(f"\nGreedy next token: id={greedy_id} text={e.decode([greedy_id])!r}")
