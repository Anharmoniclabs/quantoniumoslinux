"""Quick 7B diagnostic — single token forward + LM head check."""
import sys, numpy as np, time
sys.path.insert(0, 'quantonium_os_src')
sys.path.insert(0, 'src/rftmw_native/build_asm')
from engine.rftmw_numpy_engine import Qwen25Engine, _rms_norm

e = Qwen25Engine('qwen2.5-coder_7b.rftmwpk')
e.load()
cfg = e.config
rope = e.rope_freqs

# Single token through all 28 layers
tok_id = 9707  # 'Hello'
h = e._embed_token(tok_id)
k_caches = [np.zeros((10, cfg.num_kv_heads, cfg.head_dim), dtype=np.float32)
             for _ in range(cfg.num_layers)]
v_caches = [np.zeros((10, cfg.num_kv_heads, cfg.head_dim), dtype=np.float32)
             for _ in range(cfg.num_layers)]

t0 = time.perf_counter()
for li, layer in enumerate(e.layers):
    h = layer.forward(h, 0, rope, k_caches[li], v_caches[li], 0)
h = _rms_norm(h, e.output_norm, cfg.rms_norm_eps)
print(f"All layers: {time.perf_counter()-t0:.2f}s")
print(f"h: mean={h.mean():.6f} std={h.std():.6f} NaN={np.any(np.isnan(h))}")

# LM head
t0 = time.perf_counter()
logits = e._lm_head(h)
print(f"LM head: {time.perf_counter()-t0:.2f}s")
print(f"logits shape={logits.shape}")

e._ensure_tokenizer()
top10 = np.argsort(logits)[-10:][::-1]
for t in top10:
    print(f"  {int(t):6d}: {e.decode([int(t)])!r:20s} score={logits[t]:.3f}")
