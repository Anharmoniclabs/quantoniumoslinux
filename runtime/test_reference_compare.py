"""Compare our engine's logits against llama-cpp-python reference."""
import sys, os, numpy as np, ctypes

# === Reference: llama-cpp-python ===
print('=== Loading reference (llama.cpp) ===')
from llama_cpp import Llama

GGUF = r'C:\Users\lansky\.ollama\models\blobs\sha256-4a188102020e9c9530b687fd6400f775c45e90a0d7baafe65bd0a36963fbb7ba'
llm = Llama(model_path=GGUF, n_ctx=128, n_gpu_layers=0, logits_all=True, verbose=False)

# Tokenize "The" 
prompt = "The"
tokens_ref = llm.tokenize(prompt.encode(), add_bos=False)
print(f'Reference tokens for "{prompt}": {tokens_ref}')

# Run eval to get logits
llm.reset()
llm.eval(tokens_ref)
# Get logits for the last token
ref_logits = np.array(llm.scores[len(tokens_ref) - 1])
print(f'Reference logits: shape={ref_logits.shape} min={ref_logits.min():.4f} max={ref_logits.max():.4f}')

top10_ref = np.argsort(ref_logits)[-10:][::-1]
print('Reference top 10:')
for tid in top10_ref:
    tok = llm.detokenize([tid]).decode('utf-8', errors='replace')
    print(f'  token={tid:6d}  logit={ref_logits[tid]:8.3f}  text={repr(tok)}')

del llm  # free memory

# === Our engine ===
print('\n=== Loading our engine ===')
sys.path.insert(0, r'c:\quantoniumos\src\rftmw_native\build')
os.environ.pop('RFTMW_NO_NATIVE', None)
from quantonium_os_src.engine.rftmw_numpy_engine import Qwen25Engine, _rms_norm, _get_native_infer

e = Qwen25Engine(r'c:\quantoniumos\qwen2.5-coder_3b.rftmwpk')
e.load()
cfg = e.config

# Tokenize
our_tokens = e.encode(prompt)
print(f'Our tokens for "{prompt}": {our_tokens}')
assert our_tokens == list(tokens_ref), f'Token mismatch: {our_tokens} != {list(tokens_ref)}'

# Forward pass
tid = our_tokens[0]
h = e._embed_token(tid)
print(f'Embedding: norm={np.linalg.norm(h):.6f} first5={h[:5]}')

max_seq = 10
k_caches = [np.zeros((max_seq, cfg.num_kv_heads, cfg.head_dim), dtype=np.float32)
             for _ in range(cfg.num_layers)]
v_caches = [np.zeros((max_seq, cfg.num_kv_heads, cfg.head_dim), dtype=np.float32)
             for _ in range(cfg.num_layers)]

for li, layer in enumerate(e.layers):
    h = layer.forward(h, 0, e.rope_freqs, k_caches[li], v_caches[li], 0)

h = _rms_norm(h, e.output_norm, cfg.rms_norm_eps)
our_logits = e._lm_head(h)

print(f'Our logits: shape={our_logits.shape} min={our_logits.min():.4f} max={our_logits.max():.4f}')

top10_ours = np.argsort(our_logits)[-10:][::-1]
print('Our top 10:')
for tid in top10_ours:
    tok = e.decode([tid])
    print(f'  token={tid:6d}  logit={our_logits[tid]:8.3f}  text={repr(tok)}')

# === Compare ===
print('\n=== Comparison ===')
# Check if any ref top-10 tokens appear in our top-10
ref_set = set(top10_ref)
our_set = set(top10_ours)
overlap = ref_set & our_set
print(f'Top-10 overlap: {len(overlap)} tokens in common')

# Correlation between logit vectors
corr = np.corrcoef(ref_logits[:our_logits.shape[0]], our_logits[:ref_logits.shape[0]])[0, 1]
print(f'Logit correlation: {corr:.6f}')

# Max diff
diff = np.abs(ref_logits[:our_logits.shape[0]] - our_logits[:ref_logits.shape[0]])
print(f'Logit max_diff: {diff.max():.4f}  mean_diff: {diff.mean():.4f}')
