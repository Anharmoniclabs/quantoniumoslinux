"""Get the final hidden state (before lm_head) from llama-cpp-python 
and compare with our engine's final hidden state."""
import sys, os, numpy as np, ctypes
sys.path.insert(0, r'c:\quantoniumos\src\rftmw_native\build')
os.environ.pop('RFTMW_NO_NATIVE', None)

GGUF = r'C:\Users\lansky\.ollama\models\blobs\sha256-4a188102020e9c9530b687fd6400f775c45e90a0d7baafe65bd0a36963fbb7ba'

# Get reference embedding (last hidden state AFTER output norm)
print('=== Reference via llama-cpp-python (embedding mode) ===')
from llama_cpp import Llama
llm = Llama(model_path=GGUF, n_ctx=128, n_gpu_layers=0, embedding=True, verbose=False)

# Get embedding for "The" = token 785
ref_emb = llm.embed("The")
ref_emb = np.array(ref_emb, dtype=np.float32)
print(f'Reference hidden state shape: {ref_emb.shape}')
if ref_emb.ndim > 1:
    ref_emb = ref_emb[0]
print(f'  first 8: {ref_emb[:8]}')
print(f'  norm: {np.linalg.norm(ref_emb):.6f}')
del llm

# Get our engine's hidden state
print('\n=== Our engine ===')
from quantonium_os_src.engine.rftmw_numpy_engine import Qwen25Engine, _rms_norm, _get_native_infer

e = Qwen25Engine(r'c:\quantoniumos\qwen2.5-coder_3b.rftmwpk')
e.load()
cfg = e.config

tid = 785
h = e._embed_token(tid)

max_seq = 10
k_caches = [np.zeros((max_seq, cfg.num_kv_heads, cfg.head_dim), dtype=np.float32)
             for _ in range(cfg.num_layers)]
v_caches = [np.zeros((max_seq, cfg.num_kv_heads, cfg.head_dim), dtype=np.float32)
             for _ in range(cfg.num_layers)]

for li, layer in enumerate(e.layers):
    h = layer.forward(h, 0, e.rope_freqs, k_caches[li], v_caches[li], 0)

# Apply output norm (same as llama.cpp's final norm)
h_normed = _rms_norm(h, e.output_norm, cfg.rms_norm_eps)
print(f'Our hidden state (post output norm): shape={h_normed.shape}')
print(f'  first 8: {h_normed[:8]}')
print(f'  norm: {np.linalg.norm(h_normed):.6f}')

# Compare
if ref_emb.shape[0] == h_normed.shape[0]:
    diff = np.abs(ref_emb - h_normed)
    corr = np.corrcoef(ref_emb, h_normed)[0, 1]
    print(f'\nHidden state comparison:')
    print(f'  max_diff: {diff.max():.4f}')
    print(f'  mean_diff: {diff.mean():.4f}')
    print(f'  correlation: {corr:.6f}')
else:
    print(f'\nShape mismatch: ref={ref_emb.shape} ours={h_normed.shape}')
    # The embedding API might return the pooled output, try raw
    print(f'Ref range: [{ref_emb.min():.4f}, {ref_emb.max():.4f}]')
    print(f'Our range: [{h_normed.min():.4f}, {h_normed.max():.4f}]')

# Also: compare pre-norm hidden state
print(f'\nPre-norm hidden: norm={np.linalg.norm(h):.4f} first 4={h[:4]}')
