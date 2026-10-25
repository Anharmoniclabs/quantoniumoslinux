"""Debug SSM forward pass — corrected Mamba-2 ZOH semantics."""
import numpy as np
from quantonium_os_src.engine.qwen35_moe_engine import Qwen35MoeEngine
from quantonium_os_src.engine.rftmw_numpy_engine import _rms_norm

gguf_path = r'C:\Users\lansky\.ollama\models\blobs\sha256-ea269fa39ffe4b9f320205bbd24ea9847dad25805b7bd7c21167c7effdf4bece'
engine = Qwen35MoeEngine(gguf_path)
engine.load()

# Embed a token and trace through layer 0
x = engine._embed(0)  # [2048]
print(f'Embed: {x.shape}, range=[{x.min():.3f}, {x.max():.3f}]')

L0 = engine.layers[0]
normed = _rms_norm(x, L0.attn_norm, 1e-6)
print(f'Normed: range=[{normed.min():.3f}, {normed.max():.3f}]')

# QK projection
qk = L0.attn_qkv.matmul(normed)
print(f'QK: {qk.shape}, range=[{qk.min():.3f}, {qk.max():.3f}]')

# Z gate
z = L0.attn_gate.matmul(normed)
print(f'Z: {z.shape}, range=[{z.min():.3f}, {z.max():.3f}]')

# Alpha (raw), beta (raw)
alpha_raw = L0.ssm_alpha.matmul(normed)
beta_raw  = L0.ssm_beta.matmul(normed)
print(f'Alpha raw: range=[{alpha_raw.min():.3f}, {alpha_raw.max():.3f}]')
print(f'Beta raw:  range=[{beta_raw.min():.3f}, {beta_raw.max():.3f}]')

# Corrected Mamba-2 ZOH discretization for head 0
A_val   = L0.ssm_a[0]
dt_bias = L0.ssm_dt_bias[0]
print(f'\nHead 0: A_val={A_val:.4f}, dt_bias={dt_bias:.4f}')
print(f'  alpha_raw[0]={alpha_raw[0]:.4f}, beta_raw[0]={beta_raw[0]:.4f}')

dt_val  = np.log1p(np.exp(np.clip(alpha_raw[0] + dt_bias, -20, 20)))   # softplus
decay   = np.exp(np.clip(-np.exp(A_val) * dt_val, -20, 0))
beta_g  = 1.0 / (1.0 + np.exp(-np.clip(beta_raw[0], -20, 20)))         # sigmoid
B_bar   = dt_val * beta_g
print(f'  dt={dt_val:.4f}, decay={decay:.4f}, beta_g={beta_g:.4f}, B_bar={B_bar:.4f}')

# Run actual SSM layer
h_out = engine._forward_ssm_layer(x, L0)
print(f'\nSSM layer output: shape={h_out.shape}, range=[{h_out.min():.3f}, {h_out.max():.3f}]')
print(f'State after: sum={np.abs(engine._ssm_states[0].h).sum():.4f}')
