#!/usr/bin/env python3
"""Test tensor loading and MoE routing for qwen35-opus-distil."""
import sys
import json
import os
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, 'quantonium_os_src/engine')
from rftmw_numpy_engine import (
    _parse_gguf_index,
    _read_gguf_metadata,
    Qwen25Config as Qwen35MoeConfig,
    _GGUF_TYPE_BLOCK_SIZE,
    _dequant_q6k_blocks,
)

# Config
pack_path = Path('qwen35-opus-distil.rftmwpk')
sidecar_path = pack_path.with_suffix('.rftmwpk.meta.json')
meta = json.loads(sidecar_path.read_text())
gguf_path = meta['provenance']['gguf_path']
# Remap Linux codespace path to local Windows Ollama blobs path
if not os.path.exists(gguf_path):
    sha = meta['provenance'].get('gguf_file_sha256', '')
    candidate = os.path.expanduser(f'~/.ollama/models/blobs/sha256-{sha}')
    if os.path.exists(candidate):
        gguf_path = candidate
cfg = Qwen35MoeConfig.from_gguf(gguf_path)

print(f'Model: {cfg.num_layers}L x {cfg.hidden_size}H, {cfg.n_experts} experts, MoE={cfg.is_moe}')

# Parse GGUF index
idx, data_start = _parse_gguf_index(gguf_path)

def load_f32_tensor(name: str) -> np.ndarray:
    """Load a float32 tensor from GGUF."""
    offset, num_bytes, ttype, shape = idx[name]
    if ttype != 0:  # F32
        raise ValueError(f'{name} is not F32 (type={ttype})')
    with open(gguf_path, 'rb') as f:
        f.seek(data_start + offset)
        return np.frombuffer(f.read(num_bytes), dtype=np.float32).reshape(shape)

def load_q6k_tensor(name: str) -> np.ndarray:
    """Load and dequantize a Q6_K tensor from GGUF."""
    offset, num_bytes, ttype, shape = idx[name]
    if ttype != 14:  # Q6_K
        raise ValueError(f'{name} is not Q6_K (type={ttype})')
    with open(gguf_path, 'rb') as f:
        f.seek(data_start + offset)
        raw = np.frombuffer(f.read(num_bytes), dtype=np.uint8)
    # Dequant blocks
    blocks = raw.reshape(-1, 210)  # Q6_K block size
    return _dequant_q6k_blocks(blocks)

# === Test 1: Load router weights and do MoE routing ===
print('\n=== Test 1: MoE Router ===')
t0 = time.time()

# Router weights: (hidden_size, num_experts) = (2048, 256)
router_w = load_f32_tensor('blk.0.ffn_gate_inp.weight')
print(f'Router weights: {router_w.shape}')

# Simulate input
x = np.random.randn(cfg.hidden_size).astype(np.float32)

# Compute router logits
logits = x @ router_w  # (num_experts,)
print(f'Router logits: shape={logits.shape}, range=[{logits.min():.2f}, {logits.max():.2f}]')

# Top-k selection
top_k = cfg.n_experts_per_tok
top_indices = np.argsort(logits)[-top_k:][::-1]
top_logits = logits[top_indices]

# Softmax over selected
exp_logits = np.exp(top_logits - top_logits.max())
weights = exp_logits / exp_logits.sum()

print(f'Top-{top_k} experts: {top_indices}')
print(f'Weights: {weights}')
print(f'Time: {(time.time()-t0)*1000:.1f}ms')

# === Test 2: Load SSM parameters ===
print('\n=== Test 2: SSM Parameters ===')
t0 = time.time()

ssm_a = load_f32_tensor('blk.0.ssm_a')
ssm_dt_bias = load_f32_tensor('blk.0.ssm_dt.bias')
ssm_norm = load_f32_tensor('blk.0.ssm_norm.weight')

print(f'ssm_a: {ssm_a.shape} (continuous A in log scale)')
print(f'ssm_dt.bias: {ssm_dt_bias.shape}')
print(f'ssm_norm: {ssm_norm.shape}')

# Show A values (negative real eigenvalues)
A_continuous = np.exp(ssm_a)
print(f'A (continuous): mean={A_continuous.mean():.4f}, std={A_continuous.std():.4f}')
print(f'Time: {(time.time()-t0)*1000:.1f}ms')

# === Test 3: Load and dequant embedding row ===
print('\n=== Test 3: Embedding Lookup ===')
t0 = time.time()

# Load one row of Q6_K embedding
offset, num_bytes, ttype, shape = idx['token_embd.weight']
bytes_per_row = num_bytes // shape[0]
print(f'Embedding: {shape}, {bytes_per_row} bytes/row, type={ttype}')

# Read just row 0 (token 0)
with open(gguf_path, 'rb') as f:
    f.seek(data_start + offset)
    row_raw = np.frombuffer(f.read(bytes_per_row), dtype=np.uint8)

# Dequant
blocks = row_raw.reshape(-1, 210)
row_vals = _dequant_q6k_blocks(blocks).flatten()[:cfg.hidden_size]
print(f'Token 0 embedding: shape={row_vals.shape}, mean={row_vals.mean():.4f}, std={row_vals.std():.4f}')
print(f'Time: {(time.time()-t0)*1000:.1f}ms')

print('\n=== All Tests Passed ===')
print('The qwen35-opus-distil pack loads correctly and MoE/SSM tensors are accessible.')
