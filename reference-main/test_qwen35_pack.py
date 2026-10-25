#!/usr/bin/env python3
"""Test loading qwen35-opus-distil pack and config."""
import sys
import json
import os
import struct
from pathlib import Path

sys.path.insert(0, 'quantonium_os_src/engine')
from rftmw_numpy_engine import _parse_gguf_index, _read_gguf_metadata, Qwen35MoeConfig

# Load the pack and sidecar
pack_path = Path('qwen35-opus-distil.rftmwpk')
sidecar_path = pack_path.with_suffix('.rftmwpk.meta.json')

print('=== Pack Header ===')
with open(pack_path, 'rb') as f:
    magic = f.read(8)
    print(f'Magic: {magic}')
    hdr_len = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(hdr_len))
    print(f'Weights count: {len(hdr["weights"])}')

print('\n=== Sidecar Config ===')    
meta = json.loads(sidecar_path.read_text())
cfg = meta.get('model_config', {})
for k, v in cfg.items():
    if not k.startswith('rope'):
        print(f'  {k}: {v}')

print('\n=== GGUF Direct Load Test ===')
gguf_path = meta.get('provenance', {}).get('gguf_path', '')
if os.path.exists(gguf_path):
    cfg2 = Qwen35MoeConfig.from_gguf(gguf_path)
    if cfg2:
        print(f'  Detected: {cfg2.num_layers} layers, {cfg2.num_experts} experts, ssm_state={cfg2.ssm_state_size}')
        print(f'  Full attention interval: {cfg2.full_attention_interval}')
        print(f'  SSM inner size: {cfg2.ssm_inner_size}')
    else:
        print('  FAILED to detect')
    
    # Load tensor index
    idx, data_start = _parse_gguf_index(gguf_path)
    print(f'\n=== Tensor Index ===')
    print(f'  Total tensors: {len(idx)}')
    print(f'  Data starts at: {data_start} bytes')
    
    # Check a few key tensors
    print('\n=== Sample Tensors ===')
    for name in ['token_embd.weight', 'blk.0.ffn_gate_inp.weight', 'blk.0.ssm_a', 'output.weight']:
        if name in idx:
            offset, num_bytes, ttype, shape = idx[name]
            print(f'  {name}: shape={list(shape)}, bytes={num_bytes}, type={ttype}')
else:
    print(f'  GGUF not found: {gguf_path}')

print('\n=== Test Complete ===')
