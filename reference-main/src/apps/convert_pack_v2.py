#!/usr/bin/env python3
"""
SPDX-License-Identifier: LicenseRef-SPX-Proprietary
Copyright (C) 2025-2026 Luis M. Minier / quantoniumos

convert_pack_v2.py - Convert .rftmwpk v1 to v2 (mmap-native)
=============================================================

V2 format is designed for instant mmap loading:
- 64-byte aligned header
- Tensor descriptors with offsets
- Contiguous quantized data blocks
- No JSON, no zlib, pure binary

Usage:
    python convert_pack_v2.py input.rftmwpk output.rftmwpk2
"""

import argparse
import json
import struct
import sys
from pathlib import Path

# V2 header format (matches rftmw_mmap.hpp)
PACK_HEADER_V2 = struct.Struct(
    '<'      # Little endian
    '8s'     # magic: "RFTMWPK2"
    'Q'      # version
    'Q'      # header_size
    'Q'      # num_tensors
    'Q'      # model_type
    'Q'      # hidden_size
    'Q'      # num_layers
    'Q'      # num_heads
    'Q'      # num_kv_heads
    'Q'      # intermediate_size
    'Q'      # vocab_size
    'Q'      # rope_theta (fixed-point * 1e6)
    'Q'      # max_position
    '4Q'     # reserved
)

# Tensor descriptor format
TENSOR_DESC = struct.Struct(
    '<'
    '64s'    # name (null-terminated)
    'Q'      # offset
    'Q'      # size_bytes
    'Q'      # rows
    'Q'      # cols
    'I'      # quant_type
    'I'      # flags
)

# Quant types
QUANT_FP32 = 0
QUANT_Q4_K = 1
QUANT_Q6_K = 2
QUANT_FP16 = 3

# Model types
MODEL_QWEN25 = 0
MODEL_QWEN35MOE = 1
MODEL_LLAMA = 2


def detect_quant_type(name: str, data_len: int, shape: tuple) -> int:
    """Detect quantization type from tensor name and size."""
    rows, cols = shape if len(shape) == 2 else (shape[0], 1)
    expected_q4k = ((cols + 255) // 256) * 144 * rows
    expected_q6k = ((cols + 255) // 256) * 210 * rows
    expected_fp32 = rows * cols * 4
    
    if abs(data_len - expected_q4k) < 256:
        return QUANT_Q4_K
    elif abs(data_len - expected_q6k) < 256:
        return QUANT_Q6_K
    elif data_len == expected_fp32:
        return QUANT_FP32
    else:
        # Default to raw bytes
        return QUANT_FP32


def convert_v1_to_v2(input_path: str, output_path: str):
    """Convert v1 pack to v2 mmap-native format."""
    print(f"[convert] Reading v1 pack: {input_path}")
    
    with open(input_path, 'rb') as f:
        # Read v1 header
        magic = f.read(8)
        if magic != b'RFTMWPK1':
            raise ValueError(f"Invalid magic: {magic!r}, expected RFTMWPK1")
        
        # V1 format: 8-byte header size, then JSON header
        header_len = struct.unpack('<Q', f.read(8))[0]
        header_json = f.read(header_len).decode('utf-8')
        header = json.loads(header_json)
        
        config = header.get('config', {})
        tensors_meta = header.get('tensors', [])
        
        print(f"[convert] Model config: {config.get('model_name', 'unknown')}")
        print(f"[convert] Tensors: {len(tensors_meta)}")
        
        # Read all tensor data
        data_start = f.tell()
        tensor_data = []
        for tmeta in tensors_meta:
            size = tmeta['size']
            data = f.read(size)
            tensor_data.append(data)
    
    # Build v2 format
    print(f"[convert] Writing v2 pack: {output_path}")
    
    # Determine model type
    model_name = config.get('model_name', '').lower()
    if 'qwen35' in model_name or 'moe' in model_name:
        model_type = MODEL_QWEN35MOE
    elif 'qwen' in model_name:
        model_type = MODEL_QWEN25
    else:
        model_type = MODEL_LLAMA
    
    # Extract config values
    hidden_size = config.get('hidden_size', 2048)
    num_layers = config.get('num_layers', config.get('n_layers', 36))
    num_heads = config.get('num_heads', config.get('n_heads', 16))
    num_kv_heads = config.get('num_kv_heads', config.get('n_kv_heads', 2))
    intermediate_size = config.get('intermediate_size', 11008)
    vocab_size = config.get('vocab_size', 151936)
    rope_theta = config.get('rope_theta', 1000000.0)
    max_position = config.get('max_position', config.get('max_seq_len', 32768))
    
    num_tensors = len(tensors_meta)
    
    # Calculate header size (must be 64-byte aligned)
    base_header_size = PACK_HEADER_V2.size + num_tensors * TENSOR_DESC.size
    header_size = ((base_header_size + 63) // 64) * 64
    
    # Calculate tensor offsets
    current_offset = header_size
    tensor_descs = []
    for i, tmeta in enumerate(tensors_meta):
        name = tmeta['name']
        shape = tmeta.get('shape', [len(tensor_data[i]) // 4, 1])
        size_bytes = len(tensor_data[i])
        rows = shape[0] if len(shape) >= 1 else 1
        cols = shape[1] if len(shape) >= 2 else 1
        quant_type = detect_quant_type(name, size_bytes, shape)
        
        tensor_descs.append({
            'name': name,
            'offset': current_offset,
            'size_bytes': size_bytes,
            'rows': rows,
            'cols': cols,
            'quant_type': quant_type,
            'flags': 0
        })
        
        # Align next tensor to 64 bytes for cache efficiency
        current_offset += size_bytes
        current_offset = ((current_offset + 63) // 64) * 64
    
    # Write v2 file
    with open(output_path, 'wb') as f:
        # Write header
        header_bytes = PACK_HEADER_V2.pack(
            b'RFTMWPK2',
            2,  # version
            header_size,
            num_tensors,
            model_type,
            hidden_size,
            num_layers,
            num_heads,
            num_kv_heads,
            intermediate_size,
            vocab_size,
            int(rope_theta * 1e6),
            max_position,
            0, 0, 0, 0  # reserved
        )
        f.write(header_bytes)
        
        # Write tensor descriptors
        for desc in tensor_descs:
            name_bytes = desc['name'].encode('utf-8')[:63] + b'\x00' * (64 - min(len(desc['name']), 63))
            desc_bytes = TENSOR_DESC.pack(
                name_bytes[:64],
                desc['offset'],
                desc['size_bytes'],
                desc['rows'],
                desc['cols'],
                desc['quant_type'],
                desc['flags']
            )
            f.write(desc_bytes)
        
        # Pad to header_size
        current_pos = f.tell()
        if current_pos < header_size:
            f.write(b'\x00' * (header_size - current_pos))
        
        # Write tensor data with alignment padding
        for i, data in enumerate(tensor_data):
            # Ensure we're at the right offset
            expected_offset = tensor_descs[i]['offset']
            current_pos = f.tell()
            if current_pos < expected_offset:
                f.write(b'\x00' * (expected_offset - current_pos))
            
            f.write(data)
        
        # Final size
        final_size = f.tell()
    
    print(f"[convert] Done!")
    print(f"[convert] V1 size: {Path(input_path).stat().st_size:,} bytes")
    print(f"[convert] V2 size: {final_size:,} bytes")
    print(f"[convert] Tensors: {num_tensors}")
    print(f"[convert] Ready for mmap instant loading")


def main():
    parser = argparse.ArgumentParser(description='Convert .rftmwpk v1 to v2 (mmap-native)')
    parser.add_argument('input', help='Input v1 pack file')
    parser.add_argument('output', help='Output v2 pack file')
    args = parser.parse_args()
    
    convert_v1_to_v2(args.input, args.output)


if __name__ == '__main__':
    main()
