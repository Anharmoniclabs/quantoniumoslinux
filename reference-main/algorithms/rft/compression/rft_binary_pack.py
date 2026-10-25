# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Binary pack format for RFT tensor containers.

Format (.rftb):
  Header (32 bytes):
    magic:       4 bytes  b'RFTB'
    version:     2 bytes  uint16 = 1
    flags:       2 bytes  bit0=lossy, bit1=assembly
    dtype_len:   1 byte   length of dtype string
    dtype:       up to 16 bytes (padded to 16)
    ndim:        1 byte
    shape:       ndim * 4 bytes (uint32 each, max 8 dims)
    chunk_size:  4 bytes  uint32
    n_chunks:    4 bytes  uint32
    total_len:   8 bytes  uint64

  Per chunk (variable):
    offset:        8 bytes  uint64
    seg_len:       4 bytes  uint32
    rft_size:      4 bytes  uint32
    backend:       1 byte   0=python, 1=assembly
    codec_mode:    1 byte   0=lossless, 1=pruned, 2=quantized
    kept_count:    4 bytes  uint32
    amp_bits:      1 byte   0 = float64
    phase_bits:    1 byte   0 = float64
    amp_max:       8 bytes  float64  (max amplitude, for dequant)
    mask_bytes:    4 bytes  uint32   (0 if keep_all)
    mask_data:     mask_bytes bytes   (packbits)
    amp_nbytes:    4 bytes  uint32
    amp_data:      amp_nbytes bytes   (raw uint8/16/32 or float64)
    phase_nbytes:  4 bytes  uint32
    phase_data:    phase_nbytes bytes (raw uint8/16/32 or float64)
"""
from __future__ import annotations

import base64
import io
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

MAGIC = b'RFTB'
VERSION = 1
_DTYPE_PAD = 16
_MAX_DIMS = 8

# codec mode byte
_MODE_LOSSLESS  = 0
_MODE_PRUNED    = 1
_MODE_QUANTIZED = 2
_MODE_MAP = {'lossless': _MODE_LOSSLESS, 'pruned': _MODE_PRUNED, 'quantized': _MODE_QUANTIZED}
_MODE_INV = {v: k for k, v in _MODE_MAP.items()}


def _uint_dtype(bits: int) -> np.dtype:
    if bits <= 8:  return np.dtype('uint8')
    if bits <= 16: return np.dtype('uint16')
    if bits <= 32: return np.dtype('uint32')
    return np.dtype('uint64')


def pack_container(container: Dict[str, Any]) -> bytes:
    """Serialize an rft_vertex_tensor_container dict to compact binary bytes."""
    dtype_str = container['dtype']
    shape = tuple(container['original_shape'])
    ndim = len(shape)
    if ndim > _MAX_DIMS:
        raise ValueError(f"Too many dims: {ndim}")
    chunk_size = int(container['chunk_size'])
    chunks: List[Dict] = container['chunks']
    n_chunks = len(chunks)
    total_len = int(container['total_length'])
    backend_str = container.get('backend', 'python')
    codec = container.get('codec', {})
    lossy = bool(codec.get('mode', 'lossless') != 'lossless')
    flags = (int(lossy)) | (int('assembly' in backend_str) << 1)

    buf = io.BytesIO()

    # --- header ---
    buf.write(MAGIC)
    buf.write(struct.pack('<H', VERSION))
    buf.write(struct.pack('<H', flags))
    dtype_bytes = dtype_str.encode('ascii')
    buf.write(struct.pack('B', len(dtype_bytes)))
    buf.write(dtype_bytes.ljust(_DTYPE_PAD, b'\x00'))
    buf.write(struct.pack('B', ndim))
    for d in shape:
        buf.write(struct.pack('<I', int(d)))
    for _ in range(_MAX_DIMS - ndim):  # pad shape to MAX_DIMS
        buf.write(struct.pack('<I', 0))
    buf.write(struct.pack('<I', chunk_size))
    buf.write(struct.pack('<I', n_chunks))
    buf.write(struct.pack('<Q', total_len))

    # --- per-chunk ---
    for chunk in chunks:
        offset = int(chunk['offset'])
        seg_len = int(chunk['length'])
        rft_size = int(chunk['rft_size'])
        ch_backend = 1 if chunk.get('backend', 'python') == 'assembly' else 0
        codec_info = chunk.get('codec') or {}
        mode_str = codec_info.get('mode', 'lossless')
        # lossless chunks may still have 'vertices' key
        if 'vertices' in chunk:
            mode_str = 'lossless'
        mode_byte = _MODE_MAP.get(mode_str, _MODE_LOSSLESS)

        if mode_str == 'lossless':
            vertices = chunk.get('vertices', [])
            kept_count = len(vertices)
            amp_bits = 0
            phase_bits = 0
            amp_max = 0.0
            # build arrays from vertex dicts
            if kept_count:
                amp_arr = np.array([v['A'] for v in vertices], dtype=np.float64)
                phase_arr = np.array([v['phi'] for v in vertices], dtype=np.float64)
                idx_arr = np.array([v['idx'] for v in vertices], dtype=np.uint32)
            else:
                amp_arr = np.empty(0, dtype=np.float64)
                phase_arr = np.empty(0, dtype=np.float64)
                idx_arr = np.empty(0, dtype=np.uint32)
            mask_data = b''  # no mask for lossless — indices stored instead
        else:
            amp_info = codec_info.get('amplitude', {}) or {}
            phase_info = codec_info.get('phase', {}) or {}
            amp_bits = int(amp_info.get('bits') or 0)
            phase_bits = int(phase_info.get('bits') or 0)
            amp_max = float(amp_info.get('max_amplitude') or 0.0)

            # decode mask from existing packbits payload
            mask_payload = codec_info.get('mask')
            if mask_payload:
                raw_mask = base64.b64decode(mask_payload['data'])
                mask_data = raw_mask
                mask_length = int(codec_info.get('mask_length', rft_size))
                kept_indices = np.nonzero(
                    np.unpackbits(np.frombuffer(raw_mask, dtype=np.uint8), count=mask_length).astype(bool)
                )[0]
            else:
                mask_data = b''
                kept_indices = np.arange(rft_size, dtype=np.uint32)

            kept_count = len(kept_indices)

            # decode amplitude payload
            amp_payload = amp_info.get('payload', {})
            if amp_payload and amp_payload.get('encoding') == 'raw':
                amp_raw = base64.b64decode(amp_payload['data'])
                amp_dtype = _uint_dtype(amp_bits) if amp_bits else np.dtype('float64')
                amp_arr = np.frombuffer(amp_raw, dtype=amp_dtype).copy()
            else:
                amp_arr = np.empty(0, dtype=np.float64)

            phase_payload = phase_info.get('payload', {})
            if phase_payload and phase_payload.get('encoding') == 'raw':
                phase_raw = base64.b64decode(phase_payload['data'])
                phase_dtype = _uint_dtype(phase_bits) if phase_bits else np.dtype('float64')
                phase_arr = np.frombuffer(phase_raw, dtype=phase_dtype).copy()
            else:
                phase_arr = np.empty(0, dtype=np.float64)

            idx_arr = kept_indices.astype(np.uint32)

        buf.write(struct.pack('<Q', offset))
        buf.write(struct.pack('<I', seg_len))
        buf.write(struct.pack('<I', rft_size))
        buf.write(struct.pack('B', ch_backend))
        buf.write(struct.pack('B', mode_byte))
        buf.write(struct.pack('<I', kept_count))
        buf.write(struct.pack('B', amp_bits))
        buf.write(struct.pack('B', phase_bits))
        buf.write(struct.pack('<d', amp_max))

        buf.write(struct.pack('<I', len(mask_data)))
        if mask_data:
            buf.write(mask_data)

        # write idx array (uint32 per kept coeff)
        idx_bytes = idx_arr.astype(np.uint32).tobytes()
        buf.write(struct.pack('<I', len(idx_bytes)))
        buf.write(idx_bytes)

        amp_bytes = amp_arr.tobytes()
        buf.write(struct.pack('<I', len(amp_bytes)))
        buf.write(amp_bytes)

        phase_bytes = phase_arr.tobytes()
        buf.write(struct.pack('<I', len(phase_bytes)))
        buf.write(phase_bytes)

    return buf.getvalue()


def unpack_container(data: bytes) -> Dict[str, Any]:
    """Deserialize binary .rftb bytes back to an rft_vertex_tensor_container dict."""
    buf = io.BytesIO(data)

    magic = buf.read(4)
    if magic != MAGIC:
        raise ValueError(f"Bad magic: {magic}")
    version, = struct.unpack('<H', buf.read(2))
    flags, = struct.unpack('<H', buf.read(2))
    lossy = bool(flags & 1)
    dtype_len, = struct.unpack('B', buf.read(1))
    dtype_str = buf.read(_DTYPE_PAD)[:dtype_len].decode('ascii')
    ndim, = struct.unpack('B', buf.read(1))
    shape_raw = struct.unpack(f'<{_MAX_DIMS}I', buf.read(_MAX_DIMS * 4))
    shape = tuple(shape_raw[:ndim])
    chunk_size, n_chunks, total_len = struct.unpack('<IIQ', buf.read(16))

    chunks = []
    for _ in range(n_chunks):
        offset, = struct.unpack('<Q', buf.read(8))
        seg_len, = struct.unpack('<I', buf.read(4))
        rft_size, = struct.unpack('<I', buf.read(4))
        ch_backend_byte, = struct.unpack('B', buf.read(1))
        mode_byte, = struct.unpack('B', buf.read(1))
        kept_count, = struct.unpack('<I', buf.read(4))
        amp_bits, = struct.unpack('B', buf.read(1))
        phase_bits, = struct.unpack('B', buf.read(1))
        amp_max, = struct.unpack('<d', buf.read(8))

        mask_nbytes, = struct.unpack('<I', buf.read(4))
        mask_data = buf.read(mask_nbytes) if mask_nbytes else b''

        idx_nbytes, = struct.unpack('<I', buf.read(4))
        idx_bytes = buf.read(idx_nbytes)
        idx_arr = np.frombuffer(idx_bytes, dtype=np.uint32).copy() if idx_bytes else np.empty(0, dtype=np.uint32)

        amp_nbytes, = struct.unpack('<I', buf.read(4))
        amp_raw = buf.read(amp_nbytes)
        phase_nbytes, = struct.unpack('<I', buf.read(4))
        phase_raw = buf.read(phase_nbytes)

        backend_str = 'assembly' if ch_backend_byte else 'python'
        mode_str = _MODE_INV.get(mode_byte, 'lossless')

        if mode_str == 'lossless':
            amp_dtype = np.float64
            phase_dtype = np.float64
            amp_arr = np.frombuffer(amp_raw, dtype=amp_dtype).copy() if amp_raw else np.empty(0, dtype=amp_dtype)
            phase_arr = np.frombuffer(phase_raw, dtype=phase_dtype).copy() if phase_raw else np.empty(0, dtype=phase_dtype)
            vertices = [
                {'idx': int(idx_arr[i]), 'real': float(amp_arr[i] * np.cos(phase_arr[i])),
                 'imag': float(amp_arr[i] * np.sin(phase_arr[i])),
                 'A': float(amp_arr[i]), 'phi': float(phase_arr[i])}
                for i in range(kept_count)
            ]
            chunk = {
                'chunk_index': len(chunks), 'offset': offset, 'length': seg_len,
                'rft_size': rft_size, 'backend': backend_str,
                'codec': {'mode': 'lossless'}, 'vertices': vertices,
            }
        else:
            amp_dtype = _uint_dtype(amp_bits) if amp_bits else np.dtype('float64')
            phase_dtype = _uint_dtype(phase_bits) if phase_bits else np.dtype('float64')
            amp_arr = np.frombuffer(amp_raw, dtype=amp_dtype).copy() if amp_raw else np.empty(0, dtype=amp_dtype)
            phase_arr = np.frombuffer(phase_raw, dtype=phase_dtype).copy() if phase_raw else np.empty(0, dtype=phase_dtype)

            levels_amp = (1 << amp_bits) - 1 if amp_bits else None
            levels_phase = (1 << phase_bits) - 1 if phase_bits else None

            amp_payload = {
                'encoding': 'raw',
                'dtype': str(amp_dtype),
                'shape': [len(amp_arr)],
                'data': base64.b64encode(amp_arr.tobytes()).decode('ascii'),
            }
            phase_payload = {
                'encoding': 'raw',
                'dtype': str(phase_dtype),
                'shape': [len(phase_arr)],
                'data': base64.b64encode(phase_arr.tobytes()).decode('ascii'),
            }

            if mask_data:
                mask_payload = {
                    'encoding': 'packbits',
                    'length': rft_size,
                    'data': base64.b64encode(mask_data).decode('ascii'),
                }
            else:
                mask_payload = None

            chunk = {
                'chunk_index': len(chunks), 'offset': offset, 'length': seg_len,
                'rft_size': rft_size, 'backend': backend_str,
                'codec': {
                    'mode': mode_str,
                    'mask': mask_payload,
                    'mask_length': rft_size,
                    'kept_count': kept_count,
                    'amplitude': {
                        'bits': amp_bits or None,
                        'levels': levels_amp,
                        'max_amplitude': amp_max,
                        'payload': amp_payload,
                    },
                    'phase': {
                        'bits': phase_bits or None,
                        'levels': levels_phase,
                        'payload': phase_payload,
                    },
                },
            }
        chunks.append(chunk)

    return {
        'type': 'rft_vertex_tensor_container',
        'version': version,
        'dtype': dtype_str,
        'original_shape': list(shape),
        'total_length': total_len,
        'chunk_size': chunk_size,
        'chunks': chunks,
        'backend': 'assembly' if (flags & 2) else 'python',
        'codec': {'mode': 'lossy' if lossy else 'lossless'},
    }


def write_rftb(path: Path, container: Dict[str, Any]) -> int:
    """Pack container to binary and write to path. Returns bytes written."""
    data = pack_container(container)
    Path(path).write_bytes(data)
    return len(data)


def read_rftb(path: Path) -> Dict[str, Any]:
    """Read a .rftb file and return the container dict."""
    return unpack_container(Path(path).read_bytes())
