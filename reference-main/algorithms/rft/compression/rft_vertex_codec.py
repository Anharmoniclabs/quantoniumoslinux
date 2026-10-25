#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
RFT Vertex Codec — see module docstring for full details.
"""

from __future__ import annotations

import hashlib
import math
import base64
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .ans import RANS_PRECISION_DEFAULT, ans_decode, ans_encode

from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as rft_forward, rft_inverse_square as rft_inverse

try:  # pragma: no cover - optional native backend
    # FIX: on Windows, ensure System32 is on the DLL search path so that
    # rftmw_native.cp312-win_amd64.pyd can resolve MSVCP140/VCRUNTIME140.
    if sys.platform == 'win32':
        _system32 = os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32')
        if hasattr(os, 'add_dll_directory') and os.path.isdir(_system32):
            try:
                os.add_dll_directory(_system32)
            except OSError:
                pass
    import rftmw_native as _rftmw_native
    class UnitaryRFT:
        def __init__(self, size: int, seed: int = 0):
            self._engine = _rftmw_native.RFTMWEngine(size) if hasattr(_rftmw_native, 'RFTMWEngine') else None
            self._size = size

        def forward(self, arr):
            if self._engine is None:
                return None
            return self._engine.forward(arr.tolist())

        def inverse(self, coeffs):
            if self._engine is None:
                return None
            return self._engine.inverse(list(coeffs))
except Exception:  # pragma: no cover
    UnitaryRFT = None  # type: ignore

ASSEMBLY_AVAILABLE = UnitaryRFT is not None
ASSEMBLY_ENABLED = ASSEMBLY_AVAILABLE


@dataclass
class RFTVertex:
    idx: int
    real: float
    imag: float
    A: float
    phi: float

    def to_dict(self) -> Dict[str, float]:
        return {
            'idx': self.idx,
            'real': self.real,
            'imag': self.imag,
            'A': self.A,
            'phi': self.phi,
        }


DEFAULT_MAX_CHUNK = 4096


def _sha256_bytes(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _quantized_checksum(flat: np.ndarray, tolerance: float) -> str:
    if tolerance <= 0:
        raise ValueError("tolerance must be positive for quantized checksums")
    scale = 1.0 / tolerance
    if not np.isfinite(scale):
        raise ValueError("Invalid tolerance leading to non-finite scale")
    try:
        clip_min, clip_max = np.iinfo(np.int64).min, np.iinfo(np.int64).max
        quantized_int = np.rint(np.clip(flat * scale, clip_min, clip_max)).astype(np.int64)
        return hashlib.sha256(quantized_int.tobytes()).hexdigest()
    except (OverflowError, ValueError):
        decimals = max(0, int(round(-math.log10(tolerance))))
        quantized = np.round(flat, decimals=decimals).astype(np.float64, copy=False)
        return _sha256_bytes(quantized)


def enable_assembly_rft(enable: bool) -> bool:
    global ASSEMBLY_ENABLED
    ASSEMBLY_ENABLED = bool(enable) and ASSEMBLY_AVAILABLE
    if enable and not ASSEMBLY_ENABLED:
        warnings.warn(
            "Assembly RFT requested but rftmw_native module is unavailable; continuing with Python implementation.",
            RuntimeWarning,
            stacklevel=2,
        )
    return ASSEMBLY_ENABLED


def is_assembly_enabled() -> bool:
    return ASSEMBLY_ENABLED


def _generate_seed(size: int) -> int:
    phi = (1 + math.sqrt(5.0)) / 2.0
    return int((size * phi) % (1 << 63))


def _python_forward(segment: np.ndarray, size: int) -> np.ndarray:
    return rft_forward(segment)


def _python_inverse(coeffs: np.ndarray, size: int) -> np.ndarray:
    return rft_inverse(coeffs)


def _get_assembly_engine(size: int, seed: Optional[int]) -> Optional[Any]:
    if not ASSEMBLY_ENABLED or UnitaryRFT is None:
        return None
    try:
        engine = UnitaryRFT(size, seed=seed if seed is not None else _generate_seed(size))
        return engine
    except Exception as exc:
        warnings.warn(
            f"Assembly RFTMWEngine initialisation failed (fallback to Python): {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def _assembly_forward(segment: np.ndarray, size: int, seed: int) -> Optional[np.ndarray]:
    engine = _get_assembly_engine(size, seed)
    if engine is None:
        return None
    try:
        result = engine.forward(segment.astype(np.float64))
        if result is None:
            return None
        return np.asarray(result, dtype=np.complex128)
    except Exception:
        return None


def _assembly_inverse(coeffs: np.ndarray, size: int, seed: int) -> Optional[np.ndarray]:
    engine = _get_assembly_engine(size, seed)
    if engine is None:
        return None
    try:
        result = engine.inverse(coeffs)
        if result is None:
            return None
        return np.asarray(result, dtype=np.float64)
    except Exception:
        return None


def _select_uint_dtype(bits: int) -> np.dtype:
    if bits <= 0:
        raise ValueError("bits must be positive for quantized payloads")
    if bits <= 8:
        return np.uint8
    if bits <= 16:
        return np.uint16
    if bits <= 32:
        return np.uint32
    if bits <= 64:
        return np.uint64
    raise ValueError("Quantization with more than 64 bits is not supported")


def _pack_bool_mask(mask: np.ndarray) -> Dict[str, Any]:
    if mask.dtype != np.bool_:
        mask = mask.astype(np.bool_, copy=False)
    packed = np.packbits(mask.astype(np.uint8, copy=False))
    return {
        'encoding': 'packbits',
        'length': int(mask.size),
        'data': base64.b64encode(packed.tobytes()).decode('ascii'),
    }


def _unpack_bool_mask(payload: Optional[Dict[str, Any]], expected_length: int) -> np.ndarray:
    if not payload:
        return np.ones(expected_length, dtype=bool)
    if payload.get('encoding') != 'packbits':
        raise ValueError('Unsupported mask encoding')
    data = base64.b64decode(payload['data'])
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr, count=expected_length)
    return bits.astype(bool)


def _serialize_numeric_array(arr: np.ndarray) -> Dict[str, Any]:
    return {
        'encoding': 'raw',
        'dtype': str(arr.dtype),
        'shape': list(arr.shape),
        'data': base64.b64encode(arr.tobytes()).decode('ascii'),
    }


def _deserialize_numeric_array(payload: Dict[str, Any]) -> np.ndarray:
    dtype = np.dtype(payload['dtype'])
    data = base64.b64decode(payload['data'])
    arr = np.frombuffer(data, dtype=dtype)
    shape = payload.get('shape')
    if shape:
        arr = arr.reshape(tuple(int(x) for x in shape))
    return arr


def _decode_numeric_payload(payload: Optional[Dict[str, Any]], dtype: np.dtype) -> np.ndarray:
    if not payload or payload.get('encoding') == 'empty':
        return np.empty(0, dtype=dtype)
    encoding = payload.get('encoding')
    if encoding == 'raw':
        arr = _deserialize_numeric_array(payload)
        return arr.astype(dtype, copy=False)
    if encoding == 'ans':
        if ans_decode is None:
            raise ImportError("ANS decoder not available")
        data = base64.b64decode(payload['data'])
        freq_data = payload.get('freq_data')
        num_symbols = payload.get('num_symbols')
        if freq_data is None or num_symbols is None:
            raise ValueError("ANS payload missing frequency data or symbol count")
        symbols = ans_decode(data, freq_data, num_symbols)
        return np.asarray(symbols, dtype=dtype)
    raise ValueError(f"Unsupported numeric payload encoding '{encoding}'")


def _make_numeric_payload(
    array: np.ndarray,
    *,
    bits: Optional[int],
    ans_precision: Optional[int],
    alphabet_size: Optional[int],
) -> Dict[str, Any]:
    if array.size == 0:
        return {'encoding': 'empty'}
    if bits and bits > 0 and ans_precision and ans_precision > 0 and alphabet_size and ans_encode is not None:
        if alphabet_size > (1 << ans_precision):
            warnings.warn(
                f"Alphabet size {alphabet_size} exceeds rANS capacity for precision {ans_precision}; falling back to raw payload",
                RuntimeWarning,
                stacklevel=3,
            )
        else:
            symbols = array.astype(np.int64, copy=False).tolist()
            try:
                encoded, freq_data = ans_encode(symbols, precision=ans_precision)
                return {
                    'encoding': 'ans',
                    'dtype': str(array.dtype),
                    'shape': list(array.shape),
                    'data': base64.b64encode(encoded.tobytes()).decode('ascii'),
                    'freq_data': freq_data,
                    'num_symbols': len(symbols)
                }
            except Exception as e:
                warnings.warn(
                    f"ANS encoding failed ({e}); falling back to raw payload",
                    RuntimeWarning,
                    stacklevel=3,
                )
    return _serialize_numeric_array(array)


def _quantize_amplitudes(values: np.ndarray, bits: Optional[int]) -> Tuple[np.ndarray, Dict[str, Any]]:
    amplitudes = values.astype(np.float64, copy=False)
    meta: Dict[str, Any] = {
        'bits': int(bits) if bits and bits > 0 else None,
        'max_amplitude': float(np.max(amplitudes)) if amplitudes.size else 0.0,
    }
    if meta['bits'] is None or meta['max_amplitude'] <= 0.0:
        return amplitudes, meta
    levels = (1 << meta['bits']) - 1
    meta['levels'] = int(levels)
    dtype = _select_uint_dtype(meta['bits'])
    normalized = amplitudes / meta['max_amplitude']
    quantized = np.clip(np.rint(normalized * levels), 0, levels).astype(dtype)
    return quantized, meta


def _dequantize_amplitudes(data: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    bits = meta.get('bits')
    if not bits:
        return data.astype(np.float64, copy=False)
    levels = meta.get('levels')
    max_amp = meta.get('max_amplitude', 0.0)
    if levels is None or levels <= 0 or max_amp <= 0.0:
        return np.zeros_like(data, dtype=np.float64)
    return (data.astype(np.float64) / float(levels)) * float(max_amp)


def _quantize_phases(values: np.ndarray, bits: Optional[int]) -> Tuple[np.ndarray, Dict[str, Any]]:
    phases = values.astype(np.float64, copy=False)
    meta: Dict[str, Any] = {'bits': int(bits) if bits and bits > 0 else None}
    if meta['bits'] is None:
        return phases, meta
    levels = (1 << meta['bits']) - 1
    meta['levels'] = int(levels)
    dtype = _select_uint_dtype(meta['bits'])
    shifted = (phases + np.pi) / (2 * np.pi)
    quantized = np.clip(np.rint(shifted * levels), 0, levels).astype(dtype)
    return quantized, meta


def _dequantize_phases(data: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    bits = meta.get('bits')
    if not bits:
        return data.astype(np.float64, copy=False)
    levels = meta.get('levels')
    if levels is None or levels <= 0:
        return np.zeros_like(data, dtype=np.float64)
    normalized = data.astype(np.float64) / float(levels)
    phases = normalized * (2 * np.pi) - np.pi
    return np.clip(phases, -np.pi, np.pi)


def _encode_chunk_lossless(
    chunk_index: int, offset: int, seg_len: int,
    coeffs: np.ndarray, backend: str, seed: Optional[int],
) -> Dict[str, Any]:
    vertices: List[Dict[str, float]] = []
    for i, c in enumerate(coeffs):
        A = float(abs(c))
        phi = float(math.atan2(c.imag, c.real))
        v = RFTVertex(idx=i, real=float(c.real), imag=float(c.imag), A=A, phi=phi)
        vertices.append(v.to_dict())
    return {
        'chunk_index': int(chunk_index), 'offset': int(offset),
        'length': int(seg_len), 'rft_size': int(len(coeffs)),
        'vertices': vertices, 'backend': backend,
        'seed': int(seed) if (seed is not None and backend == 'assembly') else None,
        'codec': {'mode': 'lossless'},
    }


def _encode_chunk_lossy(
    chunk_index: int, offset: int, seg_len: int, coeffs: np.ndarray,
    backend: str, seed: Optional[int], prune_threshold: Optional[float],
    quant_bits_amplitude: Optional[int], quant_bits_phase: Optional[int],
    ans_precision: Optional[int],
) -> Optional[Dict[str, Any]]:
    rft_size = len(coeffs)
    if prune_threshold is not None and prune_threshold < 0:
        raise ValueError("prune_threshold must be non-negative")
    if ans_precision is not None and ans_precision <= 0:
        ans_precision = None

    keep_mask = np.ones(rft_size, dtype=bool)
    if prune_threshold is not None and prune_threshold > 0:
        keep_mask = np.abs(coeffs) >= prune_threshold

    kept_indices = np.nonzero(keep_mask)[0]
    kept_count = int(kept_indices.size)

    amplitudes = np.abs(coeffs[kept_indices]) if kept_count else np.empty(0, dtype=np.float64)
    phases = np.angle(coeffs[kept_indices]) if kept_count else np.empty(0, dtype=np.float64)

    amp_data, amp_meta = _quantize_amplitudes(amplitudes, quant_bits_amplitude)
    phase_data, phase_meta = _quantize_phases(phases, quant_bits_phase)

    quantized = bool(amp_meta.get('bits')) or bool(phase_meta.get('bits'))
    alphabet_amp = (amp_meta.get('levels', 0) + 1) if amp_meta.get('bits') else None
    alphabet_phase = (phase_meta.get('levels', 0) + 1) if phase_meta.get('bits') else None

    amp_payload = _make_numeric_payload(np.asarray(amp_data), bits=amp_meta.get('bits'), ans_precision=ans_precision, alphabet_size=alphabet_amp)
    phase_payload = _make_numeric_payload(np.asarray(phase_data), bits=phase_meta.get('bits'), ans_precision=ans_precision, alphabet_size=alphabet_phase)

    codec_mode = 'quantized' if quantized else 'pruned'
    mask_payload = None if keep_mask.all() else _pack_bool_mask(keep_mask)

    return {
        'chunk_index': int(chunk_index), 'offset': int(offset),
        'length': int(seg_len), 'rft_size': int(rft_size),
        'backend': backend,
        'seed': int(seed) if (seed is not None and backend == 'assembly') else None,
        'codec': {
            'mode': codec_mode, 'mask': mask_payload,
            'mask_length': int(rft_size), 'kept_count': kept_count,
            'prune_threshold': float(prune_threshold) if prune_threshold is not None else None,
            'ans_precision': int(ans_precision) if ans_precision else None,
            'amplitude': {
                'bits': amp_meta.get('bits'), 'levels': amp_meta.get('levels'),
                'max_amplitude': amp_meta.get('max_amplitude'), 'payload': amp_payload,
            },
            'phase': {
                'bits': phase_meta.get('bits'), 'levels': phase_meta.get('levels'),
                'payload': phase_payload,
            },
        },
    }


def _decode_chunk_lossy(chunk: Dict[str, Any], codec: Dict[str, Any]) -> np.ndarray:
    rft_size = int(chunk['rft_size'])
    mask_length = int(codec.get('mask_length', rft_size))
    mask = _unpack_bool_mask(codec.get('mask'), mask_length)
    if mask.size != rft_size:
        raise ValueError('Mask length mismatch with RFT size')
    kept_indices = np.nonzero(mask)[0]

    amp_meta = codec.get('amplitude', {}) or {}
    phase_meta = codec.get('phase', {}) or {}

    amp_bits = amp_meta.get('bits')
    amp_dtype = np.float64 if not amp_bits else _select_uint_dtype(int(amp_bits))
    amp_data = _decode_numeric_payload(amp_meta.get('payload'), np.dtype(amp_dtype))
    amplitudes = _dequantize_amplitudes(amp_data, amp_meta)

    phase_bits = phase_meta.get('bits')
    phase_dtype = np.float64 if not phase_bits else _select_uint_dtype(int(phase_bits))
    phase_data = _decode_numeric_payload(phase_meta.get('payload'), np.dtype(phase_dtype))
    phases = _dequantize_phases(phase_data, phase_meta)

    if amplitudes.size != kept_indices.size or phases.size != kept_indices.size:
        raise ValueError('Amplitude/phase payload sizes do not match kept coefficient count')

    coeffs = np.zeros(rft_size, dtype=np.complex128)
    if kept_indices.size:
        coeffs[kept_indices] = amplitudes * (np.cos(phases) + 1j * np.sin(phases))
    return coeffs


def encode_tensor(
    tensor: np.ndarray,
    chunk_size: Optional[int] = None,
    tolerance: float = 1e-10,
    prune_threshold: Optional[float] = None,
    quant_bits_amplitude: Optional[int] = None,
    quant_bits_phase: Optional[int] = None,
    ans_precision: Optional[int] = None,
    use_assembly: Optional[bool] = None,
    mode: str = "crypto",
    target_bpv: Optional[float] = None,
) -> Dict[str, Any]:
    if use_assembly is not None:
        _prev = ASSEMBLY_ENABLED
        enable_assembly_rft(use_assembly)

    try:
        if not isinstance(tensor, np.ndarray):
            raise TypeError("tensor must be a numpy.ndarray")
        original_shape = tuple(tensor.shape)
        original_dtype = tensor.dtype

        if ans_precision is None:
            ans_precision_effective: Optional[int] = None
        elif ans_precision <= 0:
            ans_precision_effective = RANS_PRECISION_DEFAULT
        else:
            ans_precision_effective = int(ans_precision)

        lossy_requested = any((
            prune_threshold and prune_threshold > 0,
            quant_bits_amplitude and quant_bits_amplitude > 0,
            quant_bits_phase and quant_bits_phase > 0,
            ans_precision_effective is not None,
        ))

        if tensor.dtype.kind in "fi":
            flat = tensor.reshape(-1).astype(np.float64)
            checksum = _sha256_bytes(flat)
            if chunk_size is None:
                chunk_size = min(DEFAULT_MAX_CHUNK, len(flat))
            if chunk_size <= 0:
                raise ValueError("chunk_size must be > 0")

            chunks: List[Dict[str, Any]] = []
            offset = 0
            backend_mode: Optional[str] = None
            while offset < len(flat):
                segment = flat[offset: offset + chunk_size]
                seg_len = len(segment)
                seed = _generate_seed(seg_len)
                coeffs: Optional[np.ndarray] = None
                backend = "python"
                if ASSEMBLY_ENABLED:
                    coeffs = _assembly_forward(segment, seg_len, seed)
                    if coeffs is not None:
                        backend = "assembly"
                if coeffs is None:
                    coeffs = _python_forward(segment, seg_len)
                    backend = "python"

                seed_value = int(seed) if backend == "assembly" else None

                # --- Rate control: compress mode thresholds to top-k ---
                if mode == "compress" and target_bpv is not None:
                    n_c = len(coeffs)
                    bits_per_coeff = math.log2(max(n_c, 2)) + 16  # index + complex
                    k = max(1, int(target_bpv * n_c / bits_per_coeff))
                    k = min(k, n_c)
                    mag = np.abs(coeffs)
                    if k < n_c:
                        topk_idx = np.argsort(mag)[-k:]
                        mask = np.zeros(n_c, dtype=bool)
                        mask[topk_idx] = True
                        coeffs = coeffs * mask  # zero out below top-k
                    # Derive prune_threshold from the minimum kept magnitude
                    kept_mags = mag[mag > 0] if k >= n_c else mag[topk_idx]
                    _rate_prune = float(kept_mags.min()) if kept_mags.size else 0.0
                # -------------------------------------------------------

                chunk_record: Optional[Dict[str, Any]] = None
                if mode == "compress" and target_bpv is not None:
                    # Always use lossy path in compress mode
                    chunk_record = _encode_chunk_lossy(
                        len(chunks), offset, seg_len, coeffs, backend, seed_value,
                        _rate_prune, quant_bits_amplitude, quant_bits_phase, ans_precision_effective,
                    )
                elif lossy_requested:
                    chunk_record = _encode_chunk_lossy(
                        len(chunks), offset, seg_len, coeffs, backend, seed_value,
                        prune_threshold, quant_bits_amplitude, quant_bits_phase, ans_precision_effective,
                    )
                if chunk_record is None:
                    chunk_record = _encode_chunk_lossless(len(chunks), offset, seg_len, coeffs, backend, seed_value)

                chunks.append(chunk_record)
                offset += len(segment)
                if backend_mode is None:
                    backend_mode = backend
                elif backend_mode != backend:
                    backend_mode = "mixed"

            container = {
                'type': 'rft_vertex_tensor_container', 'version': 1,
                'dtype': str(np.dtype(original_dtype)), 'original_shape': original_shape,
                'total_length': int(len(flat)), 'chunk_size': int(chunk_size),
                'checksum': checksum, 'chunks': chunks, 'backend': backend_mode or "python",
            }
            lossy_chunks = any(c.get('codec', {}).get('mode', 'lossless') != 'lossless' for c in chunks)
            codec_summary: Dict[str, Any] = {'mode': mode if mode == 'compress' else ('lossy' if lossy_chunks else 'lossless')}
            if mode == "compress" and target_bpv is not None:
                codec_summary['target_bpv'] = float(target_bpv)
            if prune_threshold and prune_threshold > 0:
                codec_summary['prune_threshold'] = float(prune_threshold)
            if quant_bits_amplitude and quant_bits_amplitude > 0:
                codec_summary['quant_bits_amplitude'] = int(quant_bits_amplitude)
            if quant_bits_phase and quant_bits_phase > 0:
                codec_summary['quant_bits_phase'] = int(quant_bits_phase)
            if ans_precision_effective is not None:
                codec_summary['ans_precision'] = int(ans_precision_effective)
            if lossy_chunks:
                container['lossy'] = True
            container['codec'] = codec_summary
            if tolerance:
                container['tolerance'] = float(tolerance)
                try:
                    container['quantized_checksum'] = _quantized_checksum(flat, tolerance)
                except ValueError:
                    pass
            return container

        raw_bytes = tensor.tobytes(order='C')
        return {
            'type': 'rft_raw_tensor_container', 'version': 1,
            'dtype': str(np.dtype(original_dtype)), 'original_shape': original_shape,
            'total_length': int(tensor.size), 'storage': 'base64',
            'data': base64.b64encode(raw_bytes).decode('ascii'),
        }
    finally:
        if use_assembly is not None:
            enable_assembly_rft(_prev)


def decode_tensor(
    container: Dict[str, Any],
    verify_checksum: bool = False,
    atol: Optional[float] = None,
    tensor_name: Optional[str] = None,
    return_status: bool = False,
) -> Any:
    container_type = container.get('type')
    if container_type == 'rft_raw_tensor_container':
        raw = base64.b64decode(container['data'])
        flat = np.frombuffer(raw, dtype=np.dtype(container['dtype'])).copy()
        result = flat.reshape(tuple(container['original_shape']))
        return (result, {'used_secondary_checksum': False}) if return_status else result

    if container_type != 'rft_vertex_tensor_container':
        raise ValueError('Not an RFT vertex tensor container')

    original_shape = tuple(container['original_shape'])
    total_length = container['total_length']
    flat = np.zeros(total_length, dtype=np.float64)
    container_backend = container.get('backend', 'python')

    for chunk in container['chunks']:
        length = chunk['length']
        rft_size = chunk['rft_size']
        offset = chunk['offset']
        codec_info = chunk.get('codec') if isinstance(chunk.get('codec'), dict) else None
        mode = codec_info.get('mode', 'lossless') if codec_info else 'lossless'
        backend = chunk.get('backend', container_backend)
        seed = chunk.get('seed')

        if mode == 'lossless' or 'vertices' in chunk:
            vertices = chunk.get('vertices')
            if vertices is None:
                raise ValueError('Lossless chunk missing vertex data')
            coeffs = np.zeros(rft_size, dtype=np.complex128)
            for v in vertices:
                coeffs[v['idx']] = v['real'] + 1j * v['imag']
        else:
            coeffs = _decode_chunk_lossy(chunk, codec_info or {})

        if backend == 'assembly':
            if not ASSEMBLY_ENABLED:
                raise RuntimeError("Encoded with assembly backend but assembly is disabled.")
            segment_full = _assembly_inverse(coeffs, rft_size, seed if seed is not None else _generate_seed(rft_size))
            if segment_full is None:
                raise RuntimeError("Assembly inverse failed.")
        else:
            segment_full = _python_inverse(coeffs, rft_size)

        flat[offset: offset + length] = segment_full[:length].real

    secondary_used = False
    if verify_checksum:
        dec_checksum = _sha256_bytes(flat)
        if dec_checksum != container['checksum']:
            tolerance = atol or container.get('tolerance')
            if not tolerance or tolerance <= 0:
                raise ValueError("Checksum mismatch and no tolerance for secondary verification")
            quant_ref = container.get('quantized_checksum')
            if not quant_ref:
                raise ValueError("No quantized checksum for secondary verification")
            if _quantized_checksum(flat, tolerance) != quant_ref:
                raise ValueError("Secondary quantized verification failed")
            secondary_used = True
            warnings.warn(f"Primary checksum mismatch; secondary accepted with tolerance {tolerance}", RuntimeWarning, stacklevel=2)

    target_dtype = np.dtype(container['dtype'])
    if np.issubdtype(target_dtype, np.integer):
        flat = np.rint(flat)
    result = flat.reshape(original_shape).astype(target_dtype, copy=False)
    return (result, {'used_secondary_checksum': secondary_used}) if return_status else result


def roundtrip_tensor(tensor: np.ndarray, atol: float = 1e-10, verify_checksum: bool = False, **encode_kwargs: Any) -> Tuple[bool, float]:
    enc = encode_tensor(tensor, **encode_kwargs)
    dec = decode_tensor(enc, verify_checksum=verify_checksum)
    max_err = float(np.max(np.abs(dec.astype(np.float64) - tensor.astype(np.float64))))
    return max_err <= atol, max_err


def encode_state_dict(state: Dict[str, np.ndarray], max_tensors: Optional[int] = None, tolerance: float = 1e-10, **kwargs) -> Dict[str, Any]:
    encoded = {}
    for name, tensor in list(state.items())[:(max_tensors or len(state))]:
        try:
            import torch as _torch
            if isinstance(tensor, _torch.Tensor):
                tensor = tensor.detach().cpu().numpy()
        except Exception:
            pass
        if not isinstance(tensor, np.ndarray):
            tensor = np.asarray(tensor)
        try:
            encoded[name] = encode_tensor(tensor, tolerance=tolerance, **kwargs)
        except Exception as e:
            encoded[name] = {'error': str(e)}
    result = {'type': 'rft_vertex_state_dict', 'version': 1, 'tensors': encoded}
    if kwargs:
        result['codec'] = {k: v for k, v in kwargs.items() if v is not None}
    return result


def decode_state_dict(encoded: Dict[str, Any]) -> Dict[str, np.ndarray]:
    if encoded.get('type') != 'rft_vertex_state_dict':
        raise ValueError('Not an RFT vertex state dict container')
    return {
        name: decode_tensor(entry)
        for name, entry in encoded['tensors'].items()
        if isinstance(entry, dict) and entry.get('type') in {'rft_vertex_tensor_container', 'rft_raw_tensor_container'}
    }


class RFTVertexCodec:
    def __init__(self, use_assembly: bool = True, **kwargs):
        enable_assembly_rft(use_assembly)
        self.encode_kwargs = kwargs

    def encode(self, tensor: np.ndarray, **kwargs) -> Dict[str, Any]:
        merged = {**self.encode_kwargs, **kwargs}
        return encode_tensor(tensor, **merged)

    def decode(self, container: Dict[str, Any]) -> np.ndarray:
        return decode_tensor(container)


__all__ = [
    'encode_tensor', 'decode_tensor', 'roundtrip_tensor',
    'encode_state_dict', 'decode_state_dict', 'RFTVertex',
    'enable_assembly_rft', 'is_assembly_enabled', 'RFTVertexCodec'
]
