# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
Unified RFT Compression Entry Point
=====================================

Single public API that sequences all compression decisions:

  DECISION 1  unified_transform_scheduler.py
              Which compute backend executes the transform?
              -> Python (always) | C/ASM (N<=8) | C++ Native (AVX2, if built)

  DECISION 2  algorithms/rft/routing.py
              Which codec variant fits this signal?
              -> CASCADE(8) | ENTROPY_GUIDED(11) | DICTIONARY(12) | ...

  R-D CHECK   algorithms/rft/compression/entropy.py
              Is RFT actually worth it vs INT8+zlib for this tensor?
              Uses calculate_rd_point() fed the *actual* scheduled transform.

  CODEC       algorithms/rft/compression/rft_vertex_codec.py
              algorithms/rft/hybrids/rft_hybrid_codec.py

  SERIALISE   algorithms/rft/compression/rft_binary_pack.py  -> .rftb bytes
  ENTROPY     algorithms/rft/compression/ans.py              -> ANS final pass

Usage::

    import numpy as np
    from quantonium_os_src.engine.rft_compress import compress, decompress

    tensor  = np.random.randn(512, 512).astype(np.float32)
    blob    = compress(tensor, quality="balanced", mse_budget=1e-4)
    recovered = decompress(blob, shape=tensor.shape, dtype=tensor.dtype)

The INT8+zlib fallback path is taken automatically when the R-D check shows
RFT would produce a higher BPP or when the reconstruction error would exceed
`mse_budget`.
"""

from __future__ import annotations

import io
import json
import os
import struct
import sys
import zlib
from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def _import_scheduler():
    from algorithms.rft.unified_transform_scheduler import (
        UnifiedTransformScheduler, Backend
    )
    return UnifiedTransformScheduler, Backend


def _import_routing():
    from algorithms.rft.routing import classify_signal, TransformType
    return classify_signal, TransformType


def _import_entropy():
    from algorithms.rft.compression.entropy import (
        calculate_rd_point, estimate_bitrate, uniform_quantizer
    )
    return calculate_rd_point, estimate_bitrate, uniform_quantizer


def _import_vertex_codec():
    from algorithms.rft.compression.rft_vertex_codec import RFTVertexCodec
    return RFTVertexCodec


def _import_hybrid_codec():
    from algorithms.rft.hybrids.rft_hybrid_codec import RFTHybridCodec
    return RFTHybridCodec


def _import_pack():
    from algorithms.rft.compression.rft_binary_pack import (
        pack_container, unpack_container
    )
    return pack_container, unpack_container


def _import_ans():
    from algorithms.rft.compression.ans import ANSCoder
    return ANSCoder


def _import_native():
    native_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'src',
        'rftmw_native',
        'build',
    )
    if os.path.isdir(native_path) and native_path not in sys.path:
        sys.path.insert(0, native_path)
    import rftmw_native
    return rftmw_native


# ---------------------------------------------------------------------------
# Singleton scheduler
# ---------------------------------------------------------------------------

_SCHEDULER = None
_BACKEND   = None


def _get_scheduler():
    global _SCHEDULER
    if _SCHEDULER is None:
        UnifiedTransformScheduler, _ = _import_scheduler()
        _SCHEDULER = UnifiedTransformScheduler(prefer_native=False)
    return _SCHEDULER


# ---------------------------------------------------------------------------
# Wire-format header
#
#   4 bytes  magic  b'RFTC'
#   1 byte   path   0x00 = RFT+ANS   0x01 = INT8+zlib
#   4 bytes  ndim   (uint32 LE)
#   ndim*4   shape  (uint32 LE each)
#   8 bytes  abs_max float64 (for INT8 dequant; 0.0 for RFT path)
#   <rest>   payload
# ---------------------------------------------------------------------------

_MAGIC     = b'RFTC'
_PATH_RFT  = 0x00
_PATH_INT8 = 0x01


def _encode_header(path_byte: int, shape: tuple, abs_max: float) -> bytes:
    buf = io.BytesIO()
    buf.write(_MAGIC)
    buf.write(struct.pack('B', path_byte))
    buf.write(struct.pack('<I', len(shape)))
    for d in shape:
        buf.write(struct.pack('<I', int(d)))
    buf.write(struct.pack('<d', abs_max))
    return buf.getvalue()


def _decode_header(data: bytes):
    buf = io.BytesIO(data)
    magic = buf.read(4)
    if magic != _MAGIC:
        raise ValueError(f'Bad magic {magic!r} -- not an rft_compress blob')
    path_byte, = struct.unpack('B', buf.read(1))
    ndim,      = struct.unpack('<I', buf.read(4))
    shape      = tuple(struct.unpack('<I', buf.read(4))[0] for _ in range(ndim))
    abs_max,   = struct.unpack('<d', buf.read(8))
    payload    = data[buf.tell():]
    return path_byte, shape, abs_max, payload


# ---------------------------------------------------------------------------
# TransformType -> scheduler variant key mapping
# ---------------------------------------------------------------------------

_TRANSFORM_TO_VARIANT = {
    "dct":              "dct",
    "fft":              "h3_cascade",
    "rft_golden":       "op_rft_golden",
    "rft_fibonacci":    "op_rft_fibonacci",
    "rft_harmonic":     "op_rft_harmonic",
    "rft_geometric":    "op_rft_geometric",
    "rft_beating":      "op_rft_beating",
    "rft_phyllotaxis":  "op_rft_phyllotaxis",
    "arft":             "adaptive_phi",
}


# ---------------------------------------------------------------------------
# Codec factory
# ---------------------------------------------------------------------------

def _make_codec(variant_str: str, n: int, prune_threshold: float = 0.05):
    """Return an encode-capable codec for the given variant string and size."""
    _MODE_MAP = {
        "op_rft_golden":     "legacy",
        "fh5_entropy":       "fh5_entropy",
        "h6_dictionary":     "h6_dictionary",
        "h3_cascade":        "h3_cascade",
        "dct":               "legacy",
        "op_rft_fibonacci":  "legacy",
        "op_rft_harmonic":   "legacy",
        "op_rft_geometric":  "legacy",
        "op_rft_beating":    "legacy",
        "op_rft_phyllotaxis":"legacy",
        "adaptive_phi":      "legacy",
    }
    mode = _MODE_MAP.get(variant_str, "legacy")
    RFTHybridCodec = _import_hybrid_codec()
    if mode == "legacy":
        return RFTHybridCodec(mode=mode, prune_threshold=prune_threshold)
    return RFTHybridCodec(mode=mode)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compress(
    tensor: np.ndarray,
    quality: str = 'balanced',
    mse_budget: float = 1e-4,
    use_ans: bool = True,
    rd_step_size: float = 0.05,
) -> bytes:
    """
    Compress a NumPy tensor using the unified RFT pipeline.

    Parameters
    ----------
    tensor      : Any shape/dtype NumPy array.
    quality     : 'speed' | 'balanced' | 'quality'  -- passed to routing.py.
    mse_budget  : Maximum acceptable mean-squared reconstruction error.
    use_ans     : Apply rANS entropy coding to the .rftb payload (default True).
    rd_step_size: Quantisation step size used by the R-D probe in entropy.py.
                  0.05 = balanced quality/ratio (default).
                  0.01 = higher quality, lower ratio.
                  0.10 = higher ratio, slightly lower quality.

    Returns
    -------
    bytes  -- RFTC-magic blob; pass to decompress() to recover the tensor.
    """
    original_shape = tensor.shape
    original_dtype = tensor.dtype
    flat           = tensor.ravel().astype(np.float64)
    n              = len(flat)

    # DECISION 1: backend
    scheduler = _get_scheduler()
    backend   = scheduler.select_backend(n, require_accuracy=True)

    # DECISION 2: codec variant
    classify_signal, TransformType = _import_routing()
    sig_type, _ = classify_signal(flat)
    variant      = sig_type.value
    variant_str  = _TRANSFORM_TO_VARIANT.get(variant, 'h3_cascade')

    # R-D check
    calculate_rd_point, estimate_bitrate, _ = _import_entropy()

    def rft_fwd(x: np.ndarray) -> np.ndarray:
        return scheduler.forward(x.astype(np.complex128), variant=variant_str, backend=backend).real

    def rft_inv(X: np.ndarray) -> np.ndarray:
        return scheduler.inverse(X.astype(np.complex128), variant=variant_str, backend=backend).real

    # Compute the forward transform ONCE via the (cached) scheduler and reuse
    # for both the R-D probe and the encoder — avoids a redundant O(n³) basis
    # construction inside encode_tensor_hybrid's rft_forward_square path.
    coeffs_full = None
    sparse_route = None
    if getattr(backend, "value", "") == "cpp":
        try:
            native = _import_native()
            concentration = native.analyze_signal_concentration(flat.astype(np.float64), 0.90)
            if concentration.get('use_sparse_route'):
                engine = native.RFTMWEngine(n)
                coeffs_full = engine.forward_sparse(
                    flat.astype(np.float64),
                    int(concentration['top_k'])
                )
                sparse_route = {
                    'mode': 'native-iht',
                    'top_k': int(concentration['top_k']),
                    'top_k_energy_ratio': float(concentration['top_k_energy_ratio']),
                    'coherence_proxy': float(concentration['coherence_proxy']),
                }
        except Exception:
            coeffs_full = None

    if coeffs_full is None:
        coeffs_full = scheduler.forward(
            flat.astype(np.complex128), variant=variant_str, backend=backend
        )

    bpp_rft, mse_rft, _, _ = calculate_rd_point(
        flat, rft_fwd, rft_inv, step_size=rd_step_size,
        precomputed_coeffs=coeffs_full.real,
    )

    abs_max   = float(np.max(np.abs(flat))) + 1e-9
    int8_syms = np.clip((flat / abs_max) * 127, -127, 127).astype(np.int8)
    bpp_int8  = estimate_bitrate(int8_syms.astype(np.float64)) / n

    use_rft = (bpp_rft < bpp_int8) and (mse_rft <= mse_budget)

    if use_rft:
        prune_threshold = rd_step_size * abs_max
        codec     = _make_codec(variant_str, n, prune_threshold=prune_threshold)
        container = codec.encode(flat, precomputed_coeffs=coeffs_full)
        # Store the scheduler variant so decompress() can use the matching inverse
        container['variant'] = variant_str
        if sparse_route is not None:
            container['sparse_route'] = sparse_route

        if mse_rft > mse_budget * 0.5:
            container = _apply_hybrid_residual(container, flat, rft_inv)

        ctype = container.get("type", "")

        if ctype == "rft_vertex_tensor_container":
            pack_container, _ = _import_pack()
            raw = pack_container(container)
            if use_ans:
                try:
                    ANSCoder = _import_ans()
                    payload  = ANSCoder().encode(raw)
                except Exception:
                    payload  = raw
            else:
                payload = raw
        else:
            payload = zlib.compress(
                json.dumps(container, separators=(",", ":")).encode("utf-8"),
                level=6
            )

        header = _encode_header(_PATH_RFT, original_shape, 0.0)
        return header + payload

    else:
        compressed = zlib.compress(int8_syms.tobytes(), level=9)
        header     = _encode_header(_PATH_INT8, original_shape, abs_max)
        return header + compressed


def decompress(
    blob: bytes,
    shape: Optional[tuple] = None,
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """
    Recover a tensor from a compress() blob.

    Parameters
    ----------
    blob    : Bytes returned by compress().
    shape   : Override output shape (default: shape stored in header).
    dtype   : Cast output to this dtype (default: float64).

    Returns
    -------
    np.ndarray  -- Reconstructed tensor.
    """
    path_byte, stored_shape, abs_max, payload = _decode_header(blob)
    out_shape = shape if shape is not None else stored_shape
    out_dtype = dtype if dtype is not None else np.float64

    if path_byte == _PATH_RFT:
        if payload[:4] == b'RFTB':
            try:
                ANSCoder = _import_ans()
                raw      = ANSCoder().decode(payload)
            except Exception:
                raw      = payload
            _, unpack_container = _import_pack()
            container   = unpack_container(raw)
            scheduler   = _get_scheduler()
            variant_str = str(container.get('variant', 'h3_cascade'))
            backend     = scheduler.select_backend(
                int(container.get('total_length', 1)), require_accuracy=True
            )
            reconstructed_chunks = []
            for chunk in container['chunks']:
                vertices = chunk.get('vertices', [])
                if vertices:
                    rft_size = chunk['rft_size']
                    X = np.zeros(rft_size, dtype=np.complex128)
                    for v in vertices:
                        X[v['idx']] = v['A'] * np.exp(1j * v['phi'])
                    seg = scheduler.inverse(X, variant=variant_str, backend=backend)
                    reconstructed_chunks.append(seg.real)
            flat = np.concatenate(reconstructed_chunks) if reconstructed_chunks else np.zeros(1)
            return flat[:int(np.prod(out_shape))].reshape(out_shape).astype(out_dtype)
        else:
            container      = json.loads(zlib.decompress(payload).decode("utf-8"))
            mode           = container.get("mode", "legacy")
            variant_str    = container.get("variant")
            RFTHybridCodec = _import_hybrid_codec()
            codec          = RFTHybridCodec(mode=mode)

            # If the container was encoded with a scheduler variant, use the
            # matching scheduler inverse so the decode is consistent.
            inv_func = None
            if variant_str:
                _sched   = _get_scheduler()
                _n       = container.get("total_coeff", int(np.prod(out_shape)))
                _backend = _sched.select_backend(_n, require_accuracy=True)
                def inv_func(X, _s=_sched, _v=variant_str, _b=_backend):
                    return _s.inverse(X.astype(np.complex128), variant=_v, backend=_b).real

            flat = codec.decode(container, inverse_func=inv_func).ravel().astype(np.float64)
            return flat[:int(np.prod(out_shape))].reshape(out_shape).astype(out_dtype)

    elif path_byte == _PATH_INT8:
        raw_bytes = zlib.decompress(payload)
        int8_arr  = np.frombuffer(raw_bytes, dtype=np.int8).copy()
        flat_f64  = int8_arr.astype(np.float64) / 127.0 * abs_max
        return flat_f64[:int(np.prod(out_shape))].reshape(out_shape).astype(out_dtype)

    else:
        raise ValueError(f'Unknown path byte 0x{path_byte:02x}')


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_hybrid_residual(container: dict, original: np.ndarray, rft_inv) -> dict:
    coarse_chunks = []
    for chunk in container.get('chunks', []):
        vertices = chunk.get('vertices', [])
        if vertices:
            rft_size = chunk['rft_size']
            X = np.zeros(rft_size, dtype=np.complex128)
            for v in vertices:
                X[v['idx']] = v['A'] * np.exp(1j * v['phi'])
            seg = rft_inv(X)
            coarse_chunks.append(seg.real)

    if coarse_chunks:
        coarse   = np.concatenate(coarse_chunks)[:len(original)]
        residual = (original - coarse).astype(np.float32)
        container['residual_f16'] = residual.astype(np.float16).tobytes().hex()

    return container


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import time

    print('rft_compress -- smoke test')
    print('=' * 50)

    rng = np.random.default_rng(42)

    for desc, tensor in [
        ('smooth 1-D', np.sin(np.linspace(0, 20 * np.pi, 1024))),
        ('random 2-D', rng.standard_normal((64, 64)).astype(np.float32)),
        ('sparse 1-D', np.where(rng.random(512) > 0.85, rng.standard_normal(512), 0.0)),
    ]:
        t0    = time.perf_counter()
        blob  = compress(tensor, quality='balanced')
        t1    = time.perf_counter()
        recon = decompress(blob, shape=tensor.shape, dtype=tensor.dtype)
        t2    = time.perf_counter()

        orig_bytes = tensor.nbytes
        comp_bytes = len(blob)
        ratio      = orig_bytes / comp_bytes
        mse        = float(np.mean((tensor.ravel().astype(np.float64)
                                    - recon.ravel().astype(np.float64)) ** 2))

        print(f'  {desc:15}  {orig_bytes:6d}B -> {comp_bytes:6d}B  '
              f'ratio={ratio:.2f}x  MSE={mse:.2e}  '
              f'enc={1000*(t1-t0):.1f}ms  dec={1000*(t2-t1):.1f}ms')

    print('Done.')
