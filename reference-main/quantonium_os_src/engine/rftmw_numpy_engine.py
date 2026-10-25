"""Pure-NumPy inference engine for RFTMWPK-compressed Qwen2.5 models.

No PyTorch dependency.  Weights stay in GGUF-quantized form in memory
(Q4_K / Q6_K byte arrays).  Dequantization happens on-the-fly during
matrix operations.  Only activations live as float32.

Compression methods handled:
  INT8_ZLIB  – group-wise uint8 quantization + zlib  (2D GGUF weights, 1D norms)
  RFT        – resonant Fourier transform binary pack (3 ffn_norm tensors)
  NONE       – raw float32 bytes                      (attention biases)
"""

from __future__ import annotations

import io
import json
import os
import re
import struct
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Project root for RFT imports ──
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Mimosa KV compression (optional) ──
try:
    from algorithms.mimosa import MimosaGate as _MimosaGate
except ImportError:
    _MimosaGate = None

# ── RFT spectral stabilization (for SSM output) ──
try:
    from algorithms.rft.core.fast_rft import fast_rft
    from quantonium_os_src.engine.three_distance_router import (
        analyze_gap_structure,
        allocate_budget,
    )
    _HAS_RFT_STABILIZE = True
except ImportError:
    _HAS_RFT_STABILIZE = False
    fast_rft = None
    analyze_gap_structure = None
    allocate_budget = None

# ── Try loading C++ inference kernels ──
# Deferred loading: native module loaded on first use, not at import time,
# to avoid heap corruption from ASM kernel init + large numpy arrays at shutdown.
_native_infer = None
_native_checked = False
_native_module_ref = None  # prevent GC of the module itself

class NativeKernelsUnavailable(RuntimeError):
    """Raised when native C++/ASM kernels are required but cannot be loaded."""


def _get_native_infer():
    """Lazy-load the C++ inference kernels on first use.

    Tries each candidate build directory with explicit importlib fallthrough
    so a broken .pyd in one build dir doesn't prevent a good one from loading.
    If `RFTMW_REQUIRE_NATIVE=1`, raises NativeKernelsUnavailable instead of
    silently falling back to NumPy.
    """
    global _native_infer, _native_checked, _native_module_ref
    if _native_checked:
        return _native_infer
    _native_checked = True

    require_native = bool(os.environ.get("RFTMW_REQUIRE_NATIVE", ""))
    if os.environ.get("RFTMW_NO_NATIVE", ""):
        msg = "[engine] C++ kernels disabled via RFTMW_NO_NATIVE"
        print(msg)
        if require_native:
            raise NativeKernelsUnavailable("RFTMW_NO_NATIVE is set")
        return _native_infer

    # Add MinGW runtime to the DLL search path so libgcc_s_seh-1.dll,
    # libstdc++-6.dll, libwinpthread-1.dll can resolve.
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        for d in (r"C:\msys64\mingw64\bin", r"C:\msys64\clang64\bin"):
            try:
                if os.path.isdir(d):
                    os.add_dll_directory(d)
            except OSError:
                pass

    native_root = _PROJECT_ROOT / "src" / "rftmw_native"
    candidate_builds = [
        native_root / "build_mmap",       # mmap-native / current Qwen3.5 SSM build
        native_root / "build_linux",      # Linux native build (preferred)
        native_root / "build_asm",        # AVX2 + ASM sgemv (Windows)
        native_root / "build",            # statically linked fallback
        native_root / "build_cpp",        # plain C++ build
        native_root / "build_cpp_nolto",  # no-LTO variant
    ]

    import importlib.util
    errors = []
    loaded_module = None
    # Python extension modules export PyInit_<modname>, so we must use the
    # real name "rftmw_native". Once a pybind11 module registers its types
    # it can't be re-imported — track whether we've already done one attempt.
    pybind_poisoned = False
    for candidate in candidate_builds:
        pyd = candidate / "rftmw_native.cp313-win_amd64.pyd"
        if not pyd.exists():
            pyd_candidates = list(candidate.glob("rftmw_native*.pyd")) + \
                             list(candidate.glob("rftmw_native*.so"))
            if not pyd_candidates:
                continue
            pyd = pyd_candidates[0]
        if pybind_poisoned:
            errors.append(f"{candidate.name}: skipped (pybind11 already registered types from prior attempt)")
            continue
        try:
            # Clear any stale entry so importlib actually re-executes.
            sys.modules.pop("rftmw_native", None)
            spec = importlib.util.spec_from_file_location("rftmw_native", str(pyd))
            if spec is None or spec.loader is None:
                raise ImportError(f"no spec for {pyd}")
            mod = importlib.util.module_from_spec(spec)
            sys.modules["rftmw_native"] = mod
            spec.loader.exec_module(mod)
            if hasattr(mod, 'infer') and getattr(mod.infer, 'available', False):
                print(f"[engine] C++ kernels loaded from {candidate.name} "
                      f"(AVX2={getattr(mod, 'HAS_AVX2', '?')})")
                loaded_module = mod
                break
            else:
                errors.append(f"{candidate.name}: loaded but no usable .infer")
                pybind_poisoned = True  # successful exec registered pybind types
        except ImportError as e:
            msg = str(e)
            errors.append(f"{candidate.name}: ImportError: {msg}")
            sys.modules.pop("rftmw_native", None)
            # DLL load failure = fine to try next; "already registered" = poisoned
            if "already registered" in msg:
                pybind_poisoned = True
            continue
        except Exception as e:
            errors.append(f"{candidate.name}: {type(e).__name__}: {e}")
            sys.modules.pop("rftmw_native", None)
            if "already registered" in str(e):
                pybind_poisoned = True
            continue

    if loaded_module is None:
        msg = "[engine] No working native module found:\n  " + "\n  ".join(errors)
        print(msg)
        if require_native:
            raise NativeKernelsUnavailable(msg)
        return _native_infer

    _native_module_ref = loaded_module
    _native_infer = loaded_module.infer
    import atexit
    def _cleanup_native():
        global _native_infer, _native_module_ref
        _native_infer = None
        _native_module_ref = None
    atexit.register(_cleanup_native)
    return _native_infer

PHI = (1 + 5**0.5) / 2

# ===================================================================
# GGUF dequantisation  (Q4_K  –  144 bytes / 256 values)
# ===================================================================

def _dequant_q4k_blocks(raw: np.ndarray) -> np.ndarray:
    """Vectorised Q4_K dequantisation.

    Parameters
    ----------
    raw : uint8 array of shape (n_blocks, 144)

    Returns
    -------
    float32 array of shape (n_blocks, 256)
    """
    nb = raw.shape[0]

    # ── fp16 scales ──
    d_bytes = raw[:, 0:2].copy()
    dmin_bytes = raw[:, 2:4].copy()
    d = np.frombuffer(d_bytes.tobytes(), dtype=np.float16).astype(np.float32).reshape(nb)
    dmin = np.frombuffer(dmin_bytes.tobytes(), dtype=np.float16).astype(np.float32).reshape(nb)

    # ── packed sub-block scales (12 bytes → 8 scales + 8 mins, 6 bits each) ──
    sc_raw = raw[:, 4:16].copy()  # (nb, 12)
    sc = np.zeros((nb, 8), dtype=np.float32)
    mn = np.zeros((nb, 8), dtype=np.float32)

    for j in range(4):
        sc[:, j] = sc_raw[:, j] & 0x3F
        mn[:, j] = sc_raw[:, j + 4] & 0x3F
    for j in range(4, 8):
        sc[:, j] = (sc_raw[:, j + 4] & 0x0F).astype(np.float32) + ((sc_raw[:, j - 4] >> 6).astype(np.float32) * 16)
        mn[:, j] = (sc_raw[:, j + 4] >> 4).astype(np.float32) + ((sc_raw[:, j] >> 6).astype(np.float32) * 16)

    # ── 4-bit quants (128 bytes → 256 nibbles) ──
    qs = raw[:, 16:144].copy()  # (nb, 128)

    out = np.empty((nb, 256), dtype=np.float32)

    # 4 groups of 64, each group = 2 sub-blocks of 32
    for group in range(4):
        is0 = group * 2
        is1 = is0 + 1
        q_slice = qs[:, group * 32 : (group + 1) * 32]  # (nb, 32)
        lo = (q_slice & 0x0F).astype(np.float32)  # (nb, 32)
        hi = (q_slice >> 4).astype(np.float32)     # (nb, 32)

        d1 = d * sc[:, is0]      # (nb,)
        m1 = dmin * mn[:, is0]   # (nb,)
        d2 = d * sc[:, is1]
        m2 = dmin * mn[:, is1]

        out[:, group * 64 : group * 64 + 32] = lo * d1[:, None] - m1[:, None]
        out[:, group * 64 + 32 : group * 64 + 64] = hi * d2[:, None] - m2[:, None]

    return out


# ===================================================================
# GGUF dequantisation  (Q6_K  –  210 bytes / 256 values)
# ===================================================================

def _dequant_q6k_blocks(raw: np.ndarray) -> np.ndarray:
    """Vectorised Q6_K dequantisation.

    Parameters
    ----------
    raw : uint8 array of shape (n_blocks, 210)

    Returns
    -------
    float32 array of shape (n_blocks, 256)

    Layout per block (210 bytes):
      ql[128]  – lower 4 bits of each 6-bit value
      qh[64]   – upper 2 bits
      scales[16] – int8 per-group scale (16 groups of 16 values)
      d[2]     – fp16 master scale
    """
    nb = raw.shape[0]

    ql = raw[:, 0:128].copy()      # (nb, 128)
    qh = raw[:, 128:192].copy()    # (nb, 64)
    scales = raw[:, 192:208].copy().view(np.int8).astype(np.float32)  # (nb, 16)
    d_bytes = raw[:, 208:210].copy()
    d = np.frombuffer(d_bytes.tobytes(), dtype=np.float16).astype(np.float32).reshape(nb)

    # Pre-multiply d * scales → (nb, 16, 1) for 16 groups of 16 values
    ds = (d[:, None] * scales).reshape(nb, 16, 1)

    # Reconstruct 6-bit values:
    # ql (nb, 128) → reshape (nb, 2, 1, 64), shift by {0, 4}, mask → (nb, 2, 2, 64) → (nb, 8, 32)
    ql4 = ql.reshape(nb, 2, 1, 64) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
    ql4 = (ql4 & np.uint8(0x0F)).reshape(nb, -1, 32)  # (nb, 8, 32)

    # qh (nb, 64) → reshape (nb, 2, 1, 32), shift by {0, 2, 4, 6}, mask → (nb, 2, 4, 32) → (nb, 8, 32)
    qh2 = qh.reshape(nb, 2, 1, 32) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)
    qh2 = (qh2 & np.uint8(0x03)).reshape(nb, -1, 32)  # (nb, 8, 32)

    # Combine: 6-bit value = lower4 | (upper2 << 4), then subtract 32
    q = (ql4 | (qh2 << np.uint8(4))).astype(np.int8) - np.int8(32)
    q = q.reshape(nb, 16, 16).astype(np.float32)  # 16 groups of 16 values

    return (ds * q).reshape(nb, 256)


# ===================================================================
# GGUF tensor index (for lossless tensor loading from source GGUF)
# ===================================================================

# Bytes per element for known GGUF quant types (for block-quantised,
# this is total_block_bytes / values_per_block).
_GGUF_TYPE_BLOCK_SIZE = {
    0:  (1, 4),      # F32:  1 value, 4 bytes
    12: (256, 144),   # Q4_K: 256 values, 144 bytes
    14: (256, 210),   # Q6_K: 256 values, 210 bytes
}

def _read_gguf_metadata(path) -> Dict[str, Any]:
    """Read the key/value metadata block of a GGUF file.

    Returns a dict of all top-level KV pairs (strings, ints, floats, lists).
    Ignores the tensor index — that's what _parse_gguf_index is for.
    Used by Qwen25Config.from_gguf() to auto-detect hidden_size, n_layers, etc.
    """
    meta: Dict[str, Any] = {}
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            return meta
        _version = struct.unpack('<I', f.read(4))[0]
        _n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]

        def _read_string():
            slen = struct.unpack('<Q', f.read(8))[0]
            return f.read(slen).decode('utf-8', errors='replace')

        def _read_value(vtype):
            if vtype == 0:    return struct.unpack('<B', f.read(1))[0]
            elif vtype == 1:  return struct.unpack('<b', f.read(1))[0]
            elif vtype == 2:  return struct.unpack('<H', f.read(2))[0]
            elif vtype == 3:  return struct.unpack('<h', f.read(2))[0]
            elif vtype == 4:  return struct.unpack('<I', f.read(4))[0]
            elif vtype == 5:  return struct.unpack('<i', f.read(4))[0]
            elif vtype == 6:  return struct.unpack('<f', f.read(4))[0]
            elif vtype == 7:  return struct.unpack('<B', f.read(1))[0] != 0
            elif vtype == 8:  return _read_string()
            elif vtype == 9:
                arr_type = struct.unpack('<I', f.read(4))[0]
                arr_len = struct.unpack('<Q', f.read(8))[0]
                return [_read_value(arr_type) for _ in range(arr_len)]
            elif vtype == 10: return struct.unpack('<Q', f.read(8))[0]
            elif vtype == 11: return struct.unpack('<q', f.read(8))[0]
            elif vtype == 12: return struct.unpack('<d', f.read(8))[0]
            else:
                raise ValueError(f"Unknown GGUF value type: {vtype}")

        for _ in range(n_kv):
            key = _read_string()
            vtype = struct.unpack('<I', f.read(4))[0]
            try:
                meta[key] = _read_value(vtype)
            except Exception:
                break
    return meta


def _parse_gguf_index(path) -> Tuple[Dict[str, Tuple[int, int, int, list]], int]:
    """Parse GGUF v3 header and return index of ALL tensors.

    Returns (index, data_start) where index maps
      tensor_name → (offset, num_bytes, type_id, ne_list).
    """
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            raise ValueError(f"Bad GGUF magic: {magic}")
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]

        def _read_string():
            slen = struct.unpack('<Q', f.read(8))[0]
            return f.read(slen).decode('utf-8')

        def _read_value(vtype):
            if vtype == 0:    return struct.unpack('<B', f.read(1))[0]
            elif vtype == 1:  return struct.unpack('<b', f.read(1))[0]
            elif vtype == 2:  return struct.unpack('<H', f.read(2))[0]
            elif vtype == 3:  return struct.unpack('<h', f.read(2))[0]
            elif vtype == 4:  return struct.unpack('<I', f.read(4))[0]
            elif vtype == 5:  return struct.unpack('<i', f.read(4))[0]
            elif vtype == 6:  return struct.unpack('<f', f.read(4))[0]
            elif vtype == 7:  return struct.unpack('<B', f.read(1))[0] != 0
            elif vtype == 8:  return _read_string()
            elif vtype == 9:
                arr_type = struct.unpack('<I', f.read(4))[0]
                arr_len = struct.unpack('<Q', f.read(8))[0]
                return [_read_value(arr_type) for _ in range(arr_len)]
            elif vtype == 10: return struct.unpack('<Q', f.read(8))[0]
            elif vtype == 11: return struct.unpack('<q', f.read(8))[0]
            elif vtype == 12: return struct.unpack('<d', f.read(8))[0]
            else:
                raise ValueError(f"Unknown GGUF value type: {vtype}")

        # Skip KV metadata
        for _ in range(n_kv):
            _read_string()
            vtype = struct.unpack('<I', f.read(4))[0]
            _read_value(vtype)

        # Read tensor info — index ALL tensors
        index = {}
        for _ in range(n_tensors):
            name = _read_string()
            n_dims = struct.unpack('<I', f.read(4))[0]
            ne = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            # Compute total bytes for this tensor
            n_elements = 1
            for d in ne:
                n_elements *= d
            if ttype in _GGUF_TYPE_BLOCK_SIZE:
                vals_per_block, bytes_per_block = _GGUF_TYPE_BLOCK_SIZE[ttype]
                n_blocks = n_elements // vals_per_block
                num_bytes = n_blocks * bytes_per_block
            else:
                # Unknown type — store 0 bytes, will skip
                num_bytes = 0
            index[name] = (offset, num_bytes, ttype, ne)

        data_start = (f.tell() + 31) & ~31

    return index, data_start


# ===================================================================
# INT8_ZLIB  /  RFT  /  NONE decompression
# ===================================================================

def _decompress_int8_zlib(blob: bytes) -> np.ndarray:
    """Decompress INT8_ZLIB blob → float32 array."""
    buf = io.BytesIO(blob)
    N, n_groups, G = struct.unpack('>III', buf.read(12))
    gmins = np.empty(n_groups, dtype=np.float64)
    scales = np.empty(n_groups, dtype=np.float64)
    for gi in range(n_groups):
        gmins[gi], scales[gi] = struct.unpack('>dd', buf.read(16))
    comp_len = struct.unpack('>I', buf.read(4))[0]
    q = np.frombuffer(zlib.decompress(buf.read(comp_len)),
                      dtype=np.uint8).reshape(n_groups, G)
    recon = (q.astype(np.float32) * scales[:, None].astype(np.float32)
             + gmins[:, None].astype(np.float32))
    return recon.flatten()[:N]


def _decompress_int8_zlib_to_uint8(blob: bytes) -> np.ndarray:
    """Decompress INT8_ZLIB blob → uint8 array (for GGUF byte recovery).

    Rounds to uint8 in-place to minimise peak memory.
    """
    buf = io.BytesIO(blob)
    N, n_groups, G = struct.unpack('>III', buf.read(12))
    gmins = np.empty(n_groups, dtype=np.float32)
    scales = np.empty(n_groups, dtype=np.float32)
    for gi in range(n_groups):
        mn, sc = struct.unpack('>dd', buf.read(16))
        gmins[gi] = mn
        scales[gi] = sc
    comp_len = struct.unpack('>I', buf.read(4))[0]
    q_all = np.frombuffer(zlib.decompress(buf.read(comp_len)),
                          dtype=np.uint8).reshape(n_groups, G)
    del buf, blob  # free compressed data immediately

    # In-place reconstruction directly to uint8
    out = np.empty(n_groups * G, dtype=np.uint8)
    CHUNK = 8192
    for start in range(0, n_groups, CHUNK):
        end = min(start + CHUNK, n_groups)
        q = q_all[start:end].astype(np.float32)
        q *= scales[start:end, None]
        q += gmins[start:end, None]
        np.round(q, out=q)
        np.clip(q, 0, 255, out=q)
        out[start * G : end * G] = q.ravel().astype(np.uint8)
    del q_all
    return out[:N]


def _hybrid_phase_vector(n: int) -> np.ndarray:
    k = np.arange(n, dtype=np.float64)
    theta = 2.0 * np.pi * np.mod((k + 1.0) * PHI, 1.0)
    return np.exp(1j * theta).astype(np.complex128)


def _hybrid_inverse(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.complex128)
    n = int(y.size)
    e = _hybrid_phase_vector(n)
    return (np.fft.ifft(np.conj(e) * y) * np.sqrt(float(n))).real


def _decompress_rft_blob(blob: bytes) -> np.ndarray:
    """Decompress an RFT blob (RFTB v2) → float64 array."""
    buf = io.BytesIO(blob)
    magic = buf.read(4)
    if magic != b"RFTB":
        raise ValueError(f"Expected RFTB magic, got {magic!r}")
    ver, impl_id = struct.unpack(">BB", buf.read(2))
    if ver != 2:
        raise ValueError(f"Unsupported RFT blob version: {ver}")
    N, BLOCK, n_blocks = struct.unpack(">III", buf.read(12))
    mag_bits, phase_bits, keep_ratio = struct.unpack(">BBd", buf.read(10))
    impl = "hybrid" if impl_id == 1 else "canonical"
    max_mag_val = (1 << mag_bits) - 1
    max_phase_val = (1 << phase_bits) - 1

    block_peaks = [struct.unpack('>d', buf.read(8))[0] for _ in range(n_blocks)]
    comp_len = struct.unpack('>I', buf.read(4))[0]
    payload = zlib.decompress(buf.read(comp_len))

    Phi = None
    if impl == "canonical":
        from algorithms.rft.core.resonant_fourier_transform import rft_basis_matrix
        Phi = rft_basis_matrix(BLOCK, BLOCK, use_gram_normalization=True)

    offset = 0
    bpb = BLOCK * 2 * 2
    out_blocks: list[np.ndarray] = []
    for bi in range(n_blocks):
        chunk = payload[offset:offset + bpb]
        offset += bpb
        mq = np.frombuffer(chunk[:BLOCK * 2], dtype=np.uint16).copy()
        pq = np.frombuffer(chunk[BLOCK * 2:], dtype=np.uint16).copy()
        mags = mq.astype(np.float64) / max_mag_val * block_peaks[bi]
        phases = pq.astype(np.float64) / max_phase_val * 2 * np.pi - np.pi
        coeffs = mags * np.exp(1j * phases)

        if impl == "hybrid":
            block = _hybrid_inverse(coeffs)
        else:
            block = (Phi @ coeffs).real  # type: ignore[operator]
        out_blocks.append(block)

    return np.concatenate(out_blocks)[:N].astype(np.float32)


# ===================================================================
# Quantised weight wrapper
# ===================================================================

class QuantWeight:
    """Holds a GGUF-quantised 2D weight as raw uint8 bytes.

    Stored shape is (out_features, bytes_per_row) in the pack.
    During matmul, rows are dequantised on-the-fly.
    """

    __slots__ = ('raw', 'out_features', 'bytes_per_row', 'in_features',
                 'quant_type', 'block_size', 'dense')

    def __init__(self, raw_uint8: np.ndarray, packed_shape: Tuple[int, int]):
        self.out_features, self.bytes_per_row = packed_shape
        self.raw = raw_uint8.reshape(self.out_features, self.bytes_per_row)
        self.dense: Optional[np.ndarray] = None
        if self.bytes_per_row % 144 == 0 and self.bytes_per_row % 210 != 0:
            self.quant_type = 'q4k'
            self.block_size = 144
            self.in_features = (self.bytes_per_row // 144) * 256
        elif self.bytes_per_row % 210 == 0:
            self.quant_type = 'q6k'
            self.block_size = 210
            self.in_features = (self.bytes_per_row // 210) * 256
        else:
            raise ValueError(f"Unknown quant: bytes_per_row={self.bytes_per_row}")

    @classmethod
    def from_f32(cls, data: np.ndarray) -> 'QuantWeight':
        """Create a QuantWeight from raw F32 data (no quantization).

        The dense array is set directly, raw is None.
        """
        obj = object.__new__(cls)
        obj.out_features = data.shape[0]
        obj.in_features = data.shape[1] if len(data.shape) > 1 else 1
        obj.bytes_per_row = 0
        obj.raw = None
        obj.quant_type = 'f32'
        obj.block_size = 0
        obj.dense = np.ascontiguousarray(data.astype(np.float32))
        return obj

    def preload_dense(self, free_raw: bool = True) -> None:
        """Materialize weights as a contiguous FP32 matrix for BLAS matmul.

        Trades ~4 bytes/param of RAM for skipping per-token dequant.
        """
        if self.dense is not None:
            return
        W = self.dequant_all().astype(np.float32, copy=False)
        self.dense = np.ascontiguousarray(W)
        if free_raw:
            self.raw = None  # type: ignore[assignment]

    def dequant_all(self) -> np.ndarray:
        """Dequantise entire matrix → float32 (out_features, in_features)."""
        nblocks_per_row = self.bytes_per_row // self.block_size
        all_blocks = self.raw.reshape(-1, self.block_size)
        if self.quant_type == 'q4k':
            vals = _dequant_q4k_blocks(all_blocks)
        else:
            vals = _dequant_q6k_blocks(all_blocks)
        return vals.reshape(self.out_features, self.in_features)

    def matmul(self, x: np.ndarray) -> np.ndarray:
        """Compute x @ W^T  where W is this quantised weight.

        Uses C++ AVX2 kernels when available, falls back to chunked NumPy.

        x : (in_features,)  or  (..., in_features)
        returns : (..., out_features)
        """
        if self.dense is not None:
            # F32 dense path - use NumPy BLAS (sgemv_f32 has bugs with large matrices)
            return self.dense @ x if x.ndim == 1 else x @ self.dense.T
        _ni = _get_native_infer()
        if x.ndim == 1 and _ni is not None:
            # === C++ fast path: fused dequant + AVX2 dot product ===
            x_f32 = np.ascontiguousarray(x, dtype=np.float32)
            raw_flat = np.ascontiguousarray(self.raw.ravel())
            if self.quant_type == 'q4k':
                return _ni.q4k_matmul(raw_flat, x_f32,
                                      self.out_features, self.bytes_per_row)
            else:
                return _ni.q6k_matmul(raw_flat, x_f32,
                                      self.out_features, self.bytes_per_row)

        if x.ndim == 1:
            return self._matmul_vec(x)
        W = self.dequant_all()
        return x @ W.T

    def _matmul_vec(self, x: np.ndarray) -> np.ndarray:
        """Chunked vector × matrix multiply (memory-efficient)."""
        CHUNK = 256  # rows per chunk
        nblocks = self.bytes_per_row // self.block_size
        result = np.empty(self.out_features, dtype=np.float32)
        dequant_fn = _dequant_q4k_blocks if self.quant_type == 'q4k' else _dequant_q6k_blocks

        for start in range(0, self.out_features, CHUNK):
            end = min(start + CHUNK, self.out_features)
            chunk_raw = self.raw[start:end]  # (chunk, bytes_per_row)
            blocks = chunk_raw.reshape(-1, self.block_size)
            vals = dequant_fn(blocks)
            W_chunk = vals.reshape(end - start, self.in_features)
            result[start:end] = W_chunk @ x
        return result


# ===================================================================
# 3D Expert Weight (MoE)
# ===================================================================

class QuantWeight3D:
    """Holds a GGUF-quantised 3D expert weight as raw uint8 bytes.

    Shape is (n_experts, out_features, in_features) in logical form.
    Storage is flattened to (n_experts * out_features, bytes_per_row).
    """

    __slots__ = ('raw', 'n_experts', 'out_features', 'in_features',
                 'bytes_per_row', 'quant_type', 'block_size', 'rows_per_expert')

    def __init__(self, raw_uint8: np.ndarray, shape: Tuple[int, int, int]):
        """
        Args:
            raw_uint8: Flattened quantized bytes
            shape: (n_experts, out_features, in_features)
        """
        if len(shape) == 3:
            self.n_experts = shape[0]
            self.out_features = shape[1]
            self.in_features = shape[2]
        else:
            raise ValueError(f"Expected 3D shape, got {shape}")

        # Calculate bytes per row from in_features
        # Try Q4_K first, then Q6_K
        total_bytes = len(raw_uint8)
        total_rows = self.n_experts * self.out_features

        # Q4_K: 256 values per 144-byte block
        bytes_per_row_q4k = (self.in_features // 256) * 144
        # Q6_K: 256 values per 210-byte block
        bytes_per_row_q6k = (self.in_features // 256) * 210

        expected_q4k = total_rows * bytes_per_row_q4k
        expected_q6k = total_rows * bytes_per_row_q6k

        # Check for exact match first, then approximate
        if total_bytes == expected_q6k:
            self.quant_type = 'q6k'
            self.block_size = 210
            self.bytes_per_row = bytes_per_row_q6k
        elif total_bytes == expected_q4k:
            self.quant_type = 'q4k'
            self.block_size = 144
            self.bytes_per_row = bytes_per_row_q4k
        elif bytes_per_row_q6k > 0 and abs(total_bytes - expected_q6k) < expected_q6k * 0.01:
            # Allow 1% tolerance for Q6_K
            self.quant_type = 'q6k'
            self.block_size = 210
            self.bytes_per_row = bytes_per_row_q6k
        elif bytes_per_row_q4k > 0 and abs(total_bytes - expected_q4k) < expected_q4k * 0.01:
            # Allow 1% tolerance for Q4_K
            self.quant_type = 'q4k'
            self.block_size = 144
            self.bytes_per_row = bytes_per_row_q4k
        else:
            # Fallback: infer from total_bytes
            self.bytes_per_row = total_bytes // total_rows if total_rows > 0 else 0
            if self.bytes_per_row % 144 == 0 and self.bytes_per_row % 210 != 0:
                self.quant_type = 'q4k'
                self.block_size = 144
            elif self.bytes_per_row % 210 == 0:
                self.quant_type = 'q6k'
                self.block_size = 210
            else:
                raise ValueError(f"Cannot determine quant type for 3D weight: "
                                 f"shape={shape}, bytes={total_bytes}")

        self.rows_per_expert = self.out_features
        # Reshape to (n_experts, rows_per_expert, bytes_per_row)
        try:
            self.raw = raw_uint8.reshape(self.n_experts, self.rows_per_expert, self.bytes_per_row)
        except ValueError:
            # Shape mismatch - try alternative interpretation
            # GGUF stores as [out, in, n_experts] → we have [in, out, n_experts] bytes
            self.out_features, self.in_features = self.in_features, self.out_features
            bytes_per_row_q4k = (self.in_features // 256) * 144
            bytes_per_row_q6k = (self.in_features // 256) * 210
            if self.quant_type == 'q4k':
                self.bytes_per_row = bytes_per_row_q4k
            else:
                self.bytes_per_row = bytes_per_row_q6k
            self.rows_per_expert = self.out_features
            self.raw = raw_uint8.reshape(self.n_experts, self.rows_per_expert, self.bytes_per_row)

    def matmul_expert(self, x: np.ndarray, expert_idx: int) -> np.ndarray:
        """Compute x @ W[expert_idx]^T using native kernels."""
        expert_raw = self.raw[expert_idx]  # (rows_per_expert, bytes_per_row)

        _ni = _get_native_infer()
        if x.ndim == 1 and _ni is not None:
            x_f32 = np.ascontiguousarray(x, dtype=np.float32)
            raw_flat = np.ascontiguousarray(expert_raw.ravel())
            if self.quant_type == 'q4k':
                return _ni.q4k_matmul(raw_flat, x_f32,
                                      self.rows_per_expert, self.bytes_per_row)
            else:
                return _ni.q6k_matmul(raw_flat, x_f32,
                                      self.rows_per_expert, self.bytes_per_row)

        # NumPy fallback
        dequant_fn = _dequant_q4k_blocks if self.quant_type == 'q4k' else _dequant_q6k_blocks
        blocks = expert_raw.reshape(-1, self.block_size)
        vals = dequant_fn(blocks)
        W = vals.reshape(self.rows_per_expert, self.in_features)
        return W @ x


# ===================================================================
# MoE FFN (Mixture of Experts Feed-Forward Network)
# ===================================================================

class MoEFFN:
    """Mixture-of-Experts FFN with top-k routing and optional shared expert."""

    __slots__ = ('router', 'gate_exps', 'up_exps', 'down_exps',
                 'shared_gate', 'shared_up', 'shared_down',
                 'shared_expert_gate',
                 'n_experts', 'n_experts_per_tok', 'hidden_size', 'expert_dim')

    def __init__(self, config: Optional[Qwen25Config] = None,
                 router: Optional[QuantWeight] = None,
                 gate_exps: Optional[QuantWeight3D] = None,
                 up_exps: Optional[QuantWeight3D] = None,
                 down_exps: Optional[QuantWeight3D] = None,
                 shared_gate: Optional[QuantWeight] = None,
                 shared_up: Optional[QuantWeight] = None,
                 shared_down: Optional[QuantWeight] = None,
                 n_experts_per_tok: int = 8):
        self.router = router
        self.gate_exps = gate_exps
        self.up_exps = up_exps
        self.down_exps = down_exps
        self.shared_gate = shared_gate
        self.shared_up = shared_up
        self.shared_down = shared_down
        self.shared_expert_gate = None  # sigmoid gate vector (hidden_size,)

        # Config-based defaults
        if config is not None:
            self.n_experts = config.n_experts
            self.n_experts_per_tok = config.n_experts_per_tok
            self.hidden_size = config.hidden_size
            self.expert_dim = config.expert_dim
        elif gate_exps is not None:
            self.n_experts = gate_exps.n_experts
            self.n_experts_per_tok = n_experts_per_tok
            self.hidden_size = router.out_features if router else 0
            self.expert_dim = gate_exps.out_features
        else:
            self.n_experts = 0
            self.n_experts_per_tok = n_experts_per_tok
            self.hidden_size = 0
            self.expert_dim = 0

    @staticmethod
    def _silu(x: np.ndarray) -> np.ndarray:
        return x / (1.0 + np.exp(-np.clip(x, -88, 88)))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """MoE FFN forward pass.

        Args:
            x: Input hidden state (hidden_size,)

        Returns:
            Output hidden state (hidden_size,)
        """
        # Compute router logits
        logits = self.router.matmul(x)  # (n_experts,)

        # Top-k expert selection with softmax
        k = min(self.n_experts_per_tok, self.n_experts)
        top_k_idx = np.argpartition(-logits, k)[:k]

        # Softmax over selected experts only (standard Qwen3.5 MoE routing)
        top_k_logits = logits[top_k_idx]
        top_k_logits = top_k_logits - np.max(top_k_logits)  # numerical stability
        exp_logits = np.exp(top_k_logits)
        weights = exp_logits / np.sum(exp_logits)

        # Compute expert outputs and weighted sum
        out = np.zeros(self.hidden_size, dtype=np.float32)
        _ni = _get_native_infer()

        for i in range(k):
            expert_idx = top_k_idx[i]
            w = weights[i]
            if w <= 1e-6: continue

            # SwiGLU: silu(gate) * up, then down
            gate = self.gate_exps.matmul_expert(x, expert_idx)
            up = self.up_exps.matmul_expert(x, expert_idx)

            if _ni is not None:
                hidden = _ni.silu_mul(
                    np.ascontiguousarray(gate, dtype=np.float32),
                    np.ascontiguousarray(up, dtype=np.float32))
            else:
                hidden = self._silu(gate) * up

            expert_out = self.down_exps.matmul_expert(np.asarray(hidden), expert_idx)
            out += w * expert_out

        # Add shared expert output if present
        if self.shared_gate is not None and self.shared_up is not None and self.shared_down is not None:
            gate = self.shared_gate.matmul(x)
            up = self.shared_up.matmul(x)

            if _ni is not None:
                hidden = _ni.silu_mul(
                    np.ascontiguousarray(gate, dtype=np.float32),
                    np.ascontiguousarray(up, dtype=np.float32))
            else:
                hidden = self._silu(gate) * up

            shexp_out = self.shared_down.matmul(np.asarray(hidden))

            # Apply sigmoid gate to shared expert output
            # shared_expert_gate is a (hidden_size,) weight vector for
            # Linear(hidden_size, 1): gate = sigmoid(w @ x) → scalar
            if self.shared_expert_gate is not None:
                gate_logit = np.dot(
                    self.shared_expert_gate.astype(np.float32), x)
                sig_gate = 1.0 / (1.0 + np.exp(-float(np.clip(
                    gate_logit, -88, 88))))
                shexp_out = shexp_out * sig_gate

            out += shexp_out

        return out


# ===================================================================
# Model configuration
# ===================================================================

@dataclass
class Qwen25Config:
    hidden_size: int = 2048
    num_heads: int = 16
    num_kv_heads: int = 2
    head_dim: int = 128
    intermediate_size: int = 11008
    num_layers: int = 36
    vocab_size: int = 151936
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    max_position: int = 32768

    # MoE-specific fields
    is_moe: bool = False
    n_experts: int = 0
    n_experts_per_tok: int = 8
    expert_dim: int = 0          # intermediate dim for expert FFN
    has_shared_expert: bool = False
    has_fused_qkv: bool = False  # attn_qkv instead of separate q/k/v
    has_ssm: bool = False        # Mamba/SSM blocks

    # Qwen3.5 MoE hybrid architecture fields
    full_attention_interval: int = 4  # every Nth layer is full attention
    # GatedDeltaNet (linear attention) config
    linear_num_k_heads: int = 16
    linear_num_v_heads: int = 32
    linear_head_k_dim: int = 128
    linear_head_v_dim: int = 128
    # Full attention config (separate from linear attention)
    attn_head_dim: int = 256      # head_dim for full attention layers
    attn_num_heads: int = 16      # Q heads for full attention
    attn_num_kv_heads: int = 2    # KV heads for full attention
    rope_dim: int = 0             # 0 = use head_dim // 2

    # ---- Known Qwen2.5 architectures ----
    # Keyed by num_layers since that's the most distinguishing parameter
    # that also appears plainly in GGUF metadata as qwen2.block_count.
    _PRESETS = {
        # num_layers: (hidden, heads, kv_heads, ffn)
        36: (2048, 16, 2, 11008),   # Qwen2.5 3B  (Coder 3B)
        28: (3584, 28, 4, 18944),   # Qwen2.5 7B  (Coder 7B, Instruct 7B)
        48: (5120, 40, 8, 13824),   # Qwen2.5 14B
        64: (5120, 40, 8, 27648),   # Qwen2.5 32B
    }

    # ---- Known Qwen3.5 MoE architectures ----
    _MOE_PRESETS = {
        # (n_layers, hidden, n_experts): (heads, kv_heads, expert_dim, experts_per_tok)
        (40, 2048, 256): (16, 2, 512, 8),   # Qwen3.5-A3B (35B total, 3B active)
    }

    @classmethod
    def from_sidecar(cls, sidecar_path) -> Optional["Qwen25Config"]:
        """Build a config from a .rftmwpk.meta.json sidecar if it exists.

        Looks for a ``model_config`` dict written by the pack builder with
        the canonical Qwen2.5 dimensions; falls back to the num_layers preset
        if only ``num_layers`` is present.
        """
        from pathlib import Path
        p = Path(sidecar_path)
        if not p.exists():
            return None
        try:
            meta = json.loads(p.read_text())
        except Exception:
            return None
        mc = meta.get("model_config") or {}
        if not mc:
            return None
        # If all core fields present, use them directly
        core = ("hidden_size", "num_heads", "num_kv_heads",
                "intermediate_size", "num_layers")
        if all(k in mc for k in core):
            return cls(
                hidden_size=int(mc["hidden_size"]),
                num_heads=int(mc["num_heads"]),
                num_kv_heads=int(mc["num_kv_heads"]),
                head_dim=int(mc.get("head_dim", int(mc["hidden_size"]) // int(mc["num_heads"]))),
                intermediate_size=int(mc["intermediate_size"]),
                num_layers=int(mc["num_layers"]),
                vocab_size=int(mc.get("vocab_size", 151936)),
                rope_theta=float(mc.get("rope_theta", 1_000_000.0)),
                rms_norm_eps=float(mc.get("rms_norm_eps", 1e-6)),
                max_position=int(mc.get("max_position", 32768)),
            )
        # Preset fallback keyed on num_layers
        n_layers = int(mc.get("num_layers", 0))
        if n_layers in cls._PRESETS:
            hidden, heads, kv, ffn = cls._PRESETS[n_layers]
            return cls(hidden_size=hidden, num_heads=heads, num_kv_heads=kv,
                       head_dim=hidden // heads, intermediate_size=ffn,
                       num_layers=n_layers)
        return None

    @classmethod
    def from_gguf(cls, gguf_path) -> Optional["Qwen25Config"]:
        """Read config directly from GGUF key/value metadata.

        Supports both ``qwen2.*`` (Qwen2.5) and ``qwen35moe.*`` (Qwen3.5 MoE)
        namespaces.
        """
        meta = _read_gguf_metadata(gguf_path)
        if not meta:
            return None

        arch = meta.get("general.architecture", "")

        # ── Qwen3.5 MoE hybrid (SSM + attention) ──
        if arch == "qwen35moe":
            ns = "qwen35moe"
            n_layers = meta.get(f"{ns}.block_count")
            hidden = meta.get(f"{ns}.embedding_length")
            heads = meta.get(f"{ns}.attention.head_count")
            kv_heads = meta.get(f"{ns}.attention.head_count_kv", heads)
            key_len = meta.get(f"{ns}.attention.key_length", 256)
            val_len = meta.get(f"{ns}.attention.value_length", 256)
            rope_theta = meta.get(f"{ns}.rope.freq_base", 10_000_000.0)
            rope_dim = meta.get(f"{ns}.rope.dimension_count", 64)
            rms_eps = meta.get(f"{ns}.attention.layer_norm_rms_epsilon", 1e-6)
            ctx_len = meta.get(f"{ns}.context_length", 262144)
            n_experts = meta.get(f"{ns}.expert_count", 256)
            expert_ffn = meta.get(f"{ns}.expert_feed_forward_length", 512)
            experts_used = meta.get(f"{ns}.expert_used_count", 8)
            full_attn_interval = meta.get(f"{ns}.full_attention_interval", 4)
            ssm_state = meta.get(f"{ns}.ssm.state_size", 128)
            ssm_inner = meta.get(f"{ns}.ssm.inner_size", 4096)
            ssm_rank = meta.get(f"{ns}.ssm.time_step_rank", 32)
            ssm_groups = meta.get(f"{ns}.ssm.group_count", 16)
            ssm_conv_k = meta.get(f"{ns}.ssm.conv_kernel", 4)
            vocab = meta.get(f"{ns}.vocab_size") or meta.get("tokenizer.ggml.tokens", [])
            if not isinstance(vocab, int):
                vocab = len(vocab) if isinstance(vocab, list) else 151936
            if None in (n_layers, hidden, heads):
                return None
            cfg = cls(
                hidden_size=int(hidden),
                num_heads=int(heads),
                num_kv_heads=int(kv_heads),
                head_dim=int(key_len),
                intermediate_size=int(expert_ffn),
                num_layers=int(n_layers),
                vocab_size=int(vocab),
                rope_theta=float(rope_theta),
                rms_norm_eps=float(rms_eps),
                max_position=int(ctx_len),
                is_moe=True,
                n_experts=int(n_experts),
                n_experts_per_tok=int(experts_used),
                expert_dim=int(expert_ffn),
                has_shared_expert=True,
                has_fused_qkv=True,
                has_ssm=True,
                full_attention_interval=int(full_attn_interval),
                attn_head_dim=int(key_len),
                attn_num_heads=int(heads),
                attn_num_kv_heads=int(kv_heads),
                linear_num_k_heads=int(ssm_groups),
                linear_num_v_heads=int(ssm_inner) // int(ssm_state),
                linear_head_k_dim=int(ssm_state),
                linear_head_v_dim=int(ssm_state),
                rope_dim=int(rope_dim),
            )
            print(f"[engine] Qwen3.5-MoE config: {n_layers}L, {hidden}D, "
                  f"attn={heads}h/{kv_heads}kv/{key_len}hd, "
                  f"ssm={ssm_groups}kh/{ssm_inner}v/{ssm_rank}r, "
                  f"{n_experts}E/{experts_used}active, "
                  f"rope_dim={rope_dim}, theta={rope_theta:.0f}")
            return cfg

        # ── Qwen2 / Qwen2.5 dense ──
        n_layers = meta.get("qwen2.block_count")
        hidden = meta.get("qwen2.embedding_length")
        heads = meta.get("qwen2.attention.head_count")
        kv_heads = meta.get("qwen2.attention.head_count_kv", heads)
        ffn = meta.get("qwen2.feed_forward_length")
        rope_theta = meta.get("qwen2.rope.freq_base", 1_000_000.0)
        rms_eps = meta.get("qwen2.attention.layer_norm_rms_epsilon", 1e-6)
        ctx_len = meta.get("qwen2.context_length", 32768)
        vocab = meta.get("qwen2.vocab_size") or meta.get("tokenizer.ggml.tokens", [])
        if not isinstance(vocab, int):
            vocab = len(vocab) if isinstance(vocab, list) else 151936
        if None in (n_layers, hidden, heads, ffn):
            return None
        return cls(
            hidden_size=int(hidden),
            num_heads=int(heads),
            num_kv_heads=int(kv_heads),
            head_dim=int(hidden) // int(heads),
            intermediate_size=int(ffn),
            num_layers=int(n_layers),
            vocab_size=int(vocab),
            rope_theta=float(rope_theta),
            rms_norm_eps=float(rms_eps),
            max_position=int(ctx_len),
        )


# ===================================================================
# Building blocks
# ===================================================================

def _rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    _ni = _get_native_infer()
    if _ni is not None:
        return _ni.rms_norm(
            np.ascontiguousarray(x, dtype=np.float32),
            np.ascontiguousarray(weight, dtype=np.float32),
            eps)
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def _silu(x: np.ndarray) -> np.ndarray:
    _ni = _get_native_infer()
    if _ni is not None:
        return _ni.silu(np.ascontiguousarray(x, dtype=np.float32))
    return x / (1.0 + np.exp(-np.clip(x, -88, 88)))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    # C++ softmax is 1-D only; use for attention scores
    _ni = _get_native_infer()
    if _ni is not None and x.ndim == 1:
        return _ni.softmax(np.ascontiguousarray(x, dtype=np.float32))
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def _rope_freqs(dim: int, max_pos: int, theta: float) -> np.ndarray:
    """Precompute RoPE complex frequencies."""
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    t = np.arange(max_pos, dtype=np.float32)
    angles = np.outer(t, freqs)  # (max_pos, dim//2)
    return np.stack([np.cos(angles), np.sin(angles)], axis=-1)  # (max_pos, dim//2, 2)


def _apply_rope(x: np.ndarray, freqs: np.ndarray, pos: int) -> np.ndarray:
    """Apply interleaved RoPE to x of shape (n_heads, head_dim).

    freqs : precomputed (max_pos, head_dim//2, 2)
    Interleaved pairing: (dim[0], dim[1]) with freq[0], (dim[2], dim[3]) with freq[1], etc.
    """
    head_dim = x.shape[-1]
    d2 = head_dim // 2
    f = freqs[pos, :d2]  # (d2, 2)
    cos_f = f[:, 0]  # (d2,)
    sin_f = f[:, 1]  # (d2,)
    # Interleaved: even dims (0,2,4,...) and odd dims (1,3,5,...)
    x_even = x[..., 0::2]  # (..., d2)
    x_odd = x[..., 1::2]   # (..., d2)
    out = np.empty_like(x)
    out[..., 0::2] = x_even * cos_f - x_odd * sin_f
    out[..., 1::2] = x_even * sin_f + x_odd * cos_f
    return out


# ===================================================================
# Transformer layer
# ===================================================================

class TransformerLayer:
    __slots__ = ('attn_norm', 'ffn_norm',
                 'q_proj', 'k_proj', 'v_proj', 'o_proj',
                 'q_bias', 'k_bias', 'v_bias',
                 'gate_proj', 'up_proj', 'down_proj',
                 'config')

    def __init__(self, config: Qwen25Config):
        self.config = config
        # Weights set externally during load
        self.attn_norm: Optional[np.ndarray] = None
        self.ffn_norm: Optional[np.ndarray] = None
        self.q_proj: Optional[QuantWeight] = None
        self.k_proj: Optional[QuantWeight] = None
        self.v_proj: Optional[QuantWeight] = None
        self.o_proj: Optional[QuantWeight] = None
        self.q_bias: Optional[np.ndarray] = None
        self.k_bias: Optional[np.ndarray] = None
        self.v_bias: Optional[np.ndarray] = None
        self.gate_proj: Optional[QuantWeight] = None
        self.up_proj: Optional[QuantWeight] = None
        self.down_proj: Optional[QuantWeight] = None

    def forward(self, x: np.ndarray, pos: int,
                rope_freqs: np.ndarray,
                k_cache: np.ndarray, v_cache: np.ndarray,
                cache_len: int) -> np.ndarray:
        """Forward pass for a single token.

        x : (hidden_size,)
        Returns : (hidden_size,)

        Uses C++ AVX2 kernels for all heavy ops when available,
        falls back to NumPy otherwise.
        """
        cfg = self.config
        h = x

        # ── Attention ──
        normed = _rms_norm(h, self.attn_norm, cfg.rms_norm_eps)

        q = self.q_proj.matmul(normed)  # (num_heads * head_dim,)
        k = self.k_proj.matmul(normed)  # (num_kv_heads * head_dim,)
        v = self.v_proj.matmul(normed)

        if self.q_bias is not None:
            q = q + self.q_bias
        if self.k_bias is not None:
            k = k + self.k_bias
        if self.v_bias is not None:
            v = v + self.v_bias

        q = q.reshape(cfg.num_heads, cfg.head_dim)
        k = k.reshape(cfg.num_kv_heads, cfg.head_dim)
        v = v.reshape(cfg.num_kv_heads, cfg.head_dim)

        # RoPE
        _ni = _get_native_infer()
        if _ni is not None:
            f = rope_freqs[pos]  # (head_dim//2, 2)
            cos_f = np.ascontiguousarray(f[:, 0], dtype=np.float32)
            sin_f = np.ascontiguousarray(f[:, 1], dtype=np.float32)
            q = np.asarray(_ni.apply_rope(
                np.ascontiguousarray(q, dtype=np.float32),
                cos_f, sin_f, cfg.num_heads, cfg.head_dim))
            k = np.asarray(_ni.apply_rope(
                np.ascontiguousarray(k, dtype=np.float32),
                cos_f, sin_f, cfg.num_kv_heads, cfg.head_dim))
        else:
            q = _apply_rope(q, rope_freqs, pos)
            k = _apply_rope(k, rope_freqs, pos)

        # KV cache update
        k_cache[cache_len] = k  # (num_kv_heads, head_dim)
        v_cache[cache_len] = v
        seq_len = cache_len + 1

        # GQA — C++ does the full attention in one call
        if _ni is not None:
            attn_out = np.asarray(_ni.gqa_attention(
                np.ascontiguousarray(q, dtype=np.float32),
                np.ascontiguousarray(k_cache[:seq_len], dtype=np.float32),
                np.ascontiguousarray(v_cache[:seq_len], dtype=np.float32),
                cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, seq_len))
        else:
            heads_per_group = cfg.num_heads // cfg.num_kv_heads
            attn_out = np.zeros((cfg.num_heads, cfg.head_dim), dtype=np.float32)
            for kv_idx in range(cfg.num_kv_heads):
                k_seq = k_cache[:seq_len, kv_idx]
                v_seq = v_cache[:seq_len, kv_idx]
                for h_off in range(heads_per_group):
                    h_idx = kv_idx * heads_per_group + h_off
                    scores = q[h_idx] @ k_seq.T / np.sqrt(np.float32(cfg.head_dim))
                    weights = _softmax(scores)
                    attn_out[h_idx] = weights @ v_seq

        attn_flat = attn_out.reshape(-1)  # (num_heads * head_dim,)
        o = self.o_proj.matmul(attn_flat)
        h = h + o

        # ── FFN (SwiGLU) ──
        normed = _rms_norm(h, self.ffn_norm, cfg.rms_norm_eps)
        gate = self.gate_proj.matmul(normed)
        up = self.up_proj.matmul(normed)
        if _ni is not None:
            swiglu = _ni.silu_mul(
                np.ascontiguousarray(gate, dtype=np.float32),
                np.ascontiguousarray(up, dtype=np.float32))
            ffn_out = self.down_proj.matmul(np.asarray(swiglu))
        else:
            ffn_out = self.down_proj.matmul(_silu(gate) * up)
        h = h + ffn_out

        return h


# ===================================================================
# MoE Transformer Layer (Qwen3.5 style)
# ===================================================================

def _l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-6) -> np.ndarray:
    """L2 normalize along axis (matches FLA library implementation)."""
    inv_norm = 1.0 / np.sqrt(np.sum(x * x, axis=axis, keepdims=True) + eps)
    return x * inv_norm


def _per_head_rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMS norm applied independently per head.

    x: (n_heads, head_dim)
    weight: (head_dim,)
    Returns: (n_heads, head_dim)
    """
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


# ── RFT-based spectral stabilization cache ──
_rft_gap_cache: dict = {}


def _rft_stabilize_output(
    y: np.ndarray,
    retain_frac: float = 0.70,
    target_rms: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """RFT-based output stabilization for SSM layers.

    Replaces the broken ssm_norm weights (which amplify 2.5x) with spectral
    filtering using the Three-Distance Theorem. The key insight from the
    "intelligence dividend" research is that RFT compression at moderate
    retention (~40-70%) can IMPROVE signal quality by separating signal-
    carrying zones from noise-carrying zones in the phi-basis.

    y:           (n_heads, head_dim) SSM output per head
    retain_frac: Fraction of coefficients to keep (0.70 = 70%)
    target_rms:  Target RMS norm for output (default 1.0 for unit variance)

    Returns stabilized output with controlled norm.
    """
    if not _HAS_RFT_STABILIZE:
        # Fallback: simple RMS norm to target
        rms = np.sqrt(np.mean(y * y, axis=-1, keepdims=True) + eps)
        return (y / np.maximum(rms, eps)) * target_rms

    n_heads, head_dim = y.shape

    # Cache gap structure per head_dim
    if head_dim not in _rft_gap_cache:
        _rft_gap_cache[head_dim] = analyze_gap_structure(head_dim)
    gap_info = _rft_gap_cache[head_dim]

    out = np.empty_like(y)
    for h in range(n_heads):
        # Forward RFT
        coeffs = fast_rft(y[h].astype(np.complex128), inverse=False)

        # Three-distance zone-aware coefficient allocation
        # This keeps energy in dense spectral zones (signal) and
        # prunes sparse zones (noise) — the "intelligence dividend"
        masked_coeffs = allocate_budget(coeffs, retain_frac, gap_info)

        # Inverse RFT
        reconstructed = fast_rft(masked_coeffs, inverse=True).real

        # Scale to target RMS
        rms = np.sqrt(np.mean(reconstructed ** 2) + eps)
        out[h] = (reconstructed / np.maximum(rms, eps)) * target_rms

    return out.astype(np.float32)


class MoETransformerLayer:
    """Qwen3.5-MoE hybrid layer — supports both GatedDeltaNet (linear
    attention) and full GQA attention layer types.

    Layer type is determined by ``layer_type`` attribute:
    - 'linear_attention': GatedDeltaNet with matrix recurrence (SSM layers)
    - 'full_attention':   Standard GQA with Q gating (attention layers)
    """

    __slots__ = ('attn_norm', 'post_attn_norm',
                 'qkv_proj', 'attn_gate', 'o_proj',
                 # GatedDeltaNet (linear attention) weights
                 'ssm_a', 'ssm_conv1d', 'ssm_alpha', 'ssm_beta',
                 'ssm_dt_bias', 'ssm_norm',
                 'deltanet_state', 'deltanet_conv_state',
                 # Full attention weights
                 'q_proj', 'k_proj', 'v_proj',
                 'q_norm', 'k_norm',
                 # Common
                 'moe_ffn', 'config', 'layer_type', 'layer_idx')

    def __init__(self, config: Qwen25Config, layer_idx: int = 0):
        self.config = config
        self.layer_idx = layer_idx
        # Determine layer type from interval pattern
        interval = config.full_attention_interval
        if interval > 0 and ((layer_idx + 1) % interval == 0):
            self.layer_type = 'full_attention'
        else:
            self.layer_type = 'linear_attention'

        # Common weights
        self.attn_norm: Optional[np.ndarray] = None
        self.post_attn_norm: Optional[np.ndarray] = None
        self.o_proj: Optional[QuantWeight] = None
        self.moe_ffn: Optional[MoEFFN] = None

        # -- GatedDeltaNet (linear attention) --
        self.qkv_proj: Optional[QuantWeight] = None  # in_proj_qkv
        self.attn_gate: Optional[QuantWeight] = None  # in_proj_z (output gate)
        self.ssm_a: Optional[np.ndarray] = None       # A_log
        self.ssm_conv1d: Optional[QuantWeight] = None  # depthwise conv
        self.ssm_alpha: Optional[QuantWeight] = None   # in_proj_a (decay driver)
        self.ssm_beta: Optional[QuantWeight] = None    # in_proj_b (write gate)
        self.ssm_dt_bias: Optional[np.ndarray] = None  # dt_bias
        self.ssm_norm: Optional[np.ndarray] = None     # gated RMS norm weight
        self.deltanet_state: Optional[np.ndarray] = None   # (n_v_heads, k_dim, v_dim)
        self.deltanet_conv_state: Optional[np.ndarray] = None  # (conv_dim, conv_k)

        # -- Full attention --
        self.q_proj: Optional[QuantWeight] = None  # produces query + gate
        self.k_proj: Optional[QuantWeight] = None
        self.v_proj: Optional[QuantWeight] = None
        self.q_norm: Optional[np.ndarray] = None   # per-head RMS norm
        self.k_norm: Optional[np.ndarray] = None

    @staticmethod
    def _softplus(x: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

    def _deltanet_forward(self, normed: np.ndarray, cache_len: int) -> np.ndarray:
        """GatedDeltaNet forward for a single token.

        normed: RMS-normed hidden state (hidden_size,)
        Returns: (hidden_size,) output after residual-ready projection.

        Uses C++ ``deltanet_step`` kernel when available (conv + recurrence
        + gated-RMS-norm fused), falling back to pure NumPy otherwise.
        """
        cfg = self.config
        n_k = cfg.linear_num_k_heads   # 16
        n_v = cfg.linear_num_v_heads   # 32
        dk = cfg.linear_head_k_dim     # 128
        dv = cfg.linear_head_v_dim     # 128
        key_dim = n_k * dk             # 2048
        value_dim = n_v * dv           # 4096
        conv_dim = key_dim * 2 + value_dim  # 8192

        # 1. Input projections (already use C++ Q4/Q6K matmul kernels)
        qkv = self.qkv_proj.matmul(normed).astype(np.float32)   # (8192,)
        z = self.attn_gate.matmul(normed).astype(np.float32)     # (4096,) output gate
        a = self.ssm_alpha.matmul(normed).astype(np.float32)     # (32,) decay driver
        b = self.ssm_beta.matmul(normed).astype(np.float32)      # (32,) write gate

        # 2. Ensure conv kernel is dequantized + contiguous
        conv_kernel = self.ssm_conv1d.dense
        if conv_kernel is None:
            conv_kernel = self.ssm_conv1d.dequant_all().astype(np.float32)
            self.ssm_conv1d.dense = np.ascontiguousarray(conv_kernel)

        # 3. Initialize states on first step
        conv_k_sz = conv_kernel.shape[1] if conv_kernel.ndim == 2 else int(self.ssm_conv1d.in_features)
        if cache_len == 0 or self.deltanet_conv_state is None:
            self.deltanet_conv_state = np.zeros((conv_dim, conv_k_sz), dtype=np.float32)
        if cache_len == 0 or self.deltanet_state is None:
            # Flat layout for C++ compatibility: (n_v * dk * dv,)
            self.deltanet_state = np.zeros(n_v * dk * dv, dtype=np.float32)

        # 4. Prepare shared weight arrays (cached as contiguous float32)
        ssm_a = np.ascontiguousarray(self.ssm_a.astype(np.float32))
        dt_bias = np.ascontiguousarray(self.ssm_dt_bias.astype(np.float32))
        if self.ssm_norm is not None and self.ssm_norm.size > 0:
            norm_w = np.ascontiguousarray(self.ssm_norm.astype(np.float32))
        else:
            norm_w = np.empty(0, dtype=np.float32)

        # Native C++ GatedDeltaNet recurrence. The Python code below is only a
        # development fallback when native kernels are explicitly optional.
        _ni = _get_native_infer()
        if _ni is not None and hasattr(_ni, 'deltanet_step'):
            output = np.asarray(_ni.deltanet_step(
                np.ascontiguousarray(qkv, dtype=np.float32),
                np.ascontiguousarray(z, dtype=np.float32),
                np.ascontiguousarray(a, dtype=np.float32),
                np.ascontiguousarray(b, dtype=np.float32),
                np.ascontiguousarray(conv_kernel, dtype=np.float32),
                ssm_a,
                dt_bias,
                norm_w,
                float(cfg.rms_norm_eps),
                self.deltanet_conv_state,
                self.deltanet_state,
                int(n_k), int(n_v), int(dk), int(dv), int(conv_k_sz),
            ), dtype=np.float32)
            return self.o_proj.matmul(output.reshape(-1))

        raise NativeKernelsUnavailable(
            "GatedDeltaNet/SSM requires native C++ deltanet_step; "
            "the Python recurrence fallback is disabled. Use "
            "src\\rftmw_native\\build_mmap\\rftmw_qwen35_native.exe for the "
            "no-Python runner, or rebuild rftmw_native."
        )

        # ── Python fallback — Mamba-2 SSD (correct Qwen3.5 SSM) ──
        #
        # Architecture (per QWEN35_SSM_IMPLEMENTATION_STATUS.md):
        #   attn_qkv → [8192] = Q[4096] + K[4096] after conv+SiLU
        #   attn_gate → V values (NOT a multiplicative output gate)
        #   ssm_alpha → dt (time step), ssm_beta → beta gate
        #   state: S[h] ∈ R^{dk × dv}  (outer-product accumulation)
        #
        # This path is intentionally not used for benchmark/inference runs.

        # V values come from attn_gate (z), NOT from conv output
        v_flat = z / (1.0 + np.exp(-np.clip(z, -88, 88)))  # SiLU(attn_gate)
        v = v_flat.reshape(n_v, dv)                          # (32, 128)

        # Causal conv1d on qkv (Q+K only, full 8192 dims)
        self.deltanet_conv_state[:, :-1] = self.deltanet_conv_state[:, 1:]
        self.deltanet_conv_state[:, -1] = qkv
        conv_out = np.sum(self.deltanet_conv_state * conv_kernel, axis=1).astype(np.float32)
        conv_act = conv_out / (1.0 + np.exp(-np.clip(conv_out, -88, 88)))  # SiLU

        # Split Q and K — each gets n_v*dk = 4096 dims
        q_dim = n_v * dk   # 4096
        q = conv_act[:q_dim].reshape(n_v, dk)          # (32, 128) — all 32 heads, unique
        k = conv_act[q_dim:q_dim * 2].reshape(n_v, dk) # (32, 128)

        # L2 normalize Q, K per head
        q = _l2norm(q, axis=-1)
        k = _l2norm(k, axis=-1)

        # Compute dt, decay, and B_bar (Mamba-2 SSD parametrization)
        # dt is CLAMPED to max=0.1 — prevents geometric state growth when Δ is large.
        # Without this clamp, large dt pushes decay toward 0 and B_bar becomes a
        # gain factor >> 1, causing the hidden state to diverge over 30 SSM layers.
        A = np.exp(ssm_a)                                           # [32] positive
        dt = np.clip(self._softplus(a + dt_bias), 0.0, 0.1)        # [32] clamped ≤ 0.1
        decay = np.exp(-A * dt)                                     # [32] in (0, 1)
        beta_g = 1.0 / (1.0 + np.exp(-np.clip(b, -88, 88)))       # [32] sigmoid
        B_bar = dt * beta_g                                         # [32] effective write gate

        # Mamba-2 SSD recurrence: S[h] = decay[h]*S[h] + B_bar[h]*outer(K[h],V[h])
        state_3d = self.deltanet_state.reshape(n_v, dk, dv)
        y = np.empty((n_v, dv), dtype=np.float32)
        for h in range(n_v):
            state_3d[h] = decay[h] * state_3d[h] + B_bar[h] * np.outer(k[h], v[h])
            y[h] = q[h] @ state_3d[h]   # (dk,) @ (dk, dv) → (dv,)

        # Per-head RMS norm using learned ssm_norm weights
        if norm_w.size > 0:
            rms = np.sqrt(np.mean(y * y, axis=-1, keepdims=True) + cfg.rms_norm_eps)
            y = (y / rms) * norm_w

        # Output projection + 1/sqrt(d_model) variance scaling.
        # SSD output variance scales as O(d_state) due to the outer-product state.
        # Dividing by sqrt(d_model) brings it back to O(1) relative to the residual
        # stream, preventing the ~37x norm explosion seen without this correction.
        out = self.o_proj.matmul(y.reshape(-1))
        return out * (1.0 / np.sqrt(float(cfg.hidden_size)))

    def _attention_forward(self, normed: np.ndarray, pos: int,
                           rope_freqs: np.ndarray,
                           k_cache: np.ndarray, v_cache: np.ndarray,
                           cache_len: int) -> np.ndarray:
        """Full GQA attention forward with Q gating (Qwen3.5 style).

        normed: RMS-normed hidden state (hidden_size,)
        Returns: (hidden_size,) output after residual-ready projection.
        """
        cfg = self.config
        n_heads = cfg.attn_num_heads       # 16
        n_kv = cfg.attn_num_kv_heads       # 2
        hd = cfg.attn_head_dim             # 256
        rope_dim = cfg.rope_dim            # 64

        # 1. Q projection → query + gate
        q_full = self.q_proj.matmul(normed).astype(np.float32)  # (8192,)
        # Qwen3.5 stores Q and gate interleaved per head:
        #   [q_head0, gate_head0, q_head1, gate_head1, ...]
        # This matches HF's view(..., n_heads, head_dim * 2).chunk(2, dim=-1)
        qg = q_full.reshape(n_heads, hd * 2)
        query = qg[:, :hd].copy()        # (16, 256)
        gate = qg[:, hd:].reshape(-1)    # (4096,) for sigmoid gating

        # 2. K, V projections
        k = self.k_proj.matmul(normed).astype(np.float32).reshape(n_kv, hd)  # (2, 256)
        v = self.v_proj.matmul(normed).astype(np.float32).reshape(n_kv, hd)  # (2, 256)

        # 3. Per-head Q/K RMS normalization
        if self.q_norm is not None:
            query = _per_head_rms_norm(query, self.q_norm.astype(np.float32), cfg.rms_norm_eps)
        if self.k_norm is not None:
            k = _per_head_rms_norm(k, self.k_norm.astype(np.float32), cfg.rms_norm_eps)

        # 4. RoPE (only first rope_dim dimensions per head).
        # Qwen3.5-MoE uses the HF/llama.cpp split-half rotate_half layout:
        # first half rotates against the second half. For text-only positions,
        # the MRoPE streams share the same position ids, so the section table
        # does not alter these frequencies.
        if rope_dim > 0 and rope_dim < hd:
            f = rope_freqs[pos]  # (rope_dim//2, 2)
            cos_f = f[:, 0]  # (32,)
            sin_f = f[:, 1]  # (32,)
            half_rot = rope_dim // 2
            for h_idx in range(n_heads):
                x1 = query[h_idx, :half_rot].copy()
                x2 = query[h_idx, half_rot:rope_dim].copy()
                query[h_idx, :half_rot] = x1 * cos_f - x2 * sin_f
                query[h_idx, half_rot:rope_dim] = x2 * cos_f + x1 * sin_f
            for h_idx in range(n_kv):
                x1 = k[h_idx, :half_rot].copy()
                x2 = k[h_idx, half_rot:rope_dim].copy()
                k[h_idx, :half_rot] = x1 * cos_f - x2 * sin_f
                k[h_idx, half_rot:rope_dim] = x2 * cos_f + x1 * sin_f
        else:
            # Full RoPE (use existing implementation)
            _ni = _get_native_infer()
            if _ni is not None:
                f = rope_freqs[pos]
                cos_f = np.ascontiguousarray(f[:, 0], dtype=np.float32)
                sin_f = np.ascontiguousarray(f[:, 1], dtype=np.float32)
                query = np.asarray(_ni.apply_rope(
                    np.ascontiguousarray(query, dtype=np.float32),
                    cos_f, sin_f, n_heads, hd))
                k = np.asarray(_ni.apply_rope(
                    np.ascontiguousarray(k, dtype=np.float32),
                    cos_f, sin_f, n_kv, hd))
            else:
                query = _apply_rope(query, rope_freqs, pos)
                k = _apply_rope(k, rope_freqs, pos)

        # 5. KV cache update
        k_cache[cache_len] = k
        v_cache[cache_len] = v
        seq_len = cache_len + 1

        # 6. GQA attention — C++ fast path when available
        _ni = _get_native_infer()
        if _ni is not None and hasattr(_ni, 'gqa_attention'):
            attn_out = np.asarray(_ni.gqa_attention(
                np.ascontiguousarray(query, dtype=np.float32),
                np.ascontiguousarray(k_cache[:seq_len], dtype=np.float32),
                np.ascontiguousarray(v_cache[:seq_len], dtype=np.float32),
                n_heads, n_kv, hd, seq_len
            ))
        else:
            # Python fallback
            heads_per_group = n_heads // n_kv
            scale = 1.0 / np.sqrt(np.float32(hd))
            attn_out = np.zeros((n_heads, hd), dtype=np.float32)
            for kv_idx in range(n_kv):
                k_seq = k_cache[:seq_len, kv_idx]
                v_seq = v_cache[:seq_len, kv_idx]
                for h_off in range(heads_per_group):
                    h_idx = kv_idx * heads_per_group + h_off
                    scores = query[h_idx] @ k_seq.T * scale
                    weights = _softmax(scores)
                    attn_out[h_idx] = weights @ v_seq

        # 7. Gate: attn_out * sigmoid(gate)
        attn_flat = attn_out.reshape(-1)  # (4096,)
        gate_sig = 1.0 / (1.0 + np.exp(-np.clip(gate, -88, 88)))
        attn_flat = attn_flat * gate_sig

        # 8. Output projection: 4096 → 2048
        return self.o_proj.matmul(attn_flat)

    def forward(self, x: np.ndarray, pos: int,
                rope_freqs: np.ndarray,
                k_cache: np.ndarray, v_cache: np.ndarray,
                cache_len: int) -> np.ndarray:
        """Forward pass dispatches to DeltaNet or full attention."""
        cfg = self.config
        h = x
        normed = _rms_norm(h, self.attn_norm, cfg.rms_norm_eps)

        if self.layer_type == 'linear_attention':
            o = self._deltanet_forward(normed, cache_len)
        else:
            o = self._attention_forward(normed, pos, rope_freqs,
                                        k_cache, v_cache, cache_len)
        h = h + o

        # ── MoE FFN branch ──
        normed = _rms_norm(h, self.post_attn_norm, cfg.rms_norm_eps)
        h_moe = self.moe_ffn.forward(normed)
        h = h + h_moe

        return h


# ===================================================================
# Top-level engine
# ===================================================================

class Qwen25Engine:
    """Pure-NumPy autoregressive engine for RFTMWPK-packed Qwen2.5 models."""

    @staticmethod
    def _resolve_gguf_from_meta(meta: dict, pack_dir: Path) -> Optional[Path]:
        """Resolve GGUF path from sidecar metadata.

        Checks both ``provenance.gguf_path`` and top-level ``gguf_path``,
        resolves relative paths against *pack_dir*, and falls back to the
        Ollama blob store via SHA-256 hash.
        """
        gp = (meta.get("provenance", {}).get("gguf_path", "")
              or meta.get("gguf_path", ""))
        sha = (meta.get("provenance", {}).get("gguf_file_sha256", "")
               or meta.get("gguf_file_sha256", ""))

        if gp:
            candidate = Path(gp)
            if candidate.exists():
                return candidate
            # Resolve relative path against pack directory and its parent
            for base in (pack_dir, pack_dir.parent):
                resolved = base / candidate
                if resolved.exists():
                    return resolved

        if sha:
            blob = Path.home() / '.ollama' / 'models' / 'blobs' / f'sha256-{sha}'
            if blob.exists():
                return blob

        return None

    def __init__(self, pack_path: str, config: Optional[Qwen25Config] = None):
        self.pack_path = Path(pack_path)

        # Auto-detect config: sidecar model_config → sidecar gguf_path → default
        detected: Optional[Qwen25Config] = None
        if config is None:
            sidecar = self._resolve_sidecar()
            detected = Qwen25Config.from_sidecar(sidecar)
            # Always check GGUF when available — has complete arch info
            # including Qwen3.5-specific fields (rope_dim, SSM config, etc.)
            if sidecar.exists():
                try:
                    meta = json.loads(sidecar.read_text())
                    resolved_gp = self._resolve_gguf_from_meta(meta, self.pack_path.parent)
                    if resolved_gp:
                        gguf_cfg = Qwen25Config.from_gguf(str(resolved_gp))
                        if gguf_cfg is not None:
                            detected = gguf_cfg
                except Exception:
                    pass
            if detected is not None:
                print(f"[engine] Auto-detected config: {detected.num_layers} layers, "
                      f"hidden={detected.hidden_size}, heads={detected.num_heads}/{detected.num_kv_heads}, "
                      f"ffn={detected.intermediate_size}")

        self.config = config or detected or Qwen25Config()
        self.layers: List[TransformerLayer] = []
        self.embed: Optional[QuantWeight] = None
        self.lm_head: Optional[QuantWeight] = None  # separate output projection (may be tied to embed)
        self.output_norm: Optional[np.ndarray] = None
        self.rope_freqs: Optional[np.ndarray] = None
        self._tokenizer = None
        self._loaded = False
        self.finetune_weights_path: Optional[Path] = None
        self.finetune_weights_loaded = 0

        # Prefix KV cache — speeds up repeated system prompts
        self._prefix_ids: Optional[List[int]] = None  # cached token IDs
        self._prefix_k: Optional[List[np.ndarray]] = None  # (prefix_len, kv_heads, head_dim) per layer
        self._prefix_v: Optional[List[np.ndarray]] = None
        self._prefix_h: Optional[np.ndarray] = None  # last hidden state after prefix

    def _resolve_sidecar(self) -> Path:
        """Resolve metadata for a pack, including finetuned sidecar fallback."""
        candidates = [
            self.pack_path.with_suffix('.rftmwpk.meta.json'),
            self.pack_path.parent / f"quai-packs{self.pack_path.stem}.rftmwpk.meta.json",
        ]
        if self.pack_path.stem.endswith("_finetuned"):
            base_stem = self.pack_path.stem[:-len("_finetuned")]
            candidates.extend([
                self.pack_path.parent / f"{base_stem}.rftmwpk.meta.json",
                self.pack_path.parent / f"quai-packs{base_stem}.rftmwpk.meta.json",
            ])
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _resolve_finetune_weights_path(self) -> Optional[Path]:
        """Find an adapter sidecar for the selected pack when one is intended."""
        candidates = [Path(str(self.pack_path) + ".finetuned_weights.npz")]
        if (
            not self.pack_path.stem.endswith("_finetuned")
            and os.environ.get("RFTMW_LOAD_FINETUNED") == "1"
        ):
            candidates.append(
                self.pack_path.with_name(
                    f"{self.pack_path.stem}_finetuned{self.pack_path.suffix}.finetuned_weights.npz"
                )
            )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _apply_finetuned_sidecar(self) -> None:
        """Overlay spectral fine-tune weights onto loaded attention matrices."""
        weights_path = self._resolve_finetune_weights_path()
        if weights_path is None:
            return

        applied = 0
        skipped = 0
        try:
            with np.load(weights_path) as data:
                for key in data.files:
                    match = re.fullmatch(r"layer_(\d+)_(q_proj|k_proj|v_proj|o_proj)", key)
                    if match is None:
                        skipped += 1
                        continue
                    layer_idx = int(match.group(1))
                    attr = match.group(2)
                    if layer_idx < 0 or layer_idx >= len(self.layers):
                        skipped += 1
                        continue
                    layer = self.layers[layer_idx]
                    current = getattr(layer, attr, None)
                    if current is None:
                        skipped += 1
                        continue
                    arr = data[key]
                    shape = tuple(arr.shape)
                    expected = (int(current.out_features), int(current.in_features))
                    if shape != expected:
                        print(
                            f"[engine] Fine-tune skip {key}: shape {shape} != {expected}",
                            flush=True,
                        )
                        skipped += 1
                        continue
                    setattr(layer, attr, QuantWeight.from_f32(arr))
                    applied += 1
        except Exception as exc:
            print(f"[engine] Fine-tune sidecar load failed: {type(exc).__name__}: {exc}", flush=True)
            return

        self.finetune_weights_path = weights_path
        self.finetune_weights_loaded = applied
        print(
            f"[engine] Applied fine-tuned sidecar: {weights_path.name} "
            f"({applied} matrices, {skipped} skipped)",
            flush=True,
        )

    # ───────────────────────────────────────────────────────────────
    # Loading
    # ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load pack, decompress INT8_ZLIB → raw GGUF uint8 bytes, build layers."""
        if self._loaded:
            return

        cfg = self.config

        with open(self.pack_path, 'rb') as f:
            magic = f.read(8)
            if magic != b"RFTMWPK1":
                raise ValueError(f"Bad pack magic: {magic!r}")
            hdr_len = struct.unpack('<Q', f.read(8))[0]
            hdr = json.loads(f.read(hdr_len))
            data_offset = 16 + hdr_len

            # Index by name
            weight_index: Dict[str, dict] = {w['name']: w for w in hdr['weights']}

            # ── Try to locate the source GGUF for lossless tensor loading ──
            # INT8_ZLIB quantization is too lossy for inference.
            # If the original GGUF is available, read tensors from it directly.
            gguf_index = {}  # name → (offset, num_bytes, type_id, ne)
            gguf_data_start = 0
            gguf_path = None

            # Check sidecar metadata for GGUF path
            sidecar = self._resolve_sidecar()
            if sidecar.exists():
                try:
                    meta = json.loads(sidecar.read_text())
                    gguf_path = self._resolve_gguf_from_meta(meta, self.pack_path.parent)
                except Exception:
                    pass

            if gguf_path is not None:
                try:
                    gguf_index, gguf_data_start = _parse_gguf_index(gguf_path)
                    if gguf_index:
                        print(f"[engine] GGUF source found: {gguf_path.name} ({len(gguf_index)} tensors)")
                except Exception as e:
                    print(f"[engine] GGUF parse failed ({e}), using pack values")
                    gguf_index = {}

            def _load_raw(name: str) -> Tuple[np.ndarray, dict]:
                """Read and decompress a tensor from the pack."""
                entry = weight_index[name]
                f.seek(data_offset + entry['offset'])
                blob = f.read(entry['length'])
                method = entry['method']
                shape = tuple(entry['original_shape'])

                if method == 'INT8_ZLIB':
                    vals = _decompress_int8_zlib(blob)
                    return vals, entry
                elif method == 'NONE':
                    arr = np.frombuffer(blob, dtype=np.float32).copy()
                    return arr, entry
                elif method == 'RFT':
                    arr = _decompress_rft_blob(blob)
                    return arr, entry
                else:
                    raise ValueError(f"Unknown method {method} for {name}")

            def _load_2d(name: str) -> QuantWeight:
                """Load a 2D weight as QuantWeight (raw GGUF bytes).

                Prefers lossless bytes from the original GGUF source when
                available, since INT8_ZLIB round-tripping of quantised
                bytes is lossy and corrupts inference.
                """
                import gc

                # ── Fast path: read raw bytes directly from GGUF ──
                if name in gguf_index and gguf_path is not None:
                    offset, num_bytes, ttype, ne = gguf_index[name]

                    # Handle F32 tensors (e.g., router weights)
                    if ttype == 0 and num_bytes > 0:  # F32
                        with open(gguf_path, 'rb') as gf:
                            gf.seek(gguf_data_start + offset)
                            raw = np.frombuffer(gf.read(num_bytes), dtype=np.float32).copy()
                        # ne[0] = in_features, ne[1] = out_features (GGUF convention)
                        out_features = ne[1] if len(ne) > 1 else 1
                        in_features = ne[0] if len(ne) > 0 else 1
                        data = raw.reshape(out_features, in_features)
                        gc.collect()
                        return QuantWeight.from_f32(data)

                    # Handle F16 tensors
                    if ttype == 1 and num_bytes > 0:  # F16
                        with open(gguf_path, 'rb') as gf:
                            gf.seek(gguf_data_start + offset)
                            raw = np.frombuffer(gf.read(num_bytes), dtype=np.float16).copy()
                        out_features = ne[1] if len(ne) > 1 else 1
                        in_features = ne[0] if len(ne) > 0 else 1
                        data = raw.reshape(out_features, in_features).astype(np.float32)
                        gc.collect()
                        return QuantWeight.from_f32(data)

                    # Handle quantized tensors
                    if num_bytes > 0 and ttype in _GGUF_TYPE_BLOCK_SIZE:
                        vals_per_block, bytes_per_block = _GGUF_TYPE_BLOCK_SIZE[ttype]
                        n_elements = 1
                        for d in ne:
                            n_elements *= d
                        # ne[0] = in_features, ne[1] = out_features (GGUF convention)
                        out_features = ne[1] if len(ne) > 1 else 1
                        bytes_per_row = (ne[0] // vals_per_block) * bytes_per_block
                        with open(gguf_path, 'rb') as gf:
                            gf.seek(gguf_data_start + offset)
                            raw = np.frombuffer(gf.read(num_bytes), dtype=np.uint8).copy()
                        gc.collect()
                        return QuantWeight(raw, (out_features, bytes_per_row))

                # ── Fallback: INT8_ZLIB from pack (lossy) ──
                entry = weight_index[name]
                shape = tuple(entry['original_shape'])
                method = entry['method']
                if method != 'INT8_ZLIB':
                    raise ValueError(f"2D weight {name} has method {method}")

                base = data_offset + entry['offset']

                # Read INT8_ZLIB sub-header directly from file
                f.seek(base)
                N, n_groups, G = struct.unpack('>III', f.read(12))
                gmins = np.empty(n_groups, dtype=np.float32)
                scales = np.empty(n_groups, dtype=np.float32)
                for gi in range(n_groups):
                    mn, sc = struct.unpack('>dd', f.read(16))
                    gmins[gi] = mn
                    scales[gi] = sc
                comp_len = struct.unpack('>I', f.read(4))[0]

                # Streaming decompression: read compressed data in 1 MB
                # chunks, feed through zlib decompressor, reconstruct
                # uint8 values group-by-group without holding the full
                # decompressed array in memory.
                out = np.empty(N, dtype=np.uint8)
                decompressor = zlib.decompressobj()
                leftover = b""
                out_pos = 0
                group_idx = 0
                bytes_per_group = G  # uint8 values
                READ_CHUNK = 1 * 1024 * 1024  # 1 MB read chunks
                remaining = comp_len

                while remaining > 0 or leftover:
                    if remaining > 0:
                        chunk_size = min(READ_CHUNK, remaining)
                        raw = f.read(chunk_size)
                        remaining -= len(raw)
                        decompressed = decompressor.decompress(raw)
                        del raw
                        buf = leftover + decompressed
                        del decompressed
                    else:
                        buf = leftover
                        leftover = b""

                    # Process complete groups from buf
                    usable = (len(buf) // bytes_per_group) * bytes_per_group
                    if usable == 0 and remaining > 0:
                        leftover = buf
                        continue

                    if usable > 0:
                        n_chunk_groups = usable // bytes_per_group
                        q = np.frombuffer(buf[:usable], dtype=np.uint8).reshape(
                            n_chunk_groups, G).astype(np.float32)
                        sc = scales[group_idx:group_idx + n_chunk_groups, None]
                        mn = gmins[group_idx:group_idx + n_chunk_groups, None]
                        q *= sc
                        q += mn
                        np.round(q, out=q)
                        np.clip(q, 0, 255, out=q)
                        count = min(n_chunk_groups * G, N - out_pos)
                        out[out_pos:out_pos + count] = q.ravel()[:count].astype(np.uint8)
                        out_pos += count
                        group_idx += n_chunk_groups
                        leftover = buf[usable:]
                    elif not remaining:
                        # Handle any trailing bytes
                        if buf:
                            q = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
                            gi = group_idx
                            if gi < n_groups:
                                q *= scales[gi]
                                q += gmins[gi]
                                np.round(q, out=q)
                                np.clip(q, 0, 255, out=q)
                                count = min(len(q), N - out_pos)
                                out[out_pos:out_pos + count] = q[:count].astype(np.uint8)
                        break

                # Flush any remaining decompressor data
                try:
                    tail = decompressor.flush()
                    if tail and leftover:
                        leftover += tail
                except Exception:
                    pass

                del decompressor, leftover
                gc.collect()

                return QuantWeight(out[:N], shape)

            def _load_1d(name: str) -> np.ndarray:
                """Load a 1D norm/bias weight as float32.

                Prefers lossless F32 values from the original GGUF source
                when available, since INT8_ZLIB quantization of norms/biases
                introduces errors that corrupt inference.
                """
                if name in gguf_index and gguf_path is not None:
                    offset, num_bytes, ttype, ne = gguf_index[name]
                    if ttype == 0 and num_bytes > 0:  # F32
                        n_elements = num_bytes // 4
                        with open(gguf_path, 'rb') as gf:
                            gf.seek(gguf_data_start + offset)
                            arr = np.frombuffer(gf.read(num_bytes), dtype=np.float32).copy()
                        entry = weight_index[name]
                        return arr[:entry['original_shape'][0]]
                vals, entry = _load_raw(name)
                return vals[:entry['original_shape'][0]].astype(np.float32)

            def _load_3d(name: str) -> QuantWeight3D:
                """Load a 3D expert weight as QuantWeight3D.

                MoE expert tensors have shape (n_experts, out_features, in_features)
                stored as GGUF Q4_K or Q6_K quantized blocks.
                """
                import gc

                if name in gguf_index and gguf_path is not None:
                    offset, num_bytes, ttype, ne = gguf_index[name]
                    if num_bytes > 0 and ttype in _GGUF_TYPE_BLOCK_SIZE:
                        # ne = [in_features, out_features, n_experts] in GGUF
                        in_feat = ne[0] if len(ne) > 0 else 1
                        out_feat = ne[1] if len(ne) > 1 else 1
                        n_experts = ne[2] if len(ne) > 2 else 1
                        with open(gguf_path, 'rb') as gf:
                            gf.seek(gguf_data_start + offset)
                            raw = np.frombuffer(gf.read(num_bytes), dtype=np.uint8).copy()
                        gc.collect()
                        # QuantWeight3D expects (n_experts, out_features, in_features)
                        return QuantWeight3D(raw, (n_experts, out_feat, in_feat))

                raise ValueError(f"3D weight {name} not found in GGUF index - required for MoE")

            # Embedding (also used as output projection – tied weights)
            print(f"[engine] Loading embeddings ...", flush=True)
            self.embed = _load_2d('token_embd.weight')
            import gc; gc.collect()

            # Output norm
            self.output_norm = _load_1d('output_norm.weight')

            # Separate output projection (untied lm_head) — used by 7B+ models
            if 'output.weight' in weight_index or (gguf_index and 'output.weight' in gguf_index):
                self.lm_head = _load_2d('output.weight')
                print(f"[engine] Loaded separate output.weight (untied lm_head)", flush=True)
            else:
                self.lm_head = None  # tied — _lm_head() uses self.embed

            # Layers
            self.layers = []

            # ── Detect MoE architecture from tensor names ──
            is_moe = False
            has_fused_qkv = False

            # Check first layer for MoE indicators
            check_names = list(weight_index.keys()) + list(gguf_index.keys())
            if any('ffn_gate_exps' in n for n in check_names):
                is_moe = True
                cfg.is_moe = True
                print(f"[engine] Detected MoE architecture (ffn_gate_exps found)")
            if any('attn_qkv' in n for n in check_names):
                has_fused_qkv = True
                cfg.has_fused_qkv = True
                print(f"[engine] Detected fused QKV attention")

            # Auto-detect MoE config from tensor shapes if not set
            if is_moe and cfg.n_experts == 0:
                # Try to get expert count from first layer's tensor shape
                gate_name = 'blk.0.ffn_gate_exps.weight'
                if gate_name in gguf_index:
                    _, _, _, ne = gguf_index[gate_name]
                    if len(ne) >= 3:
                        cfg.n_experts = ne[2]
                        cfg.expert_dim = ne[1]
                        print(f"[engine] Auto-detected: {cfg.n_experts} experts, dim={cfg.expert_dim}")

                # Check for shared expert
                if any('ffn_gate_shexp' in n or 'ffn_gate_up_shexp' in n for n in check_names):
                    cfg.has_shared_expert = True
                    print(f"[engine] Detected shared expert")

            for i in range(cfg.num_layers):
                print(f"[engine] Loading layer {i}/{cfg.num_layers} ...", end='\r', flush=True)

                if is_moe:
                    # ── MoE Layer (Qwen3.5 hybrid) ──
                    layer = MoETransformerLayer(cfg, layer_idx=i)
                    layer.attn_norm = _load_1d(f'blk.{i}.attn_norm.weight')

                    # post_attention_norm (may be named differently)
                    pan_name = f'blk.{i}.post_attention_norm.weight'
                    ffn_norm_name = f'blk.{i}.ffn_norm.weight'
                    if pan_name in weight_index or pan_name in gguf_index:
                        layer.post_attn_norm = _load_1d(pan_name)
                    elif ffn_norm_name in weight_index or ffn_norm_name in gguf_index:
                        layer.post_attn_norm = _load_1d(ffn_norm_name)

                    if layer.layer_type == 'linear_attention':
                        # ── GatedDeltaNet layer ──
                        qkv_name = f'blk.{i}.attn_qkv.weight'
                        if qkv_name in weight_index or qkv_name in gguf_index:
                            layer.qkv_proj = _load_2d(qkv_name)

                        gate_name = f'blk.{i}.attn_gate.weight'
                        if gate_name in weight_index or gate_name in gguf_index:
                            layer.attn_gate = _load_2d(gate_name)

                        ssm_out_name = f'blk.{i}.ssm_out.weight'
                        if ssm_out_name in weight_index or ssm_out_name in gguf_index:
                            layer.o_proj = _load_2d(ssm_out_name)

                        # SSM / DeltaNet tensors
                        ssm_a_name = f'blk.{i}.ssm_a'
                        ssm_conv_name = f'blk.{i}.ssm_conv1d.weight'
                        ssm_alpha_name = f'blk.{i}.ssm_alpha.weight'
                        ssm_beta_name = f'blk.{i}.ssm_beta.weight'
                        ssm_dt_bias_name = f'blk.{i}.ssm_dt.bias'
                        ssm_norm_name = f'blk.{i}.ssm_norm.weight'

                        if ssm_a_name in weight_index or ssm_a_name in gguf_index:
                            layer.ssm_a = _load_1d(ssm_a_name)
                        if ssm_conv_name in weight_index or ssm_conv_name in gguf_index:
                            layer.ssm_conv1d = _load_2d(ssm_conv_name)
                            layer.ssm_conv1d.preload_dense(free_raw=False)
                        if ssm_alpha_name in weight_index or ssm_alpha_name in gguf_index:
                            layer.ssm_alpha = _load_2d(ssm_alpha_name)
                        if ssm_beta_name in weight_index or ssm_beta_name in gguf_index:
                            layer.ssm_beta = _load_2d(ssm_beta_name)
                        if ssm_dt_bias_name in weight_index or ssm_dt_bias_name in gguf_index:
                            layer.ssm_dt_bias = _load_1d(ssm_dt_bias_name)
                        if ssm_norm_name in weight_index or ssm_norm_name in gguf_index:
                            layer.ssm_norm = _load_1d(ssm_norm_name)
                        if layer.ssm_a is not None and layer.ssm_conv1d is not None:
                            cfg.has_ssm = True
                        print(f"[engine]   L{i}: GatedDeltaNet (SSM)       ", end='\r', flush=True)

                    else:
                        # ── Full attention layer ──
                        q_name = f'blk.{i}.attn_q.weight'
                        k_name = f'blk.{i}.attn_k.weight'
                        v_name = f'blk.{i}.attn_v.weight'
                        out_name = f'blk.{i}.attn_output.weight'
                        qn_name = f'blk.{i}.attn_q_norm.weight'
                        kn_name = f'blk.{i}.attn_k_norm.weight'

                        if q_name in weight_index or q_name in gguf_index:
                            layer.q_proj = _load_2d(q_name)
                        if k_name in weight_index or k_name in gguf_index:
                            layer.k_proj = _load_2d(k_name)
                        if v_name in weight_index or v_name in gguf_index:
                            layer.v_proj = _load_2d(v_name)
                        if out_name in weight_index or out_name in gguf_index:
                            layer.o_proj = _load_2d(out_name)
                        if qn_name in weight_index or qn_name in gguf_index:
                            layer.q_norm = _load_1d(qn_name)
                        if kn_name in weight_index or kn_name in gguf_index:
                            layer.k_norm = _load_1d(kn_name)
                        print(f"[engine]   L{i}: Full Attention (GQA)      ", end='\r', flush=True)

                    # MoE FFN (shared by both layer types)
                    moe = MoEFFN(cfg)
                    moe.router = _load_2d(f'blk.{i}.ffn_gate_inp.weight')
                    moe.gate_exps = _load_3d(f'blk.{i}.ffn_gate_exps.weight')
                    moe.up_exps = _load_3d(f'blk.{i}.ffn_up_exps.weight')
                    moe.down_exps = _load_3d(f'blk.{i}.ffn_down_exps.weight')

                    # Shared expert (optional)
                    shexp_gate = f'blk.{i}.ffn_gate_shexp.weight'
                    shexp_up = f'blk.{i}.ffn_up_shexp.weight'
                    shexp_down = f'blk.{i}.ffn_down_shexp.weight'
                    if cfg.has_shared_expert:
                        if shexp_gate in weight_index or shexp_gate in gguf_index:
                            moe.shared_gate = _load_2d(shexp_gate)
                            moe.shared_up = _load_2d(shexp_up)
                            moe.shared_down = _load_2d(shexp_down)
                            # Shared expert sigmoid gate vector
                            shexp_gate_inp = f'blk.{i}.ffn_gate_inp_shexp.weight'
                            if shexp_gate_inp in weight_index or shexp_gate_inp in gguf_index:
                                moe.shared_expert_gate = _load_1d(shexp_gate_inp)

                    layer.moe_ffn = moe
                else:
                    # ── Dense Layer ──
                    layer = TransformerLayer(cfg)

                    layer.attn_norm = _load_1d(f'blk.{i}.attn_norm.weight')
                    layer.ffn_norm = _load_1d(f'blk.{i}.ffn_norm.weight')

                    layer.q_proj = _load_2d(f'blk.{i}.attn_q.weight')
                    layer.k_proj = _load_2d(f'blk.{i}.attn_k.weight')
                    layer.v_proj = _load_2d(f'blk.{i}.attn_v.weight')
                    layer.o_proj = _load_2d(f'blk.{i}.attn_output.weight')

                    # Q/K/V biases (may not exist for all layers/models)
                    q_bias_name = f'blk.{i}.attn_q.bias'
                    k_bias_name = f'blk.{i}.attn_k.bias'
                    v_bias_name = f'blk.{i}.attn_v.bias'
                    if q_bias_name in weight_index:
                        layer.q_bias = _load_1d(q_bias_name)
                    if k_bias_name in weight_index:
                        layer.k_bias = _load_1d(k_bias_name)
                    if v_bias_name in weight_index:
                        layer.v_bias = _load_1d(v_bias_name)

                    layer.gate_proj = _load_2d(f'blk.{i}.ffn_gate.weight')
                    layer.up_proj = _load_2d(f'blk.{i}.ffn_up.weight')
                    layer.down_proj = _load_2d(f'blk.{i}.ffn_down.weight')

                self.layers.append(layer)

            print(f"\n[engine] All {cfg.num_layers} layers loaded.", flush=True)
            self._apply_finetuned_sidecar()

        # RoPE frequencies
        # For Qwen3.5 MoE, use rope_dim (64) instead of head_dim
        rope_d = cfg.rope_dim if cfg.rope_dim > 0 else cfg.head_dim
        self.rope_freqs = _rope_freqs(rope_d, min(cfg.max_position, 32768), cfg.rope_theta)
        self._loaded = True

    def preload_dense(self, free_raw: bool = True) -> None:
        """Materialize every quantized weight as FP32 for BLAS matmul.

        ~12 GB RAM for a 3B model. Run once after load(); tokens thereafter
        skip the per-call dequant and hit NumPy BLAS directly.

        Note: MoE expert tensors (QuantWeight3D) are NOT preloaded as they
        are too large; only router/shared expert weights are materialized.
        """
        import gc
        weights: List[QuantWeight] = []
        if self.embed is not None:
            weights.append(self.embed)
        if self.lm_head is not None:
            weights.append(self.lm_head)

        for layer in self.layers:
            if isinstance(layer, MoETransformerLayer):
                # MoE layer: preload projections for both layer types
                for w in (layer.qkv_proj, layer.o_proj, layer.attn_gate,
                          layer.ssm_conv1d, layer.ssm_alpha, layer.ssm_beta,
                          layer.q_proj, layer.k_proj, layer.v_proj):
                    if w is not None and isinstance(w, QuantWeight):
                        weights.append(w)
                if layer.moe_ffn is not None:
                    moe = layer.moe_ffn
                    if moe.router is not None:
                        weights.append(moe.router)
                    for sw in (moe.shared_gate, moe.shared_up, moe.shared_down):
                        if sw is not None:
                            weights.append(sw)
            else:
                # Dense layer
                for w in (layer.q_proj, layer.k_proj, layer.v_proj, layer.o_proj,
                          layer.gate_proj, layer.up_proj, layer.down_proj):
                    if w is not None:
                        weights.append(w)

        total = len(weights)
        t0 = time.perf_counter()
        for i, w in enumerate(weights):
            w.preload_dense(free_raw=free_raw)
            if (i + 1) % 8 == 0 or i + 1 == total:
                print(f"[engine] Preload FP32: {i + 1}/{total}", end='\r', flush=True)
        gc.collect()
        elapsed = time.perf_counter() - t0
        print(f"\n[engine] Preload FP32 complete: {total} tensors in {elapsed:.1f}s")

    # ───────────────────────────────────────────────────────────────
    # Tokeniser (uses HF `tokenizers` library – no torch)
    # ───────────────────────────────────────────────────────────────

    def _ensure_tokenizer(self) -> None:
        if self._tokenizer is not None:
            return
        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise RuntimeError(
                "Install the 'tokenizers' package: pip install tokenizers"
            )

        # Determine required vocab size from model
        model_vocab = self.lm_head.out_features if self.lm_head else self.embed.in_features

        # Prefer local tokenizer.json when present AND vocab size matches model.
        local = self.pack_path.parent / 'tokenizer.json'
        if local.exists():
            candidate = Tokenizer.from_file(str(local))
            if candidate.get_vocab_size() == model_vocab:
                self._tokenizer = candidate
                print(f"[engine] Using local tokenizer: {local}")
                return
            else:
                print(f"[engine] Local tokenizer vocab {candidate.get_vocab_size()} "
                      f"!= model vocab {model_vocab}, skipping")

        # Try to build tokenizer from embedded GGUF vocabulary (offline, exact match)
        sidecar = self._resolve_sidecar()
        gguf_path_for_tok = None
        gguf_sha_for_tok = None
        if sidecar.exists():
            try:
                meta_j = json.loads(sidecar.read_text())
                gp = meta_j.get('provenance', {}).get('gguf_path', '')
                sha = meta_j.get('provenance', {}).get('gguf_file_sha256', '')
                gguf_sha_for_tok = sha or None
                if gp and Path(gp).exists():
                    gguf_path_for_tok = Path(gp)
                elif sha:
                    candidate_path = Path.home() / '.ollama' / 'models' / 'blobs' / f'sha256-{sha}'
                    if candidate_path.exists():
                        gguf_path_for_tok = candidate_path
            except Exception:
                pass

        tok_cache_path = None
        if gguf_path_for_tok is not None:
            cache_key = gguf_sha_for_tok or gguf_path_for_tok.name
            cache_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(cache_key))
            cache_root = Path(os.environ.get(
                "RFTMW_TOKENIZER_CACHE",
                str(Path.home() / ".cache" / "quantoniumos" / "tokenizers"),
            ))
            tok_cache_path = cache_root / f"{cache_key}.{model_vocab}.tokenizer.json"
            if tok_cache_path.exists():
                try:
                    candidate = Tokenizer.from_file(str(tok_cache_path))
                    if candidate.get_vocab_size() == model_vocab:
                        self._tokenizer = candidate
                        print(f"[engine] Using cached GGUF tokenizer: {tok_cache_path}")
                        return
                except Exception as ex:
                    print(f"[engine] Cached tokenizer load failed: {ex}")

            try:
                tok = self._build_tokenizer_from_gguf(gguf_path_for_tok, Tokenizer)
                if tok is not None and tok.get_vocab_size() == model_vocab:
                    self._tokenizer = tok
                    print(f"[engine] Built tokenizer from GGUF ({model_vocab} tokens)")
                    if tok_cache_path is not None:
                        try:
                            tok_cache_path.parent.mkdir(parents=True, exist_ok=True)
                            tok.save(str(tok_cache_path))
                            print(f"[engine] Cached GGUF tokenizer: {tok_cache_path}")
                        except Exception as ex:
                            print(f"[engine] Tokenizer cache write failed: {ex}")
                    return
            except Exception as ex:
                print(f"[engine] GGUF tokenizer build failed: {ex}")

        # Select tokenizer based on model vocab size
        # Qwen 3.5 / Qwen3 MoE models have ~248K vocab
        # Qwen 2.5 / Qwen3 dense models have ~151K vocab
        if model_vocab > 200000:
            # Qwen 3.5 MoE model - use the matching tokenizer
            self._tokenizer = Tokenizer.from_pretrained(
                "Jackrong/Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled"
            )
        else:
            self._tokenizer = Tokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B")

    @staticmethod
    def _build_tokenizer_from_gguf(gguf_path: Path, Tokenizer_cls) -> Optional[Any]:
        """Build a HuggingFace tokenizers.Tokenizer from GGUF embedded vocabulary.

        Supports GPT2/BPE-style tokenizers (used by Qwen3.5 MoE).
        Returns None if the GGUF does not contain tokenizer data.
        """
        from tokenizers import AddedToken, Regex
        from tokenizers.models import BPE
        from tokenizers import normalizers, pre_tokenizers
        from tokenizers.decoders import ByteLevel as BLDecoder

        try:
            from transformers.models.qwen3_5.tokenization_qwen3_5 import (
                PRETOKENIZE_REGEX as qwen35_pretokenize_regex,
            )
        except Exception:
            qwen35_pretokenize_regex = (
                r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?"""
                r"""[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+"""
                r"""[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
            )

        meta = _read_gguf_metadata(gguf_path)
        tokens = meta.get('tokenizer.ggml.tokens')
        merges_raw = meta.get('tokenizer.ggml.merges')
        model_type = meta.get('tokenizer.ggml.model', 'gpt2')
        token_types = meta.get('tokenizer.ggml.token_type') or []

        if tokens is None or model_type not in ('gpt2', 'bpe'):
            return None

        # Build vocab dict: token_string → id
        vocab = {tok: idx for idx, tok in enumerate(tokens)}

        # Parse merges: "A B" → (A, B)
        merges = []
        if merges_raw:
            for m in merges_raw:
                parts = m.split(' ', 1)
                if len(parts) == 2:
                    merges.append((parts[0], parts[1]))

        bpe_model = BPE(
            vocab=vocab,
            merges=merges,
            unk_token=None,
            continuing_subword_prefix="",
            end_of_word_suffix="",
            fuse_unk=False,
            byte_fallback=False,
        )
        tokenizer = Tokenizer_cls(bpe_model)
        tokenizer.normalizer = normalizers.NFC()
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(
                Regex(qwen35_pretokenize_regex),
                behavior="isolated",
                invert=False,
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        tokenizer.decoder = BLDecoder()

        # GGUF stores chat/control markers in the normal BPE vocab plus a
        # token_type side table. Register them as AddedToken specials so
        # strings like "<|im_start|>" encode to one ID instead of literal BPE
        # pieces ("<", "|", "im", ...).
        special_tokens = []
        for idx, tok in enumerate(tokens):
            ttype = token_types[idx] if idx < len(token_types) else 1
            try:
                ttype_i = int(ttype)
            except (TypeError, ValueError):
                ttype_i = 1
            if ttype_i != 1:
                special_tokens.append(
                    AddedToken(tok, single_word=False, lstrip=False,
                               rstrip=False, normalized=False, special=True)
                )
        if special_tokens:
            tokenizer.add_special_tokens(special_tokens)

        return tokenizer

    def encode(self, text: str) -> List[int]:
        self._ensure_tokenizer()
        return self._tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        self._ensure_tokenizer()
        return self._tokenizer.decode(ids)

    # ───────────────────────────────────────────────────────────────
    # Embedding / LM head
    # ───────────────────────────────────────────────────────────────

    def _embed_token(self, token_id: int) -> np.ndarray:
        """Look up embedding for a single token → (hidden_size,) float32."""
        # embed is QuantWeight with shape (vocab, bytes_per_row)
        if self.embed.dense is not None:
            return self.embed.dense[token_id, :self.config.hidden_size].copy()
        # Dequant one row
        row_bytes = self.embed.raw[token_id]  # (bytes_per_row,)
        row_bytes = row_bytes.reshape(1, -1)
        if self.embed.quant_type == 'q4k':
            blocks = row_bytes.reshape(-1, 144)
            vals = _dequant_q4k_blocks(blocks)
        else:
            blocks = row_bytes.reshape(-1, 210)
            vals = _dequant_q6k_blocks(blocks)
        return vals.flatten()[:self.config.hidden_size]

    def _lm_head(self, hidden: np.ndarray) -> np.ndarray:
        """Compute logits from final hidden state → (vocab_size,) float32.

        Uses separate output.weight when available (untied), otherwise
        tied embeddings (embed weight transposed).
        """
        w = self.lm_head if self.lm_head is not None else self.embed
        return w.matmul(hidden)

    def _format_generation_prompt(self, prompt: str,
                                  template_mode: str = "raw",
                                  final_answer_only: bool = False) -> str:
        """Apply lightweight prompt templates without touching model weights."""
        mode = (template_mode or "raw").lower().strip()
        if mode == "auto":
            mode = "final_answer" if final_answer_only else "raw"

        if mode in ("raw", "none", "plain"):
            return prompt

        if mode in ("final_answer", "answer_only"):
            return (
                "Return only the final answer. Do not include reasoning.\n"
                f"Question: {prompt.strip()}\n"
                "Final answer:"
            )

        if mode in ("ollama", "qwen35_chat", "chat"):
            return (
                "<|im_start|>system\n"
                "You are a concise assistant. Answer directly.\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                f"{prompt.strip()}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

        if mode in ("qwen35_think", "think"):
            return (
                "<|im_start|>user\n"
                f"{prompt.strip()}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
                "<think>\n"
            )

        raise ValueError(
            "template_mode must be raw, final_answer, ollama, qwen35_chat, "
            "qwen35_think, or auto"
        )

    @staticmethod
    def _extract_final_answer(text: str) -> str:
        """Trim common reasoning traces down to the final answer span."""
        cleaned = text.strip()
        if not cleaned:
            return cleaned

        cleaned = re.sub(r"(?is)<think>.*?</think>", " ", cleaned).strip()
        if "</think>" in cleaned:
            cleaned = cleaned.split("</think>")[-1].strip()

        matches = re.findall(
            r"(?im)(?:final answer|answer)\s*[:=]\s*([^\n]+)",
            cleaned,
        )
        if matches:
            return matches[-1].strip().strip("`*_ .")

        expr_matches = re.findall(r"=\s*([-+]?\d+(?:\.\d+)?)", cleaned)
        if expr_matches:
            return expr_matches[-1]

        line_matches = [
            line.strip().strip("`*_ .")
            for line in cleaned.splitlines()
            if line.strip()
        ]
        if line_matches:
            last = line_matches[-1]
            number = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", last)
            if number:
                return number.group(0)
            trailing_number = re.search(r"([-+]?\d+(?:\.\d+)?)\s*$", last)
            if trailing_number:
                return trailing_number.group(1)
            return last

        return cleaned

    # ───────────────────────────────────────────────────────────────
    # Generation
    # ───────────────────────────────────────────────────────────────

    def generate(self, prompt: str, max_tokens: int = 128,
                 temperature: float = 0.7,
                 top_k: int = 40,
                 stop_tokens: Optional[List[int]] = None,
                 use_mimosa: bool = False,
                 mimosa_cache_size: int = 256,
                 template_mode: str = "raw",
                 final_answer_only: bool = False,
                 return_raw: bool = False) -> str:
        """Autoregressive text generation.

        Returns generated text (not including the prompt).
        """
        tokens = list(self.generate_stream(prompt, max_tokens, temperature,
                                           top_k, stop_tokens,
                                           use_mimosa=use_mimosa,
                                           mimosa_cache_size=mimosa_cache_size,
                                           template_mode=template_mode,
                                           final_answer_only=final_answer_only))
        text = "".join(tokens)
        if final_answer_only and not return_raw:
            return self._extract_final_answer(text)
        return text

    def generate_stream(self, prompt: str, max_tokens: int = 128,
                        temperature: float = 0.7,
                        top_k: int = 40,
                        stop_tokens: Optional[List[int]] = None,
                        on_stage=None,
                        use_mimosa: bool = False,
                        mimosa_cache_size: int = 256,
                        template_mode: str = "raw",
                        final_answer_only: bool = False):
        """Yield decoded token strings one at a time for true streaming.

        Supports prefix KV caching: if the prompt starts with the same
        tokens as the previous call, the cached KV states are reused
        and only the new suffix tokens are prefilled.  This typically
        saves 60-80% of prefill time when using a fixed system prompt.

        on_stage: optional callable(stage_name: str, info: dict) invoked at
        key progress points — "encoded", "prefill_progress", "decode_start".
        """
        self.load()
        self._ensure_tokenizer()

        if on_stage is not None:
            try: on_stage("encoding", {})
            except Exception: pass

        prompt = self._format_generation_prompt(
            prompt,
            template_mode=template_mode,
            final_answer_only=final_answer_only,
        )
        input_ids = self.encode(prompt)
        cfg = self.config
        n_prompt = len(input_ids)

        if on_stage is not None:
            try: on_stage("encoded", {"n_tokens": n_prompt})
            except Exception: pass

        # KV caches: only needed for full attention layers
        # For MoE hybrid models, attention layers have different head config
        max_seq = n_prompt + max_tokens
        if cfg.is_moe and cfg.has_ssm:
            # Qwen3.5 hybrid: attention layers use attn_num_kv_heads, attn_head_dim
            kv_heads = cfg.attn_num_kv_heads
            kv_hd = cfg.attn_head_dim
        else:
            kv_heads = cfg.num_kv_heads
            kv_hd = cfg.head_dim
        k_caches = [np.zeros((max_seq, kv_heads, kv_hd), dtype=np.float32)
                     for _ in range(cfg.num_layers)]
        v_caches = [np.zeros((max_seq, kv_heads, kv_hd), dtype=np.float32)
                     for _ in range(cfg.num_layers)]

        # ── Mimosa KV compression setup ──
        _mimosa_gate = None
        _attn_layer_ids: List[int] = []
        if use_mimosa:
            if _MimosaGate is None:
                raise RuntimeError(
                    "use_mimosa=True but algorithms.mimosa could not be imported. "
                    "Check that algorithms/mimosa/mimosa_gate.py exists."
                )
            _mimosa_gate = _MimosaGate(threshold=1.5, propagation_decay=0.6,
                                       max_propagation_hops=4)
            # Identify which layers maintain a KV cache (attention, not SSM)
            for li, layer in enumerate(self.layers):
                if not (hasattr(layer, 'layer_type')
                        and layer.layer_type == 'linear_attention'):
                    _attn_layer_ids.append(li)
            if not _attn_layer_ids:
                # All layers are standard attention (3B, 7B)
                _attn_layer_ids = list(range(cfg.num_layers))

        # Default stop tokens: EOS variants for Qwen/Qwen3.5 vocabularies.
        if stop_tokens is None:
            if cfg.vocab_size >= 248000:
                stop_tokens = [248044, 248046]  # <|endoftext|>, <|im_end|>
            else:
                stop_tokens = [151643, 151645]  # <|endoftext|>, <|im_end|>

        # ── Prefix KV cache: reuse previous system prompt computation ──
        prefix_hit = 0
        h = None
        if (not cfg.has_ssm
            and self._prefix_ids is not None
                and len(self._prefix_ids) <= len(input_ids)
                and input_ids[:len(self._prefix_ids)] == self._prefix_ids):
            prefix_hit = len(self._prefix_ids)
            # Copy cached KV into the new caches
            for li in range(cfg.num_layers):
                k_caches[li][:prefix_hit] = self._prefix_k[li][:prefix_hit]
                v_caches[li][:prefix_hit] = self._prefix_v[li][:prefix_hit]
            h = self._prefix_h.copy()
            if on_stage is not None:
                try: on_stage("prefill_progress", {"done": prefix_hit, "total": n_prompt, "cached": prefix_hit})
                except Exception: pass

        # Process prompt (prefill) — skip cached prefix tokens
        cache_len = prefix_hit
        remaining = n_prompt - prefix_hit
        prefill_every = max(1, remaining // 20) if remaining > 0 else 1
        for i, tid in enumerate(input_ids[prefix_hit:]):
            pos_idx = prefix_hit + i
            h = self._embed_token(tid)
            for li, layer in enumerate(self.layers):
                h = layer.forward(h, pos_idx, self.rope_freqs,
                                  k_caches[li], v_caches[li], cache_len)
            h = _rms_norm(h, self.output_norm, cfg.rms_norm_eps)
            cache_len += 1
            if on_stage is not None and (i % prefill_every == 0 or pos_idx == n_prompt - 1):
                try: on_stage("prefill_progress", {"done": pos_idx + 1, "total": n_prompt, "cached": prefix_hit})
                except Exception: pass

        # ── Save prefix cache for next call ──
        # Cache the first portion of the prompt (system prompt + early turns)
        # Only cache if we have a substantial prefix (>20 tokens) to avoid
        # caching tiny prompts. Cache up to the full prompt.
        if n_prompt > 20 and not cfg.has_ssm:
            self._prefix_ids = list(input_ids)
            self._prefix_k = [kc[:n_prompt].copy() for kc in k_caches]
            self._prefix_v = [vc[:n_prompt].copy() for vc in v_caches]
            self._prefix_h = h.copy() if h is not None else None

        if on_stage is not None:
            try: on_stage("decode_start", {"n_prompt": n_prompt})
            except Exception: pass

        # Autoregressive decode — yield each token
        logits = self._lm_head(h)
        for step in range(max_tokens):
            next_id = self._sample(logits, temperature, top_k)
            if next_id in stop_tokens:
                break

            token_text = self.decode([next_id])
            yield token_text

            # Forward pass for new token
            pos = cache_len
            h = self._embed_token(next_id)
            for li, layer in enumerate(self.layers):
                h = layer.forward(h, pos, self.rope_freqs,
                                  k_caches[li], v_caches[li], cache_len)
            h = _rms_norm(h, self.output_norm, cfg.rms_norm_eps)
            cache_len += 1

            # ── Mimosa KV eviction: prune one low-salience entry when over budget ──
            if _mimosa_gate is not None and cache_len > mimosa_cache_size:
                # Score all cached entries using the first attention layer's K norms
                li0 = _attn_layer_ids[0]
                # key_repr: (cache_len, head_dim) — mean over KV heads
                key_repr = k_caches[li0][:cache_len].mean(axis=1)
                _, turgor, _ = _mimosa_gate.gate(key_repr)
                # Never evict the newest token (index cache_len-1)
                evict_idx = int(np.argmin(turgor[:-1]))
                keep = [i for i in range(cache_len) if i != evict_idx]
                for li in _attn_layer_ids:
                    k_caches[li][:cache_len - 1] = k_caches[li][keep]
                    v_caches[li][:cache_len - 1] = v_caches[li][keep]
                cache_len -= 1

            logits = self._lm_head(h)

    def _sample(self, logits: np.ndarray, temperature: float,
                top_k: int) -> int:
        """Sample next token from logits with temperature and top-k."""
        if temperature < 1e-8:
            return int(np.argmax(logits))

        logits = logits / temperature
        # Top-k filtering
        if top_k > 0 and top_k < len(logits):
            topk_idx = np.argpartition(logits, -top_k)[-top_k:]
            mask = np.full_like(logits, -1e9)
            mask[topk_idx] = logits[topk_idx]
            logits = mask
        probs = _softmax(logits)
        return int(np.random.choice(len(probs), p=probs))


# ===================================================================
# CLI smoke test
# ===================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RFTMWPK NumPy inference")
    parser.add_argument("pack", help="Path to .rftmwpk file")
    parser.add_argument("--prompt", default="Hello, world!", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--template-mode", default="raw",
                        choices=["raw", "none", "plain",
                                 "final_answer", "answer_only",
                                 "ollama", "qwen35_chat", "qwen35_think",
                                 "chat", "think", "auto"],
                        help="Prompt template control")
    parser.add_argument("--final-answer-only", action="store_true",
                        help="Post-process generation down to the final answer")
    parser.add_argument("--return-raw", action="store_true",
                        help="Print raw generated text even with --final-answer-only")
    args = parser.parse_args()

    engine = Qwen25Engine(args.pack)
    print(f"[engine] Loading {args.pack} ...")
    engine.load()
    print(f"[engine] Generating (max_tokens={args.max_tokens}) ...")
    output = engine.generate(args.prompt, max_tokens=args.max_tokens,
                              temperature=args.temperature,
                              top_k=args.top_k,
                              template_mode=args.template_mode,
                              final_answer_only=args.final_answer_only,
                              return_raw=args.return_raw)
    print(f"\n=== Prompt ===\n{args.prompt}")
    print(f"\n=== Output ===\n{output}")
