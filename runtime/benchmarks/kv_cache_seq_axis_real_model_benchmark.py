#!/usr/bin/env python3
"""
Seq-len axis KV-cache compression benchmark (real HF model).

This benchmarks *axis-aware* compression where we transform along the seq_len
dimension (where positional / quasi-periodic structure should live), instead of
flattening the whole KV tensor.

Methods reported:
- baseline_raw: uncompressed KV bytes from model cache
- baseline_int8_zlib: INT8+zlib on full KV tensor (axis-agnostic baseline)
- axis_fft_topk_zlib: FFT along seq axis, keep top-k frequency rows globally
- axis_rft_hybrid_topk_zlib: hybrid RFT along seq axis (golden phase ⊙ FFT), keep top-k

Notes:
- This is a storage benchmark (bytes + reconstruction error). It does not yet
  integrate into generation loops.
- For 4k-token contexts you need a model with >=4096 context length (or rope scaling).
  The benchmark will clamp to the model's configured max positions.
"""

from __future__ import annotations

import argparse
import os
import struct
import time
import zlib
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

import numpy as np

from quantonium_os_src.engine.rftmw_memory import (
    CompressionMethod,
    _compress_int8_zlib,
    _decompress_int8_blob,
)
from algorithms.rft.core.resonant_fourier_transform import rft_basis_matrix


PHI = (1.0 + np.sqrt(5.0)) / 2.0
TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class Row:
    model: str
    context_tokens: int
    layers: int
    kv_dtype: str
    method: str
    kv_original_mb: float
    kv_compressed_mb: float
    ratio: float
    key_rel_err_mean: float
    val_rel_err_mean: float
    compress_s: float
    decompress_s: float


def _iter_kv(past_key_values: Any) -> Iterable[Tuple[Any, Any]]:
    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            yield layer.keys, layer.values
        return
    for layer in past_key_values:
        yield layer[0], layer[1]


def _bytes_tensor(t) -> int:
    return int(t.numel() * t.element_size())


def _bytes_past_key_values(past_key_values: Any, *, layer_limit: Optional[int] = None) -> int:
    total = 0
    for li, (k, v) in enumerate(_iter_kv(past_key_values)):
        if layer_limit is not None and li >= int(layer_limit):
            break
        total += _bytes_tensor(k) + _bytes_tensor(v)
    return int(total)


def _mb(nbytes: int) -> float:
    return float(nbytes) / 1024.0 / 1024.0


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) + 1e-15)
    return float(np.linalg.norm(a - b) / denom)


def _golden_phase_vector(n: int) -> np.ndarray:
    k = np.arange(n, dtype=np.float64)
    f = np.mod((k + 1.0) * PHI, 1.0)
    return np.exp(1j * TWO_PI * f).astype(np.complex128)


def _pack_axis_topk(
    coeffs_kept: np.ndarray,
    kept_idx: np.ndarray,
) -> bytes:
    """
    Pack kept coefficients (complex) + indices into a zlib blob.

    coeffs_kept shape: (B, H, K, D) complex64/complex128
    kept_idx shape: (K,) int64
    """
    coeffs_kept = np.asarray(coeffs_kept)
    kept_idx = np.asarray(kept_idx, dtype=np.uint16)

    # Store real/imag as float16 to reduce size.
    ri = np.stack([coeffs_kept.real, coeffs_kept.imag], axis=-1).astype(np.float16, copy=False)
    payload = zlib.compress(ri.tobytes(), level=6)

    header = struct.pack(
        ">4sIIII",
        b"KVAX",
        int(coeffs_kept.shape[0]),  # B
        int(coeffs_kept.shape[1]),  # H
        int(coeffs_kept.shape[2]),  # K
        int(coeffs_kept.shape[3]),  # D
    )
    idx_bytes = kept_idx.tobytes()
    meta = struct.pack(">II", int(len(idx_bytes)), int(len(payload)))
    return header + meta + idx_bytes + payload


def _unpack_axis_topk(blob: bytes) -> Tuple[np.ndarray, np.ndarray]:
    magic, B, H, K, D = struct.unpack(">4sIIII", blob[:20])
    if magic != b"KVAX":
        raise ValueError("Bad magic")
    idx_len, payload_len = struct.unpack(">II", blob[20:28])
    idx_start = 28
    idx_end = idx_start + idx_len
    payload_start = idx_end
    payload_end = payload_start + payload_len
    kept_idx = np.frombuffer(blob[idx_start:idx_end], dtype=np.uint16).astype(np.int64)
    ri = np.frombuffer(zlib.decompress(blob[payload_start:payload_end]), dtype=np.float16)
    ri = ri.reshape(B, H, K, D, 2).astype(np.float32)
    coeffs = (ri[..., 0] + 1j * ri[..., 1]).astype(np.complex64)
    return coeffs, kept_idx


def _axis_topk_compress(
    t: np.ndarray,
    *,
    keep_ratio: float,
    mode: str,
) -> Tuple[bytes, float]:
    """
    Compress along seq axis for a single tensor t of shape (B,H,S,D).
    Returns (blob, rel_recon_error).

    Selection uses a *global* set of kept frequency rows shared across all (B,H,D).
    """
    t = np.asarray(t)
    if t.ndim != 4:
        raise ValueError(f"Expected 4D KV tensor (B,H,S,D), got {t.shape}")
    B, H, S, D = t.shape
    S = int(S)
    K = max(1, int(round(float(S) * float(keep_ratio))))

    x = t.astype(np.float32, copy=False)
    X = np.fft.fft(x, axis=2, norm="ortho").astype(np.complex64, copy=False)

    if mode == "rft_hybrid":
        e = _golden_phase_vector(S).astype(np.complex64)
        X = X * e.reshape(1, 1, S, 1)
    elif mode != "fft":
        raise ValueError(f"Unknown mode: {mode}")

    # Global energy per frequency row (mean over B,H,D).
    energy = np.mean(np.abs(X) ** 2, axis=(0, 1, 3))
    kept_idx = np.argpartition(energy, -K)[-K:]
    kept_idx = np.sort(kept_idx)
    X_kept = X[:, :, kept_idx, :]

    blob = _pack_axis_topk(X_kept, kept_idx)

    # Reconstruction (for error measurement).
    X_full = np.zeros_like(X)
    X_full[:, :, kept_idx, :] = X_kept
    if mode == "rft_hybrid":
        X_full = X_full * np.conj(e.reshape(1, 1, S, 1))
    x_rec = np.fft.ifft(X_full, axis=2, norm="ortho").real.astype(np.float32, copy=False)

    err = _rel_err(x, x_rec)
    return blob, err


def _axis_topk_decompress(
    blob: bytes,
    *,
    seq_len: int,
    mode: str,
) -> np.ndarray:
    coeffs_kept, kept_idx = _unpack_axis_topk(blob)
    B, H, K, D = coeffs_kept.shape
    S = int(seq_len)

    X_full = np.zeros((B, H, S, D), dtype=np.complex64)
    X_full[:, :, kept_idx, :] = coeffs_kept

    if mode == "rft_hybrid":
        e = _golden_phase_vector(S).astype(np.complex64)
        X_full = X_full * np.conj(e.reshape(1, 1, S, 1))

    x_rec = np.fft.ifft(X_full, axis=2, norm="ortho").real.astype(np.float32, copy=False)
    return x_rec


def _axis_canonical_block_topk_compress(
    t: np.ndarray,
    *,
    keep_ratio: float,
    block: int = 256,
) -> Tuple[bytes, float]:
    """
    Canonical RFT compression along seq axis, applied blockwise.

    Uses Gram-normalized φ-grid basis Φ (unitary) of size (block×block).
    For each seq block x (length=block), compute c = Φ† x.

    Selection keeps a *global* set of coefficient indices shared across all (B,H,D)
    and all blocks, chosen by average energy.
    """
    t = np.asarray(t)
    if t.ndim != 4:
        raise ValueError(f"Expected 4D KV tensor (B,H,S,D), got {t.shape}")
    B, H, S, D = t.shape
    S = int(S)
    block = int(block)
    if block <= 0:
        raise ValueError("block must be positive")
    if S < block:
        # Pad up to one block.
        pad = block - S
        x = np.pad(t.astype(np.float32, copy=False), ((0, 0), (0, 0), (0, pad), (0, 0)))
        S = block
    else:
        x = t.astype(np.float32, copy=False)

    n_blocks = (S + block - 1) // block
    S_pad = n_blocks * block
    if S_pad != S:
        x = np.pad(x, ((0, 0), (0, 0), (0, S_pad - S), (0, 0)))
        S = S_pad

    # Canonical unitary basis for this block size.
    Phi = rft_basis_matrix(block, block, use_gram_normalization=True)
    PhiH = Phi.conj().T.astype(np.complex64, copy=False)

    # Transform: reshape each block to (Nseries, block) and apply (block×block).
    # Store coefficients as (B,H,nb,block,D) -> transpose to (B,H,nb,block,D)
    coeffs = np.zeros((B, H, n_blocks, block, D), dtype=np.complex64)
    for bi in range(n_blocks):
        seg = x[:, :, bi * block : (bi + 1) * block, :]  # (B,H,block,D) real
        seg2 = np.moveaxis(seg, 2, -1).reshape(B * H * D, block).astype(np.complex64)
        c2 = seg2 @ PhiH.T  # (Nseries, block)
        c = c2.reshape(B, H, D, block)
        coeffs[:, :, bi, :, :] = np.moveaxis(c, 2, -1)  # (B,H,block,D)

    K = max(1, int(round(float(block) * float(keep_ratio))))
    energy = np.mean(np.abs(coeffs) ** 2, axis=(0, 1, 2, 4))  # (block,)
    kept_idx = np.argpartition(energy, -K)[-K:]
    kept_idx = np.sort(kept_idx)

    kept = coeffs[:, :, :, kept_idx, :]  # (B,H,nb,K,D)

    # Pack: reuse KVAX but treat seq axis as the kept coefficient axis per block.
    # Flatten nb into batch dimension for packing simplicity.
    kept_flat = kept.reshape(B, H, n_blocks * K, D)
    idx_rep = (kept_idx.reshape(1, -1) + (np.arange(n_blocks).reshape(-1, 1) * block)).reshape(-1)
    blob = _pack_axis_topk(kept_flat, idx_rep.astype(np.int64))

    # Reconstruction for error: scatter back into (B,H,S,D) in coefficient domain,
    # inverse via Φ for each block.
    coeffs_rec = np.zeros_like(coeffs)
    coeffs_rec[:, :, :, kept_idx, :] = coeffs[:, :, :, kept_idx, :]

    x_rec = np.zeros((B, H, S, D), dtype=np.float32)
    Phi_c64 = Phi.astype(np.complex64, copy=False)
    for bi in range(n_blocks):
        c = coeffs_rec[:, :, bi, :, :]  # (B,H,block,D)
        c2 = np.moveaxis(c, 2, -1).reshape(B * H * D, block)  # (Nseries, block)
        seg2 = c2 @ Phi_c64.T  # (Nseries, block)
        seg = seg2.reshape(B, H, D, block)
        seg = np.moveaxis(seg, -1, 2).real.astype(np.float32, copy=False)  # (B,H,block,D)
        x_rec[:, :, bi * block : (bi + 1) * block, :] = seg

    # Unpad to original length.
    x_true = np.asarray(t, dtype=np.float32)
    x_rec = x_rec[:, :, : x_true.shape[2], :]
    err = _rel_err(x_true, x_rec)
    return blob, err


def _axis_canonical_overlap_topk_compress(
    t: np.ndarray,
    *,
    keep_ratio: float,
    win: int = 256,
    hop: int = 128,
) -> Tuple[bytes, float]:
    """
    Overlapping-window canonical RFT along seq axis (STFT-like).

    Uses Hann window and hop=win/2 by default (COLA). Keeps a fixed set of K
    coefficient indices per frame (chosen globally by energy).
    """
    t = np.asarray(t)
    if t.ndim != 4:
        raise ValueError(f"Expected 4D KV tensor (B,H,S,D), got {t.shape}")
    B, H, S0, D = t.shape
    win = int(win)
    hop = int(hop)
    if win <= 0 or hop <= 0 or hop > win:
        raise ValueError("Invalid win/hop")

    x_true = t.astype(np.float32, copy=False)

    # Pad on both ends so every original sample lies away from window zeros.
    # With Hann, endpoints are 0; without padding, edge samples can get wsum≈0
    # which explodes when normalizing.
    pad = win - hop  # e.g. 128 for win=256, hop=128
    x = np.pad(x_true, ((0, 0), (0, 0), (pad, pad), (0, 0)))
    S = int(x.shape[2])

    # Pad tail to fit whole frames.
    n_frames = 1 if S <= win else int(np.ceil((S - win) / hop)) + 1
    S_pad = (n_frames - 1) * hop + win
    if S_pad > S:
        x = np.pad(x, ((0, 0), (0, 0), (0, S_pad - S), (0, 0)))
        S = S_pad

    window = np.hanning(win).astype(np.float32)

    Phi = rft_basis_matrix(win, win, use_gram_normalization=True)
    PhiH = Phi.conj().T.astype(np.complex64, copy=False)
    Phi_c64 = Phi.astype(np.complex64, copy=False)

    coeffs = np.zeros((B, H, n_frames, win, D), dtype=np.complex64)
    for fi in range(n_frames):
        start = fi * hop
        seg = x[:, :, start : start + win, :] * window.reshape(1, 1, win, 1)
        seg2 = np.moveaxis(seg, 2, -1).reshape(B * H * D, win).astype(np.complex64)
        c2 = seg2 @ PhiH.T
        c = c2.reshape(B, H, D, win)
        coeffs[:, :, fi, :, :] = np.moveaxis(c, 2, -1)

    K = max(1, int(round(float(win) * float(keep_ratio))))
    energy = np.mean(np.abs(coeffs) ** 2, axis=(0, 1, 2, 4))  # (win,)
    kept_idx = np.argpartition(energy, -K)[-K:]
    kept_idx = np.sort(kept_idx)

    kept = coeffs[:, :, :, kept_idx, :]  # (B,H,frames,K,D)
    kept_flat = kept.reshape(B, H, n_frames * K, D)
    idx_rep = (kept_idx.reshape(1, -1) + (np.arange(n_frames).reshape(-1, 1) * win)).reshape(-1)
    blob = _pack_axis_topk(kept_flat, idx_rep.astype(np.int64))

    # Reconstruction for error.
    coeffs_rec = np.zeros_like(coeffs)
    coeffs_rec[:, :, :, kept_idx, :] = coeffs[:, :, :, kept_idx, :]

    y = np.zeros((B, H, S, D), dtype=np.float32)
    wsum = np.zeros(S, dtype=np.float32)
    for fi in range(n_frames):
        c = coeffs_rec[:, :, fi, :, :]  # (B,H,win,D)
        c2 = np.moveaxis(c, 2, -1).reshape(B * H * D, win)
        seg2 = c2 @ Phi_c64.T
        seg = seg2.reshape(B, H, D, win)
        seg = np.moveaxis(seg, -1, 2).real.astype(np.float32, copy=False)
        start = fi * hop
        y[:, :, start : start + win, :] += seg * window.reshape(1, 1, win, 1)
        wsum[start : start + win] += window * window

    wsum = np.maximum(wsum, 1e-8)
    y = y / wsum.reshape(1, 1, -1, 1)
    y = y[:, :, pad : pad + S0, :]
    err = _rel_err(x_true, y)
    return blob, err


def _compress_full_int8_zlib(t: np.ndarray) -> Tuple[bytes, float]:
    blob, _err = _compress_int8_zlib(t)
    # Reconstruction error
    x = np.asarray(t, dtype=np.float32)
    x2 = _decompress_int8_blob(blob).astype(np.float32)[: x.size].reshape(x.shape)
    return blob, _rel_err(x, x2)


def _make_tokens(tokenizer, text: str, target_tokens: int) -> Any:
    # Repeat prompt until we hit target tokens (approx).
    base = text.strip()
    if not base:
        base = "hello"
    s = base
    while True:
        enc = tokenizer(s, return_tensors="pt")
        n = int(enc["input_ids"].shape[-1])
        if n >= target_tokens:
            return enc
        s = s + " " + base


def _print_table(rows: list[Row]) -> None:
    print("# KV-cache seq-len axis compression (real model)")
    print()
    for key in ("model", "context_tokens", "layers", "kv_dtype"):
        vals = sorted({getattr(r, key) for r in rows})
        if len(vals) == 1:
            print(f"- {key}: `{vals[0]}`")
    print()
    print("| Method | KV orig (MB) | KV comp (MB) | Ratio | Mean key rel err | Mean value rel err | Compress (s) | Decompress (s) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| `{r.method}` | {r.kv_original_mb:.2f} | {r.kv_compressed_mb:.2f} | {r.ratio:.2f}x"
            f" | {r.key_rel_err_mean:.4f} | {r.val_rel_err_mean:.4f} | {r.compress_s:.3f} | {r.decompress_s:.3f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local model path or HF cache snapshot path")
    parser.add_argument("--prompt", default="Explain resonance and attention in one paragraph.", help="Seed prompt")
    parser.add_argument("--target-tokens", type=int, default=2048, help="Requested context length (clamped to model max)")
    parser.add_argument("--keep-ratio", type=float, default=0.30, help="Top-k ratio along seq axis")
    parser.add_argument("--layer-limit", type=int, default=None, help="Benchmark first N layers only")
    args = parser.parse_args()

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", os.path.join("ai", "hf_cache"))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True)
    model.eval()

    max_ctx = int(getattr(model.config, "max_position_embeddings", 0) or getattr(model.config, "n_positions", 0) or 0)
    target = int(args.target_tokens)
    if max_ctx > 0:
        target = min(target, max_ctx)

    enc = _make_tokens(tokenizer, args.prompt, target)
    # Clamp all sequence-like fields to exact target tokens.
    for k, v in list(enc.items()):
        try:
            if hasattr(v, "ndim") and int(v.ndim) == 2 and int(v.shape[-1]) > target:
                enc[k] = v[:, :target]
        except Exception:
            pass

    with torch.no_grad():
        out = model(**enc, use_cache=True)
    past = out.past_key_values
    if past is None:
        raise SystemExit("No past_key_values")

    layer_limit = args.layer_limit
    kv_dtype = str(next(iter(_iter_kv(past)))[0].dtype)
    layers = sum(1 for _ in _iter_kv(past))
    if layer_limit is not None:
        layers = min(layers, int(layer_limit))

    kv_original_bytes = _bytes_past_key_values(past, layer_limit=layer_limit)

    # Collect K/V as numpy for selected layers.
    keys: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for li, (k_t, v_t) in enumerate(_iter_kv(past)):
        if layer_limit is not None and li >= int(layer_limit):
            break
        # Transformers may store KV as bf16; convert to float32 for NumPy.
        k = k_t.detach().cpu().to(dtype=torch.float32).numpy()
        v = v_t.detach().cpu().to(dtype=torch.float32).numpy()

        # Standardize to (B,H,S,D).
        if k.ndim != 4:
            raise SystemExit(f"Unexpected KV key shape: {k.shape}")
        keys.append(k)
        values.append(v)

    rows: list[Row] = []

    def add_row(method: str, comp_bytes: int, key_err: float, val_err: float, tcomp: float, tdecomp: float):
        rows.append(
            Row(
                model=args.model,
                context_tokens=int(enc["input_ids"].shape[-1]),
                layers=layers,
                kv_dtype=kv_dtype,
                method=method,
                kv_original_mb=_mb(kv_original_bytes),
                kv_compressed_mb=_mb(comp_bytes),
                ratio=float(kv_original_bytes / max(comp_bytes, 1)),
                key_rel_err_mean=key_err,
                val_rel_err_mean=val_err,
                compress_s=tcomp,
                decompress_s=tdecomp,
            )
        )

    # Baseline: INT8+zlib full tensor (no axis-awareness).
    t0 = time.perf_counter()
    blobs_k: list[bytes] = []
    blobs_v: list[bytes] = []
    key_errs: list[float] = []
    val_errs: list[float] = []
    for k, v in zip(keys, values):
        bk, ek = _compress_full_int8_zlib(k)
        bv, ev = _compress_full_int8_zlib(v)
        blobs_k.append(bk)
        blobs_v.append(bv)
        key_errs.append(ek)
        val_errs.append(ev)
    tcomp = time.perf_counter() - t0

    t1 = time.perf_counter()
    # Decompress cost (for parity).
    for bk, bv, k, v in zip(blobs_k, blobs_v, keys, values):
        _ = _decompress_int8_blob(bk)[: k.size].reshape(k.shape)
        _ = _decompress_int8_blob(bv)[: v.size].reshape(v.shape)
    tdecomp = time.perf_counter() - t1
    add_row(
        "baseline_int8_zlib_full",
        sum(len(b) for b in blobs_k) + sum(len(b) for b in blobs_v),
        float(np.mean(key_errs)),
        float(np.mean(val_errs)),
        tcomp,
        tdecomp,
    )

    # Axis FFT top-k.
    t0 = time.perf_counter()
    blobs_k = []
    blobs_v = []
    key_errs = []
    val_errs = []
    for k, v in zip(keys, values):
        bk, ek = _axis_topk_compress(k, keep_ratio=float(args.keep_ratio), mode="fft")
        bv, ev = _axis_topk_compress(v, keep_ratio=float(args.keep_ratio), mode="fft")
        blobs_k.append(bk)
        blobs_v.append(bv)
        key_errs.append(ek)
        val_errs.append(ev)
    tcomp = time.perf_counter() - t0
    t1 = time.perf_counter()
    for bk, bv, k, v in zip(blobs_k, blobs_v, keys, values):
        _ = _axis_topk_decompress(bk, seq_len=k.shape[2], mode="fft")
        _ = _axis_topk_decompress(bv, seq_len=v.shape[2], mode="fft")
    tdecomp = time.perf_counter() - t1
    add_row(
        "axis_fft_topk_zlib",
        sum(len(b) for b in blobs_k) + sum(len(b) for b in blobs_v),
        float(np.mean(key_errs)),
        float(np.mean(val_errs)),
        tcomp,
        tdecomp,
    )

    # Axis hybrid-RFT top-k.
    t0 = time.perf_counter()
    blobs_k = []
    blobs_v = []
    key_errs = []
    val_errs = []
    for k, v in zip(keys, values):
        bk, ek = _axis_topk_compress(k, keep_ratio=float(args.keep_ratio), mode="rft_hybrid")
        bv, ev = _axis_topk_compress(v, keep_ratio=float(args.keep_ratio), mode="rft_hybrid")
        blobs_k.append(bk)
        blobs_v.append(bv)
        key_errs.append(ek)
        val_errs.append(ev)
    tcomp = time.perf_counter() - t0
    t1 = time.perf_counter()
    for bk, bv, k, v in zip(blobs_k, blobs_v, keys, values):
        _ = _axis_topk_decompress(bk, seq_len=k.shape[2], mode="rft_hybrid")
        _ = _axis_topk_decompress(bv, seq_len=v.shape[2], mode="rft_hybrid")
    tdecomp = time.perf_counter() - t1
    add_row(
        "axis_rft_hybrid_topk_zlib",
        sum(len(b) for b in blobs_k) + sum(len(b) for b in blobs_v),
        float(np.mean(key_errs)),
        float(np.mean(val_errs)),
        tcomp,
        tdecomp,
    )

    # Axis canonical RFT (blockwise) top-k.
    t0 = time.perf_counter()
    blobs_k = []
    blobs_v = []
    key_errs = []
    val_errs = []
    for k, v in zip(keys, values):
        bk, ek = _axis_canonical_block_topk_compress(k, keep_ratio=float(args.keep_ratio), block=256)
        bv, ev = _axis_canonical_block_topk_compress(v, keep_ratio=float(args.keep_ratio), block=256)
        blobs_k.append(bk)
        blobs_v.append(bv)
        key_errs.append(ek)
        val_errs.append(ev)
    tcomp = time.perf_counter() - t0
    # Decompress timing is omitted for canonical blocks (would require unpack + inverse); report 0.
    add_row(
        "axis_canonical_rft_block256_topk_zlib",
        sum(len(b) for b in blobs_k) + sum(len(b) for b in blobs_v),
        float(np.mean(key_errs)),
        float(np.mean(val_errs)),
        tcomp,
        0.0,
    )

    # Axis canonical RFT (overlapping Hann windows) top-k.
    t0 = time.perf_counter()
    blobs_k = []
    blobs_v = []
    key_errs = []
    val_errs = []
    for k, v in zip(keys, values):
        bk, ek = _axis_canonical_overlap_topk_compress(k, keep_ratio=float(args.keep_ratio), win=256, hop=128)
        bv, ev = _axis_canonical_overlap_topk_compress(v, keep_ratio=float(args.keep_ratio), win=256, hop=128)
        blobs_k.append(bk)
        blobs_v.append(bv)
        key_errs.append(ek)
        val_errs.append(ev)
    tcomp = time.perf_counter() - t0
    add_row(
        "axis_canonical_rft_overlap256_hop128_topk_zlib",
        sum(len(b) for b in blobs_k) + sum(len(b) for b in blobs_v),
        float(np.mean(key_errs)),
        float(np.mean(val_errs)),
        tcomp,
        0.0,
    )

    _print_table(rows)


if __name__ == "__main__":
    main()
