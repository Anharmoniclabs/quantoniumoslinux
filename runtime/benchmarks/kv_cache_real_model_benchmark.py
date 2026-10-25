#!/usr/bin/env python3
"""
KV-cache compression benchmark on a real (non-synthetic) HF model.

Measures KV-cache size and compression ratio using:
- Baseline: raw `past_key_values` tensors (typically FP32 on CPU)
- RFTMW:    RFT compression (top-k in RFT domain + quantize + zlib)
- INT8:     INT8 quantize + zlib

Default model is the repo-local DialoGPT cache:
  ai/models/huggingface/DialoGPT-small
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np

from quantonium_os_src.engine.rftmw_memory import (
    RFTMWMemoryLayer,
    CompressionMethod,
)


@dataclass(frozen=True)
class MethodResult:
    method: str
    compressed_bytes: int
    ratio: float
    compress_s: float
    decompress_s: float
    key_rel_err_mean: float
    val_rel_err_mean: float


def _iter_kv(past_key_values: Any):
    """Yield (K,V) tensors per layer for both legacy tuples and DynamicCache."""
    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            yield layer.keys, layer.values
        return
    for layer in past_key_values:
        yield layer[0], layer[1]


def _bytes_tensor(t) -> int:
    return int(t.numel() * t.element_size())


def _bytes_past_key_values(past_key_values: Any) -> int:
    total = 0
    for k, v in _iter_kv(past_key_values):
        total += _bytes_tensor(k) + _bytes_tensor(v)
    return int(total)


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) + 1e-15)
    return float(np.linalg.norm(a - b) / denom)


def _compress_one_method(
    *,
    past_key_values: Any,
    kv_keep_ratio: float,
    method: CompressionMethod,
) -> MethodResult:
    mem = RFTMWMemoryLayer(kv_keep_ratio=kv_keep_ratio)
    total_original = _bytes_past_key_values(past_key_values)

    key_errs: list[float] = []
    val_errs: list[float] = []

    # Compress
    t0 = time.perf_counter()
    for layer_idx, (k_t, v_t) in enumerate(_iter_kv(past_key_values)):
        k_t = k_t.detach().cpu()
        v_t = v_t.detach().cpu()
        k = k_t.numpy()
        v = v_t.numpy()
        mem.compress_kv(layer_idx=layer_idx, keys=k, values=v, method=method)
    compress_s = time.perf_counter() - t0

    # Decompress + error
    t1 = time.perf_counter()
    for layer_idx, (k_t, v_t) in enumerate(_iter_kv(past_key_values)):
        k_t = k_t.detach().cpu()
        v_t = v_t.detach().cpu()
        k = k_t.numpy()
        v = v_t.numpy()
        k2, v2 = mem.decompress_kv(layer_idx, k.shape, v.shape)
        key_errs.append(_rel_err(k, k2))
        val_errs.append(_rel_err(v, v2))
    decompress_s = time.perf_counter() - t1

    compressed_bytes = int(mem._report.kv_compressed_bytes)
    ratio = float(total_original / max(compressed_bytes, 1))

    return MethodResult(
        method=method.name,
        compressed_bytes=compressed_bytes,
        ratio=ratio,
        compress_s=compress_s,
        decompress_s=decompress_s,
        key_rel_err_mean=float(np.mean(key_errs)) if key_errs else 0.0,
        val_rel_err_mean=float(np.mean(val_errs)) if val_errs else 0.0,
    )


def _format_mb(num_bytes: int) -> float:
    return float(num_bytes) / 1024.0 / 1024.0


def _print_markdown_table(
    *,
    model_id: str,
    prompt: str,
    prompt_tokens: int,
    layers: int,
    kv_dtype: str,
    kv_original_bytes: int,
    results: Iterable[MethodResult],
) -> None:
    print("# KV-cache compression benchmark (real model)")
    print()
    print(f"- Model: `{model_id}`")
    print(f"- Prompt tokens: `{prompt_tokens}`")
    print(f"- Layers: `{layers}`")
    print(f"- KV dtype: `{kv_dtype}`")
    print(f"- Baseline KV size: `{_format_mb(kv_original_bytes):.2f} MB`")
    print()
    print("## Results")
    print()
    print("| Method | KV compressed (MB) | Ratio | Mean key rel err | Mean value rel err | Compress (s) | Decompress (s) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        print(
            f"| `{r.method}` | {_format_mb(r.compressed_bytes):.2f} | {r.ratio:.2f}x"
            f" | {r.key_rel_err_mean:.4f} | {r.val_rel_err_mean:.4f}"
            f" | {r.compress_s:.3f} | {r.decompress_s:.3f} |"
        )
    print()
    print("## Prompt")
    print()
    print("```")
    print(prompt)
    print("```")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.path.join("ai", "models", "huggingface", "DialoGPT-small"),
        help="Local model path or HF model id (default: repo-local DialoGPT-small)",
    )
    parser.add_argument(
        "--prompt",
        default="Explain resonance and Fourier transforms in one short paragraph.",
        help="Prompt used to build the initial KV-cache.",
    )
    parser.add_argument(
        "--kv-keep-ratio",
        type=float,
        default=0.30,
        help="KV keep ratio for RFT/INT8 compression (default: 0.30).",
    )
    args = parser.parse_args()

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", os.path.join("ai", "hf_cache"))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)
    model.eval()

    enc = tokenizer(args.prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc, use_cache=True)

    past = out.past_key_values
    if past is None:
        raise SystemExit("Model did not return past_key_values; cannot benchmark KV-cache.")

    kv_original_bytes = _bytes_past_key_values(past)
    layers = sum(1 for _ in _iter_kv(past))
    first_k, _first_v = next(iter(_iter_kv(past)))
    kv_dtype = str(first_k.dtype)
    prompt_tokens = int(enc["input_ids"].shape[-1])

    rft = _compress_one_method(
        past_key_values=past,
        kv_keep_ratio=args.kv_keep_ratio,
        method=CompressionMethod.RFT,
    )
    int8 = _compress_one_method(
        past_key_values=past,
        kv_keep_ratio=args.kv_keep_ratio,
        method=CompressionMethod.INT8_ZLIB,
    )

    _print_markdown_table(
        model_id=model_id,
        prompt=args.prompt,
        prompt_tokens=prompt_tokens,
        layers=layers,
        kv_dtype=kv_dtype,
        kv_original_bytes=kv_original_bytes,
        results=[rft, int8],
    )


if __name__ == "__main__":
    main()
