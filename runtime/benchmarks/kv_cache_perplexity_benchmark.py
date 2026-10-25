#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
KV-Cache Perplexity Benchmark — Variant-Scheduled RFT vs DCT vs FFT
====================================================================

Runs GPT-2 (124M, Apache 2.0) and measures perplexity degradation when
the KV cache is compressed at various retention levels using:

  1. Baseline     — uncompressed KV cache
  2. DCT          — hard-block DCT top-k (the standard)
  3. FFT          — hard-block FFT top-k
  4. RFT Canonical — phi-grid Gram-normalized basis
  5. RFT Auto     — variant-scheduled (probes 27 variants per head-dim,
                    picks lowest K_90)

For each method, sweeps retention fractions [0.20, 0.40, 0.60, 0.80]
and reports: perplexity, delta%, MSE, compression ratio.

Usage:
    python benchmarks/kv_cache_perplexity_benchmark.py
    python benchmarks/kv_cache_perplexity_benchmark.py --retain 0.40
    python benchmarks/kv_cache_perplexity_benchmark.py --context 80 --target 30
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

# Windows console encoding safety
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Optional deps ────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn.functional as F
    from transformers import DynamicCache, AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("ERROR: pip install torch transformers")
    sys.exit(1)

from scipy.fft import dct, idct

# RFT imports
from algorithms.rft.core.resonant_fourier_transform import (
    PHI,
    rft_basis_matrix,
)
from algorithms.rft.variants.registry import VARIANTS
from algorithms.rft.variants.variant_scheduler import (
    variant_probe,
    build_variant_basis,
    FAST_CANDIDATES,
)
from quantonium_os_src.engine.three_distance_router import (
    analyze_gap_structure,
    three_distance_compress,
)

# ── Constants ────────────────────────────────────────────────────────────────
CONTEXT_LEN = 80
TARGET_LEN = 30
RETAIN_FRACS = [0.20, 0.40, 0.60, 0.80]
_VARIANT_OVERRIDE = None  # set via --variant CLI flag (str or None)
MODEL_NAMES = {
    "gpt2": "GPT-2 (124M)",
    "gpt2-medium": "GPT-2 Medium (355M)",
    "gpt2-large": "GPT-2 Large (774M)",
    "gpt2-xl": "GPT-2 XL (1.5B)",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "TinyLlama 1.1B Chat",
    "microsoft/phi-2": "Phi-2 (2.7B)",
}

TEST_PASSAGE = (
    "The attention mechanism in transformer models computes queries, keys, "
    "and values from the input sequence. Each head independently attends to "
    "different aspects of the representation. The key matrix captures what "
    "information each token offers, while the value matrix carries the actual "
    "content. Long-context inference requires storing all past key-value pairs, "
    "which grows linearly with sequence length. This memory bottleneck has "
    "motivated research into KV-cache compression techniques. Methods such as "
    "token eviction, quantization, and spectral compression have been proposed "
    "to reduce the memory footprint without significant degradation in quality."
)


# ── Basis builders (cached) ──────────────────────────────────────────────────

_basis_cache = {}


def _get_rft_canonical_basis(n: int) -> np.ndarray:
    """Gram-normalized phi-grid basis (canonical RFT)."""
    key = ("canonical", n)
    if key not in _basis_cache:
        _basis_cache[key] = rft_basis_matrix(n, n, use_gram_normalization=True)
    return _basis_cache[key]


def _get_variant_basis_cached(n: int, override_variant=None) -> tuple:
    """Variant-scheduled basis — probes once per dimension size.

    If *override_variant* is set, skip probing and use that variant directly.
    """
    key = ("auto", n, override_variant)
    if key not in _basis_cache:
        if override_variant:
            variant_id = override_variant
            basis = build_variant_basis(variant_id, n)
        else:
            # Probe with a representative signal (sweep of frequencies)
            probe_signal = np.sin(2 * np.pi * np.arange(n) * PHI / n)
            variant_id, _, _ = variant_probe(probe_signal)
            basis = build_variant_basis(variant_id, n)
        _basis_cache[key] = (variant_id, basis)
    return _basis_cache[key]


# ── Compression primitives ───────────────────────────────────────────────────

def _top_k_retain(coeffs: np.ndarray, retain_frac: float) -> np.ndarray:
    """Keep top-k coefficients by magnitude, zero the rest."""
    k = max(1, int(len(coeffs) * retain_frac))
    idx = np.argsort(np.abs(coeffs))[::-1][:k]
    out = np.zeros_like(coeffs)
    out[idx] = coeffs[idx]
    return out


def compress_dct(signal: np.ndarray, retain_frac: float) -> np.ndarray:
    """DCT top-k compression."""
    c = dct(signal, norm="ortho")
    c = _top_k_retain(c, retain_frac)
    return idct(c, norm="ortho")


def compress_fft(signal: np.ndarray, retain_frac: float) -> np.ndarray:
    """FFT top-k compression."""
    n = len(signal)
    c = np.fft.fft(signal) / np.sqrt(n)
    c = _top_k_retain(c, retain_frac)
    return np.fft.ifft(c * np.sqrt(n)).real


def compress_rft_canonical(signal: np.ndarray, retain_frac: float) -> np.ndarray:
    """RFT canonical (phi-grid Gram-normalized) top-k compression."""
    n = len(signal)
    U = _get_rft_canonical_basis(n)
    c = U.conj().T @ signal.astype(complex)
    c = _top_k_retain(c, retain_frac)
    return np.real(U @ c)


def compress_rft_auto(signal: np.ndarray, retain_frac: float) -> np.ndarray:
    """RFT variant-scheduled top-k compression."""
    n = len(signal)
    variant_id, basis = _get_variant_basis_cached(n, _VARIANT_OVERRIDE)
    c = basis.conj().T @ signal.astype(complex)
    c = _top_k_retain(c, retain_frac)
    return np.real(basis @ c)


def compress_rft_3dist(signal: np.ndarray, retain_frac: float) -> np.ndarray:
    """RFT + Three-Distance spectral router (RFTMW Decision 3).

    Uses the variant-scheduled basis (Decision 2) but replaces flat
    top-k with the three_distance_router engine layer.  The router
    partitions coefficients into spectral zones defined by the
    Steinhaus-Sos gap structure and allocates retention budget
    proportional to per-zone energy density.

    This is the same geometric property that yields 47-80% cross-window
    leakage reduction in positional encoding -- applied to coefficient
    retention during head-dim compression.
    """
    n = len(signal)
    variant_id, basis = _get_variant_basis_cached(n, _VARIANT_OVERRIDE)
    gap_info = analyze_gap_structure(n)
    reconstructed, _ = three_distance_compress(signal, basis, retain_frac, gap_info)
    return reconstructed


METHODS = {
    "dct": compress_dct,
    "fft": compress_fft,
    "rft_canonical": compress_rft_canonical,
    "rft_auto": compress_rft_auto,
    "rft_3dist": compress_rft_3dist,
}


# ── KV cache compression ────────────────────────────────────────────────────

def compress_kv_cache(past_kv, method_fn, retain_frac: float):
    """Compress all KV cache layers along the head_dim dimension."""
    new_cache = DynamicCache()

    # Handle both DynamicCache (has .layers) and tuple-of-tuples format
    if hasattr(past_kv, 'key_cache'):
        # Modern DynamicCache (transformers >= 4.38)
        n_layers = len(past_kv.key_cache)
        for layer_idx in range(n_layers):
            key_tensor = past_kv.key_cache[layer_idx]
            val_tensor = past_kv.value_cache[layer_idx]
            new_key = _compress_tensor(key_tensor, method_fn, retain_frac)
            new_val = _compress_tensor(val_tensor, method_fn, retain_frac)
            new_cache.update(new_key, new_val, layer_idx)
    elif hasattr(past_kv, 'layers'):
        for layer_idx, layer in enumerate(past_kv.layers):
            new_key = _compress_tensor(layer.keys, method_fn, retain_frac)
            new_val = _compress_tensor(layer.values, method_fn, retain_frac)
            new_cache.update(new_key, new_val, layer_idx)
    else:
        # Tuple-of-tuples fallback: ((k, v), (k, v), ...)
        for layer_idx, (k, v) in enumerate(past_kv):
            new_key = _compress_tensor(k, method_fn, retain_frac)
            new_val = _compress_tensor(v, method_fn, retain_frac)
            new_cache.update(new_key, new_val, layer_idx)

    return new_cache


def _compress_tensor(tensor, method_fn, retain_frac: float):
    """Compress a KV tensor along head_dim (last axis)."""
    arr = tensor.cpu().numpy()  # (batch, heads, seq, head_dim)
    batch, heads, seq, hdim = arr.shape
    out = arr.copy()

    for b in range(batch):
        for h in range(heads):
            for s in range(seq):
                sig = arr[b, h, s, :]  # one head_dim vector
                out[b, h, s, :] = method_fn(sig, retain_frac)

    return torch.tensor(out, dtype=tensor.dtype)


# ── Perplexity ───────────────────────────────────────────────────────────────

def compute_perplexity(model, target_ids, past_kv) -> float:
    """Perplexity of target_ids conditioned on past_kv."""
    with torch.no_grad():
        out = model(target_ids, past_key_values=past_kv, use_cache=False)
    logits = out.logits[0]
    labels = target_ids[0]
    nll = F.cross_entropy(logits[:-1], labels[1:])
    return math.exp(nll.item())


def _get_kv_pairs(past_kv):
    """Extract list of (key_tensor, val_tensor) from any cache format."""
    if hasattr(past_kv, 'key_cache'):
        return list(zip(past_kv.key_cache, past_kv.value_cache))
    elif hasattr(past_kv, 'layers'):
        return [(l.keys, l.values) for l in past_kv.layers]
    else:
        return list(past_kv)


def compute_mse(past_kv_orig, past_kv_comp) -> float:
    """Mean squared error across all KV tensors."""
    total_mse = 0.0
    total_elements = 0
    for (ok, ov), (ck, cv) in zip(_get_kv_pairs(past_kv_orig),
                                   _get_kv_pairs(past_kv_comp)):
        for orig_t, comp_t in [(ok, ck), (ov, cv)]:
            diff = (orig_t - comp_t).float()
            total_mse += (diff ** 2).sum().item()
            total_elements += diff.numel()
    return total_mse / max(total_elements, 1)


# ── Main benchmark ───────────────────────────────────────────────────────────

def run_benchmark(context_len=CONTEXT_LEN, target_len=TARGET_LEN,
                  retain_fracs=None, model_id="gpt2"):
    if retain_fracs is None:
        retain_fracs = RETAIN_FRACS

    model_label = MODEL_NAMES.get(model_id, model_id)
    print("=" * 80)
    print("  KV-CACHE PERPLEXITY BENCHMARK")
    print(f"  {model_label} + Variant-Scheduled RFT vs DCT/FFT")
    print("=" * 80)
    print()

    # Load model
    print(f"Loading {model_label}...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    # Tokenize
    ids = tokenizer(TEST_PASSAGE, return_tensors="pt")["input_ids"]
    assert ids.shape[1] >= context_len + target_len, (
        f"Passage too short: {ids.shape[1]} tokens, need {context_len + target_len}"
    )
    context_ids = ids[:, :context_len]
    target_ids = ids[:, context_len: context_len + target_len]

    # Build KV cache from context
    print(f"  Context: {context_len} tokens | Target: {target_len} tokens")
    with torch.no_grad():
        ctx_out = model(context_ids, use_cache=True)
    past_kv = ctx_out.past_key_values
    kv_pairs = _get_kv_pairs(past_kv)
    kv_shape = kv_pairs[0][0].shape
    n_layers = len(kv_pairs)
    head_dim = kv_shape[-1]
    print(f"  KV layers: {n_layers}, shape per layer: {kv_shape}")
    print(f"  head_dim: {head_dim} (compression target dimension)")

    # Pre-probe variant for head_dim
    if _VARIANT_OVERRIDE:
        print(f"\n  Variant override: {_VARIANT_OVERRIDE}")
    else:
        print(f"\n  Probing {len(VARIANTS)} RFT variants for head_dim={head_dim}...")
    variant_id, _ = _get_variant_basis_cached(head_dim, _VARIANT_OVERRIDE)
    print(f"  Selected variant: {variant_id}")

    # Baseline perplexity (deepcopy so past_kv is not mutated)
    ppl_base = compute_perplexity(model, target_ids, copy.deepcopy(past_kv))
    print(f"\n  Baseline perplexity (uncompressed): {ppl_base:.2f}")

    # Results table
    print()
    print("=" * 80)
    print(f"  {'Method':<16s} {'Retain':>7s} {'PPL':>8s} {'Delta%':>8s} "
          f"{'MSE':>10s} {'Ratio':>7s}")
    print(f"  {'-'*16} {'-'*7} {'-'*8} {'-'*8} {'-'*10} {'-'*7}")

    all_results = []

    for retain in retain_fracs:
        for method_name, method_fn in METHODS.items():
            t0 = time.perf_counter()
            comp_kv = compress_kv_cache(past_kv, method_fn, retain)
            compress_time = time.perf_counter() - t0

            mse = compute_mse(past_kv, comp_kv)
            ppl = compute_perplexity(model, target_ids, copy.deepcopy(comp_kv))
            delta_pct = (ppl - ppl_base) / ppl_base * 100
            ratio = 1.0 / max(retain, 0.01)  # approximate compression ratio

            result = {
                "method": method_name,
                "retain": retain,
                "ppl": ppl,
                "delta_pct": delta_pct,
                "mse": mse,
                "ratio": ratio,
                "time_s": compress_time,
            }
            all_results.append(result)

            print(f"  {method_name:<16s} {retain:>6.0%} {ppl:>8.2f} "
                  f"{delta_pct:>+7.1f}% {mse:>10.2e} {ratio:>6.1f}x")

        print()  # blank line between retention levels

    # Summary table
    print("=" * 80)
    print("  SUMMARY: Best perplexity per retention level")
    print("=" * 80)
    for retain in retain_fracs:
        group = [r for r in all_results if r["retain"] == retain]
        group.sort(key=lambda r: r["ppl"])
        best = group[0]
        worst = group[-1]
        print(f"  {retain:>5.0%} retain: BEST={best['method']:<16s} "
              f"PPL={best['ppl']:.2f} ({best['delta_pct']:+.1f}%) | "
              f"WORST={worst['method']:<16s} PPL={worst['ppl']:.2f} ({worst['delta_pct']:+.1f}%)")

    # RFT Auto vs DCT head-to-head
    print()
    print("=" * 80)
    print("  HEAD-TO-HEAD: RFT Auto vs DCT")
    print("=" * 80)
    for retain in retain_fracs:
        rft = next(r for r in all_results if r["method"] == "rft_auto" and r["retain"] == retain)
        dct_r = next(r for r in all_results if r["method"] == "dct" and r["retain"] == retain)
        diff = rft["ppl"] - dct_r["ppl"]
        winner = "RFT Auto" if diff < 0 else "DCT"
        print(f"  {retain:>5.0%}: RFT Auto PPL={rft['ppl']:.2f}  DCT PPL={dct_r['ppl']:.2f}  "
              f"diff={diff:+.2f}  WINNER={winner}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KV-Cache Perplexity Benchmark")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="HuggingFace model ID (default: gpt2). "
                             "Presets: gpt2, gpt2-medium, gpt2-large, gpt2-xl, "
                             "TinyLlama/TinyLlama-1.1B-Chat-v1.0, microsoft/phi-2")
    parser.add_argument("--context", type=int, default=CONTEXT_LEN)
    parser.add_argument("--target", type=int, default=TARGET_LEN)
    parser.add_argument("--retain", type=float, nargs="*", default=None,
                        help="Retention fractions (e.g. 0.20 0.40 0.60)")
    parser.add_argument("--variant", type=str, default=None,
                        help="Override RFT variant for rft_auto/rft_3dist "
                             "(e.g. pat_rft_phase_coherent, op_rft_geometric)")
    args = parser.parse_args()

    if args.variant:
        _VARIANT_OVERRIDE = args.variant  # noqa: F841 — module-level already

    retain = args.retain if args.retain else RETAIN_FRACS
    run_benchmark(model_id=args.model, context_len=args.context,
                  target_len=args.target, retain_fracs=retain)
