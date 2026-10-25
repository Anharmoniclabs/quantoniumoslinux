#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
RoPE Variant Probe — Find the best RFT variant for RoPE-based models.
=====================================================================

Loads a RoPE model (default: TinyLlama-1.1B), extracts real KV-cache
vectors from the head_dim axis, and probes ALL registered RFT variants
(including the 6 newly-registered patent variants designed for rotary
positional embeddings) to find which concentrates the most energy into
the fewest coefficients (lowest K_90).

This answers the question: which phi-basis variant best handles the
rotary phase structure that defeated op_rft_golden on TinyLlama?

Usage:
    python benchmarks/probe_rope_variants.py
    python benchmarks/probe_rope_variants.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python benchmarks/probe_rope_variants.py --model gpt2  # comparison baseline
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

# Windows console encoding safety
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("ERROR: pip install torch transformers")
    sys.exit(1)

from algorithms.rft.variants.registry import VARIANTS
from algorithms.rft.variants.variant_scheduler import build_variant_basis

# ── Constants ────────────────────────────────────────────────────────────────

CONTEXT_LEN = 80

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


def extract_kv_vectors(model_id: str, n_sample: int = 50):
    """Load model, run context, extract n_sample head_dim vectors from KV cache."""
    print(f"Loading {model_id}...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    ids = tokenizer(TEST_PASSAGE, return_tensors="pt")["input_ids"]
    context_ids = ids[:, :CONTEXT_LEN]

    with torch.no_grad():
        out = model(context_ids, use_cache=True)

    past_kv = out.past_key_values

    # Extract KV pairs
    if hasattr(past_kv, 'key_cache'):
        kv_pairs = list(zip(past_kv.key_cache, past_kv.value_cache))
    elif hasattr(past_kv, 'layers'):
        kv_pairs = [(l.keys, l.values) for l in past_kv.layers]
    else:
        kv_pairs = list(past_kv)

    kv_shape = kv_pairs[0][0].shape
    head_dim = kv_shape[-1]
    n_layers = len(kv_pairs)
    n_heads = kv_shape[1]

    print(f"  KV layers: {n_layers}, heads: {n_heads}, head_dim: {head_dim}")
    print(f"  KV shape: {kv_shape}")

    # Sample head_dim vectors from keys across layers/heads/positions
    vectors = []
    for layer_idx in range(n_layers):
        k_tensor = kv_pairs[layer_idx][0].cpu().numpy()
        for h in range(n_heads):
            for s in range(min(kv_shape[2], CONTEXT_LEN)):
                vectors.append(k_tensor[0, h, s, :])
                if len(vectors) >= n_sample:
                    break
            if len(vectors) >= n_sample:
                break
        if len(vectors) >= n_sample:
            break

    print(f"  Extracted {len(vectors)} head_dim vectors for probing")
    return np.array(vectors), head_dim


def probe_all_variants(vectors: np.ndarray, head_dim: int, energy_threshold: float = 0.90):
    """Probe every registered variant on sampled KV vectors. Return sorted results."""
    print(f"\nProbing {len(VARIANTS)} variants on {len(vectors)} vectors (head_dim={head_dim})...")
    print(f"  Energy threshold: K_{int(energy_threshold*100)}")
    print()

    results = {}

    for vid in sorted(VARIANTS.keys()):
        try:
            t0 = time.perf_counter()
            basis = build_variant_basis(vid, head_dim)
            elapsed = time.perf_counter() - t0

            k_scores = []
            for vec in vectors:
                signal = vec.astype(np.complex128)
                coeffs = basis.conj().T @ signal
                mag_sq = np.abs(coeffs) ** 2
                total_energy = mag_sq.sum()
                if total_energy < 1e-30:
                    k_scores.append(head_dim)
                    continue
                sorted_energy = np.sort(mag_sq)[::-1]
                cumulative = np.cumsum(sorted_energy) / total_energy
                k_val = int(np.searchsorted(cumulative, energy_threshold) + 1)
                k_scores.append(k_val)

            avg_k90 = np.mean(k_scores)
            median_k90 = np.median(k_scores)
            min_k90 = np.min(k_scores)
            max_k90 = np.max(k_scores)
            sparsity = avg_k90 / head_dim

            results[vid] = {
                "avg_k90": avg_k90,
                "median_k90": median_k90,
                "min_k90": min_k90,
                "max_k90": max_k90,
                "sparsity": sparsity,
                "build_time": elapsed,
                "is_patent": vid.startswith("pat_"),
            }

            marker = " *** PATENT (RoPE-targeted)" if vid.startswith("pat_") else ""
            print(f"  {vid:<28s}  K_90={avg_k90:>5.1f}  sparsity={sparsity:.3f}  "
                  f"[{min_k90}-{max_k90}]  {elapsed:.3f}s{marker}")

        except Exception as exc:
            print(f"  {vid:<28s}  ERROR: {exc}")
            results[vid] = {"avg_k90": head_dim, "error": str(exc)}

    return results


def print_rankings(results: dict, head_dim: int, top_n: int = 15):
    """Print sorted rankings."""
    valid = {k: v for k, v in results.items() if "error" not in v}
    ranked = sorted(valid.items(), key=lambda x: x[1]["avg_k90"])

    print()
    print("=" * 80)
    print(f"  VARIANT RANKINGS BY K_90 (lower = better, head_dim={head_dim})")
    print("=" * 80)
    print(f"  {'Rank':<5s} {'Variant':<30s} {'Avg K_90':>8s} {'Sparsity':>9s} "
          f"{'Range':>10s} {'Type':>10s}")
    print(f"  {'-'*5} {'-'*30} {'-'*8} {'-'*9} {'-'*10} {'-'*10}")

    for i, (vid, info) in enumerate(ranked[:top_n], 1):
        vtype = "PATENT" if info["is_patent"] else "standard"
        print(f"  {i:<5d} {vid:<30s} {info['avg_k90']:>8.1f} {info['sparsity']:>9.3f} "
              f"  [{info['min_k90']:.0f}-{info['max_k90']:.0f}]  {vtype:>10s}")

    winner = ranked[0]
    print()
    print(f"  WINNER: {winner[0]}  (K_90={winner[1]['avg_k90']:.1f}, "
          f"sparsity={winner[1]['sparsity']:.3f})")

    # Patent variant rankings
    patent_ranked = [(k, v) for k, v in ranked if v["is_patent"]]
    if patent_ranked:
        print()
        print("  PATENT VARIANT RANKINGS (RoPE-targeted):")
        for i, (vid, info) in enumerate(patent_ranked, 1):
            overall_rank = next(j for j, (k, _) in enumerate(ranked, 1) if k == vid)
            print(f"    {i}. {vid:<28s}  K_90={info['avg_k90']:>5.1f}  "
                  f"(overall rank #{overall_rank}/{len(ranked)})")

    # Best patent vs best standard
    standard_ranked = [(k, v) for k, v in ranked if not v["is_patent"]]
    if patent_ranked and standard_ranked:
        best_pat = patent_ranked[0]
        best_std = standard_ranked[0]
        print()
        if best_pat[1]["avg_k90"] < best_std[1]["avg_k90"]:
            diff = best_std[1]["avg_k90"] - best_pat[1]["avg_k90"]
            print(f"  >>> PATENT VARIANT WINS: {best_pat[0]} beats {best_std[0]} "
                  f"by {diff:.1f} K_90 points")
        else:
            diff = best_pat[1]["avg_k90"] - best_std[1]["avg_k90"]
            print(f"  >>> STANDARD VARIANT WINS: {best_std[0]} beats best patent "
                  f"{best_pat[0]} by {diff:.1f} K_90 points")
            print(f"      Best patent variant for benchmark: {best_pat[0]}")

    return ranked[0][0]  # return winner variant ID


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoPE Variant Probe")
    parser.add_argument("--model", type=str,
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="HuggingFace model ID")
    parser.add_argument("--n-sample", type=int, default=50,
                        help="Number of head_dim vectors to sample")
    parser.add_argument("--energy", type=float, default=0.90,
                        help="Energy threshold for K metric (default: 0.90)")
    args = parser.parse_args()

    vectors, head_dim = extract_kv_vectors(args.model, n_sample=args.n_sample)
    results = probe_all_variants(vectors, head_dim, energy_threshold=args.energy)
    winner = print_rankings(results, head_dim)

    print()
    print(f"  Next step: run perplexity benchmark with winner")
    print(f"  python benchmarks/kv_cache_perplexity_benchmark.py "
          f"--model {args.model} --retain 0.40")
