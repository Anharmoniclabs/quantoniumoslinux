#!/usr/bin/env python3
"""
H3 Cascade KV Buffer Test — QuantoniumOS Research
==================================================

Tests the Mimosa evict → H3 compress → retrieval buffer pipeline.

Phase 1: Extract real KV cache state from llama.cpp via llama_state_seq_save_file
Phase 2: Apply H3 cascade compression to the evicted KV data
Phase 3: Measure compression ratio, reconstruction error, and viability

This answers: "Can H3 cascade compress KV cache data well enough to store
evicted entries in a retrieval buffer for later re-injection?"
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import struct
import time
import subprocess
import tempfile
from pathlib import Path

# Import H3 cascade from the repo
from algorithms.rft.hybrids.cascade_hybrids import (
    H3HierarchicalCascade, FH5EntropyGuided, CascadeResult
)


def extract_kv_state_from_llama(model_path: str, n_ctx: int = 768,
                                 n_keep: int = 48) -> bytes:
    """
    Run llama.cpp inference and extract KV cache state at the eviction point.
    Uses llama_state_seq_save_file to dump KV state to a temp file.
    """
    # We'll use the modified C++ binary that dumps state
    dump_path = "/tmp/kv_state_dump.bin"
    bin_path = "/workspaces/llama.cpp/build/bin/llama-mimosa-kv-test"

    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Build the test first: {bin_path}")

    # For now, we'll simulate KV vectors by extracting from the state dump
    # The C++ test will dump via llama_state_seq_save_file
    return dump_path


def load_kv_dump(path: str) -> np.ndarray:
    """Load binary KV dump and reshape into vectors."""
    data = np.fromfile(path, dtype=np.float16)
    print(f"  Raw dump: {len(data)} float16 values ({len(data)*2} bytes)")
    return data.astype(np.float32)


def test_h3_on_kv_vectors(kv_data: np.ndarray, head_dim: int = 128):
    """
    Test H3 cascade compression on KV cache vectors.

    KV cache shape: [n_positions, head_dim] per layer per head.
    We test compression along head_dim (each 128-dim vector is a signal).
    Also test compression along the position dimension (across time).
    """
    print("\n" + "=" * 70)
    print("H3 CASCADE COMPRESSION ON KV CACHE VECTORS")
    print("=" * 70)

    h3 = H3HierarchicalCascade()
    fh5 = FH5EntropyGuided()

    # Test 1: Compress individual 128-dim KV vectors (head_dim axis)
    print(f"\n  Test 1: Per-vector compression (head_dim={head_dim})")
    print(f"  {'─' * 60}")

    n_vectors = len(kv_data) // head_dim
    if n_vectors == 0:
        print("  Not enough data for head_dim vectors")
        return {}

    vectors = kv_data[:n_vectors * head_dim].reshape(n_vectors, head_dim)
    print(f"  Vectors: {n_vectors} × {head_dim}")

    results = {}
    for sparsity in [0.80, 0.90, 0.95]:
        bpps, psnrs, errors = [], [], []
        t0 = time.perf_counter()

        for i in range(min(n_vectors, 500)):  # sample up to 500
            vec = vectors[i]
            result = h3.encode(vec, sparsity=sparsity)
            recon = h3.decode(result.coefficients, head_dim)
            
            bpps.append(result.bpp)
            psnrs.append(result.psnr if result.psnr != float('inf') else 80.0)

            # Relative error
            norm_orig = np.linalg.norm(vec)
            if norm_orig > 1e-8:
                rel_err = np.linalg.norm(vec - recon[:head_dim]) / norm_orig
            else:
                rel_err = 0.0
            errors.append(rel_err)

        elapsed = (time.perf_counter() - t0) * 1000
        n_tested = min(n_vectors, 500)

        # Compression ratio: original = head_dim * 2 bytes (f16)
        # compressed = bpp * head_dim / 8 bytes
        avg_bpp = np.mean(bpps)
        orig_bytes = head_dim * 2  # float16
        comp_bytes = avg_bpp * head_dim / 8
        ratio = orig_bytes / max(comp_bytes, 1)

        results[f"h3_s{int(sparsity*100)}"] = {
            "bpp": avg_bpp,
            "psnr": np.mean(psnrs),
            "rel_error": np.mean(errors),
            "max_error": np.max(errors),
            "ratio": ratio,
            "time_per_vec_us": elapsed / n_tested * 1000,
        }

        print(f"  Sparsity {sparsity:.0%}:  BPP={avg_bpp:.3f}  "
              f"PSNR={np.mean(psnrs):.1f}dB  "
              f"RelErr={np.mean(errors):.4f} (max {np.max(errors):.4f})  "
              f"Ratio={ratio:.1f}×  "
              f"{elapsed/n_tested*1000:.0f}µs/vec")

    # Test 2: Compress blocks of vectors (position axis)
    # Group 32 positions together → treat the 32×128 block as 32 signals of length 128
    print(f"\n  Test 2: Block compression (32 positions × {head_dim})")
    print(f"  {'─' * 60}")

    block_size = 32
    n_blocks = n_vectors // block_size
    if n_blocks > 0:
        for sparsity in [0.80, 0.90, 0.95]:
            total_orig = 0
            total_comp = 0
            block_errors = []

            for b in range(min(n_blocks, 50)):
                block = vectors[b*block_size:(b+1)*block_size]  # [32, 128]
                block_orig_bytes = block_size * head_dim * 2  # float16
                total_orig += block_orig_bytes

                block_comp_bytes = 0
                block_recon = np.zeros_like(block)
                for row in range(block_size):
                    result = h3.encode(block[row], sparsity=sparsity)
                    recon = h3.decode(result.coefficients, head_dim)[:head_dim]
                    block_recon[row] = recon
                    block_comp_bytes += result.bpp * head_dim / 8

                total_comp += block_comp_bytes
                rel_err = np.linalg.norm(block - block_recon) / max(np.linalg.norm(block), 1e-8)
                block_errors.append(rel_err)

            ratio = total_orig / max(total_comp, 1)
            print(f"  Sparsity {sparsity:.0%}:  "
                  f"Ratio={ratio:.1f}×  "
                  f"BlockErr={np.mean(block_errors):.4f} (max {np.max(block_errors):.4f})")
            results[f"h3_block_s{int(sparsity*100)}"] = {
                "ratio": ratio,
                "block_error": np.mean(block_errors),
                "max_block_error": np.max(block_errors),
            }
    else:
        print("  Not enough vectors for block test")

    # Test 3: Compare with simple INT8+zlib (the baseline)
    print(f"\n  Test 3: Comparison baselines")
    print(f"  {'─' * 60}")

    import zlib
    sample = vectors[:min(500, n_vectors)]
    orig_bytes_total = sample.nbytes

    # INT8 quantization
    scale = np.max(np.abs(sample), axis=1, keepdims=True)
    scale = np.where(scale == 0, 1, scale)
    int8_data = np.round(sample / scale * 127).astype(np.int8)
    int8_bytes = int8_data.tobytes() + scale.astype(np.float32).tobytes()
    int8_compressed = zlib.compress(int8_bytes, 9)

    # INT4 quantization (simulated)
    int4_vals = np.round(sample / scale * 7).clip(-8, 7).astype(np.int8)
    # Pack 2 per byte
    int4_bytes = int4_vals.tobytes()  # not actually packed, but count as half
    int4_size = len(int4_bytes) // 2 + scale.astype(np.float32).nbytes

    # Reconstruction error for INT8
    int8_recon = int8_data.astype(np.float32) / 127.0 * scale
    int8_err = np.linalg.norm(sample - int8_recon) / np.linalg.norm(sample)

    print(f"  Original (f16):     {orig_bytes_total:>10,} bytes")
    print(f"  INT8+zlib:          {len(int8_compressed):>10,} bytes  "
          f"({orig_bytes_total/len(int8_compressed):.1f}×)  err={int8_err:.4f}")
    print(f"  INT4 (estimated):   {int4_size:>10,} bytes  "
          f"({orig_bytes_total/int4_size:.1f}×)")

    results["int8_zlib"] = {
        "ratio": orig_bytes_total / len(int8_compressed),
        "rel_error": int8_err,
    }

    return results


def generate_synthetic_kv_vectors(n_layers: int = 4, n_kv_heads: int = 4,
                                    head_dim: int = 128, n_positions: int = 64,
                                    seed: int = 42) -> np.ndarray:
    """
    Generate synthetic KV vectors that statistically match real transformer KV cache.

    Real KV vectors are approximately gaussian with:
    - Mean near 0
    - Std ~0.05-0.5 depending on layer
    - Some structure along head_dim (not pure noise)
    - RoPE creates sinusoidal position-dependent patterns in K vectors
    """
    rng = np.random.RandomState(seed)

    all_vectors = []
    for layer in range(n_layers):
        layer_scale = 0.1 + 0.3 * (layer / n_layers)  # deeper layers have larger values

        for head in range(n_kv_heads):
            for pos in range(n_positions):
                # Base gaussian
                vec = rng.randn(head_dim).astype(np.float32) * layer_scale

                # Add RoPE-like structure: sinusoidal position encoding
                # RoPE rotates pairs of dimensions by position-dependent angles
                theta_base = 10000.0
                for d in range(0, head_dim, 2):
                    freq = 1.0 / (theta_base ** (d / head_dim))
                    angle = pos * freq
                    cos_a, sin_a = np.cos(angle), np.sin(angle)
                    v0, v1 = vec[d], vec[d+1] if d+1 < head_dim else 0
                    vec[d] = v0 * cos_a - v1 * sin_a
                    if d + 1 < head_dim:
                        vec[d+1] = v0 * sin_a + v1 * cos_a

                all_vectors.append(vec)

    return np.array(all_vectors, dtype=np.float32)


def test_synthetic():
    """Test H3 on synthetic KV vectors with RoPE structure."""
    print("\n" + "═" * 70)
    print("  PHASE 1: H3 on Synthetic KV Vectors (RoPE-structured)")
    print("═" * 70)

    # Generate a moderate amount: 4 layers × 4 heads × 64 positions = 1024 vectors
    kv = generate_synthetic_kv_vectors(n_layers=4, n_kv_heads=4,
                                        head_dim=128, n_positions=64)
    print(f"\n  Generated {len(kv)} synthetic KV vectors: {kv.shape}")
    print(f"  Stats: mean={kv.mean():.4f}, std={kv.std():.4f}, "
          f"min={kv.min():.3f}, max={kv.max():.3f}")

    results = test_h3_on_kv_vectors(kv.flatten(), head_dim=128)
    return results


def test_real_kv_dump():
    """
    Test H3 on real KV data dumped from llama.cpp.
    Runs the C++ test with KV dump enabled, then processes the dump.
    """
    print("\n" + "═" * 70)
    print("  PHASE 2: H3 on Real R1-7B KV Vectors")
    print("═" * 70)

    model_path = "/workspaces/quantoniumos/models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
    dump_path = "/tmp/kv_state_evicted.bin"

    if not os.path.exists(model_path):
        print("  Model not found, skipping real KV test")
        return {}

    bin_path = "/workspaces/llama.cpp/build/bin/llama-mimosa-kv-test"
    if not os.path.exists(bin_path):
        print("  Test binary not found, skipping")
        return {}

    # Run the C++ test with KV dump flag
    print(f"\n  Running llama.cpp to dump KV state at eviction point...")
    env = os.environ.copy()
    env["KV_DUMP_PATH"] = dump_path
    try:
        proc = subprocess.run(
            [bin_path, "-m", model_path, "-c", "768", "--n-keep", "48", "-n", "50"],
            env=env, capture_output=True, text=True, timeout=300
        )
        # Check for dump
        if os.path.exists(dump_path):
            size = os.path.getsize(dump_path)
            print(f"  KV state dump: {size:,} bytes ({size/1024:.1f} KB)")
            data = np.fromfile(dump_path, dtype=np.float32)
            print(f"  Loaded {len(data)} float32 values")
            results = test_h3_on_kv_vectors(data, head_dim=128)
            return results
        else:
            print("  No KV dump produced (C++ binary needs --kv-dump flag)")
            print("  Falling back to llama_state_seq approach...")
            return test_state_seq_dump(model_path)
    except subprocess.TimeoutExpired:
        print("  Timeout running test")
        return {}
    except Exception as e:
        print(f"  Error: {e}")
        return {}


def test_state_seq_dump(model_path: str) -> dict:
    """
    Use llama_state_seq_save_file to dump KV state, then analyze.
    This creates a modified binary that saves state before eviction.
    """
    dump_path = "/tmp/kv_seq_state.bin"
    bin_path = "/workspaces/llama.cpp/build/bin/llama-mimosa-kv-test"

    # First run the test — it saves state via the C++ code we'll add
    print(f"\n  Using state-seq dump approach...")
    proc = subprocess.run(
        [bin_path, "-m", model_path, "-c", "768", "--n-keep", "48", "-n", "10",
         "--dump-kv", dump_path],
        capture_output=True, text=True, timeout=300
    )

    stderr = proc.stderr
    print(f"  Test output (last 10 lines):")
    for line in stderr.strip().split('\n')[-10:]:
        print(f"    {line}")

    if os.path.exists(dump_path):
        size = os.path.getsize(dump_path)
        print(f"\n  KV state dump: {size:,} bytes ({size/1024:.1f} KB)")

        # llama state format is opaque binary — parse what we can
        data = np.fromfile(dump_path, dtype=np.float32)
        if len(data) > 128:
            print(f"  Loaded {len(data)} float32 values")
            results = test_h3_on_kv_vectors(data, head_dim=128)
            return results
        else:
            print(f"  Dump too small ({len(data)} values), falling back to float16")
            data = np.fromfile(dump_path, dtype=np.float16).astype(np.float32)
            if len(data) > 128:
                results = test_h3_on_kv_vectors(data, head_dim=128)
                return results

    print("  No valid dump found, using synthetic data only")
    return {}


def memory_savings_analysis(results: dict):
    """Calculate what the H3 buffer approach saves for R1 7B."""
    print("\n" + "═" * 70)
    print("  MEMORY SAVINGS ANALYSIS: R1 7B with H3 Eviction Buffer")
    print("═" * 70)

    n_layers = 28
    n_kv_heads = 4
    head_dim = 128

    print(f"\n  Architecture: {n_layers} layers × {n_kv_heads} KV heads × {head_dim} head_dim")

    for n_ctx in [2048, 4096]:
        kv_f16 = 2 * n_layers * n_ctx * n_kv_heads * head_dim * 2
        kv_q4 = kv_f16 * 0.25

        print(f"\n  ── n_ctx = {n_ctx} ──")
        print(f"  Baseline (F16):    {kv_f16/1024**2:>6.1f} MB")
        print(f"  Q4 KV quant:       {kv_q4/1024**2:>6.1f} MB  (−{(1-kv_q4/kv_f16)*100:.0f}%)")

        # With Mimosa: runs at n_ctx but with better quality
        # Eviction buffer: stores evicted positions compressed
        # Assume Mimosa evicts ~40% of positions over a session
        n_evicted = int(n_ctx * 0.4)
        evicted_f16 = 2 * n_layers * n_evicted * n_kv_heads * head_dim * 2

        # H3 compression ratios from test
        for key, r in results.items():
            if "ratio" in r and key.startswith("h3_s"):
                ratio = r["ratio"]
                err = r.get("rel_error", 0)
                buffer_size = evicted_f16 / ratio

                print(f"\n  Mimosa + H3 buffer ({key}):")
                print(f"    Active KV (Q4):       {kv_q4/1024**2:>6.1f} MB")
                print(f"    Eviction buffer:      {buffer_size/1024**2:>6.1f} MB  "
                      f"({n_evicted} positions × {ratio:.1f}× compressed)")
                print(f"    Total:                {(kv_q4+buffer_size)/1024**2:>6.1f} MB  "
                      f"(vs {kv_f16/1024**2:.1f} MB baseline)")
                print(f"    Net savings:          {(1-(kv_q4+buffer_size)/kv_f16)*100:>5.1f}%")
                print(f"    Reconstruction err:   {err:.4f}")
                print(f"    ↳ Can retrieve {n_evicted} evicted positions on demand")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  H3 Cascade KV Eviction Buffer Test — QuantoniumOS Research    ║")
    print("║  Mimosa evict → H3 compress → retrieval buffer pipeline       ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # Phase 1: Synthetic test (always works, baseline)
    synthetic_results = test_synthetic()

    # Phase 2: Real KV vectors (if model available)
    real_results = test_real_kv_dump()

    # Use whichever has data
    results = real_results if real_results else synthetic_results

    # Phase 3: Memory analysis
    if results:
        memory_savings_analysis(results)

    print("\n" + "═" * 70)
    print("  DONE")
    print("═" * 70)
