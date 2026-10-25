# =============================================================================
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Luis M. Minier.
#
# KV-Cache Positional Boundary Error Benchmark
# =============================================
# Measures cross-window positional confusion (mean squared inner product
# between unit-normalized position vectors INSIDE vs OUTSIDE the attention
# window) and spectral tail energy for RFT vs RoPE positional encodings.
#
# Verified results (sandbox run, March 25, 2026):
#   Cross-window leakage:  RFT wins 11/11 configs, mean 63.5% reduction
#   At seq=4096, win=128:  RFT reduces leakage by 47.7% vs RoPE
#   Spectral tail energy:  RFT wins 11/11 configs, mean 86.2% reduction
#
# Reproducibility:
#   seed = 42 (fixed throughout)
#   head_dim = 64, seq_lens = [512, 1024, 2048, 4096], windows = [128, 256, 512]
#   No model weights required — encoding geometry only
#   Runtime: < 60s on CPU
#
# Output: benchmarks/results/kv_boundary_results.csv
# =============================================================================

import numpy as np
import pandas as pd
from pathlib import Path

PHI = (1 + np.sqrt(5)) / 2


# ── Positional Encodings ────────────────────────────────────────────────────

def rope_encoding(seq_len: int, head_dim: int):
    """
    Standard RoPE (Rotary Position Embedding).
    Su et al. (2021) — baseline for comparison.
    Returns cos, sin each of shape (seq_len, head_dim//2).
    """
    D = head_dim // 2
    theta = 1.0 / (10000 ** (np.arange(D) / D))
    t = np.arange(seq_len)
    return np.cos(np.outer(t, theta)), np.sin(np.outer(t, theta))


def rft_encoding(seq_len: int, head_dim: int):
    """
    RFT positional encoding using phi-grid frequencies.
    f_k = frac((k+1) * phi) — three-distance theorem optimal spacing.
    Returns cos, sin each of shape (seq_len, head_dim//2).
    """
    D = head_dim // 2
    freqs = ((np.arange(1, D + 1)) * PHI) % 1.0
    t = np.arange(seq_len)
    phases = 2 * np.pi * np.outer(t, freqs)
    return np.cos(phases), np.sin(phases)


# ── Metrics ─────────────────────────────────────────────────────────────────

def cross_window_leakage(cos_enc, sin_enc, window_size: int, seed: int = 42) -> float:
    """
    Cross-window positional confusion metric.

    Measures the mean squared inner product between unit-normalized position
    vectors INSIDE the attention window vs OUTSIDE. High value = positions
    across the boundary are confusable = more boundary error.

    Definition:
        leakage = E_{i in inside, j in outside} |<p_i, p_j>|^2

    where p_i are unit-normalized rows of the complex encoding matrix.

    Args:
        cos_enc, sin_enc: shape (seq_len, D) — encoding components
        window_size: attention window boundary position
        seed: RNG seed for reproducibility (300 samples from each side)

    Returns:
        Scalar leakage value in [0, 1]. Lower = less cross-boundary confusion.
    """
    enc = (cos_enc + 1j * sin_enc).astype(np.complex128)
    inside  = enc[:window_size]
    outside = enc[window_size:]
    inside  = inside  / (np.linalg.norm(inside,  axis=1, keepdims=True) + 1e-15)
    outside = outside / (np.linalg.norm(outside, axis=1, keepdims=True) + 1e-15)
    rng = np.random.default_rng(seed)
    n = min(300, len(inside), len(outside))
    ii = rng.choice(len(inside),  n, replace=False)
    io = rng.choice(len(outside), n, replace=False)
    gram = inside[ii] @ outside[io].conj().T    # (n, n)
    return float(np.mean(np.abs(gram) ** 2))


def spectral_tail_energy(cos_enc, sin_enc, tail_fraction: float = 0.25) -> float:
    """
    Per-position spectral tail energy.

    For each position vector p_i (dim D), computes the fraction of its
    FFT energy in the bottom `tail_fraction` of sorted frequencies.
    High tail energy = spectrally leaky = poor locality.

    Returns mean tail energy ratio across all positions.
    """
    enc = cos_enc + 1j * sin_enc
    seq_len, D = enc.shape
    tail_k = int(D * (1 - tail_fraction))
    tail_ratios = []
    for i in range(seq_len):
        spec = np.fft.fft(enc[i])
        energy = np.abs(spec) ** 2
        sorted_e = np.sort(energy)[::-1]
        tail_ratios.append(sorted_e[tail_k:].sum() / (sorted_e.sum() + 1e-15))
    return float(np.mean(tail_ratios))


# ── Benchmark Runner ────────────────────────────────────────────────────────

def run_benchmark(
    head_dim: int = 64,
    seq_lens: list = None,
    windows: list = None,
    seed: int = 42,
) -> pd.DataFrame:
    if seq_lens is None:
        seq_lens = [512, 1024, 2048, 4096]
    if windows is None:
        windows = [128, 256, 512]

    print(f"{'seq':>5} {'win':>4} | {'RoPE_leak':>10} {'RFT_leak':>9} | "
          f"{'Δ_leak%':>8} | {'RoPE_tail':>10} {'RFT_tail':>9} | {'Δ_tail%':>8}")
    print("-" * 78)

    rows = []
    for seq_len in seq_lens:
        for window in windows:
            if window >= seq_len:
                continue
            cr, sr = rope_encoding(seq_len, head_dim)
            cf, sf = rft_encoding(seq_len, head_dim)

            leak_r = cross_window_leakage(cr, sr, window, seed=seed)
            leak_f = cross_window_leakage(cf, sf, window, seed=seed)
            tail_r = spectral_tail_energy(cr, sr)
            tail_f = spectral_tail_energy(cf, sf)

            delta_leak = (leak_r - leak_f) / (leak_r + 1e-15) * 100
            delta_tail = (tail_r - tail_f) / (tail_r + 1e-15) * 100

            rows.append(dict(
                seq_len=seq_len, window=window,
                rope_leak=leak_r, rft_leak=leak_f,
                rope_tail=tail_r, rft_tail=tail_f,
                delta_leak_pct=delta_leak,
                delta_tail_pct=delta_tail,
            ))
            print(f"{seq_len:5d} {window:4d} | {leak_r:10.4f} {leak_f:9.4f} | "
                  f"{delta_leak:+8.1f}% | {tail_r:10.4f} {tail_f:9.4f} | {delta_tail:+8.1f}%")

    df = pd.DataFrame(rows)

    print(f"\n── Cross-window leakage: RFT vs RoPE ───")
    print(f"  RFT wins (lower leakage): {(df.delta_leak_pct > 0).sum()}/{len(df)} configs")
    print(f"  Mean reduction : {df.delta_leak_pct.mean():.1f}%")
    print(f"  Range          : {df.delta_leak_pct.min():.1f}% to {df.delta_leak_pct.max():.1f}%")
    print(f"  At seq=4096, win=128: {df[(df.seq_len==4096) & (df.window==128)].delta_leak_pct.values[0]:.1f}%")

    print(f"\n── Spectral tail energy: RFT vs RoPE ───")
    print(f"  RFT wins (lower tail): {(df.delta_tail_pct > 0).sum()}/{len(df)} configs")
    print(f"  Mean reduction : {df.delta_tail_pct.mean():.1f}%")

    return df


if __name__ == "__main__":
    df = run_benchmark()
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "kv_boundary_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
