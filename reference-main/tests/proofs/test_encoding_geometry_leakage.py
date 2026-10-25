"""
test_encoding_geometry_leakage.py
==================================
Formal test for the 47.7% cross-window positional leakage reduction
claim (RFT vs RoPE positional encodings).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCOPE DISCLAIMER — READ BEFORE CITING THIS RESULT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This test measures ENCODING GEOMETRY ONLY.

What it measures:
  - Cross-window inner-product confusion between unit-normalised
    position vectors (RFT φ-grid vs RoPE) across an attention
    window boundary.
  - Spectral tail energy of the per-position frequency spectrum.

What it does NOT measure:
  - Model weights or trained parameters of any kind
  - Perplexity, loss, accuracy, or any language-model quality metric
  - End-to-end LLM benchmark scores (MMLU, HellaSwag, etc.)
  - Inference latency or throughput in a real transformer stack

The 47.7% figure is a property of the positional encoding design
space, computed analytically from the encoding matrices alone.
It says nothing about whether a model trained with RFT encodings
would outperform one trained with RoPE on any downstream task.

If you cite this number publicly, please lead with:
  "47.7% reduction in cross-window positional confusion
   (encoding geometry metric; no model weights evaluated)"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Verified results (sandbox run, March 25 2026, seed=42):
  Cross-window leakage : RFT wins 11/11 configs, mean 63.5% reduction
  At seq=4096, win=128 : RFT reduces leakage by 47.7% vs RoPE
  Spectral tail energy : RFT wins 11/11 configs, mean 86.2% reduction
  Runtime              : < 60 s on CPU

Reproducibility:
  seed = 42  (fixed)
  head_dim = 64
  seq_lens = [512, 1024, 2048, 4096]
  windows  = [128, 256, 512]
  No model weights required — encoding geometry only.

Source: benchmarks/kv_cache_boundary_error.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# ── Import the benchmark helpers from benchmarks/ ───────────────────────────
_BENCH_DIR = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(_BENCH_DIR))

from kv_cache_boundary_error import (  # noqa: E402
    cross_window_leakage,
    rft_encoding,
    rope_encoding,
    spectral_tail_energy,
)

# ── Scope constant (visible in -v output and CI logs) ────────────────────────
SCOPE = (
    "ENCODING GEOMETRY ONLY — no model weights, no perplexity, "
    "no LLM quality metric. 47.7% = cross-window positional confusion "
    "reduction at seq=4096/win=128 (RFT φ-grid vs RoPE, seed=42)."
)

HEAD_DIM = 64
SEED = 42
SEQ_LENS = [512, 1024, 2048, 4096]
WINDOWS = [128, 256, 512]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _configs():
    """Return all (seq_len, window) pairs actually tested."""
    return [(s, w) for s in SEQ_LENS for w in WINDOWS if w < s]


def _leakage_pair(seq_len, window):
    cr, sr = rope_encoding(seq_len, HEAD_DIM)
    cf, sf = rft_encoding(seq_len, HEAD_DIM)
    leak_r = cross_window_leakage(cr, sr, window, seed=SEED)
    leak_f = cross_window_leakage(cf, sf, window, seed=SEED)
    return leak_r, leak_f


# ── Core claim: 47.7% at seq=4096, win=128 ──────────────────────────────────

def test_scope_disclaimer():
    """Canary test: prints the scope string so it appears in -v output."""
    print(f"\n[SCOPE] {SCOPE}")
    assert SCOPE  # always passes; purpose is visibility


def test_headline_leakage_reduction_at_4096_128():
    """
    ENCODING GEOMETRY ONLY (see module docstring).

    At seq_len=4096, window=128 the RFT φ-grid encoding reduces
    cross-window positional leakage vs RoPE by ≥ 40%.
    Verified at 47.7% on March 25 2026 (seed=42).

    Threshold is set to 40% (7+ percentage points below verified)
    to absorb any floating-point platform variation while still
    being a meaningful assertion.
    """
    leak_r, leak_f = _leakage_pair(4096, 128)
    delta_pct = (leak_r - leak_f) / (leak_r + 1e-15) * 100
    print(
        f"\n[ENCODING GEOMETRY] seq=4096 win=128 | "
        f"RoPE={leak_r:.6f}  RFT={leak_f:.6f}  Δ={delta_pct:+.1f}%"
    )
    assert delta_pct >= 40.0, (
        f"Expected ≥40% leakage reduction (encoding geometry, no model weights); "
        f"got {delta_pct:.1f}%. {SCOPE}"
    )


# ── Win rate across all configs ───────────────────────────────────────────────

def test_rft_wins_all_configs():
    """
    ENCODING GEOMETRY ONLY.

    RFT must produce lower cross-window leakage than RoPE on every
    tested (seq_len, window) configuration.  Verified: 11/11.
    """
    configs = _configs()
    failures = []
    for seq_len, window in configs:
        leak_r, leak_f = _leakage_pair(seq_len, window)
        if leak_f >= leak_r:
            failures.append((seq_len, window, leak_r, leak_f))
    assert not failures, (
        f"RFT did not beat RoPE on {len(failures)}/{len(configs)} configs "
        f"(encoding geometry metric). Failures: {failures}"
    )


@pytest.mark.slow
def test_mean_leakage_reduction_above_50pct():
    """
    ENCODING GEOMETRY ONLY.

    Mean cross-window leakage reduction across all 11 configs ≥ 50%.
    Verified at 63.5% on March 25 2026.

    Marked slow because it runs the full 11-config suite.
    """
    configs = _configs()
    deltas = []
    for seq_len, window in configs:
        leak_r, leak_f = _leakage_pair(seq_len, window)
        deltas.append((leak_r - leak_f) / (leak_r + 1e-15) * 100)
    mean_delta = float(np.mean(deltas))
    print(
        f"\n[ENCODING GEOMETRY] mean Δ leakage across {len(configs)} configs: "
        f"{mean_delta:.1f}%"
    )
    assert mean_delta >= 50.0, (
        f"Expected ≥50% mean leakage reduction (encoding geometry); "
        f"got {mean_delta:.1f}%."
    )


# ── Spectral tail energy ─────────────────────────────────────────────────────

def test_rft_spectral_tail_lower_at_4096():
    """
    ENCODING GEOMETRY ONLY.

    At seq_len=4096, RFT per-position spectral tail energy must be
    lower than RoPE (verified: RFT wins 11/11 configs, mean −86.2%).
    """
    seq_len = 4096
    cr, sr = rope_encoding(seq_len, HEAD_DIM)
    cf, sf = rft_encoding(seq_len, HEAD_DIM)
    tail_r = spectral_tail_energy(cr, sr)
    tail_f = spectral_tail_energy(cf, sf)
    assert tail_f < tail_r, (
        f"Expected RFT tail energy < RoPE tail energy at seq={seq_len}; "
        f"got RFT={tail_f:.6f}, RoPE={tail_r:.6f}."
    )


# ── Parametric smoke tests ────────────────────────────────────────────────────

@pytest.mark.parametrize("seq_len,window", _configs())
def test_rft_leakage_lower_parametric(seq_len, window):
    """
    ENCODING GEOMETRY ONLY.

    Parametric version: RFT leakage < RoPE leakage for every config.
    """
    leak_r, leak_f = _leakage_pair(seq_len, window)
    assert leak_f < leak_r, (
        f"seq={seq_len} win={window}: RFT leakage ({leak_f:.6f}) >= "
        f"RoPE leakage ({leak_r:.6f}). Encoding geometry only — "
        f"no model weights involved."
    )
