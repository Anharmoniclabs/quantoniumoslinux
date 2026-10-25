"""
test_model_weights_fidelity.py
================================
Model weight fidelity under STRT and DCT spectral compression.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCOPE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
What this measures:
  - Roundtrip reconstruction fidelity (relative L2 error) when GPT-2-scale
    weight matrices are compressed row-by-row using STRT and DCT at varying
    coefficient retention rates.

What this does NOT measure:
  - Downstream task accuracy (perplexity, GLUE, etc.) — see
    test_perplexity_kvcache.py and tests/benchmarks/downstream_performance.py.
  - Positional encoding geometry — see test_encoding_geometry_leakage.py.
  - Trained model weights (uses synthetic matrices at GPT-2 dimensions).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HONEST RESULT — READ BEFORE CITING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For MODEL WEIGHT compression, DCT BEATS STRT uniformly.

Verified results (March 26 2026, seed=42, block=32, hop=16):
  Retain  STRT rel-err (best)  DCT rel-err   Winner
  ──────  ───────────────────  ───────────   ──────
  80%     0.239 (ffn_768x3072)   0.087       DCT
  60%     0.435 (ffn_768x3072)   0.207       DCT
  40%     0.650 (ffn_768x3072)   0.399       DCT

Why this is EXPECTED (not a bug):
  GPT-2 weights are initialised ≈ N(0, 0.02) — high-entropy, near-white-noise
  distributions with no quasi-periodic structure.  This is exactly the signal
  class NON_CLAIMS.md documents as "no advantage" for STRT/RFT:
    "White noise: ❌ No advantage"
    "High-entropy random: ❌ No advantage"
  DCT's cosine basis is better matched to arbitrary smooth signals.

Where STRT DOES win:
  ✅ KV-cache activations in deep transformer layers (layers 10-11, V-cache):
     STRT wins 8/8 (100%) on boundary error — see test_strt_kvcache_real_activations.py
  ✅ Positional encoding cross-window confusion:
     STRT wins 11/11 configs, mean 63.5% reduction — see test_encoding_geometry_leakage.py
  ✅ Golden-ratio quasi-periodic signals (in-family signal class).

Do NOT claim STRT improves weight compression.  It does not.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Why synthetic weights:
  GPT-2 weights require torch/transformers (network access).
  N(0, 0.02) synthetic matrices at the correct dimensions are statistically
  equivalent for fidelity measurement and run without GPU or network.

No torch or transformers required — numpy + scipy only.
"""

import math

import numpy as np
import pytest
from scipy.fft import dct, idct

PHI        = (1 + math.sqrt(5)) / 2
BLOCK_SIZE = 32
HOP_SIZE   = 16
SEED       = 42

# GPT-2 weight matrix dimensions: (name, rows, cols)
GPT2_WEIGHT_SHAPES = [
    ("embed_proj_768x768",    768,  768),   # attention QKV projection
    ("ffn_768x3072",          768, 3072),   # FFN up-projection
    ("ffn_3072x768",         3072,  768),   # FFN down-projection
    ("embed_patch_256x768",   256,  768),   # token embedding subset
]

RETENTION_RATES = [0.80, 0.60, 0.40]


# ── Core math ────────────────────────────────────────────────────────────────

def _build_rft_basis(N: int) -> np.ndarray:
    k = np.arange(N)
    freqs = np.mod((k + 1) * PHI, 1.0)
    n = np.arange(N)
    Phi = np.exp(1j * 2 * np.pi * np.outer(n, freqs)) / math.sqrt(N)
    G = Phi.conj().T @ Phi
    eigvals, eigvecs = np.linalg.eigh(G)
    eigvals = np.maximum(eigvals, 1e-14)
    G_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.conj().T
    return Phi @ G_inv_sqrt


def _top_k(coeffs: np.ndarray, k: int) -> np.ndarray:
    idx = np.argsort(np.abs(coeffs))[::-1]
    out = np.zeros_like(coeffs)
    out[idx[:k]] = coeffs[idx[:k]]
    return out


def _strt_row(row: np.ndarray, U: np.ndarray, retain_frac: float) -> np.ndarray:
    N = len(row)
    window = np.hanning(BLOCK_SIZE)
    output = np.zeros(N, dtype=np.float64)
    weight = np.zeros(N, dtype=np.float64)
    pos = 0
    while pos + BLOCK_SIZE <= N:
        frame = row[pos:pos + BLOCK_SIZE] * window
        k = max(1, int(BLOCK_SIZE * retain_frac))
        c = _top_k(U.conj().T @ frame.astype(complex), k)
        output[pos:pos + BLOCK_SIZE] += np.real(U @ c) * window
        weight[pos:pos + BLOCK_SIZE] += window ** 2
        pos += HOP_SIZE
    valid = weight > 1e-10
    output[valid] /= weight[valid]
    return output


def _dct_row(row: np.ndarray, retain_frac: float) -> np.ndarray:
    N = len(row)
    output = np.zeros(N, dtype=np.float64)
    pos = 0
    while pos + BLOCK_SIZE <= N:
        c = dct(row[pos:pos + BLOCK_SIZE], norm="ortho")
        k = max(1, int(BLOCK_SIZE * retain_frac))
        idx = np.argsort(np.abs(c))[::-1]
        cs = np.zeros_like(c)
        cs[idx[:k]] = c[idx[:k]]
        output[pos:pos + BLOCK_SIZE] = idct(cs, norm="ortho")
        pos += BLOCK_SIZE
    return output


def _compress_matrix(W: np.ndarray, method: str, U: np.ndarray,
                     retain_frac: float) -> np.ndarray:
    rows, cols = W.shape
    if cols < BLOCK_SIZE * 2:
        return W.copy()
    W_rec = np.empty_like(W)
    for i in range(rows):
        if method == "strt":
            W_rec[i] = _strt_row(W[i], U, retain_frac)[:cols]
        else:
            W_rec[i] = _dct_row(W[i], retain_frac)[:cols]
    return W_rec


def _rel_error(orig: np.ndarray, rec: np.ndarray) -> float:
    norm = np.linalg.norm(orig)
    return float(np.linalg.norm(orig - rec) / norm) if norm > 1e-15 else 0.0


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rft_basis_weights():
    U = _build_rft_basis(BLOCK_SIZE)
    err = np.max(np.abs(U.conj().T @ U - np.eye(BLOCK_SIZE)))
    assert err < 1e-10
    return U


@pytest.fixture(scope="module")
def synthetic_weights():
    rng = np.random.default_rng(SEED)
    return {name: rng.normal(0.0, 0.02, size=(rows, cols))
            for name, rows, cols in GPT2_WEIGHT_SHAPES}


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_rft_basis_unitarity(rft_basis_weights):
    """RFT basis must be unitary to machine precision."""
    err = np.max(np.abs(rft_basis_weights.conj().T @ rft_basis_weights - np.eye(BLOCK_SIZE)))
    print(f"\n[WEIGHTS] RFT basis max off-diagonal U†U error: {err:.2e}")
    assert err < 1e-10


def test_full_fidelity_table(synthetic_weights, rft_basis_weights):
    """
    Diagnostic — always passes.
    Prints the honest fidelity table for both STRT and DCT.
    Result: DCT wins on all Gaussian weight matrices at all retention rates.
    This is expected — weights have no quasi-periodic structure.
    """
    U = rft_basis_weights
    print("\n" + "=" * 100)
    print("WEIGHT COMPRESSION FIDELITY — GPT-2-scale synthetic weights (seed=42, N(0,0.02))")
    print(f"Block={BLOCK_SIZE}, Hop={HOP_SIZE}")
    print("=" * 100)
    header = f"  {'Name':<28}  {'Shape':>10}  "
    for r in RETENTION_RATES:
        header += f"STRT_{int(r*100)}%  DCT_{int(r*100)}%  "
    print(header)
    print("  " + "-" * 95)

    for name, W in synthetic_weights.items():
        row = f"  {name:<28}  {str(W.shape[0])+'×'+str(W.shape[1]):>10}  "
        for r in RETENTION_RATES:
            e_strt = _rel_error(W, _compress_matrix(W, "strt", U, r))
            e_dct  = _rel_error(W, _compress_matrix(W, "dct",  U, r))
            row += f"{e_strt:6.3f}  {e_dct:6.3f}  "
        print(row)

    print("=" * 100)
    print("Relative L2 error = ||W - W_reconstructed||₂ / ||W||₂  (lower = better)")
    print("DCT wins uniformly — expected, weights are near-Gaussian (out-of-family for STRT).")
    print("STRT wins on: KV-cache activations in deep layers, quasi-periodic signals.")


def test_dct_beats_strt_on_gaussian_weights(synthetic_weights, rft_basis_weights):
    """
    HONEST NEGATIVE RESULT for STRT on weight matrices.

    DCT must beat STRT on all Gaussian weight matrices.
    This is CORRECT and EXPECTED per NON_CLAIMS.md:
      'High-entropy random signals: ❌ No advantage for STRT/RFT'

    Asserting the negative result prevents anyone from
    incorrectly claiming STRT improves weight compression.
    """
    U = rft_basis_weights
    retain = 0.60     # representative mid-point
    dct_wins = 0
    total = 0
    print(f"\n[WEIGHTS] DCT vs STRT on Gaussian weights (retain={retain*100:.0f}%)")
    for name, W in synthetic_weights.items():
        e_strt = _rel_error(W, _compress_matrix(W, "strt", U, retain))
        e_dct  = _rel_error(W, _compress_matrix(W, "dct",  U, retain))
        if e_dct < e_strt:
            dct_wins += 1
        total += 1
        print(f"  {name:<28}  STRT={e_strt:.4f}  DCT={e_dct:.4f}  "
              f"{'DCT wins (expected)' if e_dct < e_strt else 'STRT wins (unexpected!)'}")

    print(f"\n  DCT wins: {dct_wins}/{total}  — Gaussian weights are out-of-family for STRT.")
    assert dct_wins == total, (
        f"Expected DCT to beat STRT on ALL Gaussian weight matrices "
        f"(high-entropy signals are out-of-family for STRT/RFT). "
        f"Got DCT wins {dct_wins}/{total}."
    )


def test_dct_error_bounds_at_each_retention(synthetic_weights, rft_basis_weights):
    """
    DCT weight reconstruction error stays within measured bounds.
    These are the honest, measured values from March 26 2026.

    Retention  Max observed rel-L2 error (DCT)
    ─────────  ────────────────────────────────
    80%        0.09
    60%        0.22
    40%        0.41
    """
    U = rft_basis_weights
    bounds = {0.80: 0.12, 0.60: 0.25, 0.40: 0.45}

    for retain, max_err in bounds.items():
        errs = {}
        for name, W in synthetic_weights.items():
            e = _rel_error(W, _compress_matrix(W, "dct", U, retain))
            errs[name] = e

        worst = max(errs.values())
        print(f"\n[WEIGHTS] DCT retain={retain*100:.0f}%  worst_rel_err={worst:.4f}  "
              f"bound={max_err:.2f}  {'PASS' if worst <= max_err else 'FAIL'}")
        assert worst <= max_err, (
            f"DCT rel-L2 error {worst:.4f} > {max_err:.2f} at retain={retain*100:.0f}%"
        )


def test_error_monotone_in_retention(synthetic_weights, rft_basis_weights):
    """
    Sanity: error must increase as retention decreases, for both methods.
    Tests on ffn_768x3072 (largest weight matrix).
    """
    U = rft_basis_weights
    W = synthetic_weights["ffn_768x3072"]
    for method in ("strt", "dct"):
        errs = {r: _rel_error(W, _compress_matrix(W, method, U, r))
                for r in RETENTION_RATES}
        print(f"\n[WEIGHTS] {method.upper()} monotone check: "
              + "  ".join(f"{r*100:.0f}%→{errs[r]:.4f}" for r in RETENTION_RATES))
        for i in range(len(RETENTION_RATES) - 1):
            r_hi, r_lo = RETENTION_RATES[i], RETENTION_RATES[i + 1]
            assert errs[r_hi] <= errs[r_lo], (
                f"{method.upper()}: error should increase as retention drops: "
                f"{r_hi*100:.0f}%={errs[r_hi]:.4f} vs {r_lo*100:.0f}%={errs[r_lo]:.4f}"
            )


def test_scope_summary():
    """
    Scope canary — prints the honest weight fidelity summary to CI log.
    Always passes.
    """
    msg = (
        "WEIGHT FIDELITY SCOPE: "
        "DCT beats STRT on Gaussian model weights (expected — out-of-family signal class). "
        "STRT advantage is domain-specific: KV-cache activations (deep V-layers, 8/8 wins) "
        "and positional encoding geometry (11/11 configs, 63.5% leakage reduction). "
        "Do NOT cite STRT weight compression results as an LM performance improvement."
    )
    print(f"\n[SCOPE] {msg}")
    assert msg
