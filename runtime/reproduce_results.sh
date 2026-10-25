#!/usr/bin/env bash
# =============================================================================
# reproduce_results.sh — quantoniumos Full Proof & Benchmark Reproduction
# =============================================================================
# Reproduces all 255 formal proof tests + STRT KV-cache research benchmark
# documented in docs/PROOF_RESULTS.md
#
# Usage:
#   bash reproduce_results.sh
#
# Requirements:
#   Python 3.12+, pip, virtualenv
# =============================================================================

set -e

echo "============================================================"
echo " quantoniumos — Full Results Reproduction"
echo " See docs/PROOF_RESULTS.md for expected output"
echo "============================================================"
echo ""

# ── Setup ────────────────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  echo "[setup] Creating virtual environment..."
  python -m venv .venv
fi

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
  source .venv/Scripts/activate
else
  source .venv/bin/activate
fi

echo "[setup] Installing dependencies..."
pip install -e . --quiet
pip install transformers torch scipy --quiet

echo ""
echo "============================================================"
echo " PART 1: Formal Proof Suites (255 tests)"
echo "============================================================"
echo ""

# Suite 1 — Theorem 15: Exact Sparsity
echo "[1/10] Theorem 15 — Exact Sparsity"
pytest tests/proofs/test_theorem_15_exact_sparsity.py -v
echo ""

# Suite 2 — Formal Proofs (Theorems 1, 2, 6, 9, 10, 11, A, Conjecture 12)
echo "[2/10] Formal Proofs — Theorems 1, 2, 6, 9, 10, 11, A, Conjecture 12"
pytest tests/proofs/test_formal_proofs.py -v
echo ""

# Suite 3 — Fast RFT D3 Unitarity (Theorem 3)
echo "[3/10] Theorem 3 — Fast RFT D3 Unitarity"
pytest tests/proofs/test_fast_rft_d3_unitarity.py -v
echo ""

# Suite 4 — Diophantine RFT Extension (Theorem 6 + K99)
echo "[4/10] Diophantine RFT Extension — Theorem 6 + K99 structure"
pytest tests/proofs/test_diophantine_rft_extension.py -v
echo ""

# Suite 5 — Theorem 8 Diophantine (Lemmas 8.4a-f)
echo "[5/10] Theorem 8 — Diophantine (Lemmas 8.4a-f)"
pytest tests/proofs/test_theorem8_diophantine.py -v
echo ""

# Suite 6 — Theorem 8 Formal Proof (Lemmas 8.3a-b)
echo "[6/10] Theorem 8 — Formal Proof (Lemmas 8.3a-b)"
pytest tests/proofs/test_theorem8_formal_proof.py -v
echo ""

# Suite 7 — Theorem 8 Rank Truncation
echo "[7/10] Theorem 8 — Rank Truncation"
pytest tests/proofs/test_theorem_8_rank_truncation.py -v
echo ""

# Suite 8 — Theorem 8 Eigenvalue Tail from Truncation
echo "[8/10] Theorem 8 — Eigenvalue Tail from Truncation"
pytest tests/proofs/test_theorem_8_eigen_tail_from_truncation.py -v
echo ""

# Suite 9 — Theorems 10, 11, 12
echo "[9/10] Theorems 10, 11, 12 — Polar uniqueness, no exact diag, variational minimality"
pytest tests/proofs/test_theorems_10_12.py -v
echo ""

# Suite 10 — Theorem E: K0.99 Concentration (RIP Proof)
echo "[10/10] Theorem E — K0.99 Concentration (RIP proof via Candes-Tao threshold)"
pytest tests/proofs/test_theorem_e_rip_proof.py -v
echo ""

echo "============================================================"
echo " PART 1 COMPLETE — Running full suite summary"
echo "============================================================"
pytest tests/proofs/ -v --tb=short
echo ""

echo "============================================================"
echo " PART 2: STRT KV-Cache Research Benchmark"
echo " Model: GPT-2 124M (Apache 2.0 via HuggingFace)"
echo " Expected: deep V-layers (9-11) STRT wins 91.7%"
echo "============================================================"
echo ""
pytest tests/research/test_strt_kvcache_real_activations.py -v -s
echo ""

echo "============================================================"
echo " ALL DONE"
echo " Expected: 255 passed (proofs) + 3 passed (STRT benchmark)"
echo " See docs/PROOF_RESULTS.md for full tabulated results"
echo "============================================================"
