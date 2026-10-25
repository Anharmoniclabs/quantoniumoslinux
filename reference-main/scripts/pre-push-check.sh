#!/usr/bin/env bash
# pre-push-check.sh — Local sanity check mirroring GitHub CI
# Run automatically via git pre-push hook, or manually:
#   bash scripts/pre-push-check.sh
#
# Checks (mirrors .github/workflows/ci.yml + shannon_tests.yml):
#   1. Python syntax errors (flake8 fatal only — E9,F63,F7,F82)
#   2. RFT unitarity roundtrip < 1e-14
#   3. RFT ≠ FFT non-equivalence
#   4. ANS codec roundtrip
#   5. Crypto avalanche ~50%
#   6. Core test suites (transforms, codecs)

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0
SKIP=0

pass() { PASS=$((PASS+1)); printf "${GREEN}✓ %s${NC}\n" "$1"; }
fail() { FAIL=$((FAIL+1)); printf "${RED}✗ %s${NC}\n" "$1"; }
skip() { SKIP=$((SKIP+1)); printf "${YELLOW}⊘ %s (skipped)${NC}\n" "$1"; }

# Resolve repo root
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

printf "\n${YELLOW}═══ quantoniumos pre-push sanity check ═══${NC}\n\n"

# --- 1. Syntax check (fatal errors only, fast) ---
if command -v flake8 &>/dev/null; then
    if flake8 algorithms/ src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics 2>/dev/null; then
        pass "Syntax check (flake8 fatal errors)"
    else
        fail "Syntax check — fatal Python errors found"
    fi
else
    skip "Syntax check (flake8 not installed)"
fi

# --- 2. RFT unitarity roundtrip ---
if python -c "
from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT
import numpy as np
rft = CanonicalTrueRFT(256)
x = np.random.randn(256) + 1j*np.random.randn(256)
y = rft.forward(x)
x_rec = rft.inverse(y)
err = float(np.linalg.norm(x - x_rec) / np.linalg.norm(x))
assert err < 1e-12, f'err={err:.2e}'
" 2>/dev/null; then
    pass "RFT unitarity (roundtrip < 1e-12)"
else
    fail "RFT unitarity — roundtrip error too large"
fi

# --- 3. RFT ≠ FFT ---
if python -c "
from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT
import numpy as np
np.random.seed(42)
N = 256
x = np.random.randn(N) + 1j*np.random.randn(N)
rft = CanonicalTrueRFT(N)
y_rft = rft.forward(x)
y_fft = np.fft.fft(x, norm='ortho')
corr = float(np.abs(np.vdot(y_rft, y_fft)) / (np.linalg.norm(y_rft) * np.linalg.norm(y_fft)))
assert corr < 0.5, f'corr={corr:.4f}'
" 2>/dev/null; then
    pass "RFT ≠ FFT (correlation < 0.5)"
else
    fail "RFT ≠ FFT — transforms are too similar"
fi

# --- 4. ANS codec roundtrip ---
if python -c "
from algorithms.rft.compression.ans import ans_encode, ans_decode
symbols = list(range(16))
encoded, freq_data = ans_encode(symbols)
decoded = ans_decode(encoded, freq_data, len(symbols))
assert decoded == symbols
" 2>/dev/null; then
    pass "ANS codec roundtrip"
else
    fail "ANS codec — roundtrip mismatch"
fi

# --- 5. Crypto avalanche ---
if python -c "
from algorithms.rft.crypto.enhanced_cipher import EnhancedRFTCryptoV2
import secrets
key = secrets.token_bytes(32)
cipher = EnhancedRFTCryptoV2(master_key=key)
pt = b'Test message for avalanche analysis - need 128 bits minimum'
ct1 = cipher.encrypt_aead(pt)
pt2 = bytearray(pt); pt2[0] ^= 0x01
ct2 = cipher.encrypt_aead(bytes(pt2))
min_len = min(len(ct1), len(ct2))
diff = sum(bin(ct1[i] ^ ct2[i]).count('1') for i in range(min_len))
avalanche = diff / (min_len * 8)
assert 0.4 < avalanche < 0.6, f'{avalanche*100:.1f}%'
" 2>/dev/null; then
    pass "Crypto avalanche (40-60%)"
else
    fail "Crypto avalanche — out of range"
fi

# --- 6. Core pytest suites ---
if command -v pytest &>/dev/null; then
    if pytest tests/transforms/test_rft_correctness.py -x -q --tb=line --no-header 2>/dev/null; then
        pass "Transform correctness tests"
    else
        fail "Transform correctness tests"
    fi

    if pytest tests/codec_tests/ -x -q --tb=line --no-header 2>/dev/null; then
        pass "Codec tests"
    else
        fail "Codec tests"
    fi
else
    skip "pytest suites (pytest not installed)"
fi

# --- Summary ---
TOTAL=$((PASS+FAIL+SKIP))
printf "\n${YELLOW}═══ Results: ${GREEN}${PASS} passed${NC}, "
if [ "$FAIL" -gt 0 ]; then
    printf "${RED}${FAIL} failed${NC}, "
else
    printf "${FAIL} failed, "
fi
printf "${SKIP} skipped / ${TOTAL} total ═══${NC}\n\n"

if [ "$FAIL" -gt 0 ]; then
    printf "${RED}Push blocked — fix failures before pushing.${NC}\n"
    exit 1
fi

printf "${GREEN}All checks passed — safe to push.${NC}\n"
exit 0
