# RFT Terminology Glossary

**Authority**: This document is the **single source of truth** for RFT terminology in quantoniumos.

**Last Updated**: 2026-04-13

> **Agent Instruction**: When encountering conflicting terminology in other documents, defer to this glossary. When writing new documentation, use these exact terms.

---

## Quick Reference Table

| Term | Definition | Complexity | Theorems | Status |
|------|------------|------------|----------|--------|
| **Canonical RFT (U)** | Gram-normalized φ-grid basis | O(N³) | 3, 4, 5, 6 | ✓ PROVEN |
| **Fast Phi-RFT (Ψ)** | FFT-factorized D_φ C_σ F | O(N log N) | 1, 2, 8, 9 | ✓ PROVEN (legacy) |
| **Fibonacci Fast RFT** | Fibonacci-indexed O(N log N) variant | O(N log N)* | — | EXPERIMENTAL |
| **Fibonacci Tilt RFT** | QR-normalized exponential rate basis | O(N³)* | — | EXPERIMENTAL |

---

## 1. Canonical RFT (U)

### Definition
The **Canonical RFT** is the mathematically rigorous, Gram/Löwdin-normalized φ-grid exponential basis:

$$
\Phi[n,k] = \frac{1}{\sqrt{N}} \exp\left(i \cdot 2\pi \cdot \mathrm{frac}((k+1)\varphi) \cdot n\right), \quad k,n = 0, \ldots, N-1
$$

$$
U_\varphi = \Phi \cdot (\Phi^H \Phi)^{-1/2} \quad \text{(Löwdin orthonormalization)}
$$

### Properties
- **Unitary**: $U^H U = I$ (proven via Gram–Schmidt or SVD)
- **Complexity**: O(N³) for explicit matrix construction
- **Energy Preservation**: $\|U x\|_2 = \|x\|_2$ (Parseval)

### Implementation
- Primary: `algorithms/rft/core/canonical_true_rft.py` → `CanonicalTrueRFT`
- API: `.forward(x)`, `.inverse(y)`

### Theorems Proved
- **Theorem 3**: $U^H U = I$ (unitarity)
- **Theorem 4**: Frame-correct inversion for non-unitary Φ
- **Theorem 5**: Energy preservation
- **Theorem 6**: Bounded eigenvalue spread

---

## 2. Fast Phi-RFT (Ψ) — LEGACY

### Definition
The **Fast Phi-RFT** is the FFT-factorized O(N log N) approximation:

$$
\Psi = D_\varphi \cdot C_\sigma \cdot F
$$

Where:
- $F$ = DFT matrix (FFT)
- $C_\sigma$ = diagonal chirp phase
- $D_\varphi$ = diagonal φ-phase kernel: $E[k] = \exp(i \cdot 2\pi \cdot \mathrm{frac}((k+1)\varphi))$

### ⚠️ CRITICAL WARNING
**Ψ is NOT the Canonical RFT**. They are mathematically distinct operators:
- Ψ uses FFT frequencies with post-modulation
- U uses true φ-grid frequencies from the start

The "bridge path" attempting to prove Ψ → U convergence was **abandoned**. See `docs/research/STATUS_AFTER_BINDEL_REVIEW.md`.

### Properties
- **Unitary**: Yes, Ψ is exactly unitary (proven)
- **Complexity**: O(N log N)
- **Relation to Canonical**: No formal convergence proof; they are separate constructions

### Implementation
- `algorithms/rft/core/fast_rft.py` → `FastRFT`, `fast_rft()`
- API: `.forward(x)`, `.inverse(y)`

### Theorems Proved (for Ψ specifically)
- **Theorem 1**: Ψ is unitary
- **Theorem 2**: Error bounds on Ψ vs U coefficient-space discrepancy
- **Theorem 8**: O(N log N) complexity
- **Theorem 9**: Parseval holds for Ψ

### Status
**LEGACY** — Use for speed-critical applications where exact Canonical match is not required. For proofs and precision work, use Canonical RFT.

---

## 3. Fibonacci Fast RFT — EXPERIMENTAL

### Definition
A Fibonacci-indexed fast transform using Zeckendorf decomposition and golden-ratio modular arithmetic for O(N log N) computation at Fibonacci-sized N.

### Key Idea
At $N = F_k$ (Fibonacci numbers), the φ-grid frequencies exhibit special lattice structure enabling fast algorithms via Fibonacci recursion.

### Implementation
- `algorithms/rft/core/fibonacci_fast_rft.py` → `fast_rft_fibonacci()`, `FibonacciRFTResult`

### Properties
- **Complexity**: O(N log N) *only at Fibonacci N*
- **Unitarity**: Not separately proven; relies on canonical construction
- **Arbitrary N**: Falls back to O(N²) dense matrix

### Status
**EXPERIMENTAL** — Not production-ready. Use for research on Fibonacci/lattice signal domains.

---

## 4. Fibonacci Tilt RFT — EXPERIMENTAL

### Definition
A variant RFT basis using Fibonacci sequence ratios {1, 1, 2, 3, 5, 8, 13, 21, ...} as exponentially-spaced frequency indices, QR-normalized to unitarity.

$$
T^{(\text{raw})}[n,k] = \exp\left(i \cdot 2\pi \cdot F_k \cdot n / N\right)
$$
$$
U_F = \text{QR}(T^{(\text{raw})})
$$

### Properties
- **Unitary**: Yes (via QR factorization)
- **Structure**: Integer-lattice frequencies, suited for crypto/hash applications
- **Sparsity**: 100% for integer-lattice signals (ties DFT)

### Implementation
- `algorithms/rft/variants/operator_variants.py` → `generate_rft_fibonacci()`
- Tested in `tests/proofs/test_fibonacci_fast_rft.py`

### Theorems
- **Theorem 7 (Fibonacci Tilt)**: Achieves ~52% avalanche diffusion for lattice hashing
- See `docs/proofs/RFT_THEOREMS.md` + `PHI_RFT_MATHEMATICAL_PROOFS.md`

### Status
**EXPERIMENTAL** — Used for crypto/hash research. Not the canonical for general signal processing.

---

## 5. RFT Variants (Operator-Eigenbasis)

These are **not** fast transforms but research variants exploring different kernel structures:

| Variant Key | Generator | Use Case |
|-------------|-----------|----------|
| `op_rft_golden` | Golden-angle correlation kernel | Default canonical analog |
| `op_rft_fibonacci` | Fibonacci-ratio kernel | Integer lattice signals |
| `op_rft_harmonic` | Harmonic-phase kernel | Audio/music processing |
| `op_rft_geometric` | Geometric progression kernel | Scale-invariant signals |
| `op_rft_cascade` | Hierarchical cascade kernel | Multi-resolution analysis |

See `algorithms/rft/variants/manifest.py` for the complete list.

---

## 6. DEPRECATED / LEGACY Markers

### Files Marked LEGACY
| File | Reason | Replacement |
|------|--------|-------------|
| `algorithms/rft/core/rft_phi_legacy.py` | Old φ-phase formula (non-unitary) | `canonical_true_rft.py` |
| `algorithms/rft/core/fast_rft.py` | Fast Ψ surrogate, not canonical | Use only for speed |

### Deprecated Functions
| Function | In File | Reason |
|----------|---------|--------|
| `rft_forward()` (T≠N mode) | `resonant_fourier_transform.py` | Waveform synthesis, not RFT |
| `_phi_phase_fft()` | (removed) | Subsumed into FastRFT class |

---

## 7. Common Confusion Points

### "Is Fast RFT the same as Canonical RFT?"
**NO.** They are distinct mathematical operators sharing the φ-phase philosophy:
- Canonical U = Gram-normalized φ-grid basis
- Fast Ψ = D_φ C_σ F factorization

### "Which should I use?"
| Scenario | Use |
|----------|-----|
| Proofs / precision work | Canonical RFT (U) |
| Real-time / large-scale | Fast Phi-RFT (Ψ) |
| Fibonacci-sized signals | Fibonacci Fast RFT (research only) |
| Crypto / hash applications | Fibonacci Tilt RFT |

### "What is the O(N log N) RFT?"
The **Fast Phi-RFT (Ψ)** has O(N log N) complexity. The Canonical RFT (U) is O(N³).

Claiming "RFT is O(N log N)" without qualification is **incorrect** — you must specify which variant.

---

## 8. References

- **Canonical Definition**: `docs/proofs/DEFINITION_FIREWALL.md`
- **Theorem Proofs**: `docs/THEOREMS_RFT_IRONCLAD.md`
- **Research Gaps**: `docs/private/INTERNAL_RESEARCH_GAP_MEMO.md`
- **Transform Inventory**: `docs/RFT_TRANSFORM_INVENTORY.md`

---

## 9. Agent Usage Guidelines

When writing documentation or code:

1. **Always qualify "RFT"** — say "Canonical RFT" or "Fast Phi-RFT", never just "RFT" when precision matters.

2. **Theorem attribution** — Theorems 1-2, 8-9 apply to Ψ (Fast). Theorems 3-6 apply to U (Canonical).

3. **Complexity claims** — Only claim O(N log N) for the Fast form.

4. **"Fast RFT" = Ψ** — In this codebase, "Fast RFT" means the legacy FFT-factorized form unless otherwise specified.

5. **"Fibonacci RFT"** — Clarify whether you mean:
   - Fibonacci Fast RFT (O(N log N) at Fibonacci N)
   - Fibonacci Tilt RFT (variant basis with Fibonacci frequencies)
