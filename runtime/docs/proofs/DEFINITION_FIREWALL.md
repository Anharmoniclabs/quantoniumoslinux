# DEFINITION FIREWALL: Canonical RFT

**Version 1.0 — February 2026**

This document establishes the **immutable canonical definition** of the Resonant Fourier Transform (RFT) for the quantoniumos repository. Any deviation from this definition requires explicit labeling.

---

## 🔒 THE CANONICAL DEFINITION (NON-NEGOTIABLE)

### Canonical RFT Basis $U_\varphi$

$$
U_\varphi := \Phi (\Phi^H \Phi)^{-1/2}
$$

where $\Phi$ is the raw φ-grid exponential basis:

$$
\Phi_{n,k} := \frac{1}{\sqrt{N}} \exp\left(i 2\pi f_k n\right), \quad f_k = \text{frac}((k+1)\varphi)
$$

and $\varphi = (1+\sqrt{5})/2$ is the golden ratio.

### Canonical RFT Transform

$$
\text{RFT}(x) = U_\varphi^H x, \quad \text{RFT}^{-1}(\hat{x}) = U_\varphi \hat{x}
$$

### Key Properties (Proven)

| Property | Statement | Proof |
|----------|-----------|-------|
| **Unitarity** | $U_\varphi^H U_\varphi = I$ | Theorem 2 |
| **Nearest Unitary** | $U_\varphi = \text{polar}(\Phi)$ | Theorem A |
| **Non-DFT** | $U_\varphi \neq D P F$ for any phases $D$, permutation $P$ | Theorem 6 |

---

## ⚠️ WHAT THE CANONICAL RFT IS **NOT**

### ❌ NOT the φ-Phase FFT

The formula $\Psi = D_\varphi C_\sigma F$ (where $F$ is the DFT) is **NOT** the canonical RFT.

| Property | φ-Phase FFT | Canonical RFT |
|----------|-------------|---------------|
| **Definition** | $\Psi = D_\varphi C_\sigma F$ | $U_\varphi = \Phi(\Phi^H\Phi)^{-1/2}$ |
| **Magnitude spectrum** | $\|\Psi x\| = \|Fx\|$ | Different from DFT |
| **Sparsity advantage** | ❌ None | ✅ On golden QP signals |
| **Status** | **LEGACY** | **CANONICAL** |

**The φ-phase FFT is kept for backwards compatibility ONLY and must always be labeled "legacy" or "deprecated".**

### ❌ NOT the Vandermonde Eigenvector Matrix

The non-orthogonal matrix $V$ (Vandermonde with roots $z_k = e^{i2\pi f_k}$) satisfies:
- $V$ diagonalizes the companion matrix $C_\varphi$
- $V = \sqrt{N} \Phi$ (up to column scaling)

But $V$ is **not unitary**. The canonical RFT uses the orthonormalized version $U_\varphi$.

---

## 🚫 FORBIDDEN CLAIMS

The following claims are **mathematically incorrect** and must not appear in documentation:

### 1. "The canonical RFT exactly diagonalizes $C_\varphi$"

**FALSE.** Only the non-orthogonal $V$ diagonalizes $C_\varphi$ exactly. The unitary $U_\varphi$ does not.

```
V^{-1} C_φ V = diag(z_k)       ← TRUE (exact)
U_φ^H C_φ U_φ = diag(z_k)      ← FALSE (off-diagonal ~ O(1))
```

### 2. "RFT achieves a smaller Heisenberg bound than DFT"

**MISLEADING.** The continuous Heisenberg bound $\Delta x \cdot \Delta p \geq \hbar/2$ does not directly apply to discrete finite-N transforms. Use the Maassen-Uffink entropic bound instead.

### 3. "φ-phase FFT is the RFT"

**FALSE.** The φ-phase FFT has the same magnitude spectrum as the DFT and provides no sparsity advantage. It is **not** the canonical RFT.

---

## ✅ SAFE CLAIMS (WHAT YOU CAN SAY)

| Claim | Status | Evidence |
|-------|--------|----------|
| "$U_\varphi$ is unitary" | ✅ Proven | Theorem 2 |
| "$U_\varphi$ is the nearest unitary to $\Phi$" | ✅ Proven | Theorem A |
| "$U_\varphi \neq$ permuted/phased DFT" | ✅ Proven | Theorem 6 |
| "RFT concentrates golden QP signals better than FFT" | ✅ Proven (Diophantine) | Theorem 8 (5+6 lemmas, 33+46 tests; Hurwitz 1891) |
| "The Vandermonde $V$ diagonalizes $C_\varphi$" | ✅ Proven | Theorem B |
| "$U_\varphi$ satisfies Maassen-Uffink uncertainty" | ✅ Proven | Theorem 9 |

---

## 📁 AUTHORITATIVE FILES

| Purpose | File | Status |
|---------|------|--------|
| **Definition** | `algorithms/rft/README_RFT.md` | Canonical |
| **Proofs** | `THEOREMS_RFT_IRONCLAD.md` | Canonical |
| **Implementation** | `algorithms/rft/core/resonant_fourier_transform.py` | Reference |
| **Tests** | `tests/proofs/test_rft_transform_theorems.py` | CI-verified |

---

## 🏷️ LEGACY FILES (Must Be Labeled)

| File | Status | Label Required |
|------|--------|----------------|
| `algorithms/rft/core/phi_phase_fft_optimized.py` | **Removed** | Was deprecated; now deleted |
| `algorithms/rft/core/rft_phi_legacy.py` | Legacy | "LEGACY: Pre-canonical" |
| Any file using $\Psi = D_\varphi C_\sigma F$ | Legacy | Must note "not canonical RFT" |

---

## 🧪 DEFINITION ENFORCEMENT TESTS

The CI suite enforces canonical definitions:

```python
# tests/proofs/test_definition_firewall.py

def test_canonical_rft_is_unitary():
    """U_φ^H U_φ = I to machine precision."""
    
def test_canonical_rft_is_polar_factor():
    """U_φ = polar(Φ).U"""
    
def test_phi_phase_fft_equals_dft_magnitude():
    """|Ψx| = |Fx| proves φ-phase FFT is NOT sparse-improving."""
    
def test_vandermonde_diagonalizes_companion():
    """V^{-1} C_φ V is diagonal."""
    
def test_canonical_rft_does_not_diagonalize_companion():
    """U_φ^H C_φ U_φ has off-diagonal > 0.1"""
```

---

## 📋 CHECKLIST FOR NEW DOCUMENTATION

Before merging any documentation changes:

- [ ] Does it use "RFT" to mean the canonical $U_\varphi$?
- [ ] Is φ-phase FFT labeled as "legacy"?
- [ ] Does it avoid claiming $U_\varphi$ diagonalizes $C_\varphi$?
- [ ] Does any uncertainty principle use Maassen-Uffink (not Heisenberg)?
- [ ] Are concentration claims marked as "empirical, CI-verified"?

---

## CONTACT

Questions about the canonical definition: luisminier79@gmail.com

**Last Updated:** February 3, 2026
