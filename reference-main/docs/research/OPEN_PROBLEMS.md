# Open Problems in RFT Theory

**Date:** April 2026 (updated April 9, 2026 with gap audit results)
**Purpose:** Honest catalog of unsolved problems, ranked by importance. Each entry states the problem, why it matters, current evidence, and what counts as resolution.

---

## Problem 1: Continuous-Time Analogue

**Status: PARTIALLY RESOLVED (Fibonacci N proved, arbitrary N open)**

**Problem:** Does a continuous-time RFT operator exist with a well-defined sampling theorem?

**Why it matters:** Without a continuous-time theory, there is no aliasing bound, no band-limited analogue, no interpolation formula, and no fundamental justification for the discrete φ-grid beyond "it works numerically." Every classical transform (Fourier, wavelet, Gabor) has a continuous-time parent. Bindel identified this as the single largest theoretical gap.

**Resolution at Fibonacci N (April 2026 Gap Audit):**
At Fibonacci N, Φ is square and G = Φ†Φ is invertible. The RFT Aliasing Theorem is proved:
1. x̂(t) = Φ(t)G⁻¹Φ†x is the unique minimum-‖·‖_H norm interpolant in the φ-frequency RKHS.
2. At sample points: x̂(t_n) = x_n exactly for all x ∈ ℂ^N (verified: ‖P−I‖_F = 6.25×10⁻¹⁵ at N=89).
3. Pythagorean decomposition: ‖x_cont − x̂_cont‖²_{L²} = ‖(x_cont)_⊥‖²_{L²} (energy outside φ-span).
4. This is the direct analogue of Shannon aliasing: RFT aliasing error = energy outside φ-frequency span.

**What remains open:**
- Extension to arbitrary (non-Fibonacci) N, where G may be ill-conditioned.
- An integral operator T_φ on L²(ℝ) or L²([0,1]) (the current result uses an RKHS, not L² directly).
- Whether the RKHS approach fully satisfies the "continuous-time parent operator" requirement.

---

## Problem 2: Gram Matrix Conditioning Asymptotics

**Status: EMPIRICALLY CLOSED (Fibonacci N) / OPEN (arbitrary N) — one lemma remaining**

**Problem:** What is the behavior of λ_min(Φ†Φ) as N → ∞?

**Why it matters:** If κ(Φ†Φ) grows without bound, then the canonical construction U = Φ(Φ†Φ)^{-1/2} becomes numerically unstable at large N. The entire canonical pipeline (compression, coefficients, exact reconstruction) depends on well-conditioned Gram factorization.

**April 2026 Gap Audit findings:**
At Fibonacci N = F_m: the Three-Distance Theorem guarantees the φ-grid has exactly 2 distinct gap sizes with ratio φ. This prevents the frequency clustering that causes λ_min collapse at arbitrary N.

| Group | λ_min range | κ(G) range |
|-------|------------|------------|
| Fibonacci N | [0.5218, 0.5423] | [2.74, 2.85] |
| Arbitrary N | [0.0015, 0.1368] | [12.9, 1178.0] |

Convergence rate: κ(F_m) = κ_∞ − C/φ^{2m} + O(φ^{−4m}), where κ_∞ < 2.849.
Proof sketch: Beurling density D⁻ = 1 at Fibonacci N → frame lower bound via Kadec-type theorem → λ_min(G) ≥ (5+√5)/10 − O(φ^{−m}) > 0.

**Single remaining lemma:** Formalize the finite-N error term O(φ^{−m}) into an explicit computable bound. This one lemma also closes the fast canonical transform (Problem 4).

**What counts as full resolution:**
- A lower bound: λ_min(Φ†Φ) ≥ c·N^{-α} for explicit c, α at arbitrary N.
- OR proof that the Fibonacci restriction is inherent (arbitrary N cannot be well-conditioned).

---

## Problem 3: Signal Class Characterization

**Problem:** What natural or physical signals are efficiently representable in the φ-basis?

**Why it matters:** Theorem 15 proves that S_φ(K) signals (K-sparse in φ-grid) are exactly sparse in RFT and dense in DFT. But S_φ(K) is defined tautologically by the basis. The question is whether any real-world signal class — EEG, bearing vibrations, speech, quasi-periodic oscillations — falls naturally into S_φ(K) for small K.

**Current evidence:** Empirical compression benchmarks show +0.1 to +1.8 dB advantage on specific datasets (CWRU bearing faults, MIT-BIH arrhythmia). These are suggestive but domain-limited. No theoretical characterization analogous to "bandlimited ↔ Fourier sparse" or "piecewise smooth ↔ wavelet sparse."

**What counts as resolution:**
- A proved inclusion: signal class X ⊆ S_φ(K) for some explicit K(N).
- OR a statistical model: signals drawn from distribution D have expected φ-sparsity K(N) with high probability.
- Physical explanation of *why* golden-ratio spacing matches some natural process.

---

## Problem 4: Fast Canonical Transform

**Status: CONDITIONALLY CLOSED (Fibonacci N) — depends on Problem 2 lemma**

**Problem:** Can the canonical RFT (U = Φ(Φ†Φ)^{-1/2}) be computed in O(N log N) or O(N log²N) time?

**Why it matters:** Currently, forming U costs O(N³) via SVD/polar decomposition. The fast RFT Ψ = D_φ C_σ F achieves O(N log N) but is a different operator. The canonical U has all the proven spectral properties. If it cannot be computed fast, it is impractical for large N.

**April 2026 Gap Audit findings:**
At Fibonacci N, Newton-Schulz iteration for the polar factor converges in O(log(1/ε)) iterations, each costing O(N log N) via Toeplitz FFT (G is Toeplitz). Empirical iteration counts:

| N | Iterations | λ_min |
|---|-----------|-------|
| 8 | 39 | 0.5423 |
| 55 | 40 | 0.5231 |
| 144 | 41 | 0.5221 |
| 233 | 41 | 0.5219 |
| 377 | 42 | 0.5218 |

Iteration count is flat (39–42). Predicted ceiling from κ = 2.85: 44 iterations to ε = 10⁻¹⁴.

**Conditional on:** Proving κ(G) ≤ 2.849 analytically for all Fibonacci N (the shared lemma from Problem 2). Once that lemma is proved, the fast canonical transform at Fibonacci N is fully established.

**What remains open:**
- Fast canonical transform at arbitrary (non-Fibonacci) N.
- OR a proof that no subquadratic algorithm exists for arbitrary N.

---

## Problem 5: Column Interpretability

**Problem:** Are the columns of U quasi-monochromatic in any rigorous sense?

**Why it matters:** If columns of U are frequency-localized, the canonical basis has a physical interpretation like DFT modes. This would strengthen the claim that RFT coefficients represent "resonant modes" rather than being algebraically arbitrary.

**Current evidence:** Empirical DFT of columns shows spectral peaks near φ-related frequencies. No theorem. Observation 11 (IRONCLAD) notes quasi-orthogonality of Gram off-diagonals, which is related but not identical.

**What counts as resolution:**
- For each column u_k of U, a bound: ‖u_k - p_k‖ ≤ ε(N) where p_k is a pure-frequency signal.
- OR: the spectral measure of u_k concentrates on an interval of width δ(N) → 0.

---

## Problem 6: Structured-SIS Hardness

**Problem:** Is the structured-SIS assumption (short vectors in lattice(A_φ)) hard?

**Why it matters:** The crypto subsystem (Claim 2, patent) uses RFT-SIS hashing with structured matrix A_φ. Standard SIS hardness proofs assume random A. Structured A may have lattice shortcuts that random matrices avoid.

**Current evidence:** Theorem 7.2 (IRONCLAD) explicitly states: "For the specific deterministic A_φ, a new assumption or an indistinguishability proof is required." No such proof exists. Internal testing shows avalanche 40-60%, but that is implementation evidence, not cryptographic hardness.

**What counts as resolution:**
- A reduction: breaking structured-SIS(A_φ) ⊇ breaking some known-hard problem.
- OR: a lattice attack exploiting the φ-structure, which would be a negative result.
- OR: an indistinguishability argument: A_φ + R is statistically close to random (partially addressed by Theorem 7.4 for the hybrid construction).

---

## Problem 7: Shift-Diagonalization Gap Quantification

**Problem:** Exactly how far is U from diagonalizing cyclic shifts, and does the gap shrink or grow with N?

**Why it matters:** This determines whether any RFT-based convolution-like theorem can be salvaged. If the gap → 0 as N → ∞, there may be an approximate convolution result. If it stays bounded or grows, RFT has no convolution path at all.

**Current evidence:** At N=64, diagonal energy of U†SU is ~0.9519, off-diagonal ~0.0481. Single data point. No scaling study.

**What counts as resolution:**
- ‖off-diag(U†SU)‖_F as a function of N for N = 16, 32, 64, 128, 256, 512, 1024.
- Asymptotic fit or bound.
- Connection to the equidistribution of {frac((k+1)φ)} on [0,1].

---

## Problem 8: Domain-Specific Compression Validation

**Problem:** Do the empirical compression advantages replicate on standard benchmark datasets under peer review conditions?

**Why it matters:** Internal benchmarks show RFT advantages in specific domains. But "we tested our own basis on our own pipeline" is not independently validated science.

**Current evidence:** Internal benchmarks on CWRU bearing faults, MIT-BIH, PhysioNet Sleep-EDF, and synthetic chirps. Results show modest advantages (+0.1 to +1.8 dB vs FFT-based baselines). No external replication.

**What counts as resolution:**
- Published benchmark comparison using standard evaluation methodology (e.g., PESQ for audio, SSIM for images, PRD for ECG).
- Independent replication by a second research group.
- Statistical significance testing (not just single-run comparisons).

---

## Priority Ranking

| Rank | Problem | Status | Estimated Difficulty | Impact if Solved |
|------|---------|--------|---------------------|------------------|
| 1 | Gram conditioning lemma (Prob 2) | One lemma remaining | Medium (Diophantine tools exist) | Closes Problems 2 AND 4 simultaneously |
| 2 | Signal class characterization | OPEN | Medium-Hard | Converts tautological sparsity to real-world relevance |
| 3 | Continuous-time at arbitrary N (Prob 1) | Fibonacci N proved | Medium | Extends aliasing theorem beyond Fibonacci restriction |
| 4 | Domain-specific validation | OPEN | Low (engineering effort) | Establishes empirical credibility |
| 5 | Column interpretability | OPEN | Medium | Physical meaning of basis vectors |
| 6 | Structured-SIS hardness | OPEN | Hard (crypto-specific) | Determines if crypto claims are defensible |
| 7 | Shift-diag gap quantification | OPEN | Low (numerical experiment) | Clarifies whether convolution results are possible |

---

## Related Documents

- PROVEN_EMPIRICAL_OPEN_MATRIX.md — classification table

---

*Last updated: April 2026. Update when any problem is resolved or new evidence appears.*
