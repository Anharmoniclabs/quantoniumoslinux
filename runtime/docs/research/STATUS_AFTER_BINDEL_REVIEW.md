# Research Status After External Review (Bindel Thread)

**Date:** April 2026
**Scope:** Honest assessment of RFT research maturity after critical external review
**Audience:** Internal research team, patent counsel, potential collaborators
**Status:** Active — reflects current understanding

---

## Executive Summary

A sustained technical exchange with David Bindel (Cornell, numerical linear algebra)
forced a fundamental recalibration of how the RFT work is framed. The original pitch
— "fixes Gibbs, beats FFT" — did not survive scrutiny. What *did* survive, and what
emerged from the pressure, is a **credible finite-dimensional canonical construction**
with a cleaner mathematical story than what existed before the review.

The research is **past the hand-wavy concept stage** but **not at finished
theorem-complete Fourier-replacement stage**. It is a finite-dimensional transform
with real structure, real audits, and one or two hard mathematical bottlenecks
separating it from a convincing theory paper.

---

## What Is Now Solid in the Discrete Setting

These results have complete proofs (see `docs/THEOREMS_RFT_IRONCLAD.md`):

1. **Full rank of raw φ-grid basis Φ** (Theorem 1).
   Vandermonde argument from irrationality of φ. The matrix Φ is invertible for every N.

2. **Canonical RFT U is unitary** (Theorem 2).
   U = Φ(Φ†Φ)^{-1/2} satisfies U†U = I by construction (Löwdin polar orthogonalization).

3. **Coefficient story is pinned down.**
   - Raw frame coefficients: c_raw = G^{-1}Φ†x (dual-frame recovery, exact).
   - Canonical coefficients: c = U†x (unitary projection, Parseval holds).
   - Reconstruction is exact at full dimension: x = Uc.
   - These are two *different* coefficient vectors for the same signal. The distinction
     matters and was confused in earlier materials.

4. **Unitarity numerics are clean.**
   - Unitarity defect ‖U†U − I‖_F < 2×10^{-13} up to N ≤ 256.
   - Reconstruction error < 10^{-14} up to N ≤ 256.
   - These are machine-precision results, not approximate.

5. **RFT is provably not a DFT kernel** (Theorem 6).
   No rephasing/permutation maps Φ to the DFT matrix. Proof by rational/irrational
   frequency contradiction.

6. **RFT is provably not an LCT/FrFT variant** (Theorem 6 + P2).
   Irrational φ-phases cannot arise from any affine-phase LCT construction.

7. **Exact sparsity vs DFT leakage** (Theorem 15).
   Signals in S_φ(K) have exactly K nonzero φ-grid coefficients but dense DFT
   coefficients. This is the cleanest statement of where the basis actually wins.

8. **Twisted convolution theorem** (Theorem 4).
   Ψ(x ⋆_Ψ h) = (Ψx) ⊙ (Ψh). Proved. But see "Narrowed" section below.

9. **Entropic uncertainty** (Theorem 8).
   Maassen–Uffink bound applies: H(|x|²) + H(|U†x|²) ≥ −2 log(μ(U)).

---

## What Was Narrowed or Rejected After Critique

1. **U does NOT diagonalize cyclic shifts.**
   Direct test at N=64: diagonal energy fraction of U†SU is ~0.9519, off-diagonal
   ~0.0481. The DFT achieves exact diagonalization. This kills the "RFT convolution
   theorem = FFT convolution theorem" analogy permanently.

2. **Twisted convolution is tautological.**
   Theorem 4 holds for *any* unitary — it is not a special property of RFT. Bindel
   correctly identified this as a trivial consequence of unitarity, not a structural
   result. It does not give RFT an FFT-style fast convolution path.

3. **No general superiority over FFT/DFT.**
   The claim "RFT beats FFT" was dropped. RFT wins on φ-quasi-periodic signals
   (Theorem 15) and loses on everything else. The honest framing is
   *signal-class-dependent*, not universal.

4. **No Gibbs phenomenon fix.**
   The original "fixes Gibbs" claim was abandoned. The canonical construction does
   not address Gibbs — it is a different basis, not a windowing fix.

5. **Surrogate/bridge path died.**
   The attempt to bridge fast RFT Ψ = D_φ C_σ F to canonical U via convergence
   arguments did not hold up. The fast and canonical constructions are separate
   objects. Conditioning vs unitarity are separate problems.

---

## What Is Still Empirical

These findings are supported by benchmarks and internal experiments but do not
have theorem-level proofs:

1. **Column quasi-monochromaticity.**
   Columns of U appear to be structured, quasi-monochromatic objects (not random
   garbage). Empirically observed but not proved. Interpretability of the columns
   is an open research question.

2. **Concentration/compression on φ-quasi-periodic signals.**
   Reported >95% energy in <0.1N coefficients for N ∈ {32, 64, 128, 256, 512}.
   Labeled "Theorem 3" in the LaTeX paper — this is **empirical, not proved**.
   Should be relabeled as Observation or Empirical Finding.

3. **Fibonacci fast regime.**
   Wrapped-frequency Fibonacci regime shows favorable Gram conditioning behavior.
   Empirically observed. No theorem.

4. **Toeplitz Gram fast path.**
   The Gram matrix G = Φ†Φ is Toeplitz (for the wrapped construction). Standard
   Toeplitz solvers give O(N log²N) for G^{-1/2}. This is algorithmically
   plausible but not yet implemented as a complete fast canonical RFT.

5. **Compression wins on specific domains.**
   Medical (ECG, EEG), vibration (bearing fault), audio — reported advantages are
   domain-specific and do not generalize. Benchmarks exist but are not independently
   validated.

6. **Crypto avalanche and mixing quality.**
   RFT-SIS hash + Feistel cipher pass internal validation (40-60% avalanche).
   Not externally audited. Structured-SIS hardness assumption is non-standard.

---

## What Remains Open

1. **Continuous-time RFT analogue.**
   No interpolation formula between sample points. No aliasing bounds. No
   band-limited signal analogue. Bindel pressed on this specifically. It is the
   biggest theoretical gap.

2. **Natural signal class for φ-grid.**
   S_φ(K) is a mathematically clean class but is it *natural*? What real-world
   signals live there? This needs characterization, not assumption.

3. **Fast canonical RFT algorithm.**
   Current canonical construction is O(N²) via dense Gram inverse square root.
   The Toeplitz structure is plausible but:
   - Needs a spectral lower bound on eigenvalues of G for stability
   - Needs a complete implementation, not just a plausibility argument
   - The "only one theorem remains" claim is self-assessed, not externally validated

4. **Column interpretability theorem.**
   What do the columns of U *mean* as basis functions? Quasi-monochromaticity
   is observed. A theorem characterizing their frequency content would
   substantially strengthen the mathematical story.

5. **Conditioning asymptotics.**
   How does the condition number of G = Φ†Φ scale with N? Observation 11 shows
   average decorrelation, but worst-case coherence is sensitive to Diophantine
   near-collisions (Non-theorem 12). This is marked "not yet proved" in
   THEOREMS_RFT_IRONCLAD.md.

6. **Shift-diagonalization gap.**
   U†SU has ~4.8% off-diagonal energy at N=64. Does this improve or worsen with
   N? Is there a useful approximate shift structure? Unknown.

---

## Next Concrete Research Targets

Ranked by impact on the theory's credibility:

1. **Prove or disprove the spectral eigenvalue lower bound for G.**
   This is the gatekeeper for the fast algorithm path. If λ_min(G) = Ω(1/poly(N)),
   the Toeplitz fast path works. If not, conditioning is a fundamental obstacle.

2. **Characterize the natural signal class.**
   Move beyond S_φ(K) (which is tautologically matched to the basis) to a class
   defined by physical or statistical properties. This would answer "what signals
   does RFT work for?"

3. **Prove column quasi-monochromaticity.**
   This would make the columns interpretable as frequency-localized atoms, giving
   the basis a physical meaning beyond "unitary rotation of Φ."

4. **Address the continuous-time gap.**
   Even a partial result — e.g. interpolation error bounds for moderate
   oversampling — would show direction.

5. **External validation of crypto primitives.**
   The structured-SIS assumption is non-standard. Either prove indistinguishability
   from uniform or get external cryptographic review.

---

## Relationship to Existing Repo Documents

- `docs/THEOREMS_RFT_IRONCLAD.md` — **Ground truth for proven results**. Theorems 1–9, 13, 15 are proved. Observations 11, 14 and Non-theorem 12 are correctly labeled.
- `docs/research/RFT_PERFORMANCE_THEOREM.md` — **Model for tone and rigor**. Correctly separates hard theorems, conditional propositions, empirical findings.
- `docs/NON_CLAIMS.md` — **Correctly denies** universal superiority, quantum computing, and quantum speedup.
- `docs/research/RFT_WAVE_COMPUTER_THEOREM.md` — ⚠️ Title says "Theorem" but status is "Experimental/Proposed." Needs status header correction.
- `docs/NOVEL_ALGORITHMS.md` — ⚠️ Theorem 1.3 is empirical, not proved. Needs relabeling.
- `docs/research/THE_PHI_RFT_FRAMEWORK_PAPER.tex` — ⚠️ "Theorem 3 (Massive Sparsity)" is empirical. Needs relabeling.

---

*This document does not contain legal advice. Patent scope is determined by the
claims as allowed, not by this research status memo.*
