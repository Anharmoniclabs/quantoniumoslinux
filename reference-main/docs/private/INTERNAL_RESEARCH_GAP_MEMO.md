# Internal Research Gap Memo

**Date:** April 2026
**Classification:** Tier 3 — Do Not Share
**From:** Automated audit (post-Bindel review)
**Purpose:** Brutally honest founder-level memo on what the project has, what it doesn't, and what to do about it.

---

## Executive Summary

After the Bindel exchange, the project's position is: mathematically interesting niche transform with solid finite-dimensional proofs, no continuous-time theory, no independently validated compression story, and a crypto subsystem resting on an unproved assumption. The overclaim language throughout the repo was significantly worse than the actual results warranted.

This memo catalogs every known gap, rates its severity, and suggests next steps.

---

## Gap 1: No Continuous-Time Theory

**Severity: Critical**

Every serious transform (Fourier, wavelet, Gabor) has a continuous-time parent. RFT does not. This means:
- No sampling theorem — cannot state when discrete samples suffice.
- No aliasing bound — cannot quantify discretization error.
- No interpolation formula — cannot reconstruct between sample points.
- No Hilbert-space formulation — cannot connect to functional analysis literature.

Without this, the entire framework sits in a "numerical curiosity" bucket. It's a well-defined finite-dimensional construction, but the claim that it represents something fundamental about signal structure is unsupported.

**Next step:** This is a pure math problem. Needs a collaborator from functional analysis or harmonic analysis. Not solvable by writing more code.

---

## Gap 2: Gram Conditioning Unknown

**Severity: High**

The canonical construction U = Φ(Φ†Φ)^{-1/2} depends on inverting the Gram matrix. If κ(G) grows superpolynomially, the canonical path is numerically useless at large N. We have empirical data up to N=512 and it looks okay, but "looks okay" is not a bound.

This connects to deep number theory (Diophantine approximation of φ-related numbers). The three-distance theorem guarantees some regularity, but translating that into spectral bounds on Vandermonde Gram matrices is open math.

**Next step:** Numerical study of κ(G) for N up to 4096 or 8192. Plot and look for scaling law. If polynomial — conjecture and cite Diophantine literature. If superpolynomial — the canonical path has a ceiling.

---

## Gap 3: Compression Story Is Thin

**Severity: High**

The compression benchmarks show +0.1 to +1.8 dB over FFT on specific datasets. That's real but modest, and:
- Only tested on internal pipeline.
- No standard benchmark methodology (PESQ, SSIM, PRD).
- No independent replication.
- The advantage comes from the φ-grid matching φ-quasi-periodic structure — which is tautological without a natural signal class theorem (Gap 4).

If someone asks "why would I use this instead of wavelets or DCT?", the honest answer is currently: "On these specific datasets, it does slightly better. We don't know why in general."

**Next step:** Run standardized benchmarks on public datasets with published methodology. Submit as a short paper without any overclaim. Let the numbers speak.

---

## Gap 4: No Natural Signal Class

**Severity: High**

Theorem 15 proves that φ-sparse signals are DFT-dense. True, but tautological: any signal constructed to be sparse in basis X will be dense in basis Y if X ≠ Y. The question is whether any real-world process produces signals that are naturally φ-sparse.

Empirically, bearing vibration data and certain ECG signals show better RFT concentration. But "better" doesn't mean "natural" — it could be that any non-harmonic quasi-periodic signal shows this, and the golden ratio is just one of many viable incommensurate grids.

**Next step:** Test compression with other irrational-grid transforms (π-grid, e-grid, √2-grid). If φ is special, understand why. If not, the claim narrows to "irrational grids are sometimes better than rational grids."

---

## Gap 5: Crypto Rests on Unproved Assumption

**Severity: Medium-High**

The RFT-SIS hash uses a structured matrix A_φ. Standard SIS hardness assumes random A. The hybrid construction A_φ + R (Theorem 7.4) is uniform, but that only means the matrix *looks* random — it doesn't prove that solving SIS on it is hard.

Internal avalanche testing (40–60%) is necessary but not sufficient for security. Real crypto needs:
- A reduction from a known-hard problem (LWE, standard SIS, etc.).
- OR external audit by a lattice cryptography expert.

**Next step:** Either prove a reduction or get an external opinion from a lattice cryptographer. Until then, crypto claims should carry a "NOT AUDITED" disclaimer.

---

## Gap 6: Overclaim Debt

**Severity: Medium (being fixed)

The repo contains 13+ files with overclaim language inherited from early development:
- "quantum advantage" (it's classical)
- "genuine quantum entanglement" (it's phase correlation)
- "exponential speedup" (unqualified)
- "paradigm shift" (hype)
- "breakthrough" (hype)

This language is being systematically corrected as part of the current documentation update.

**Next step:** Complete the overclaim corrections. Verify with grep. Never use unqualified superlatives in technical documentation again.

---

## Gap 7: No External Validation

**Severity: Medium**

No external researcher has replicated any result. No peer-reviewed publication exists. The closest to external review was the Bindel exchange, which was constructive but is not peer review.

**Next step:** Write up the finite-dimensional results (Theorems 1-2, 6, 15) as a short note and submit to a signal processing or applied math venue. Keep claims minimal. Let reviewers push back.

---

## Gap 8: Fast Canonical Transform Missing

**Severity: Medium**

The canonical U is the correct mathematical object, but it costs O(N³) to compute. The fast Ψ is O(N log N) but is a different operator. Working at scale requires either:
- A fast algorithm for U (via Toeplitz structure of G), or
- A proof that Ψ approximates U in some useful norm, or
- Accepting that the canonical U is theoretical and Ψ is practical.

Currently taking option 3 by default, but this should be explicit.

**Next step:** Investigate Toeplitz inverse square root algorithms. Even O(N log²N) would be transformative. If it works, the canonical path becomes practical.

---

## Summary Table

| Gap | Severity | Type | Fixable by Code? |
|-----|----------|------|------------------|
| Continuous-time theory | Critical | Math | No — needs collaborator |
| Gram conditioning | High | Math + Numerics | Partially — numerical study, then need proof |
| Compression story | High | Engineering + Science | Partially — better benchmarks, need theory |
| Natural signal class | High | Math | No — needs theorem |
| Crypto assumption | Medium-High | Crypto | No — needs reduction or audit |
| Overclaim language | Medium | Documentation | Yes — in progress |
| No external validation | Medium | Publication | Yes — write and submit |
| Fast canonical transform | Medium | Math + CS | Partially — Toeplitz path |

---

*This memo is for the founder and core contributors. It intentionally avoids
softening language. If something isn't proved, it says so.*
