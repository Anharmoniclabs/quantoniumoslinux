# Current Research Position

**Date:** April 2026 (updated April 9, 2026 with gap audit results)
**Purpose:** Public-research-safe summary of what the RFT project has established. Written for a technically literate reader who has not seen the codebase. No hype.

---

## What RFT Is

The Resonant Fourier Transform (RFT) is a unitary transform on ℂ^N defined by:

1. Choose φ-spaced frequencies: f_k = frac((k+1)φ), where φ = (1+√5)/2 (golden ratio).
2. Form the raw basis matrix: Φ[n,k] = exp(i2πf_k n) / √N.
3. Canonicalize: U = Φ(Φ†Φ)^{-1/2} (Löwdin / symmetric orthogonalization).

The result is the unique unitary matrix closest to Φ in Frobenius norm (polar factor). It is not a DFT kernel (frequencies are irrational), not a fractional Fourier transform, and not a linear canonical transform.

## What Is Proved

All proofs are in `docs/THEOREMS_RFT_IRONCLAD.md`.

- **Φ has full rank** for all N ≥ 1 (Vandermonde + irrationality argument). (Thm 1)
- **U is unitary.** (Thm 2)
- **U is the unique polar factor** of Φ. (Thm 9)
- **Φ-sparse signals are DFT-dense.** Signals K-sparse in the φ-basis require dense DFT representation. (Thm 15)
- **The fast variant Ψ = D_φ C_σ F is unitary** and computable in O(N log N). (Thms 3, 13)
- **U does not arise from DFT, LCT, or FrFT families.** (Thm 6 + Prior Art analysis P1–P2)
- **Entropic uncertainty relation** holds for U with explicit bound. (Thm 8)
- A **twisted convolution** identity holds — but this is trivially true for any unitary. (Thm 4)
- **phi-RIP bound:** δ_s ≤ (s−1)·√5/k — sparse recovery guaranteed for k ≥ 8. (April 2026)
- **Sparsity / approximation bound (Fibonacci N):** For 1-sparse signals in the φ-basis, the dominant canonical coefficient captures α₁ ≥ (G^{1/2})_{jj}² ≥ 0.9513 of total energy (verified tight at N=89). Union bound extends to k-sparse for k ≤ 20. (Gap Audit, April 2026)
- **RFT Aliasing Theorem (Fibonacci N):** At Fibonacci N, the continuous interpolant x̂(t) = Φ(t)G⁻¹Φ†x satisfies: (1) x̂(t_n) = x_n exactly for all x ∈ ℂ^N (discrete reconstruction is exact, ‖P−I‖_F = 6.25×10⁻¹⁵ at N=89); (2) Pythagorean decomposition: ‖x_cont − x̂_cont‖²_{L²} = ‖(x_cont)_⊥‖²_{L²} (energy outside φ-frequency span); (3) direct analogue of Shannon aliasing. (Gap Audit, April 2026)

## What Is Not Proved

- Gram conditioning at arbitrary (non-Fibonacci) N — κ(Φ†Φ) can exceed 1000 at non-Fibonacci lengths. At Fibonacci N, κ < 2.849 empirically up to N=1597, with a proof sketch via Three-Distance Theorem and Beurling density. One lemma remains: formalize κ(G) ≤ 2.849 analytically for all Fibonacci N.
- Fast canonical transform for arbitrary N. At Fibonacci N, Newton-Schulz converges in 39-42 iterations (each O(N log N) via Toeplitz FFT), giving O(N log N) total — but this is conditional on the Gram conditioning lemma above.
- Column interpretability (why U columns appear quasi-monochromatic).
- Natural signal class characterization (what real signals are φ-sparse).
- Hardness of the structured-SIS problem underlying the crypto subsystem.
- Any form of "quantum advantage." RFT is entirely classical.
- Extension of the aliasing theorem and sparsity bounds beyond Fibonacci N.

## What Is Empirical

- Compression benchmarks showing modest advantages (+0.1 to +1.8 dB) over FFT/DCT on specific datasets (bearing vibrations, ECG, synthetic chirps). Not independently replicated.
- Column quasi-monochromaticity observed in DFT analysis of U columns.
- Fibonacci-index fast regime in recursive variant. Studied numerically, no theorem.
- Crypto avalanche metrics (40–60%) from internal testing. Not externally audited.
- **Gram conditioning at Fibonacci N:** κ(G) converges to ~2.849, λ_min ∈ [0.5218, 0.5423], stable up to N=1597. At non-Fibonacci N, λ_min can collapse to 0.0015 (κ up to 1178). The Fibonacci restriction is essential.
- **EEG biomarker detection:** Beta band SNR +1.65 dB over DFT on 36 Alzheimer's patients (p=1.4×10⁻²¹), Gini sparsity separation across 4 bands. Single dataset (OpenNeuro), not independently replicated.
- **SU(2) torus period finding:** Classical simulation using Wilson loop operators on a torus mesh factors semiprimes N<1000 with 100% recovery. Empirical complexity O(log³N) to O(log⁴N) (R²=0.988). **This is classical floating-point computation, not a quantum algorithm.** Rank-1 detection rate 27–67% depending on method. Untested beyond N<1000.

## Relationship to Prior Art

| Prior art family | Relationship to RFT |
|-----------------|---------------------|
| DFT / FFT | Different frequency grid (irrational vs rational). Proved distinct (Thm 6). |
| NUDFT (non-uniform DFT) | Φ is a specific NUDFT matrix. U is its polar factor — a new step. |
| LCT / FrFT | Affine symplectic structure cannot produce irrational φ-phases (Thm 6, P2). |
| Frame theory | Φ is a tight-ish frame; U is its canonical tight frame. Standard construction, specific grid. |
| Wavelets | Different: RFT has no multi-resolution structure. Both offer alternatives to DFT. |

## What This Project Does Not Claim

- No quantum computation or quantum advantage. The SU(2) torus work is a classical simulation — it does not claim to break RSA or factor large integers efficiently on classical hardware.
- No universal superiority over FFT/DCT.
- No production-ready cryptographic security.
- No claim that the fast variant Ψ inherits canonical U's spectral properties.
- No claim that Gram conditioning or sparsity results extend beyond Fibonacci N without further proof.

## Patent

USPTO Application 19/169,399 ("Hybrid Computational Framework for Quantum and Resonance Simulation"), priority date April 3, 2025. Four claims covering: (1) symbolic RFT engine, (2) crypto subsystem, (3) geometric hashing, (4) hybrid integration. The patent covers specific implementations, not the mathematics.

## License

Dual-licensed: AGPL-3.0 for non-claiming code; SPX-Proprietary + Claims-NC for patent-practicing files.

---

*This document is the public-facing research summary. For the open
problem catalog, see `docs/research/OPEN_PROBLEMS.md`.*
