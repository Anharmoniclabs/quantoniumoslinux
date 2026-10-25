# Proven / Empirical / Open — Classification Matrix

**Date:** April 2026 (updated April 9, 2026 with gap audit results)
**Purpose:** Single-source-of-truth matrix separating what is proved, what is empirical, and what is open.
**Rule:** "Proven" means a complete proof exists in `docs/THEOREMS_RFT_IRONCLAD.md` or equivalent. Everything else is honestly labeled.

---

## Status Key

| Label | Meaning |
|-------|---------|
| **PROVEN** | Complete mathematical proof exists. Theorem number cited. |
| **EMPIRICAL** | Supported by benchmarks or numerical experiments. No proof. |
| **OPEN** | Unsolved. May have partial evidence or conjectures. |
| **INTERNAL-IMPL** | Repo implementation detail. Not a mathematical claim. |

---

## Classification Matrix

| # | Topic | Status | Evidence / Notes |
|---|-------|--------|------------------|
| 1 | **Canonical discrete construction** (U = Φ(Φ†Φ)^{-1/2}) | **PROVEN** | Theorem 2 (IRONCLAD). Löwdin polar orthogonalization of full-rank Φ. |
| 2 | **Full rank of raw Φ** | **PROVEN** | Theorem 1 (IRONCLAD). Vandermonde + irrationality of φ. |
| 3 | **Raw vs canonical coefficients** | **PROVEN** | c_raw = G^{-1}Φ†x (dual frame, Thm 15 Part 1). c = U†x (canonical). Two distinct objects. |
| 4 | **Exact full-dimension reconstruction** | **PROVEN** | x = Uc where c = U†x. Direct consequence of unitarity (Thm 2). |
| 5 | **Unitarity of canonical U** | **PROVEN** | Theorem 2 (IRONCLAD). U†U = I by construction. Numerical defect < 2×10⁻¹³ at N ≤ 256. |
| 6 | **Unitarity of fast RFT Ψ** | **PROVEN** | Theorem 3 (IRONCLAD). Ψ = D_φ C_σ F. Product of unitaries. |
| 7 | **Gram conditioning behavior** | **EMPIRICAL (Fibonacci N) / OPEN (arbitrary N)** | At Fibonacci N: κ(G) < 2.849, λ_min ∈ [0.5218, 0.5423] up to N=1597. Proof sketch via Three-Distance Theorem + Beurling density. One analytic lemma remaining. At arbitrary N: κ can exceed 1000. Non-theorem 12 (IRONCLAD) still applies for general N. |
| 8 | **Fast canonical transform** (O(N log N) or better) | **CONDITIONAL (Fibonacci N) / OPEN (arbitrary N)** | At Fibonacci N: Newton-Schulz converges in 39-42 iterations, each O(N log N) via Toeplitz FFT. Total O(N log N). Conditional on proving κ(G) ≤ 2.849 analytically (the shared lemma from row 7). Not established for arbitrary N. |
| 9 | **Fast RFT Ψ complexity** | **PROVEN** | Theorem 13 (IRONCLAD). O(N log N) — one FFT + diagonal scaling. |
| 10 | **Column interpretability** (quasi-monochromaticity of U columns) | **EMPIRICAL** | Columns of U appear quasi-monochromatic in frequency domain. No theorem. |
| 11 | **Shift/circulant diagonalization** (U†SU diagonal?) | **PROVEN NEGATIVE** | Direct computation at N=64: diagonal energy ~0.9519, off-diagonal ~0.0481. U does NOT diagonalize cyclic shifts. DFT does. |
| 12 | **Twisted convolution theorem** | **PROVEN (trivial)** | Theorem 4 (IRONCLAD). Holds for ANY unitary by definition. Not a structural property of RFT specifically. |
| 13 | **Convolution/FFT-style fast path via shift diag** | **OPEN (unlikely)** | Would require U†SU to be diagonal. It is not (row 11). No fast convolution analogue to FFT exists for RFT currently. |
| 14 | **Concentration/compression on φ-quasi-periodic signals** | **PROVEN (Fibonacci N, 1-sparse) / EMPIRICAL (general)** | Sparsity bound proved: α₁ ≥ (G^{1/2})_{jj}² ≥ 0.9513 at N=89 Fibonacci. Union bound: α_k ≥ 1 − k(1−α₁), valid k ≤ 20. >95% energy in <0.1N coefficients observed at general N. phi-RIP: δ_s ≤ (s−1)√5/k, sparse recovery for k ≥ 8. |
| 15 | **Exact sparsity of S_φ(K) in φ-grid vs dense DFT leakage** | **PROVEN** | Theorem 15 (IRONCLAD). φ-grid signals have K-sparse φ-coefficients and dense DFT coefficients. |
| 16 | **Not a DFT kernel** | **PROVEN** | Theorem 6 (IRONCLAD). Rational/irrational frequency contradiction. |
| 17 | **Not an LCT/FrFT variant** | **PROVEN** | Theorem 6 + P2 (IRONCLAD). Irrational φ-phases cannot arise from affine LCT structure. |
| 18 | **Entropic uncertainty bound** | **PROVEN** | Theorem 8 (IRONCLAD). Maassen–Uffink applied to canonical U. |
| 19 | **Uniqueness of polar factor** | **PROVEN** | Theorem 9 (IRONCLAD). U is the unique unitary in polar decomposition of Φ (full rank). |
| 20 | **Crypto: Feistel + RFT-SIS hash** | **INTERNAL-IMPL** | 48-round Feistel with RFT-SIS mixing. Avalanche 40-60%. Passes internal validation. NOT externally audited. Structured-SIS assumption is non-standard (Thm 7.2). |
| 21 | **Crypto: hybrid uniform construction** | **PROVEN** | Theorem 7.4 (IRONCLAD). A = A_φ + R (mod q) is uniform by group-shift invariance. |
| 22 | **Crypto: structured-SIS hardness** | **OPEN** | Theorem 7.2 (IRONCLAD) explicitly states: structured A needs a new assumption or indistinguishability proof. Not resolved. |
| 23 | **Geometric hashing** | **INTERNAL-IMPL** | Topological embedding + geometric hash. No external validation or formal analysis. |
| 24 | **Routing / variant selection (RFTMW)** | **INTERNAL-IMPL** | Signal classifier routes to best variant. Engineering heuristic, not a mathematical result. |
| 25 | **Hardware / RTL / systolic arrays** | **INTERNAL-IMPL** | SystemVerilog implementations of weight-stationary systolic arrays (v1–v23). FPGA synthesis reports exist. No formal verification of transform correctness in hardware (only testbench validation). |
| 26 | **Continuous-time analogue** | **PROVEN (Fibonacci N) / OPEN (arbitrary N)** | RFT Aliasing Theorem proved at Fibonacci N: x̂(t_n) = x_n exactly (‖P−I‖_F = 6.25×10⁻¹⁵ at N=89). Pythagorean decomposition: aliasing error = energy outside φ-frequency span. RKHS interpolant is the Shannon analogue. Arbitrary N and integral-operator formulation remain open. |
| 27 | **Natural signal class** (what real signals match S_φ(K)?) | **OPEN** | S_φ(K) is tautologically matched to the basis. Physical or statistical characterization of "naturally φ-quasi-periodic" signals does not exist. |
| 28 | **Interpolation / aliasing** | **PROVEN (Fibonacci N) / OPEN (arbitrary N)** | At Fibonacci N: exact sample-point interpolation proved; continuous energy capture verified (100% for φ-sparse, 89-99% for general signals). Aliasing bounds via Pythagorean decomposition. Arbitrary N not addressed. |
| 29 | **Medical domain compression** (ECG, EEG, bearing fault) | **EMPIRICAL** | Reported +0.1 to +1.8 dB over FFT on CWRU bearing faults. Not independently validated. |
| 30 | **KV-cache / LLM integration** | **EMPIRICAL** | Perplexity benchmarks exist. RFT-based weight compression tested. Results are internal. |
| 31 | **Convergence of fast Ψ to canonical U** | **OPEN (likely false)** | The surrogate bridge path was abandoned. Ψ and U are separate constructions. No convergence proof exists or is expected. |
| 32 | **phi-RIP bound** | **PROVEN** | δ_s ≤ (s−1)·√5/k. Sparse recovery guaranteed for k ≥ 8. (April 2026) |
| 33 | **EEG biomarker detection** | **EMPIRICAL** | Beta band SNR +1.65 dB over DFT on 36 Alzheimer's patients (p=1.4×10⁻²¹), Gini sparsity separation across 4 bands. Single dataset (OpenNeuro). Not independently replicated. |
| 34 | **SU(2) torus period finding** | **EMPIRICAL** | Classical SU(2) Wilson loop simulation factors semiprimes N<1000 with 100% recovery. O(log³–log⁴N) empirical complexity (R²=0.988). Rank-1 detection 27–67%. NOT a quantum algorithm — classical floating-point only. Untested beyond N<1000. |
| 35 | **ECDLP holonomy rate dip** | **EMPIRICAL (no speedup)** | Discrete log detectable via zero-rate dip in SU(2) holonomy (10/10 recovery on toy curve). Cost: O(ord_P) = O(p) — same as brute force. No speedup over BSGS. Signal confirmation only. |

---

## Summary Statistics

| Status | Count |
|--------|------:|
| PROVEN (full or Fibonacci-N) | 17 |
| EMPIRICAL | 7 |
| CONDITIONAL | 1 |
| OPEN | 6 |
| INTERNAL-IMPL | 4 |
| **Total** | **35** |

---

## Notes

- Theorem numbers reference the project's canonical theorem document.
- Benchmark and implementation details are maintained in the private repository.
- Rows marked INTERNAL-IMPL are engineering artifacts, not mathematical claims.

---

*"Proven" means a mathematical proof exists and has been written down, not that
someone ran pytest. Test passage is evidence, not proof.*
