<!-- SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC -->

# Fast RFT Algorithm — Validation Hypothesis

> **Status: WORK IN PROGRESS**  
> The findings documented here are derived from numerical experiments and constitute a working hypothesis. They are pending independent peer review and extended validation at larger N and on hardware targets (GPU, FPGA).

**Patent:** USPTO 19/169,399  
**Author:** Luis M. Minier  
**Date:** March 2026

---

## How to Run

```bash
# From repo root
python algorithms/rft/fast/rft_fft_benchmark.py
```

All results are deterministic (fixed seed=42). Requires only NumPy.

---

## Overview

The Rational Frequency Transform (RFT) is a unitary tight-frame operator defined on a golden-ratio irrational-frequency lattice. This document summarizes the current state of numerical validation of the Final Fast RFT Algorithm (Tier-1 exact implementation) and outlines the open hypotheses that remain to be verified.

---

## Validated Findings

### 1. Existence and Core Properties

The RFT synthesis matrix is defined as:

```
Phi_tilde = Phi (Phi^dag Phi)^{-1/2}   [Lowdin symmetric orthogonalization]
```

where the frequency lattice is `omega_k = frac((k+1) * phi)` for `k = 0..N-1` and `phi = (1+sqrt(5))/2`.

Numerical tests confirm:
- `Phi_tilde^dag Phi_tilde = I` holds to within double-precision machine epsilon across all tested N
- The tight-frame identity is satisfied for all tested dimensions

### 2. Perfect Reconstruction

For any input `x` in C^N with unit Euclidean norm:
- Forward: `x_hat = Phi_tilde^dag x`
- Inverse: `x_rec = Phi_tilde x_hat`
- Reconstruction error `||x - x_rec||_2 <= 1.0e-14` confirmed across all tested N

### 3. Numerical Stability (Real Run)

The Gram matrix `S = Phi^dag Phi` remains positive definite across all tested N:

| N   | Frame Error `\|\|Phi~^dag Phi~ - I\|\|_F` | Condition number kappa(S) |
|-----|------------------------------------------|---------------------------|
| 64  | 5.411e-14                                | 149.14                    |
| 128 | 7.804e-14                                | 103.48                    |
| 256 | 1.823e-13                                | 435.24                    |

All frame errors are within double-precision tolerance (~N * machine_epsilon). Condition numbers confirm practical invertibility for the offline Lowdin step.

### 4. Benchmark vs FFT — Real Results (Tier-1 Exact)

> Results from local run on development machine, March 26 2026.  
> 20 trials, 3 warmup, float64, precomputation excluded from timing.

| N   | RFT Recon Error | FFT Recon Error | RFT Time (ms) | FFT Time (ms) | RFT Std | FFT Std | Ratio  |
|-----|-----------------|-----------------|---------------|---------------|---------|---------|--------|
| 64  | 7.847e-15       | 2.383e-16       | 0.452         | 0.030         | 0.150   | 0.014   | 15.27x |
| 128 | 6.351e-15       | 2.781e-16       | 0.460         | 0.025         | 0.194   | 0.010   | 18.45x |
| 256 | 9.814e-15       | 2.607e-16       | 0.967         | 0.030         | 0.197   | 0.008   | 31.97x |

**What the numbers mean:**

- **Reconstruction accuracy:** Both RFT and FFT achieve machine-epsilon-level reconstruction. RFT errors (7.8–9.8e-15) are ~30× larger than FFT (2.4–2.8e-16) but both are well within double-precision tolerance. This confirms the tight-frame property is exact.
- **Runtime:** Tier-1 RFT is 15–32x slower than FFT at these dimensions. This is expected and predicted by the O(N^2) dense matrix-vector complexity vs FFT's O(N log N). The timing ratio growing from 15x at N=64 to 32x at N=256 is consistent with the O(N^2) vs O(N log N) scaling crossover.
- **The gap is algorithmic, not fundamental.** The Tier-1 implementation is an exact reference implementation. Tier-2 (NUFFT/Toeplitz) and Tier-3 (Fibonacci-recursive) are the paths to close this gap while preserving exact reconstruction.

> **Note on prior theoretical projections:** Earlier projections suggested RFT would be competitive or faster than FFT at N<=128 on some machines due to BLAS cache effects. The real run shows this is environment-dependent — on this machine, Python overhead dominates at these small N. The accuracy results match projections exactly.

---

## Open Hypotheses (Work in Progress)

The following have been analytically derived but not yet numerically confirmed at scale:

### H1 — Tier-2 Acceleration
**Hypothesis:** A NUFFT + Toeplitz exploitation of the Gram matrix structure reduces online complexity from O(N^2) to O(N log N) while preserving exact reconstruction.  
**Status:** Theoretical blueprint complete. Implementation pending.

### H2 — Tier-3 Fibonacci-Recursive Decomposition
**Hypothesis:** The phi-lattice admits a Fibonacci-recursive fast algorithm analogous to Cooley-Tukey FFT, achieving O(N log N) with exact arithmetic.  
**Status:** Theoretical only. Not yet implemented or benchmarked.

### H3 — Large-N Stability
**Hypothesis:** Condition number kappa(S) grows sub-exponentially for N > 256, remaining practically invertible through N = 4096.  
**Status:** Tested to N=256 only. Extended validation needed.

### H4 — Hardware Portability
**Hypothesis:** Phi_tilde and its conjugate transpose can be precomputed once and deployed as static weight matrices on GPU (CUDA) and FPGA with no loss of reconstruction accuracy.  
**Status:** Not yet tested on hardware.

### H5 — Streaming / Overlap-Add
**Hypothesis:** An overlap-add extension of the RFT supports causal real-time processing with the same reconstruction guarantees.  
**Status:** Not yet designed or validated.

---

## Algorithmic Forms (F1–F5)

All five forms are mathematically equivalent. F1 is implemented in Tier-1:

| Form | Expression | Complexity | Status |
|------|------------|------------|--------|
| F1 | `x_hat = Phi_tilde^dag x`, `x = Phi_tilde x_hat` | O(N^2) | ✅ Implemented, benchmarked |
| F2 | Via explicit S^{-1/2} Cholesky | O(N^2) | Numerically equivalent to F1 |
| F3 | Via eigendecomposition of S | O(N^2) | Equivalent, higher constant |
| F4 | NUFFT approximation of Phi products | O(N log N) | ⚠️ Tier-2, not yet implemented |
| F5 | Fibonacci-recursive decomposition | O(N log N) | ⚠️ Tier-3, theoretical only |

---

## Citation

```text
@misc{rft_fast_algorithm_2026,
  title  = {Fast RFT Algorithm: Tier-1 Exact Implementation and Validation},
  author = {Minier, Luis M.},
  year   = {2026},
  note   = {USPTO Patent Application 19/169,399 | Work in Progress},
  url    = {https://github.com/LMMinier/quantoniumos}
}
```
