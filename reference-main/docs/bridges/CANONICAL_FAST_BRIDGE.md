# Canonical ↔ Fast Bridge (RFT)

This note defines the two main RFT-related operators in the repo, explains why they are currently **distinct mathematical objects**, and provides a reproducible evidence workflow.

## Definitions (as implemented)

### Canonical RFT (Gram-normalized φ-grid)

- Implemented in `algorithms/rft/core/resonant_fourier_transform.py` as `rft_basis_matrix(N,N,use_gram_normalization=True)`.
- Builds an exponential basis using folded φ-frequencies `frac((k+1)·φ)` and then applies **Gram / Löwdin normalization** to obtain a unitary operator at finite `N`.

### Fast / Hybrid FFT-RFT (Claim 1 — O(N log N) optimization)

- Specified in `docs/HYBRID_FFT_RFT_ALGORITHM.md`.
- Implemented natively in `src/rftmw_native/` as `forward_hybrid()` / `inverse_hybrid()` (unitary, `O(N log N)`).
- Operator form:
  - `H = Dφ · F` where `F` is the unitary FFT matrix and `Dφ[k,k] = exp(i·2π·frac((k+1)·φ))`.

## What is “the gap”?

`docs/algorithms/rft/RFT_DEVELOPER_MANUAL.md` states:
- the **Canonical Form** (Gram-normalized) is the mathematical reference, and
- the **Fast Form** (factorized / hybrid) is the practical implementation,
- but there is no formal proof that the fast form is a limit of the canonical form.

In this repo, the canonical operator depends on φ through **folded φ-frequencies** inside the exponent, whereas the hybrid operator is **FFT followed by a diagonal φ-phase modulation**. These are generally not the same operator.

## Reproducible evidence

Run:

```bash
python scripts/bridge_canonical_vs_hybrid.py
```

This writes a JSON report under `results/` that includes:
- unitarity checks for both operators,
- operator-level difference metrics (Frobenius, spectral norm),
- action-level differences on random unit vectors and on a φ-quasi-periodic signal.

## Closing the gap (what “done” looks like)

There are two valid directions, depending on what you want to claim and validate:

1. **Prove equivalence or convergence**: provide a theorem + proof that a specific fast operator equals (or converges to) the canonical operator under stated assumptions/normalizations.
2. **Prove controlled approximation for target workloads**: show that for a defined class of signals/tensors (e.g., φ-structured low-entropy tensors used by RFTMW routing), the hybrid operator is an ε-approximation to the canonical operator in the relevant metric (e.g., reconstruction error, downstream task loss), with empirical + theoretical bounds.

Until then, treat “canonical” (reference math) and “hybrid” (fast implementation) as **separate, explicitly-labeled operators** with different validation obligations.

