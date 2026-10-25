# Reproducibility Guide: Unitarity and Gram Normalization

> **Audience:** Independent reviewers, peer reviewers, and anyone attempting to
> reproduce the machine-precision unitarity claims for the Resonant Fourier
> Transform (RFT) canonical basis.

---

## 1. The Claim

The canonical RFT basis matrix `Phi_tilde` (shape N×N) satisfies:

```
||Phi_tilde^H  Phi_tilde - I||_F  <  1e-12   for all N in {16, 32, 64, 128, 256, 512, 1024}
kappa(Phi_tilde) = 1.0000000000     (condition number, 10 decimal places)
```

These are verified by the CI tests:
- `test_conditioning_canonical_U_is_unitary` (N=16…512)
- `test_unitarity_error_at_N512`
- `test_unitarity_error_at_N1024` (locally verified: error=1.84e-13)

and by `rft_unitary_error_canonical(N)` in
`algorithms/rft/core/resonant_fourier_transform.py`.

---

## 2. The Normalization Formula

The raw (non-unitary) basis is the φ-grid resonance kernel:

```
Phi[n, k] = (1/sqrt(N)) * exp(2*pi*i * frac((k+1)*phi) * n)

where phi = (1 + sqrt(5)) / 2  (golden ratio)
      frac(x) = x mod 1
      n = 0 .. N-1   (time index)
      k = 0 .. N-1   (frequency index)
```

The Gram-normalized (canonical unitary) basis is:

```
Phi_tilde = Phi * G^{-1/2}    where G = Phi^H Phi
```

This is the standard Löwdin orthogonalization formula. The critical detail
that determines whether you reach machine precision or not is **how G^{-1/2}
is computed**.

---

## 3. Why Naive Implementations Fail

A common independent implementation attempt uses:

```python
# WRONG — do not use this
import scipy.linalg
G = Phi.conj().T @ Phi
G_inv_sqrt = np.linalg.inv(scipy.linalg.sqrtm(G))   # ill-conditioned path
Phi_tilde = Phi @ G_inv_sqrt
```

This **does not yield machine-precision unitarity** for the φ-grid Gram
matrix because:

1. `scipy.linalg.sqrtm` uses a Schur decomposition that is not
   backward-stable for nearly degenerate or ill-conditioned matrices.
2. `np.linalg.inv` applied to an already nearly-singular result amplifies
   floating-point error by `kappa(G)`, which grows with N
   (verified by `test_conditioning_gram_matrix_scaling`).
3. The compounded error can exceed 0.1 in `||Phi_tilde^H Phi_tilde - I||_F`,
   making the result appear non-unitary.

---

## 4. The Correct Implementation

The repository uses the **spectral (eigendecomposition) form** of G^{-1/2}:

```python
import numpy as np

N = 128   # any N
n = np.arange(N, dtype=np.float64)
from algorithms.rft.core.resonant_fourier_transform import rft_frequency
freqs = np.array([rft_frequency(k) for k in range(N)])
Phi = np.exp(2j * np.pi * np.outer(n, freqs)) / np.sqrt(N)

# --- Gram normalization (spectral form) ---
G = Phi.conj().T @ Phi                          # 1. Gram matrix G = Phi^H Phi
eigvals, eigvecs = np.linalg.eigh(G)            # 2. Hermitian eigendecomposition G = V L V^H
eigvals = np.maximum(eigvals, 0.0)              # 3. clamp negative numerical noise to zero
G_inv_sqrt = (eigvecs
              @ np.diag(1.0 / np.sqrt(eigvals + 1e-300))
              @ eigvecs.conj().T)               # 4. G^{-1/2} = V L^{-1/2} V^H
Phi_tilde = Phi @ G_inv_sqrt                    # 5. Phi_tilde = Phi G^{-1/2}

# Verify
err = np.linalg.norm(Phi_tilde.conj().T @ Phi_tilde - np.eye(N), ord='fro')
print(f'unitarity error = {err:.2e}')            # expect < 1e-12
```

### Why this works

- `np.linalg.eigh` exploits Hermitian (self-adjoint) structure. It returns
  an orthonormal eigenvector matrix V satisfying `V^H V = I` to ~1e-16.
- G^{-1/2} is formed by **per-eigenvalue scalar operations** — no matrix
  inversion of an ill-conditioned matrix.
- The `+ 1e-300` floor prevents division by zero without meaningfully
  perturbing any eigenvalue above machine epsilon.
- Because V is numerically orthonormal, the product `V diag(...) V^H` is
  also Hermitian and well-conditioned.

**Mathematical identity (exact arithmetic):**

```
Phi_tilde^H Phi_tilde
  = (G^{-1/2})^H  Phi^H Phi  G^{-1/2}
  = G^{-1/2} G G^{-1/2}
  = I
```

---

## 5. Independent Cross-Check: Polar Decomposition

The file `algorithms/rft/core/transform_theorems.py` provides an
**independent route** to the same unitary via `scipy.linalg.polar`:

```python
import scipy.linalg
from algorithms.rft.core.transform_theorems import raw_phi_basis

R = raw_phi_basis(N)            # same as Phi above
U, P = scipy.linalg.polar(R)   # R = U P,  U unitary,  P pos. semidefinite
```

`scipy.linalg.polar` internally computes the SVD `R = W Sigma V^H` and
returns `U = W V^H`. Unitarity follows from the orthonormality of W and V —
no Gram matrix involved. Both routes produce the same matrix:

```python
err_cross = np.linalg.norm(U - Phi_tilde, ord='fro')
print(f'cross-method agreement = {err_cross:.2e}')   # expect < 1e-13
```

Agreement to 1e-13 between two algebraically independent algorithms is
strong evidence both are correct.

---

## 6. Verified Numerical Results

| N    | Method          | `‖Phi_tilde^H Phi_tilde - I‖_F` | `kappa(U)`           |
|------|-----------------|----------------------------------|----------------------|
| 16   | eigh spectral   | < 1e-14                          | 1.0000000000         |
| 64   | eigh spectral   | < 1e-13                          | 1.0000000000         |
| 128  | eigh spectral   | < 1e-13                          | 1.0000000000         |
| 256  | eigh spectral   | < 1e-13                          | 1.0000000000         |
| 512  | eigh spectral   | < 1e-13                          | 1.0000000000         |
| 1024 | eigh spectral   | 1.84e-13 (locally verified)      | 1.0000000000         |
| 64   | polar (SVD)     | < 1e-13                          | 1.0000000000         |
| 512  | polar (SVD)     | < 1e-13                          | 1.0000000000         |

All CI results are reproducible with:

```bash
pytest tests/proofs/test_rft_transform_theorems.py -v -k "conditioning or unitarity"
```

---

## 7. Round-Trip Reconstruction

For the Gram-normalized unitary basis:

```
x_reconstructed = Phi_tilde @ (Phi_tilde^H @ x) = Phi_tilde @ Phi_tilde^H @ x
```

Because `Phi_tilde^H Phi_tilde = I`, and since Phi_tilde is N×N square
and unitary, `Phi_tilde Phi_tilde^H = I` also holds. Round-trip error is
therefore bounded by the unitarity error:

```
||x - x_reconstructed||_2 / ||x||_2  <=  ||Phi_tilde Phi_tilde^H - I||_F  <  1e-12
```

Quick verification:

```python
from algorithms.rft.core.resonant_fourier_transform import rft_forward, rft_inverse
import numpy as np

rng = np.random.default_rng(42)
x = rng.standard_normal(128) + 1j * rng.standard_normal(128)
err = np.linalg.norm(x - rft_inverse(rft_forward(x))) / np.linalg.norm(x)
print(f'round-trip error = {err:.2e}')   # expect < 1e-12
```

---

## 8. Common Pitfalls for Independent Reproducers

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Using `scipy.linalg.sqrtm` + `np.linalg.inv` | Unitarity error > 0.01 | Use `np.linalg.eigh` spectral form |
| Using `np.linalg.eig` instead of `eigh` | Complex eigenvectors, non-Hermitian V^H V | Use `eigh` (Hermitian-aware) |
| Not clamping negative eigenvalues | NaN or large error at large N | Add `np.maximum(eigvals, 0.0)` |
| Comparing raw Phi (not Phi_tilde) | Appears non-unitary | Ensure `use_gram_normalization=True` |
| Using float32 | Precision floor ~1e-7, not 1e-13 | Use `complex128` throughout |

---

## 9. Source References

| File | What it contains |
|------|------------------|
| `algorithms/rft/core/resonant_fourier_transform.py` | `_rft_basis_matrix_cached` — spectral Gram normalization |
| `algorithms/rft/core/transform_theorems.py` | `canonical_unitary_basis` — polar decomposition cross-check |
| `tests/proofs/test_rft_transform_theorems.py` | `test_conditioning_canonical_U_is_unitary`, `test_unitarity_error_at_N1024` |
| `reproduce_results.sh` | Shell script to run full verification suite |
