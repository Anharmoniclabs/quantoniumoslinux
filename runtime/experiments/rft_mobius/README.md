# Fibonacci-frame Mobius prefix experiment

**EXPERIMENTAL.** This computes finite floating-point estimates. It does not
prove or disprove the Riemann hypothesis, provide interval certificates, or
establish constants valid over an infinite range.

From the repository root in Codespaces:

```bash
source .phistate-env/bin/activate
bash runtime/experiments/rft_mobius/run_fibonacci_prefix.sh --limit 2584
```

The launcher uses the active environment, or the repository's `.phistate-env`
if none is active. It installs NumPy and Matplotlib only when either import
is unavailable. It executes the checked-in Python file directly.

Each integer cutoff `m` uses the smallest Fibonacci frame `F >= m`. For the
canonical golden-grid polar factor `U_F`, define the prefix indicator `p_m`
and the signed vector `mu_F = (mu(1), ..., mu(F))`. In exact arithmetic,

```text
M(m) = (U_F^* p_m)^* (U_F^* mu_F)
B_F(m) = sum_k |(U_F^* p_m)_k| |(U_F^* mu_F)_k|
|M(m)| <= B_F(m)
```

The vector `mu_F` includes every value through the frame size, even when
`m < F`. Every cutoff from 1 through `--limit` is included exactly once;
non-Fibonacci limits use a larger containing frame and stop at the requested
cutoff. For a limit of 2000, for example, the largest frame is 2584.

Outputs are stored in separate timestamped directories under `results/`
(ignored by git):

- `prefixes.csv`: one row per cutoff, frame assignment, exact integer Mertens
  sum, numerical bound estimate, normalized estimate, reconstruction error.
- `progress.json`: completed cutoff and frame diagnostics, updated after each
  completed frame. This is a progress record, not an automatic resume feature.
- `summary.json`: complete coverage, finite epsilon constants and numerical
  diagnostics.
- `fibonacci_prefix.png`: normalized bounds and finite running maxima.

Use `--output-root PATH` to choose another parent directory. Dense SVD costs
O(F^3) time and O(F^2) memory per frame; this is not the fast Toeplitz path.
Set `RFT_THREADS` to change the default single-thread BLAS setting.

## Validation

[EMPIRICAL] The supplied Python implementation was run through cutoff 2584:
maximum estimated `B_F(m)/sqrt(m)` was 2.84160924 and maximum reconstruction
error was approximately 2.43e-13. Finite constants for epsilon 0.01, 0.05 and
0.1 were 2.62688724, 1.95219059 and 1.46820032 respectively. These maxima are
over the tested range only. Numerical values can vary slightly by platform.

Möbius values were independently checked by trial division through 2584.
The SVD polar factor and prefix bounds were compared with a Gram-eigenvalue
inverse-square-root construction in a small frame. Non-Fibonacci limit
handling was also checked. Runtime checks cover the divisor identity,
unitarity, reconstruction, constant-signal control and triangle/Cauchy bounds.

Repeat the small independent checks with:

```bash
python3 runtime/experiments/rft_mobius/test_fibonacci_prefix.py
```
