# Fibonacci-frame Mobius prefix experiment

**EXPERIMENTAL.** This computes finite floating-point estimates. It does not
prove or disprove the Riemann hypothesis, provide interval certificates, or
establish constants valid over an infinite range.

See [ANALYTIC_BOUNDS.md](ANALYTIC_BOUNDS.md) for the separate mathematical
derivations: a conservative uniform conditioning bound at Fibonacci sizes,
a raw prefix coefficient estimate, and the explicitly unproved arithmetic
hypothesis that would be sufficient for RH. These do not establish an
infinite-range bound for the plotted records at every positive epsilon.
See [WEIGHTED_INTERACTIONS.md](WEIGHTED_INTERACTIONS.md) for the exact
interaction bound, the limits of transferring Davenport's estimate, and
the proof that the proposed coefficient hypothesis would also imply
uniform additive cancellation and Dirichlet GRH.

See [ARITHMETIC_ROUTES.md](ARITHMETIC_ROUTES.md) for the structural audit:
exact prime pairing; signed multiplicative controls that match any finite
Mobius prefix yet eventually have linear sums; a concentration bound for
independent random signs; and an RH-equivalent spatial approximation criterion
with explicit infinite-tail certificates. It also gives an exact example of
zero finite-grid error with positive full error. Run the small algebra checks
without extending the growth scan:

```bash
python3 runtime/experiments/rft_mobius/check_arithmetic_routes.py
```

## Test the coefficient and interaction hypothesis

Run the separate raw-coefficient probe in Codespaces:

```bash
bash runtime/experiments/rft_mobius/run_weighted_probe.sh --limit 2584
```

[EMPIRICAL] It measures every entry of `c_q=Phi_q^(-1)mu_q` at every
Fibonacci dimension at or below the limit. `max |c|` is the largest
coefficient; `max R` is the largest signed off-diagonal interaction from
WEIGHTED_INTERACTIONS.md. `RMS` describes average coefficient size.
`random max` uses random signs on the same squarefree support with a fixed
seed. A pure-wave control checks the expected single coefficient `sqrt(q)`.
These are raw coefficients, not the original polar-prefix bound B.

Results default to `runtime/experiments/rft_mobius/results/weighted_probe/`:
`frames.csv`, `summary.json`, `weighted_probe.png`, and one JSON/NPZ pair
per completed frame. The NPZ contains all coefficients, diagonal terms,
interactions, and random-control coefficients. Completed frames resume
automatically; an interrupted current frame is recomputed. Extend a run with:

```bash
bash runtime/experiments/rft_mobius/run_weighted_probe.sh --limit 4181
```

Changing the probe code or random seed requires a different `--output`
directory; existing checkpoints are not silently mixed. `--eps 0.01 0.05 0.1`
sets the finite records to report. The output checks reconstruction, selected
inverse rows, and direct cross-term expansions at extrema. It is floating-point
evidence, not rigorous interval arithmetic. A flat record is not an infinite
bound; exceeding a previously observed record does not disprove RH.
Dense inversion costs O(q^3) time and O(q^2) memory. Non-Fibonacci dimensions
are not tested by this probe.

[EMPIRICAL] The probe was run through q=2584 with seed 20261030. The largest
coefficient over those frames was 3.3493226 at q=610; at q=2584 it was
3.2643228. Finite coefficient constants for epsilon 0.01, 0.05, and 0.1 were
3.14125850, 2.44895533, and 1.86290504. Numerical checks passed. Resume
preserved completed checkpoints, and extending 13 to 34 matched a fresh run.

## Original prefix-bound experiment

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
