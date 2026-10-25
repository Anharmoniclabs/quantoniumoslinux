# Technical Details

**Status:** cleaned for the April 2026 repo audit.

This page gives a conservative technical snapshot of the repository. It avoids stale path references and separates proven results from empirical ones.

---

## 1. Canonical RFT definition

The strongest mathematical object in the repo remains the discrete canonical RFT built from a φ-spaced basis and Löwdin/polar orthogonalization:

$$
\Phi_{n,k} = \frac{1}{\sqrt{N}} \exp\bigl(j 2\pi\,\mathrm{frac}((k+1)\phi) n\bigr),
\qquad
U = \Phi (\Phi^\dagger \Phi)^{-1/2}
$$

with forward/inverse relations

$$
\hat{x} = U^\dagger x,
\qquad
x = U \hat{x}.
$$

For proof status, use:
- `docs/research/CURRENT_RESEARCH_POSITION.md`
- `docs/THEOREMS_RFT_IRONCLAD.md`
- `docs/PROOF_RESULTS.md`

---

## 2. Current implementation areas

### Inference stack

The active inference-related code in this snapshot is centered on:
- `quantonium_os_src/engine/rftmw_inference.py`
- `quantonium_os_src/engine/rftmw_numpy_engine.py`
- `quantonium_os_src/engine/rftmw_mmap_engine.py`
- `src/rftmw_native/`

These implement tokenizer loading, pack handling, NumPy/native inference paths, and generation logic.

### Desktop and app layer

Desktop and application logic lives under:
- `quantonium_os_src/apps/`
- `quantonium_os_src/frontend/`
- `src/apps/`

### Hardware program

Hardware and FPGA materials live under `hardware/` and should be read as a mix of validated artifacts and forward-looking specifications.

---

## 3. Claim boundaries

### Strongest / proven
- discrete canonical RFT definition
- full-rank raw φ-grid basis
- unitarity and exact reconstruction
- discrete prior-art boundary against plain DFT relabeling/permutation

### Empirical or still open
- conditioning behavior beyond the resolved Fibonacci cases
- continuous-time parent and aliasing theory in full generality
- KV-cache intelligence-dividend claims
- RoPE-gap closure and medical compression gains

### Non-claims or explicitly limited
- universal FFT replacement
- wave-computer theorem as an established result
- standard cryptographic security proofs for the RFT-SIS work
- production silicon efficiency claims for the hardware program

The canonical non-claim source is `docs/NON_CLAIMS.md`.

---

## 4. Repo map note

If you see documentation referring to paths such as `src/core/`, `src/rftmw_memory.py`, `src/RFTMW.py`, or `quantonium-mobile/`, that content is from an older topology and should be treated as historical unless revalidated.
