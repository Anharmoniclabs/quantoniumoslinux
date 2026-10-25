# RFT Core — File Map

> **For reviewers and contributors:** This directory contains the canonical
> Resonant Fourier Transform implementation plus files pending final deletion
> after migration is confirmed safe.
> The active IP is exactly 4 files. Everything else is either migrated (with
> originals preserved for import safety) or pending migration.

---

## ✅ Active IP — Files That Practice Patent Claims

| File | Purpose | Status |
|------|---------|--------|
| `resonant_fourier_transform.py` | **THE canonical RFT** — Gram-normalized φ-grid basis, forward/inverse transforms, BinaryRFT class | **Core IP** |
| `gram_utils.py` | Löwdin normalization (Gram inverse square root). Imported directly by the above. | **Core IP** |
| `canonical_true_rft.py` | `CanonicalTrueRFT` class — backward-compatible wrapper around `resonant_fourier_transform.py`. No independent logic. | **Wrapper** |
| `fibonacci_fast_rft.py` | O(N log N) fast RFT surrogate via Fibonacci decomposition | **Engineering** |

All four are listed in `CLAIMS_PRACTICING_FILES.txt` and licensed under `LICENSE-CLAIMS-NC.md`.

---

## ✅ Migrated to `algorithms/rft/legacy/` (originals preserved here)

| File in core/ | Migrated to | Notes |
|--------------|-------------|-------|
| `rft_phi_legacy.py` | `legacy/rft_phi_legacy.py` | Pre-Gram locked implementation |
| `absolute_novelty.py` | `legacy/dft_distance_certificate.py` | **RENAMED** — content is a valid certified DFT-distance tool. Old import names kept as aliases. |

---

## ✅ Migrated to `algorithms/rft/utils/` (originals preserved here)

| File in core/ | Migrated to | Notes |
|--------------|-------------|-------|
| `bloom_filter.py` | `utils/bloom_filter.py` | Imports unchanged |
| `oscillator.py` | `utils/oscillator.py` | Imports unchanged |
| `geometric_container.py` | `utils/geometric_container.py` | Circular dep on symbolic_wave_computer removed; now uses BinaryRFT |
| `shard.py` | `utils/shard.py` | Internal imports updated to utils/ |
| `vibrational_engine.py` | `utils/vibrational_engine.py` | Internal imports updated to utils/ |

---

## 📐 Pending Physical Migration → `algorithms/rft/theorems/`

| File | What It Is |
|------|------------|
| `transform_theorems.py` | Theorem implementations (Theorems 6, 8, 15, Theorem E) |
| `golden_uncertainty_principle.py` | Uncertainty principle (φ-basis Heisenberg bound) |
| `maassen_uffink_uncertainty.py` | Maassen-Uffink entropic uncertainty bounds |
| `sharp_coherence_bounds.py` | Sharp coherence bounds for RFT vs DFT |
| `theorem8_bootstrap_verification.py` | Empirical bootstrap verification of Theorem 8 |

Re-exported via `algorithms/rft/theorems/__init__.py` already.
Physical move blocked until test suite confirms no import breaks.

---

## 🧪 Pending Physical Migration → `algorithms/rft/experimental/`

| File | What It Is |
|------|------------|
| `unified_continuous.py` | Continuous-domain RFT extension (26KB) |
| `continuous_compute.py` | Continuous computation variant (22KB) |
| `true_wave_compute.py` | Wave-domain computation research |
| `symbolic_wave_computer.py` | Symbolic computation layer |
| `golden_reservoir.py` | Golden reservoir model |
| `diophantine_rft_extension.py` | Diophantine number theory extension |
| `kernel_truncation.py` | Kernel truncation approximation |
| `fast_rft.py` | Fast variant stub |
| `fast_rft_surrogate.py` | Alternative fast surrogate |

---

## Migration Checklist

- [x] legacy/ files migrated (`rft_phi_legacy`, `dft_distance_certificate`)
- [x] utils/ cluster migrated (bloom\_filter, oscillator, geometric\_container, shard, vibrational\_engine)
- [ ] theorems/ physical move (re-exported; move after test run)
- [ ] experimental/ physical move
- [ ] Delete originals in core/ after full test suite passes
- [ ] Update CLAIMS\_PRACTICING\_FILES.txt

---

*Last updated: March 2026. See git log for change history.*
