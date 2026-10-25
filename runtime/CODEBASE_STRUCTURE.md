# quantoniumos — Codebase Structure

This document is the authoritative map of what lives where and why.
Generated March 2026 after full audit. Updated March 27, 2026.

---

## Design Principle: Frequency as Infrastructure

The φ-basis is the infrastructure, not just a tool. The same unitary matrix Φ̃ = Φ(Φ†Φ)^{-1/2} serves four roles simultaneously:

| Role | What it does | Primary code locations |
|------|-------------|----------------------|
| **STORE** | Positional encoding via φ-grid geometry (KV-cache boundaries) | `benchmarks/`, `tests/proofs/test_rft_positional_leakage_canonical.py` |
| **COMPUTE** | Binary→Wave→Compute→Binary pipeline, quantum gate simulation | `quantonium_os_src/engine/RFTMW.py`, `algorithms/rft/quantum_inspired/` |
| **COMPOSE** | Lattice cryptography, geometric waveform hashing | `algorithms/rft/crypto/`, `algorithms/rft/quantum_inspired/geometric_waveform_hash.py` |
| **ROUTE** | 27 variant bases selected per signal class | `algorithms/rft/routing.py`, `algorithms/rft/variants/registry.py` |

Every directory below implements one or more of these roles.

---

## The Core IP — What Everything Is Built On

```
algorithms/rft/core/
├── resonant_fourier_transform.py   ← THE canonical RFT (Patent 19/169,399)
├── gram_utils.py                   ← Löwdin normalization (imported by above)
├── canonical_true_rft.py           ← CanonicalTrueRFT class wrapper
├── fibonacci_fast_rft.py           ← O(N log N) fast surrogate
└── STRUCTURE.md                    ← Per-file migration map
```

Everything else in `core/` is pending migration. See `algorithms/rft/core/STRUCTURE.md`.

---

## The STRT Result — The Strongest Empirical Finding

```
algorithms/rft/routing.py           ← STRT depth-gated routing (91.7% win on GPT-2 V-cache)
algorithms/rft/routing/             ← Routing subpackage
```

This is the result to lead with in the applied paper.

---

## Theory

```
algorithms/rft/theorems/            ← Theorem implementations (target location)
algorithms/rft/theory/              ← Theory research files
```

---

## Experiments and Tests

```
algorithms/rft/experiments/         ← Experimental scripts
tests/proofs/                       ← Formal proof test suite
    test_theorem_e_rip_proof.py     ← RIP delta_8=0.1603 < 0.2929 (Candès-Tao)
    test_theorem_15_exact_sparsity.py
    test_formal_proofs.py
    test_rft_transform_theorems.py
    test_sharp_coherence_bounds.py
    test_maassen_uffink_uncertainty.py
    test_fast_rft_d3_unitarity.py
```

---

## What This Project Is (Honest Summary)

| Component | Status | Venue |
|-----------|--------|-------|
| Canonical RFT (φ-grid basis, Gram-normalized) | Implemented, patent filed | Theory paper |
| Theorem E (RIP path, δ₈=0.1603) | Empirically confirmed | Needs analytic bound |
| STRT depth-gated routing (91.7% GPT-2 V-cache) | **Strongest result** | Applied ML paper |
| Fast RFT O(N log N) surrogate | Implemented, no benchmark vs FFTW | Engineering work |
| Continuous RFT extension | Experimental, no peer-review target | Not ready |

---

## Recommended Paper

**Do not submit** "Formal Properties of the Resonant Fourier Transform" to TETC.

**Submit** "Depth-Gated Resonance Transform Routing for KV-Cache Compression"
to IEEE Signal Processing Letters or a NeurIPS Efficient LLM Systems workshop.

- Lead with: STRT 91.7% win on GPT-2 deep V-layers vs DCT baseline
- Theory section: Theorem 8 + quasi-periodicity of deep activations
- Comparison: STRT vs hard-block DCT on K-vectors (honest — STRT loses there)
- Code: open-source, reproducible from `algorithms/rft/routing.py`

---

*Last updated: March 2026*
