# Patent Claims Implementation Analysis
**Application No.:** 19/169,399
**Original Analysis Date:** October 10, 2025
**Last Updated:** March 24, 2026
**Status:** Pre-Exam Formalities — Active / Implementation Verified Against Current Codebase

---

> ⚠️ **Repo Structural Note (March 2026 Update)**
> Since the October 2025 analysis, core claim implementation files have migrated from `src/core/` to
> `algorithms/rft/core/` and `algorithms/rft/crypto/`. All paths below reflect the current
> verified repo state as of commit `2b8ad2f`. Legacy path references in the October document
> have been corrected throughout.

---

## License Boundary (March 2026)

Per owner decision, the repository uses a **dual-license model**:

| Zone | License | Files |
|---|---|---|
| **Proprietary** | `LicenseRef-SPX-Proprietary` | All files that implement or directly orchestrate the 4 patent claims |
| **Open / Verifiable** | `AGPL-3.0-or-later` | Supporting stack — infra, pipelines, benchmarks, scripts, training helpers |

The proprietary zone is defined by which code embodies a claimed behaviour. If removing a file
would gut a claimed behaviour, it is proprietary. If it only supports running or visualising
results, it is AGPL for verifiability and reproducibility.

---

## 🎯 Patent Claims → Codebase Mapping (Current)

---

### CLAIM 1: Symbolic Resonance Fourier Transform Engine ✅ IMPLEMENTED

**Patent Language:** *"symbolic transformation engine for quantum amplitude decomposition"*

**Primary Implementations (Proprietary — `LicenseRef-SPX-Proprietary`):**

```
algorithms/rft/core/canonical_true_rft.py          # Core RFT — Claim 1 anchor
algorithms/rft/core/resonant_fourier_transform.py  # Forward/inverse RFT, matrix construction
algorithms/rft/core/fast_rft.py                    # Performance-optimised RFT path
algorithms/rft/core/fibonacci_fast_rft.py          # Fibonacci-resonance RFT variant
algorithms/rft/core/sharp_coherence_bounds.py      # Phase-space coherence retention
algorithms/rft/core/transform_theorems.py          # Mathematical theorem support
algorithms/rft/unified_transform_scheduler.py      # High-level orchestration of RFT paths
```

**Supporting / Assembly Layer:**
```
algorithms/rft/kernels/                            # Assembly kernel bindings (C/NASM)
src/rftmw_native/                                  # Native RFTMW bindings layer
```

**Supporting Verifiable Stack (AGPL):**
```
scripts/rft_quick_reference.py                     # Reference guide — no claim logic
scripts/run_research_suite.py                      # Orchestrates validation runs
algorithms/rft/rft_status.py                       # Status/routing helper
```

**Claim Elements Verified in Codebase:**
- ✅ Symbolic amplitude decomposition: `canonical_true_rft.py` — unitary matrix construction
- ✅ Frobenius distance from DFT: mathematically verified ≥ 9 (documented in `RFT_RESEARCH_SUMMARY.md`)
- ✅ Phase-space coherence retention: `sharp_coherence_bounds.py`, `resonance_state` preservation
- ✅ Golden-ratio phase structure: `φ = (1+√5)/2` embedded in phase kernel `D_φ[k]`
- ✅ Symbolic gate propagation without collapse: Hadamard/Pauli-X symbolic simulation

**October → March Δ:**
- Path corrected: `src/core/canonical_true_rft.py` → `algorithms/rft/core/canonical_true_rft.py`
- New files added: `fibonacci_fast_rft.py`, `sharp_coherence_bounds.py`, `transform_theorems.py`
- `unified_transform_scheduler.py` now serves as the high-level RFT orchestration layer

---

### CLAIM 2: Resonance-Based Cryptographic Subsystem ✅ IMPLEMENTED

**Patent Language:** *"symbolic waveform generation unit configured to construct amplitude-phase modulated signatures"*

**Primary Implementations (Proprietary — `LicenseRef-SPX-Proprietary`):**

```
algorithms/rft/crypto/enhanced_cipher.py           # Full RFT-based cipher — Claim 2 anchor
algorithms/rft/crypto/rft_sis_hash.py              # RFT-SIS lattice hash
algorithms/rft/crypto/theorem_verification.py      # Crypto-property theorem proofs
algorithms/rft/kernels/engines/crypto/src/feistel_round48.c   # Assembly Feistel engine
```

**Supporting Verifiable Stack (AGPL):**
```
tests/crypto/scripts/comprehensive_crypto_suite.py  # End-to-end validation
scripts/evidence_pack_track_b.py                   # Evidence packaging
```

**Claim Elements Verified in Codebase:**
- ✅ Symbolic waveform generation: `_bytes_to_signal()` → complex exponential waveforms via `np.exp(1j * phase)`
- ✅ Amplitude-phase modulated signatures: RFT coefficient-based entropy modulation
- ✅ Dynamic entropy mapping: `enhanced_cipher.py` — real-time waveform key material
- ✅ Recursive modulation controller: Feistel round structure (`feistel_round48.c`)
- ✅ Topological hashing module: `rft_sis_hash.py` — `RFTSISHash` class with projection mapping

**October → March Δ:**
- Path corrected: `src/core/geometric_waveform_hash.py` → `algorithms/rft/crypto/enhanced_cipher.py` + `rft_sis_hash.py`
- `geometric_waveform_hash.py` no longer exists at the old path; its functionality is split across `enhanced_cipher.py` (cipher) and `rft_sis_hash.py` (hash primitive)
- `theorem_verification.py` added — formally verifies crypto security properties in-code

---

### CLAIM 3: Geometric Structures for RFT-Based Cryptographic Waveform Hashing ✅ IMPLEMENTED

**Patent Language:** *"geometric coordinate transformations map waveform features through projection mappings"*

**Primary Implementations (Proprietary — `LicenseRef-SPX-Proprietary`):**

```
algorithms/rft/crypto/enhanced_cipher.py           # Geometric pipeline — Claim 3 anchor
algorithms/rft/crypto/rft_sis_hash.py              # Projection-based hash generation
algorithms/rft/core/gram_utils.py                  # Gram matrix / geometric structure utilities
algorithms/rft/core/kernel_truncation.py           # Kernel projection truncation
```

**Claim Elements Verified in Codebase:**
- ✅ Polar-to-Cartesian with φ scaling: `phi_resonance = np.sum(np.cos(phases * self.phi))`
- ✅ Complex geometric coordinate generation: `np.exp(1j * phase)` transforms throughout
- ✅ Phase-winding tag generation: heuristic phase summary labels in `rft_sis_hash.py`
- ✅ Projection-based hash generation: `_topological_embedding()` — structure-preserving projection
- ✅ Gram matrix eigenstructure: `gram_utils.py` — geometric inner-product structure

**October → March Δ:**
- Path corrected: `src/core/enhanced_topological_qubit.py` — file absorbed into `enhanced_cipher.py` and `gram_utils.py`
- `gram_utils.py` and `kernel_truncation.py` are new additions since October that directly support Claim 3

---

### CLAIM 4: Hybrid Mode Integration ✅ IMPLEMENTED

**Patent Language:** *"unified computational framework comprising the symbolic transformation engine... cryptographic subsystem... and geometric structures"*

**Primary Implementations (Proprietary — `LicenseRef-SPX-Proprietary`):**

```
algorithms/rft/unified_transform_scheduler.py      # Unified RFT+Crypto framework — Claim 4 anchor
algorithms/rft/hybrids/                            # Hybrid canonical/RFTMW mode variants
scripts/bridge_canonical_vs_hybrid.py              # Canonical ↔ Hybrid bridge (proprietary boundary)
src/rftmw_native/                                  # Native RFTMW integration layer
```

**Supporting Verifiable Stack (AGPL):**
```
tests/benchmarks/complete_validation_suite.py      # Integrated system validation
tests/crypto/scripts/comprehensive_crypto_suite.py # End-to-end integration tests
scripts/run_research_suite.py                      # Research validation orchestration
```

**Claim Elements Verified in Codebase:**
- ✅ Unified computational framework: `unified_transform_scheduler.py` integrates all subsystems
- ✅ Coherent resonance_state propagation: maintained across RFT → cipher → hash layers
- ✅ Dynamic resource allocation: automatic engine selection (canonical / RFTMW / hybrid)
- ✅ Modular phase-aware architecture: Symbolic + crypto + nonbinary data paths
- ✅ Assembly integration with Python fallback: `src/rftmw_native/` with graceful fallback

**October → March Δ:**
- Path corrected: `src/apps/symbolic_simulator.py` — unified framework now primarily lives in `unified_transform_scheduler.py`
- `algorithms/rft/hybrids/` directory added — contains hybrid-mode variants for Claim 4
- `src/rftmw_native/` confirmed as assembly integration layer
- `scripts/bridge_canonical_vs_hybrid.py` is a new addition directly embodying the hybrid-mode claim

---

*Last updated: March 26, 2026 — repo state verified against current directory structure.*
