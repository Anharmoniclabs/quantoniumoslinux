# Four-Claim Repo Map

**Date:** April 2026
**Purpose:** Map the repository structure to the four patent claims. This is an internal organizational tool, not a legal document.

> **Disclaimer:** This file classifies repo directories and files by the patent claim they most closely relate to. This classification is for internal housekeeping and disclosure management. It does NOT define the legal scope of any claim — that is determined solely by the USPTO-issued claim language and prosecution history.

---

## Patent Reference

- **Application:** 19/169,399
- **Title:** Hybrid Computational Framework for Quantum and Resonance Simulation
- **Filing Date:** April 3, 2025
- **Claims:** 4 independent claims

---

## Claim 1 — Symbolic Resonance Fourier Transform Engine

**Scope:** Golden-ratio frequency grid, Gram-normalized unitary basis, O(N²) exact / O(N log N) approximate transform.

### Core Implementation
| Path | Description |
|------|-------------|
| `algorithms/rft/core/` | Primary RFT: canonical construction, Gram utilities, φ-grid |
| `algorithms/rft/theory/` | Proof engine, theorem implementations |
| `algorithms/rft/variants/` | Unitary variants (entropic, golden-ratio, symbolic) |
| `algorithms/rft/fast/` | Fast RFT implementations (cached basis, structured) |
| `algorithms/rft/kernels/` | Native kernels, SIMD, ASM, Python bindings |
| `src/rftmw_native/` | C++/SIMD RFT core + Python bindings |

### Theory & Documentation
| Path | Description |
|------|-------------|
| `docs/THEOREMS_RFT_IRONCLAD.md` | Canonical theorem statements and proofs |
| `docs/algorithms/rft/RFT_DEVELOPER_MANUAL.md` | Developer reference |
| `docs/research/RFT_PERFORMANCE_THEOREM.md` | Performance propositions (correctly graded) |
| `docs/research/CURRENT_RESEARCH_POSITION.md` | Public-facing summary |

### Tests
| Path | Description |
|------|-------------|
| `tests/unit/test_rft_correctness.py` | Core RFT correctness tests |
| `tests/validation/` | Validation test suites |

---

## Claim 2 — Resonance-Based Cryptographic Subsystem

**Scope:** RFT-SIS hash function, 48-round Feistel cipher with RFT mixing, chaos-based key derivation.

### Implementation
| Path | Description |
|------|-------------|
| `algorithms/rft/crypto/` | RFT-SIS hash, enhanced cipher, theorem verification |
| `algorithms/rft/crypto/primitives/` | PRNG |
| `algorithms/rft/crypto/benchmarks/` | Avalanche, validation, performance suites |
| `algorithms/rft/kernels/engines/crypto/` | C/ASM Feistel implementation |

### Status Notes
- Structured-SIS hardness is **NOT proved** (see Open Problems #6)
- Avalanche metrics (40–60%) from internal tests only
- NOT externally audited or peer-reviewed

---

## Claim 3 — Geometric Structures for Cryptographic Waveform Hashing

**Scope:** Topological waveform embedding, geometric hash functions, φ-phase signatures.

### Implementation
| Path | Description |
|------|-------------|
| `algorithms/rft/quantum/geometric_waveform_hash.py` | Geometric hashing core |
| `algorithms/rft/quantum/geometric_hashing.py` | Geometric hashing utilities |
| `algorithms/rft/quantum/theorem_verification.py` | Verification for geometric claims |
| `algorithms/rft/quantum/topological_quantum_kernel.py` | Topological embedding |
| `algorithms/rft/quantum/enhanced_topological_qubit.py` | Topological qubit simulation |

### Status Notes
- Implementation exists. No external validation or formal analysis.
- "Topological" refers to embedding topology, not topological quantum computing.

---

## Claim 4 — Hybrid Mode Integration

**Scope:** DCT-RFT cascade codecs, zero-coherence routing, adaptive basis selection.

### Implementation
| Path | Description |
|------|-------------|
| `algorithms/rft/hybrids/` | Cascade hybrids, H3-ARFT, residual predictor, codec |
| `algorithms/rft/compression/` | Entropy coding, codecs, lossless streams |
| `algorithms/rft/routing/` | Signal classifier, routing logic |
| `algorithms/rft/unified_transform_scheduler.py` | Adaptive basis selection |
| `tools/compression/` | Compression tools (TinyLlama, sparse, lossless) |

### Status Notes
- Routing is engineering heuristic, not a mathematical result.
- Compression benchmarks exist (internal) but not independently validated.

---

## Files NOT Practicing Any Claim

These files are licensed under AGPL-3.0 and do not practice patent claims:

| Path | Description |
|------|-------------|
| `benchmarks/` | Benchmark scripts (measurement, not implementation) |
| `docs/` (most files) | Documentation, research notes, guides |
| `tests/` (most files) | Test harnesses |
| `scripts/` | Utility scripts |
| `ui/` | UI components |
| `data/` | Data fetch scripts and sample data |
| `examples/` | Example usage scripts |
| `quantonium_os_src/` (non-crypto) | OS shell, boot, monitoring (not claim-specific) |

---

## Cross-References

- Full file-level registry: `docs/project/CLAIMS_PRACTICING_FILES.txt`
- License details: `LICENSE-CLAIMS-NC.md`, `LICENSE-SPX-PROPRIETARY.md`
- Disclosure tiers: `docs/project/DISCLOSURE_BOUNDARY.md`

---

*This map is for internal organization. For the authoritative file-level registry,
see `docs/project/CLAIMS_PRACTICING_FILES.txt`.*
