# Patent Claim Cross-Reference Map
**USPTO Application No.:** 19/169,399  
**Provisional Priority:** 63/749,644 (filed January 26, 2025)  
**Title:** Hybrid Computational Framework for Quantum and Resonance Simulation  
**Inventor:** Luis Michael Minier  
**Last Updated:** March 24, 2026  

> **Purpose:** This document maps every element of Claims 1–4 of the filed patent application
> to the concrete file(s), class(es), and function(s) in this repository that implement or
> demonstrate that element. It is intended to support office-action responses, examiner
> interviews, and future IDS filings — at zero additional USPTO cost.

---

## CLAIM 1 — Symbolic Resonance Fourier Transform (RFT) Engine

> *"A symbolic transformation engine for quantum amplitude decomposition..."*

| Claim Element | Repo File(s) | Key Class / Function | Notes |
|---|---|---|---|
| Symbolic representation module — expresses quantum state amplitudes as algebraic forms | `algorithms/rft/core/canonical_true_rft.py` | `CanonicalRFT` | Core RFT basis: Φ̃ = Φ(ΦᴴΦ)^{-1/2}; unitarity error < 5e-15 |
| Symbolic representation module (alternate/legacy) | `algorithms/rft/core/resonant_fourier_transform.py` | `ResonantFourierTransform` | Earlier symbolic form; cross-validates canonical |
| Symbolic representation module (phi-legacy) | `algorithms/rft/core/rft_phi_legacy.py` | — | Legacy golden-ratio basis implementation |
| Phase-space coherence retention mechanism | `algorithms/rft/core/canonical_true_rft.py` | `forward()`, `inverse()` | Unitarity of Φ̃ guarantees \|⟨T(x),T(y)⟩\| = \|⟨x,y)⟩\| |
| Phase-space coherence retention mechanism (bounds) | `algorithms/rft/core/sharp_coherence_bounds.py` | `SharpCoherenceBounds` | Formal coherence bounds; supports phase-space claim |
| Topological embedding layer — maps amplitudes into structured manifolds | `algorithms/rft/core/gram_utils.py` | `GramUtils` | G = ΦᴴΦ; G^{-1/2} normalization; embedding topology |
| Topological embedding layer (geometric container) | `algorithms/rft/core/geometric_container.py` | — | Vertex-based (Re,Im,\|Y\|) ∈ ℝ³ embedding |
| Symbolic gate propagation subsystem (Hadamard, Pauli-X) | `algorithms/rft/quantum/` | — | Quantum-gate symbolic operations directory |
| Symbolic gate propagation subsystem (quantum-inspired) | `algorithms/rft/quantum_inspired/` | — | Classical simulation of quantum gate behavior |
| Kernel construction (Toeplitz, φ-structured) | `algorithms/rft/core/canonical_true_rft.py` | `build_kernel()` | K = T(r·d), r[k]=cos(2πφ^k f₀)+i·sin(2πφ^k f₀) |
| Kernel construction (fast path) | `algorithms/rft/core/fast_rft.py` | `FastRFT` | Optimised kernel path for N≤256 |
| Kernel construction (Fibonacci variant) | `algorithms/rft/core/fibonacci_fast_rft.py` | `FibonacciFastRFT` | Alternative φ-structured kernel |
| Kernel truncation / approximation | `algorithms/rft/core/kernel_truncation.py` | `KernelTruncation` | Low-rank approximation; effective rank 15–42 at 99% energy |
| Novelty attestation | `algorithms/rft/core/absolute_novelty.py` | — | Documents Frobenius distance from DFT ≥ 9; irrational freq. spacing |
| Transform theorems / proofs | `algorithms/rft/core/transform_theorems.py` | — | Formal mathematical properties of the RFT |
| Bootstrap verification | `algorithms/rft/core/theorem8_bootstrap_verification.py` | — | Numerical verification of Theorem 8 (unitarity, reconstruction) |
| Research summary (examiner-ready data) | `algorithms/rft/RFT_RESEARCH_SUMMARY.md` | — | K Hermitian error=0, U unitary error≈1e-13, perfect recon ≈1e-15 |

---

## CLAIM 2 — Resonance-Based Cryptographic Subsystem Using Topological Waveform Hashing

> *"A cryptographic system comprising a symbolic waveform generation unit..."*

| Claim Element | Repo File(s) | Key Class / Function | Notes |
|---|---|---|---|
| Symbolic waveform generation unit — constructs amplitude-phase modulated signatures | `algorithms/rft/crypto/enhanced_cipher.py` | `EnhancedCipher` / waveform pipeline | SHA3 hash chain → signal s → RFT → amplitude-phase feature f[k] |
| Symbolic waveform generation unit (SIS hash entry point) | `algorithms/rft/crypto/rft_sis_hash.py` | `RFTSISHash` | Full waveform → SIS lattice → SHA3-256 output pipeline |
| Topological hashing module — Bloom-like filters for cryptographic identity | `algorithms/rft/core/bloom_filter.py` | `BloomFilter` | Waveform feature extraction into Bloom-like identity filter |
| Dynamic entropy mapping engine — modulates key material via resonance state | `algorithms/rft/crypto/enhanced_cipher.py` | entropy modulation section | φ·cos(∠Y[k]) term — novel entropy mapping step |
| Recursive modulation controller — 48-round Feistel structure | `algorithms/rft/crypto/enhanced_cipher.py` | Feistel round logic | 48-round recursive waveform modification; assembly-level integration |
| Theorem verification (crypto) | `algorithms/rft/crypto/theorem_verification.py` | — | Security properties and lattice theorem verification |
| Crypto benchmarks | `algorithms/rft/crypto/benchmarks/` | — | Empirical security measurements (avalanche, entropy, collision) |
| Crypto primitives | `algorithms/rft/crypto/primitives/` | — | Low-level cryptographic building blocks |

> ⚠️ **Honest caveat recorded here:** Per `rft_sis_hash.py` docstring, NumPy RNG is used —
> NOT a CSPRNG. Security claims are empirical only. Formal proof is future work.
> SIS parameters: n=512, m=1024, q=3329 (Kyber prime), β=100.

---

## CLAIM 3 — Geometric Symbolic Data Storage Framework

> *"A data storage architecture comprising an array of topological storage units structured as tetrahedral or higher-dimensional geometric simplices..."*

| Claim Element | Repo File(s) | Key Class / Function | Notes |
|---|---|---|---|
| Topological storage units — tetrahedral / higher-dimensional simplices | `algorithms/rft/core/geometric_container.py` | `GeometricContainer` | Vertex array: (Re(Y[k]), Im(Y[k]), \|Y[k]\|) ∈ ℝ³ per coefficient |
| Symbolic node encoding — algebraic amplitude values at simplex vertices | `algorithms/rft/core/canonical_true_rft.py` | `forward()` output Y | Y[k] = Φ̃ · x; each Y[k] is a simplex vertex value |
| Edge and face structures — phase-path relationships and resonance envelopes | `algorithms/rft/core/gram_utils.py` | `GramUtils.gram_matrix()` | G = ΦᴴΦ encodes phase-path relationships between basis vectors |
| Edge and face structures (golden resonance) | `algorithms/rft/core/golden_reservoir.py` | — | φ-resonance projection and winding components |
| Real-time coherence tracking | `algorithms/rft/core/sharp_coherence_bounds.py` | `SharpCoherenceBounds` | Tracks unitarity / coherence bounds dynamically |
| Symbolic compression | `algorithms/rft/compression/` | — | RFT-based compression leveraging low-rank structure |
| Topological fault tolerance / redundancy | `algorithms/rft/core/kernel_truncation.py` | `KernelTruncation` | Low-rank approximation enables redundancy-resilient storage |
| Coordinate transformation pipeline | `algorithms/rft/crypto/rft_sis_hash.py` | polar decomposition section | r[k]=\|s[k]\|, θ[k]=∠s[k] → φ-resonance projection → Cartesian re-embed |
| Gram matrix role | `algorithms/rft/core/gram_utils.py` | `GramUtils` | G^{-1/2} normalization; eigenstructure verifies near-orthogonality |
| Oscillator / vibrational model (simplex dynamics) | `algorithms/rft/core/oscillator.py` | — | Vibrational state model underlying simplex edge dynamics |
| Vibrational engine | `algorithms/rft/core/vibrational_engine.py` | — | Supports real-time amplitude propagation across simplex faces |

---

## CLAIM 4 — Hybrid Mode Integration (Unified Computational Framework)

> *"A unified computational framework comprising the symbolic transformation engine of claim 1,  
> the cryptographic subsystem of claim 2, and the geometric data storage framework of claim 3..."*

| Claim Element | Repo File(s) | Key Class / Function | Notes |
|---|---|---|---|
| **Engine A** — Symbolic RFT (Claim 1 core) | `algorithms/rft/core/canonical_true_rft.py` | `CanonicalRFT` | resonance_state = RFT.forward(quantum_state) |
| **Engine B** — Crypto subsystem (Claim 2 core) | `algorithms/rft/crypto/enhanced_cipher.py` | `EnhancedCipher` | cipher_state = Crypto.encrypt(resonance_state) |
| **Engine C** — Geometric hash (Claim 3 core) | `algorithms/rft/crypto/rft_sis_hash.py` | `RFTSISHash` | hash_output = GeoHash.hash(cipher_state.bytes) |
| **Scheduler** — dynamic resource allocator | `algorithms/rft/unified_transform_scheduler.py` | `UnifiedTransformScheduler` | Selects engine path: N≤256 canonical / N≤2048 native / N>2048 hybrid |
| Hybrid execution paths | `algorithms/rft/hybrids/` | — | Chunked execution for large N with fallback to Python reference |
| Native path (RFTMW) | `src/rftmw_native/` (see `src/`) | — | Assembly-level RFTMW native path |
| Coherent resonance_state propagation across layers | `algorithms/rft/unified_transform_scheduler.py` | `run_pipeline()` | A→B→C chain; inner-product preservation documented |
| Routing logic between engines | `algorithms/rft/routing.py` | — | Top-level routing; maps signal properties to engine selection |
| Routing directory (sub-modules) | `algorithms/rft/routing/` | — | Sub-module routing logic |
| Python reference fallback | `algorithms/rft/core/canonical_true_rft.py` | `CanonicalRFT` | Always available; used when native path unavailable |
| Fast surrogate path | `algorithms/rft/core/fast_rft_surrogate.py` | — | Approximation surrogate for throughput-critical paths |
| RFT status / health monitoring | `algorithms/rft/rft_status.py` | `RFTStatus` | Runtime status checks; validates scheduler decisions |

---

## Supporting Infrastructure (Cross-Claim)

| Purpose | File | Relevance |
|---|---|---|
| Research summary (all claims) | `algorithms/rft/RFT_RESEARCH_SUMMARY.md` | Empirical data: unitarity, approximation decay, DFT distance |
| Codebase structure overview | `CODEBASE_STRUCTURE.md` | High-level architecture; supports FIG. 4 in patent spec |
| Patent claims analysis (prior) | `PATENT_CLAIMS_IMPLEMENTATION_ANALYSIS.md` | Prior self-analysis of claim coverage |
| Repo compression / analysis | `REPO_FULL_ANALYSIS_AND_RFTMW_COMPRESSION.md` | Documents RFTMW compression layer |
| Formal verification runner | `run_formal_verification.py` | Runs theorem proofs numerically |
| Theorem E proof test | `theorem_e_proof_test.py` | Proves Theorem E; supports Claim 1 unitarity guarantee |
| Reproduce results script | `reproduce_results.sh` | Reproducibility; all benchmark outputs can be re-generated |
| Stubs and legacy tracking | `STUBS_AND_LEGACY_FILES.md` | Documents stub files; prevents confusion during examination |
| Continuous compute | `algorithms/rft/core/continuous_compute.py` | Continuous-domain extension of RFT |
| Shard / distributed | `algorithms/rft/core/shard.py` | Distributed/sharded execution support |
| Diophantine extension | `algorithms/rft/core/diophantine_rft_extension.py` | Number-theoretic extension; supports φ-irrational freq. claim |
| Uncertainty principles | `algorithms/rft/core/golden_uncertainty_principle.py` | φ-based uncertainty bounds |
| Uncertainty (Maassen-Uffink) | `algorithms/rft/core/maassen_uffink_uncertainty.py` | Standard uncertainty bound; validates coherence claims |
| Transform variants | `algorithms/rft/variants/` | Additional RFT variants for different signal classes |
| Experiments | `algorithms/rft/experiments/` | Experimental runs with results |
| Kernel directory | `algorithms/rft/kernels/` | Kernel construction modules |
| Theory directory | `algorithms/rft/theory/` | Mathematical theory documents |
| Theorem directory | `algorithms/rft/theorems/` | Formal theorem files |
| Applications | `algorithms/rft/applications/` | Applied use cases of RFT |
| UI visualization layer | `ui/` | Supports FIG. 1 (Q-Wave Debugger) and FIG. 4 (UI layer) |
| QVault | `QVault/` | Secure storage module; supports Claim 3 storage architecture |
| Tools | `tools/` | Development and analysis tools |
| Examples | `examples/` | Runnable examples for each claim's core operation |

---

## Figure-to-Implementation Map

| Patent Figure | Description | Primary Implementation |
|---|---|---|
| FIG. 1 | Q-Wave Debugger — live 3D amplitude visualization | `ui/` — real-time resonance state visualization |
| FIG. 2 | Data Processing Flowchart (raw input → resonance → encryption → output) | `algorithms/rft/unified_transform_scheduler.py` pipeline |
| FIG. 3 | System dataflow: Symbolic Resonance Processor → Geometric Amplitude Processor → Dynamic Resource Allocator → Encryption | `algorithms/rft/routing.py` + `unified_transform_scheduler.py` |
| FIG. 4 | System Architecture: UI → Computation Engine → Resonance Layer → Storage | `CODEBASE_STRUCTURE.md`; `ui/` + `algorithms/rft/` + `QVault/` |

---

## Honest Constraints (Must Be Reflected in Any Office-Action Response)

These are documented here so no overclaiming occurs in USPTO filings:

1. **Complexity:** RFT is O(N²) per transform, O(N³) kernel construction (one-time). It is NOT O(N log N). Claims hold at all N with pre-computed kernel.
2. **Crypto security:** NumPy RNG used — not a CSPRNG. Empirical security only. Formal proof pending.
3. **Scope:** RFT outperforms FFT/DCT only for φ-quasi-periodic / matched signals. Not universally superior.
4. **Publication status:** Peer review not yet complete as of filing date.
5. **Correct term:** The engine is the **Resonance Fourier Transform (RFT)**, not "RET" — the filed spec contains a typo ("RET") that should be noted in any examiner communication.

---

*Generated: March 24, 2026 | Repo: LMMinier/quantoniumos | Branch: main*  
*This file is internal documentation only and does not constitute a new USPTO filing or amendment.*
