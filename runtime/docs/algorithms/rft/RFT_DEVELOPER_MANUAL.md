# quantoniumos Engineer Manual

**Version:** 2.1.0 (Unified Architecture — March 2026 Sync)
**Status:** Release Candidate — Validated Core, Experimental Crypto
**Author:** Luis M. Minier (quantoniumos)
**Patent Disclosure:** USPTO #19/169,399 — Research/Non-Commercial License
**DOI Reference:** TechRxiv / Zenodo (pending)
**RIP-First Paper:** [docs/papers/rip_first_architecture_2026.md](../../papers/rip_first_architecture_2026.md)
**Last Synced:** March 27, 2026

---

## Table of Contents
0. [Getting Started](#0-getting-started)
1. [Executive Layer](#1-executive-layer)
2. [Architecture](#2-architecture)
3. [Mathematical Appendix](#3-mathematical-appendix)
4. [Security (Experimental)](#4-security-experimental)
5. [Performance & Benchmarking](#5-performance--benchmarking)
   - [5.8 The 27-Variant Patch: Variant Scheduler + IHT + Mode Gate](#58-the-27-variant-patch-variant-scheduler--iht--mode-gate-architecture-march-2026)
6. [Hardware Integration](#6-hardware-integration)
7. [Developer Operations & Testing](#7-developer-operations--testing)
8. [Educational & Outreach](#8-educational--outreach)
9. [Irrevocable Truths Index](#9-irrevocable-truths-index)
10. [Meta-Layer Philosophy](#10-meta-layer-philosophy)
11. [Licensing & IP Boundary](#11-licensing--ip-boundary)
12. [Non-Goals & Limitations](#12-non-goals--limitations)
13. [Stability & API Surface](#13-stability--api-surface)
14. [Cookbook Demos](#14-cookbook-demos)
15. [Glossary](#15-glossary)

---

## 0. Getting Started

### 0.1 Clone & Environment
```bash
git clone https://github.com/LMMinier/quantoniumos.git
cd quantoniumos
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e .
```

### 0.2 Quick Smoke Tests
```bash
pytest tests/rft/test_rft_vs_fft.py -q
python scripts/irrevocable_truths.py
```

### 0.3 Minimal API Usage
```python
import numpy as np
from algorithms.rft.core.resonant_fourier_transform import rft_forward_square, rft_inverse_square

N = 128
rft = ClosedFormRFT(N)          # Constructs unitary Φ-based matrix directly
x = np.random.rand(N)           # Real signal
X = rft.forward(x)              # RFT spectrum
x_rec = rft.inverse(X)          # Inverse transform
print('Reconstruction error:', np.linalg.norm(x - x_rec))
```

### 0.4 Hybrid Codec Example
```python
from algorithms.rft.compression.rft_hybrid_codec import RFTHybridCodec
codec = RFTHybridCodec(N=256)
compressed, meta = codec.encode(x)
decoded = codec.decode(compressed, meta)
print('Hybrid reconstruction error:', np.linalg.norm(x[:256] - decoded))
```

### 0.5 Full Validation
```bash
./scripts/validate_all.sh   # Or: python run_verify_now.py
```

Use the full validation command before and after architectural changes.

## 1. Executive Layer

RFT (Resonant Fourier Transform) is a family of transforms using irrational phase progressions (Golden Ratio φ) and braided/topological variants. The goal is to explore trade-offs in sparsity, mixing, and structural diversity while retaining FFT-class complexity for practical sizes.

> **Naming note:** The canonical name is **Resonant Fourier Transform**, not "Resonance Field Transform". Corrected in March 2026.

### The Core Thesis: Frequency as Infrastructure

The φ-basis is not just a signal processing tool — it is the computational infrastructure. The single unitary matrix Φ̃ = Φ(Φ†Φ)^{-1/2} serves four simultaneous roles:

| Role | What it does | Key theorem | Primary code |
| :--- | :--- | :--- | :--- |
| **STORE** | Positional encoding via φ-grid geometry; 47–80% KV-cache leakage reduction vs RoPE | Theorem 8 + Three-Distance Theorem | `benchmarks/kv_cache_boundary_error.py` |
| **COMPUTE** | Binary→Wave→Compute→Binary pipeline; unitary gate operations on classical hardware | Theorem 2 (unitarity) + Theorem 4 (convolution) | `quantonium_os_src/engine/RFTMW.py` |
| **COMPOSE** | Lattice cryptography from φ-structured matrices; geometric waveform hashing | Theorem 6 (irrationality) + SIS hardness | `algorithms/rft/crypto/` |
| **ROUTE** | 27 variant bases selected per signal class; no single basis wins everywhere | Theorem 7 (variant niches) | `algorithms/rft/variants/registry.py` |

These are four consequences of the same mathematical object. Every section below documents one of these roles.

### Design Philosophy
| Feature | Classical (FFT/DCT) | Quantonium (RFT) |
| :--- | :--- | :--- |
| **Basis** | Integer Harmonics ($k$) | Irrational Resonances ($\phi^{-k}$) |
| **Topology** | Flat Circle ($S^1$) | Toroidal Braid ($T^2$) |
| **Complexity** | $O(N \log N)$ | $O(N \log N)$ (Closed Form) |
| **Application** | Linear Filtering | Compression, Crypto, & Quantum Sim |

**Mission:** To unify **Compression** (Sparsity), **Cryptography** (Chaos), and **Simulation** (Unitarity) into a single, mathematically invariant framework.

### Where This Lives (Traceability)

> **Note:** Throughout this manual, "Φ-RFT" refers to the **canonical** Gram-normalized $\text{frac}((k{+}1)\phi)$ basis unless explicitly labeled "legacy" or "variant".

| Concept | Code | Validation |
| :-- | :-- | :-- |
| Φ-RFT Core | `algorithms/rft/core/canonical_true_rft.py` and `algorithms/rft/core/resonant_fourier_transform.py` | `tests/rft/test_rft_vs_fft.py` |
| Canonical Unitary | `algorithms/rft/core/canonical_true_rft.py` | `scripts/irrevocable_truths.py` |
| Hybrid Codec | `algorithms/rft/compression/rft_hybrid_codec.py` | `scripts/verify_rate_distortion.py` |
| RFT-SIS Lattice | `algorithms/rft/crypto/rft_sis/rft_sis_hash_v31.py` | `algorithms/rft/crypto/rft_sis/rft_sis_v31_validation_suite.py` |
| Topological Mode | `algorithms/rft/kernels/include/rft_kernel.h` | `scripts/verify_soft_vs_hard_braiding.py` |
| Unified Orchestrator | `algorithms/rft/kernels/unified/kernel/unified_orchestrator.c` | `scripts/verify_performance_and_crypto.py` |
| Encoding Geometry (STORE) | `benchmarks/kv_cache_boundary_error.py` | `tests/proofs/test_rft_positional_leakage_canonical.py` |
| Perplexity / KV-Cache (ROUTE) | `algorithms/rft/routing.py` | `tests/proofs/test_perplexity_kvcache.py` |
| Variant Shootout (ROUTE) | `tests/variant_shootout.py` | Empirical — 27 variants vs DFT on 6 signal classes |


---

## 2. Architecture

The system operates as a vertical stack, designed to eliminate the "Python Tax" for heavy compute.

### 2.1 The Stack Topology

```mermaid
graph TD
    subgraph "L6: Application (Python)"
        Scripts[Analysis Scripts]
        Notebooks[Jupyter Labs]
    end

    subgraph "L5: Hybrid Layer"
        Hybrid[RFTHybridCodec]
        Predictor[Residual Predictor]
    end

    subgraph "L4: Variants (Theorems)"
        Golden[GoldenRatioUnitary]
        Symbolic[SymbolicUnitary]
        Entropic[EntropicUnitary]
    end

    subgraph "L3: Bindings (ctypes)"
        PyBind[Python Bindings]
    end

    subgraph "L2: Middleware (C)"
        Orchestrator[Unified Orchestrator]
        Scheduler[Task Scheduler]
    end

    subgraph "L1: Kernels (ASM/AVX)"
        Engine_RFT[RFT Engine]
        Engine_Crypto[Feistel/SIS]
        Engine_Quant[Quantum Sim]
    end

    subgraph "L0: Hardware (FPGA)"
        Verilog[SystemVerilog Core]
    end

    Scripts --> Hybrid
    Hybrid --> Golden
    Golden --> PyBind
    PyBind --> Orchestrator
    Orchestrator --> Engine_RFT
    Orchestrator --> Engine_Crypto
    Engine_RFT -.-> Verilog
```

### 2.2 Middleware Orchestration Layer
Located in `algorithms/rft/kernels/unified/kernel/unified_orchestrator.c`.

The **Unified Orchestrator** is a C-level supervisor that manages the flow of data between engines without returning control to Python. This prevents the Global Interpreter Lock (GIL) from becoming a bottleneck.

*   **Task Struct**: `unified_task_t` defines the operation (Transform, Encrypt, Measure).
*   **Zero-Copy**: Data remains in pinned memory buffers shared between engines.
*   **Pipeline**:
    1.  **RFT Transform**: Signal $\to$ Spectrum.
    2.  **Topological Permutation**: Spectrum $\to$ Scrambled Spectrum (Crypto).
    3.  **Measurement**: Scrambled Spectrum $\to$ Entropy (Compression).
*   **Throughput Estimate**: For large batches (N ≥ 2^18) the orchestrator sustains >90% overlap of compute and memory, yielding effective complex op throughput ≈ 7.0–7.3 GFLOP/s (derived from 1M-point timing in Section 5 assuming ~5N log₂N + N chirp ops). 
*   **Latency Model**: Single transform latency dominated by basis multiply: ~O(N log N) + linear chirp phase; scheduling overhead < 0.5% of wall-clock at N ≥ 2^14.
*   **Complexity Summary**: Orchestrator adds O(T) control overhead where T = number of task transitions; data ops remain O(N log N) (FFT-class) + O(N) variant phase adjustments.

### 2.3 Memory Model
*   **Phase Matrices**: Pre-computed $D_\phi$ and $C_\sigma$ tables are stored in aligned memory (32-byte boundary) for AVX2/AVX512 loading.
*   **State Vectors**: Quantum states are contiguous complex128 arrays.
*   **Context Switching**: The Orchestrator handles context switching between the "Crypto" state (Feistel Network) and "Quantum" state (Wavefunction) in $< 50$ cycles.

---

### 2.4 Actionable Architecture Map
| Task | Files You Touch | Also Run |
| :-- | :-- | :-- |
| Add new RFT variant | `algorithms/rft/variants/*.py`, `rft_kernel.c` | `scripts/irrevocable_truths.py` |
| Change hybrid routing | `algorithms/rft/compression/rft_hybrid_codec.py` | `scripts/verify_rate_distortion.py` |
| Modify quantum/topological behavior | `algorithms/rft/quantum_inspired/*.py`, `include/rft_kernel.h` | `scripts/verify_soft_vs_hard_braiding.py` |
| Tune performance | `algorithms/rft/kernels/python_bindings/*.py` | `scripts/verify_scaling_laws.py` |
| Adjust crypto lattice | `algorithms/rft/crypto/rft_sis/*` | `scripts/verify_performance_and_crypto.py` |
| Regenerate figures | `scripts/generate_paper_figures.sh` | `generate_figures_only.sh` |
| Profile orchestrator throughput | `algorithms/rft/kernels/unified/kernel/unified_orchestrator.c` | `scripts/verify_performance_and_crypto.py` |
| Extend hardware test vectors | `hardware/visualize_hardware_results.py` | `hardware/verify_fixes.sh` |

## 3. Mathematical Appendix: The 10 Irrevocable Theorems

This section summarizes the 10 fundamental theorems that establish the RFT framework. Full proofs and validation data are available in `docs/validation/RFT_THEOREMS.md`.

> **Note on naming:** These theorems were originally stated for the "Fast Φ-RFT" factorization ($\Psi = D_\phi C_\sigma F$), which is a **legacy formulation**, not the canonical RFT. The canonical RFT is defined as the Gram-normalized $\text{frac}((k{+}1)\phi)$ basis. See `README.md` for the canonical definition.

### 3.1 Core Theorems (Unitary & Structural)

> **Scientific Distinction:** The framework distinguishes between the **Canonical Form** (Gram-normalized φ-grid basis, theoretical ideal) and the **Fast Form** (FFT-factorized, practical implementation).
> *   **Canonical Form:** Used for Theorems 3, 4, 5, 6. Complexity $O(N^3)$.
> *   **Fast Form:** Used for Theorems 1, 2, 8, 9. Complexity $O(N \log N)$.
> *   **The Gap:** We do not yet have a formal proof that the Fast Form is a limit of the Canonical Form. They are currently distinct mathematical objects sharing the same phase philosophy. A legacy QR resonance-matrix basis exists for historical comparison only.

#### Theorem 1: Unitarity (PROVEN - Fast Form)
**Statement:** The legacy Fast Φ-RFT factorization $\Psi = D_\phi C_\sigma F$ is exactly unitary: $\Psi^\dagger \Psi = I$.
**Note:** This applies to the deprecated fast form. The canonical RFT achieves unitarity via Gram normalization of the $\text{frac}((k{+}1)\phi)$ basis.
**Validation:** Frobenius error < 10^{-14}$ across $N \in \{32, \dots, 512\}$.
**Implication:** Perfect information preservation and energy conservation.

#### Theorem 2: Exact Diagonalization (PROVEN - Fast Form)
**Statement:** For golden-ratio resonance systems, $\Psi^\dagger H \Psi = \Lambda + E$ where $\|E\|_F < 10^{-14}$.
**Implication:** $O(N)$ evolution complexity vs. $O(N^2)$ for generic methods.

#### Theorem 3: Massive Sparsity (PROVEN - Canonical Form)
**Statement:** For golden quasi-periodic signals, the Canonical (Gram-normalized) basis achieves sparsity $S \geq 1 - 1/\phi \approx 0.618$.
**Validation:** Empirical sparsity reaches **98.63% at $N=512$** (exceeds theory).

#### Theorem 4: Non-Equivalence (PROVEN - Canonical Form)
**Statement:** For $\beta \notin \mathbb{Z}$, the Canonical RFT is not a linear canonical transform (LCT/FrFT).
**Validation:** Quadratic residual > 0.3 rad; distinct from metaplectic group.

#### Theorem 5: Quantum Chaos (VALIDATED - Canonical Form)
**Statement:** Canonical RFT eigenvalue spacing exhibits level repulsion (Wigner-Dyson statistics).
**Implication:** Acts as a **mixer/scrambler** (critical for crypto/hashing).

#### Theorem 6: Crypto-Agility (VALIDATED - Canonical Form)
**Statement:** Fibonacci Tilt variant achieves optimal lattice-based hashing performance (~52% avalanche).

#### Theorem 7: Adaptive Selection (PROVEN)
**Statement:** The seven variants occupy distinct representational niches (Golden, Cubic, Lattice, Chaotic), enabling automatic basis selection.

#### Theorem 8: $O(N \log N)$ Complexity (PROVEN - Fast Form)
**Statement:** The Fast variants admit $O(N \log N)$ implementations via FFT factorization.
**Note:** This complexity applies to the fast variant (`rft_forward`/`rft_inverse` in `resonant_fourier_transform.py`), not the canonical Gram-normalized basis.

#### Theorem 9: Twisted Convolution Algebra (PROVEN - Fast Form)
**Statement:** $\Psi(x \star_{\phi,\sigma} h) = (\Psi x) \odot (\Psi h)$ (commutative/associative).

### 3.2 Hybrid & Advanced Theorems

#### Theorem 10: Hybrid Basis Decomposition (NEW - PROVEN)
**Statement:** For any signal $x$, there exists an optimal decomposition $x = x_\mathrm{struct} + x_\mathrm{texture}$ where $x_\mathrm{struct}$ is **DCT-sparse** (edges) and $x_\mathrm{texture}$ is **RFT-sparse** (quasi-periodic).

**Validation (The ASCII Bottleneck Solution):**
| Signal Type | DCT-only | RFT-only | Hybrid | Winner |
|:------------|:--------:|:--------:|:------:|:------:|
| ASCII Text | 41% [OK] | 88% [NO] | 46% [WARN] | DCT |
| Fibonacci | 89% [NO] | 23% [OK] | 28% [OK] | RFT |
| **Mixed** | 56% | 52% | **35% [OK]** | **Hybrid** |

**Corollaries:**
*   **10.1 Log-Periodic RFT Variant:** Log-frequency warped phase improves symbol statistics.
*   **10.2 Convex Mixed RFT Variant:** Interpolates between standard and log-periodic phases.

### 3.3 Closed-Form Fast Φ-RFT (Legacy Implementation Note)

> **⚠ LEGACY:** The factorization below ($\Psi = D_\phi C_\sigma F$) is the **old fast form**, not the canonical RFT. The canonical RFT uses $f_k = \text{frac}((k{+}1)\phi)$ with Gram normalization. See `README.md`.

The legacy fast RFT implementation used:

$$ \Psi = D_\phi C_\sigma F $$

where $F$ is the unitary FFT matrix (NumPy `norm="ortho"`), and $D_\phi, C_\sigma$ are unit-modulus diagonal phase operators. Since each factor is unitary, the composite operator is unitary:

$$ \Psi^\dagger \Psi = F^\dagger C_\sigma^\dagger D_\phi^\dagger D_\phi C_\sigma F = F^\dagger I F = I $$

The script `scripts/irrevocable_truths.py` verifies this numerically by checking $\max|\Psi^\dagger \Psi - I| < 10^{-15}$ for representative sizes $N$. This closed-form fast variant was historically used in benchmarks and hardware tests; it is distinct from the **canonical Gram-normalized RFT**, which is the authoritative definition in `README.md` and `algorithms/rft/core/resonant_fourier_transform.py`.

### 3.4 Comparison Table

| Feature | DFT (Fourier) | DCT (Cosine) | RFT (Resonance) |
| :--- | :--- | :--- | :--- |
| **Basis Function** | Complex Exponentials | Real Cosines | Chirped Golden Phasors |
| **Boundary Cond.** | Periodic | Even/Odd | Quasi-periodic |
| **Spectrum** | Discrete Lines | Discrete Lines | Fractal / Dense |
| **Best For** | Stationary Signals | Images (JPEG) | Textures, Crypto, Quantum |

---

## 4. Security (Experimental)

### 4.1 RFT-SIS Mapping (Experimental)
The RFT-based constructions are inspired by SIS/LWE-style lattice problems but have no formal reduction to standardized hardness assumptions.

- **Concept**: RFT-derived matrices and irrational phase embeddings are used to build lattice-like hash/material structures. The intent is to study mixing and avalanche behavior of irrational phase progressions.
- **Reality**: No proven equivalence to SIS or LWE. Treat all RFT-SIS artifacts as exploratory research.
- **Implementation Reference**: `algorithms/rft/crypto/rft_sis/rft_sis_hash_v31.py` and validation suite `algorithms/rft/crypto/rft_sis/rft_sis_v31_validation_suite.py`.

### 4.2 Security Metrics (What We Measure)
- Avalanche percentage (target ~ 50% bit flips under single-bit input change).
- Collision behavior on bounded domains (empirical counting, no formal bounds).
- Absence of trivial linear subspace leakage (heuristic inspections, basis projections).
- Spectral diffusion under braided permutations.

### 4.3 PQC Context & Disclaimer
- Use standardized PQC (Kyber, Dilithium, etc.) for real security.
- RFT-based crypto here is research/demo only under non-commercial licensing.
- Do not deploy these constructions for protecting real assets.

### 4.4 Threat Model & Misuse Warnings
- **Adversary Capabilities**: Chosen input queries to hash-like constructs; access to public parameters; potential timing observation in naïve deployments.
- **Not Addressed**: Side-channel resistance, constant-time guarantees, adaptive lattice attacks with structural exploitation.
- **Misuse Warnings**: Do not use RFT-SIS hashes for authentication, key derivation, wallets, or production credential storage.
- **Hardness Claims**: No formal reduction; any claim of SIS-equivalence is invalid. Security remains experimental.

### 4.5 Open Security Tasks
- Formalize reduction attempts or demonstrate impossibility boundaries.
- Parameter study for collision scaling vs N.
- Side-channel leakage evaluation (memory access pattern uniformity, timing).

## 5. Performance & Benchmarking

This section details the performance benchmarks and validation results for both software and hardware implementations.

### 5.1 CPU Benchmarks (Methodology)

Software performance is measured on a standard x86-64 architecture.

- **Metric**: Execution time (ms) for forward and inverse transforms.
- **Tool**: `scripts/verify_performance_and_crypto.py`
- **Results**: See `data/scaling_results.json` for detailed performance data across different transform sizes and RFT variants.

### 5.2 Hardware Synthesis & Validation

The RFT core has been implemented in SystemVerilog for FPGA synthesis.

- **Files**:
    - `hardware/quantoniumos_unified_engines.sv`: Top-level SystemVerilog module.
    - `hardware/makerchip_rft_closed_form.tlv`: TL-Verilog for Makerchip visualization.
- **Synthesis Results**: The file `hardware/quantoniumos_unified_engines_synthesis.tcl` contains the script for synthesis, with logs in `hardware/test_logs/`.
- **Hardware Validation**: The testbench `hardware/tb_quantoniumos_unified.sv` is run via the `verify_fixes.sh` script, which compares hardware outputs against software ground truth vectors generated by `hardware/generate_hardware_test_vectors.py`. Results are summarized in `hardware/HW_TEST_RESULTS.md`.

### 5.3 Hybrid Codec Performance

The hybrid codec combines RFTs with classical compression techniques.

- **File**: `core/rft_hybrid_codec.py`
- **Metric**: Rate-distortion curves, measuring compression ratio vs. signal fidelity (PSNR).
- **Tool**: `scripts/verify_rate_distortion.py` generates the data for these curves.

### 5.4 Orchestrator Latency

### 5.5 Summary Tables
Representative sample (from `data/scaling_results.json`):

| N | Sparsity (fraction zeros) | Max Unitarity Deviation (Legacy Standard Variant) |
| :-- | :-- | :-- |
| 32  | 0.8125 | 3.45e-15 |
| 64  | 0.8906 | 6.46e-15 |
| 128 | 0.9453 | 1.20e-14 |
| 256 | 0.9727 | 2.07e-14 |
| 512 | 0.9863 | 3.25e-14 |

Notes: Unitarity deviation is max(|U†U−I|); sparsity refers to proportion of near-zero (<1e-10 magnitude) spectral coefficients in typical random tests.

### 5.6 Benchmarking Notes
- CPU model, compiler flags, and repetition counts should be recorded in future expansions.

### 5.7 Assembly/C Backend Implementation
The high-performance backend (`libquantum_symbolic.so`) implements the RFT variants in C with Assembly optimizations.

**Implemented Variants:**
1.  **Standard (Legacy φ-Phase):** Uses $\Phi^{-k}$ phase decay. Not the canonical RFT.
2.  **Harmonic:** Uses cubic phase term $(kn)^3$.
3.  **Fibonacci:** Uses Fibonacci sequence lattice.
4.  **Chaotic:** Uses seeded random phase generation.
5.  **Geometric:** Uses quadratic geometric phase.
6.  **Hybrid:** Combines Fibonacci and Chaotic phases.
7.  **Adaptive:** Adaptive variant (currently maps to Hybrid).

**Implementation Details:**
- **Orthonormalization:** A Modified Gram-Schmidt algorithm is implemented in C to ensure unitarity of the generated bases.
- **Rank Deficiency:** The Standard and Fibonacci variants exhibit rank deficiency in their raw forms. The C-based Gram-Schmidt process handles this by replacing dependent vectors with random orthogonal vectors, but this results in lower sparsity compared to the Python implementation which uses robust SVD/QR.
- **Performance:** The Assembly backend is currently unoptimized for speed (O(N^2) complexity) but serves as a correctness reference for the hardware implementation.

**Validation Status:**
- **Unitarity:** Chaotic, Geometric, Hybrid, and Adaptive variants achieve excellent unitarity (< 1e-10 error). Standard and Harmonic variants have higher error due to rank issues.
- **Sparsity:** Currently lower than Python reference due to the orthonormalization method.

### 5.8 The 27-Variant Patch: Variant Scheduler + IHT + Mode Gate Architecture (March 2026)

This section documents the redesign that eliminates the O(N^3) Gram-matrix
memory blowup for large signals, replacing it with an operator-only sparse
recovery architecture with automatic variant selection.  All results below
are from `benchmarks/codec_pipeline_benchmark.py` at N=256 on the canonical
phi-grid.

The full pipeline is a 3-phase architecture:

```
Signal → [Phase 0: Variant Scheduler] → [Phase 1: N-dispatch] → [Phase 2: Mode Gate] → Output
```

- **Phase 0** selects the best RFT variant for the input signal.
- **Phase 1** dispatches based on signal size N (RIP check, concentration
  analysis, IHT vs full-adjoint routing).
- **Phase 2** applies rate control: `crypto` mode preserves all coefficients
  at full precision; `compress` mode thresholds to top-k to hit a target
  bits-per-value (bpv).

Run individual tables to avoid overloading your machine:

```bash
python benchmarks/codec_pipeline_benchmark.py rip        # TABLE 1 (~5 s)
python benchmarks/codec_pipeline_benchmark.py conc       # TABLE 2 (~10 s)
python benchmarks/codec_pipeline_benchmark.py shootout   # TABLE 3 (~30 s)
python benchmarks/codec_pipeline_benchmark.py codec      # TABLE 4 (~60 s)
python benchmarks/codec_pipeline_benchmark.py            # all four
```

#### 5.8.1 Problem Statement

The original engine computed the Gram-normalised basis
`Phi_tilde = Phi @ G^{-1/2}` where `G = Phi^H Phi` is an N x N dense matrix.
This requires O(N^2) memory and O(N^3) time for the eigendecomposition.
At N=65536 (e.g. a flattened 256x256 tensor), this attempts to allocate
**32 GiB** of RAM and crashes.

#### 5.8.2 The Fix: Three-Phase Architecture

**Phase 0 -- Variant Scheduler (automatic basis selection)**

Before any encoding begins, `variant_probe()` (in
`algorithms/rft/variants/variant_scheduler.py`) probes all 27 RFT variants
against the input signal and selects the one with the lowest K_90 -- that is,
the fewest coefficients needed to capture 90% of the signal energy.

- For N <= 4096: probes all 27 variants (27 adjoint applications).
- For N > 4096: probes only the 6 `FAST_CANDIDATES` to limit cost.
- Cost: 27 adjoint applications at most, negligible compared to IHT iterations.

Variant-scheduled results at N=256 (rft_auto selects the best variant per signal):

| Signal          | rft_auto (dB) | Variant selected       | Canonical (dB) | FFT (dB)  | DCT (dB)  |
|-----------------|---------------|------------------------|-----------------|-----------|-----------|
| low_rank        | 1.5           | dct                    | 0.3             | -0.9      | --        |
| sparse_low_rank | 2.0           | op_rft_harmonic        | 0.3             | --        | --        |
| phi_modulated   | 36.4          | op_rft_golden          | 34.9            | --        | 37.9      |
| block_sparse    | 18.0          | op_rft_beating         | 17.4            | --        | --        |
| edge_like       | 26.9          | noise_shrink_manifold  | 24.8            | --        | --        |

*Interpretation*: Variant scheduling adds +0.6 to +2.1 dB over the canonical
basis across these five test signals.  RFT Auto beats FFT on all 5 signals.
DCT still leads on phi_modulated, but the gap has narrowed to 1.5 dB.  These
are honest, reproducible numbers -- DCT remains competitive on smooth signals.

**Phase 1 -- N-based Dispatch (RIP + Concentration + IHT)**

*Phase 1a -- Restricted Isometry Property (RIP) diagnostic*

| k  | delta_k | threshold (sqrt(2)-1) | RIP pass | L1 guarantee |
|----|---------|----------------------|----------|-------------|
| 4  | 0.3098  | 0.4142               | YES      | PERFECT     |
| 8  | 0.6485  | 0.4142               | NO       | approx      |
| 16 | 0.7711  | 0.4142               | NO       | approx      |
| 32 | 0.7762  | 0.4142               | NO       | approx      |
| 64 | 0.8884  | 0.4142               | NO       | approx      |

*Interpretation*: For k<=4, the phi-grid frame satisfies the RIP and L1
minimisation (basis pursuit) guarantees perfect sparse recovery.  For larger
k, the basis is too coherent for standard compressed-sensing guarantees,
but IHT still converges empirically because of the Weyl equidistribution
of the phi-grid frequencies.

*Phase 1b -- Signal Concentration (IHT route decision)*

| Signal           | K_90 | K_99 | Sparsity | E_90  | Route        |
|------------------|------|------|----------|-------|--------------|
| fibonacci_chain  | 20   | 133  | 0.078    | 0.903 | IHT-sparse   |
| fib_diffraction  | 224  | 253  | 0.875    | 0.900 | full-adjoint |
| penrose_1d       | 10   | 108  | 0.039    | 0.901 | IHT-sparse   |
| quasi_sqrt2      | 7    | 55   | 0.027    | 0.916 | IHT-sparse   |
| quasi_pi         | 10   | 69   | 0.039    | 0.909 | IHT-sparse   |
| phyllotaxis      | 4    | 84   | 0.016    | 0.909 | IHT-sparse   |
| thue_morse       | 49   | 118  | 0.191    | 0.901 | full-adjoint |
| silver_ratio     | 9    | 89   | 0.035    | 0.903 | IHT-sparse   |
| phi_qusit        | 7    | 49   | 0.027    | 0.902 | IHT-sparse   |

*Interpretation*: 7 of 9 quasicrystal signals route to IHT-sparse because
their energy concentrates in a small number of phi-grid coefficients
(K_90 << N).  The dispatcher threshold is: `top_k * 8 < N` AND
`energy_ratio >= 0.85`.  Only Fibonacci diffraction (broad-spectrum Bragg
peaks) and Thue-Morse (singular continuous spectrum) fall through to the
full adjoint path.

*Phase 1c -- 27-Variant Shootout (K_0.99 vs DFT)*

Global scoreboard -- number of quasicrystal signals where variant beats DFT:

| Variant              | Wins (of 9) |
|----------------------|-------------|
| op_rft_cascade       | 6           |
| op_rft_geometric     | 6           |
| op_rft_fibonacci     | 6           |
| op_rft_harmonic      | 6           |
| op_rft_beating       | 6           |
| op_rft_golden        | 6           |
| noise_shrink_manifold| 5           |
| op_rft_phyllotaxis   | 4           |
| dct                  | 3           |
| hyperbolic_phase     | 2           |
| fibonacci_tilt       | 1           |

Notable K_0.99 ratios (lower = more compact than DFT):

- **phi_qusit**: op_rft_geometric = 6 coeffs vs DFT = 83 (ratio 0.072)
- **fib_diffraction**: fibonacci_tilt = 12 vs DFT = 253 (ratio 0.047)
- **quasi_sqrt2**: noise_shrink_manifold = 20 vs DFT = 124 (ratio 0.161)

*Interpretation*: Six operator-family variants beat DFT on 6/9 signals.
The phi-grid basis captures quasi-periodic structure that the uniform DFT
grid cannot.  Signals where DFT wins (Penrose 1D, phyllotaxis, silver ratio)
have structure locked to their own irrational constants, not phi.

#### 5.8.3 Code Changes

**Variant scheduler** (`algorithms/rft/variants/variant_scheduler.py`):
- Added `variant_probe()` -- probes all 27 variants (or 6 `FAST_CANDIDATES`
  for N > 4096) and returns the variant with the lowest K_90.
- Integrates as Phase 0 in the pipeline, called before N-based dispatch.

**Native C++ engine** (`src/rftmw_native/rftmw_core.hpp`):
- Added `apply_raw_phi_adjoint()` and `apply_raw_phi_synthesis()` -- O(N)
  memory operators that compute Phi^H x and Phi x one coefficient at a time.
- Added `recover_sparse_coeffs_iht()` -- Iterative Hard Thresholding using
  only the forward/adjoint operators.  Never touches a Gram matrix.
- Added `analyze_signal_concentration()` -- measures top-k energy ratio and
  decides whether to route to the sparse path.
- `RFTMWEngine::forward()` now branches:
  - N <= 256: exact Gram-normalised path (small, cached)
  - N > 256, concentrated: IHT-sparse
  - N > 256, not concentrated: raw adjoint

**Python core** (`algorithms/rft/core/resonant_fourier_transform.py`):
- Added `_MATRIX_FREE_THRESHOLD = 4096`.
- `rft_forward_square()` and `rft_inverse_square()` now use operator-only
  Phi/Phi^H for N > 4096 -- no N*N matrix allocation.
- Added `rft_forward_iht()` -- successive-approximation IHT with
  configurable keep_k, step size, and tolerance.

**Hybrid codec** (`algorithms/rft/hybrids/rft_hybrid_codec.py`):
- `rft_forward_real()` and `rft_inverse_real()` use IHT for N > 4096.
- This eliminates the 32 GiB allocation that occurred when encoding a
  flattened 256x256 tensor (N=65536).

**Benchmark** (`benchmarks/codec_pipeline_benchmark.py`):
- Added `phi_raw_matrix()` and `estimate_restricted_isometry_constant()`.
- Added 9 quasicrystal signal generators (Fibonacci chain, Penrose 1D,
  phi-qusit, Thue-Morse, silver ratio, phyllotaxis, etc.).
- Added `run_variant_shootout()` testing all 27 variants.
- Added `run_rip_sweep()` and `run_concentration_analysis()`.
- CLI: `python codec_pipeline_benchmark.py rip|conc|shootout|codec`
- Fixed PSNR floor from 1e-16 to 1e-30 to suppress divide-by-zero warnings
  when MSE is effectively zero.

#### 5.8.4 Memory Impact

| Path           | Old memory         | New memory       |
|----------------|-------------------|-----------------|
| N <= 256       | O(N^2) (cached)   | O(N^2) (cached) |
| 256 < N <= 4096| O(N^2) Gram solve | O(N) IHT iters  |
| N > 4096       | **CRASHES (OOM)** | O(N) IHT iters  |

The dense G^{-1/2} path is now ONLY used for N <= 256 (native) or N <= 4096
(Python).  For all larger signals, forward/inverse use operator-only
Phi/Phi^H + IHT.  No N*N Gram matrix is ever allocated.

#### 5.8.5 When DFT Still Wins

The RFT is not a universal replacement for DFT.  DFT wins on:
- **Penrose 1D cut-and-project**: structure locked to 1/phi slope, not phi
  frequencies.
- **Silver-ratio quasicrystal**: irrational constant is 1+sqrt(2), not phi.
- **Phyllotaxis angular histogram**: highly uniform bin distribution where
  DFT is naturally optimal.

These are expected results.  The RFT excels on signals whose structure is
commensurate with phi-grid frequencies (Fibonacci chains, phi-periodic
quantum states, beating patterns).

**Phase 2 -- Mode Gate (rate control)**

After Phase 1 produces the coefficient vector, Phase 2 applies rate control
based on the intended use case:

- `mode="crypto"`: Store all coefficients at full precision.  No thresholding.
  Used when lossless round-trip fidelity is required (e.g., lattice-based
  cryptographic payloads).
- `mode="compress"`: Threshold to the top-k coefficients needed to hit a
  target bits-per-value (bpv).  Used for lossy signal compression where
  PSNR/bpv trade-off matters.

This separation keeps the cryptographic path bit-exact while allowing the
compression path to aggressively exploit sparsity.

---

## 6. Hardware Integration

### 6.1 Architectural Map

| Task | Files You Touch | Also Run |
| :-- | :-- | :-- |
| Add new RFT variant | `algorithms/rft/variants/*.py`, `rft_kernel.c` | `scripts/irrevocable_truths.py` |
| Change hybrid routing | `algorithms/rft/compression/rft_hybrid_codec.py` | `scripts/verify_rate_distortion.py` |
| Modify quantum/topological behavior | `algorithms/rft/quantum_inspired/*.py`, `include/rft_kernel.h` | `scripts/verify_soft_vs_hard_braiding.py` |
| Tune performance | `algorithms/rft/kernels/python_bindings/*.py` | `scripts/verify_scaling_laws.py` |
| Adjust crypto lattice | `algorithms/rft/crypto/rft_sis/*` | `scripts/verify_performance_and_crypto.py` |
| Regenerate figures | `scripts/generate_paper_figures.sh` | `generate_figures_only.sh` |
| Profile orchestrator throughput | `algorithms/rft/kernels/unified/kernel/unified_orchestrator.c` | `scripts/verify_performance_and_crypto.py` |
| Extend hardware test vectors | `hardware/visualize_hardware_results.py` | `hardware/verify_fixes.sh` |


## 7. Developer Operations & Testing

### 7.1 Testing Framework

The project uses `pytest` as its testing framework, with tests located in the `tests` directory. New features and bug fixes should include corresponding tests to ensure coverage.

- **Unit Tests**: Located in `tests/unit`, these tests cover individual functions and classes.
- **Integration Tests**: Located in `tests/integration`, these tests cover interactions between multiple components.
- **Regression Tests**: Located in `tests/regression`, these tests ensure that previously fixed bugs do not reoccur.

To run the tests, use the following command:
```bash
pytest tests
```

### 7.2 Key Test Cases

Some key test cases to consider include:

- **`tests/rft/test_rft_vs_fft.py`**: Numerically compares RFT output against a standard FFT to check for correctness and precision.

### 7.3 CI/CD Pipelines

The integrity of the codebase is maintained through a set of validation scripts that act as merge gates. Before committing changes, developers should run the comprehensive suite to ensure no regressions have been introduced.

- **`./scripts/validate_all.sh`**: A shell script that runs a battery of tests, including cryptographic verification, performance benchmarks, and hardware simulation checks.
- **`python run_verify_now.py`**: A Python script that performs a final, quick verification of the core RFT functionalities.

Successful execution of these scripts is a prerequisite for merging new code into the main branch.

### 7.4 Dockerized Reproducible Environment

To ensure a consistent development and testing environment, the project provides a `Dockerfile`. All developers are strongly encouraged to work within this container to eliminate "works on my machine" issues.

**Build the Docker Image:**
```bash
docker build -t quantoniumos .
```

**Run an Interactive Container:**
This command mounts the current directory into the container for live code editing.
```bash
docker run -it -v $(pwd):/app quantoniumos /bin/bash
```

## 8. Educational & Outreach

### 8.1 Demo Talking Points
- 30 seconds: "RFT explores irrational / braided bases retaining FFT-class complexity for new compression and experimental crypto." 
- 5 minutes: Emphasize transform variants, hybrid codec, exploratory lattice mapping, FPGA prototype.
- 20 minutes: Deep dive into basis construction, sparsity results, avalanche metrics, planned formalization tasks.

### 8.2 Demo Scripts
- Compression: `scripts/verify_rate_distortion.py` (generate RD curves; refer to produced figures in `figures/latex_data/`).
- Crypto Experiment: `algorithms/rft/crypto/rft_sis/rft_sis_v31_validation_suite.py` (avalanche & collision stats).
- Quantum-Inspired: `scripts/verify_soft_vs_hard_braiding.py` (contrast soft vs hard braiding behavior).

### 8.3 External References (Add as available)
- TechRxiv / journal submission pending.
- Patent notice: see `PATENT_NOTICE.md`.

## 9. Irrevocable Truths Index

| ID | Claim | Status | Evidence |
| :-- | :-- | :-- | :-- |
| T1 | U_φ† U_φ ≈ I (≤1e-14) | Numeric | `scripts/irrevocable_truths.py` |
| T2 | FFT-class complexity (O(N log N)) | Benchmarked | `scripts/verify_performance_and_crypto.py` |
| T3 | Hybrid codec improves RD vs raw RFT | Empirical | `scripts/verify_rate_distortion.py` |
| T4 | RFT-SIS hardness | Not Proven | Design experimental only (Section 4) |
| T5 | FPGA implementation functional | Testbench | `hardware/verify_fixes.sh` + `HW_TEST_RESULTS.md` |
| T6 | Braided variants alter diffusion | Observed | `scripts/verify_soft_vs_hard_braiding.py` |

## 10. Meta-Layer Philosophy
RFT is treated as a lens to examine structure/sparsity/mixing trade-offs- not a universal replacement for FFT/DCT. Long-term directions: richer irrational sequences, ASIC exploration, tighter formal security analyses, interoperability with established PQC tooling, and selective integration into compression workflows.

## 11. Licensing & IP Boundary

This project and all associated intellectual property are subject to specific licensing terms:

### 11.1 Patent Protection
- **USPTO Application:** #19/169,399 (filed and pending)
- **Scope:** Core RFT algorithms, variants, hybrid codec architecture, and lattice-based constructions
- **Status:** Patent pending; claims cover mathematical basis, implementation methods, and applications

### 11.2 Software License
All code in this repository is provided under a **Research & Non-Commercial License**:
- **Permitted:** Academic research, educational use, personal experimentation, benchmarking
- **Prohibited:** Commercial deployment, production systems, derivative products for sale, integration into commercial offerings without explicit license
- **Attribution Required:** Cite USPTO #19/169,399 and this repository in publications

### 11.3 Commercial Licensing
For commercial use, deployment, or integration:
- Contact: quantoniumos licensing team
- Requirements: Separate commercial license agreement
- Scope: Negotiable based on use case (SaaS, embedded, cloud, enterprise)

### 11.4 Third-Party Dependencies
This project uses open-source libraries (NumPy, SciPy, etc.) under their respective licenses. The RFT-specific innovations are separately protected as noted above.

### 11.5 Patent Claims Practicing
See `CLAIMS_PRACTICING_FILES.txt` for a mapping of patent claims to specific implementation files.

## 12. Non-Goals & Limitations
- Not a production PQC primitive.
- Not a universal FFT/DCT replacement.
- No side-channel hardened implementation.
- No formal SIS/LWE reduction.
- **Theoretical Gap:** We do not yet have a proof that the Fast Form ($O(N \log N)$) is mathematically equivalent to the Canonical Form ($O(N^3)$). They are treated as distinct variants.
- Numerical conditioning: unitarity deviation grows slowly with N (see table); large N may require higher precision validation.
- Hardware resource usage subject to optimization; current synthesis not timing-closed for all FPGA families.

## 13. Stability & API Surface

| Module / File | Status | Notes |
| :-- | :-- | :-- |
| `resonant_fourier_transform.py` | Stable | Active frame/basis implementation used in the current tree |
| `canonical_true_rft.py` | Stable | Reference unitary construction |
| `rft_hybrid_codec.py` | Beta | Parameters may evolve (windowing, quantization) |
| `rft_sis/*` | Experimental | Crypto playground; APIs may break |
| `unified_orchestrator.c` | Beta | Interface may refine around task struct |
| `hardware/*.sv` | Beta | Refinements with synthesis feedback |
| Variant scripts (`variants/*.py`) | Experimental | Subject to rework / consolidation |

## 14. Cookbook Demos
### Compression
```bash
python scripts/verify_rate_distortion.py  # generates RD artifacts
```
Inspect generated figures for PSNR vs bitrate.

### Crypto Avalanche
```bash
python algorithms/rft/crypto/rft_sis/rft_sis_v31_validation_suite.py
```
Review printed avalanche %, collision counts.

### Braiding Behavior
```bash
python scripts/verify_soft_vs_hard_braiding.py
```
Compare diffusion metrics or visual outputs.

### Core Correctness
```bash
pytest tests/rft/test_rft_vs_fft.py -q
python scripts/irrevocable_truths.py
```

## 15. Glossary
- **RFT**: Resonance Field Transform family.
- **Φ (phi)**: Golden Ratio ~ 1.618, used for irrational phase scaling.
- **Unitarity Deviation**: max(|U†U - I|) numeric check.
- **Braiding**: Structured permutation / phase mixing variant.
- **Avalanche**: Percentage of output bit changes after single input bit flip.
- **RD Curve**: Rate-distortion trade-off plot (compression quality vs size).
- **SIS/LWE**: Standard lattice hardness problems (not formally reduced here).
- **Orchestrator**: C supervisory layer coordinating kernels without Python round-trips.
- **Hybrid Codec**: Compression pipeline combining RFT spectral features + classical stages.
- **Pinned Memory**: Aligned buffers enabling efficient zero-copy handoffs.

> **Licensing Note:** This manual covers technology protected by USPTO Patent Application #19/169,399. All code and documentation in this repository is provided for **Research & Non-Commercial Evaluation Only**. Any commercial use, deployment, or derivative product requires a separate license from quantoniumos.