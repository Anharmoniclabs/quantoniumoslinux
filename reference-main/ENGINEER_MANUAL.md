# quantoniumos Engineer Manual

**Version:** 2.1.0 (Unified Architecture — March 2026 Sync)  
**Status:** Release Candidate — Validated Core, Experimental Crypto  
**Author:** Luis M. Minier (quantoniumos)  
**Patent Disclosure:** USPTO #19/169,399 — Research/Non-Commercial License  
**DOI Reference:** TechRxiv / Zenodo (pending)  
**Last Synced:** March 27, 2026 (reflects PRs #24 and #25)

---

## Table of Contents
0. [Getting Started](#0-getting-started)
1. [Executive Layer](#1-executive-layer)
   - 1.1 [AI Inference Architecture — Performance & Memory](#11-ai-inference-architecture--performance--memory)
2. [Architecture](#2-architecture)
3. [Mathematical Appendix](#3-mathematical-appendix)
4. [Security (Experimental)](#4-security-experimental)
5. [Performance & Benchmarking](#5-performance--benchmarking)
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
source .venv/bin/activate  # Windows: .venv\Scripts\activate
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
from algorithms.rft.core.closed_form_rft import ClosedFormRFT

N = 128
rft = ClosedFormRFT(N)          # Constructs unitary Phi-based matrix directly
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
# Alternatively:
python run_formal_verification.py
```

Use the full validation command before and after architectural changes.

### 0.6 Dockerized Quickstart
```bash
docker build -t quantoniumos .
docker run -it -v $(pwd):/app quantoniumos /bin/bash
```
See `Dockerfile` and `Dockerfile.papers` in the repo root for full environment specs.

---

## 1. Executive Layer

RFT (Resonant Fourier Transform) is a family of transforms using irrational phase progressions (Golden Ratio φ) and braided/topological variants. The goal is to explore trade-offs in sparsity, mixing, and structural diversity while retaining FFT-class complexity for practical sizes.

> **Naming note (PR #25):** The canonical name is **Resonant Fourier Transform**, not "Resonance Field Theory" or "Resonance Field Transform". All docstrings in `unitary_rft.py` and related modules were corrected in March 2026 to reflect this.

### The Core Thesis: Frequency as Infrastructure

Most systems use frequency transforms as a **tool** — analyze a signal, extract features, move on. quantoniumos uses the φ-basis as the **infrastructure itself**. The single unitary matrix Φ̃ = Φ(Φ†Φ)^{-1/2} simultaneously serves three roles that conventional architectures split across separate subsystems:

| Role | What the φ-basis does | Theorems | Where it lives |
| :--- | :--- | :--- | :--- |
| **STORE** | Encodes positional state via φ-grid geometry (Three-Distance Theorem optimality minimizes cross-window confusion in KV-caches) | Theorem 8, Three-Distance Theorem | `benchmarks/kv_cache_boundary_error.py`, `tests/proofs/test_rft_positional_leakage_canonical.py` |
| **COMPUTE** | Acts as a unitary gate in the Binary→Wave→Compute→Binary pipeline; twisted convolution (Theorem 4) enables algebraic operations directly in the frequency domain | Theorem 2 (unitarity), Theorem 4 (convolution) | `quantonium_os_src/engine/RFTMW.py`, `algorithms/rft/quantum_inspired/quantum_gates.py` |
| **COMPOSE** | Builds cryptographic lattice structures from φ-structured matrices; security derives from SIS hardness of the frequency lattice | Theorem 6 (irrationality), SIS lattice hardness | `algorithms/rft/crypto/rft_sis_hash.py`, `algorithms/rft/quantum_inspired/geometric_waveform_hash.py` |
| **ROUTE** | Middleware selects from 27 φ-derived variant bases per signal class; no single basis wins everywhere | Theorem 7 (variant niches) | `algorithms/rft/routing.py`, `algorithms/rft/variants/registry.py` |

These are **not separate features** — they are four consequences of the same mathematical object. Every section of this manual documents one of these roles in detail.

### Design Philosophy
| Feature | Classical (FFT/DCT) | Quantonium (RFT) |
| :--- | :--- | :--- |
| **Basis** | Integer Harmonics (k) | Irrational Resonances (frac((k+1)φ)) |
| **Topology** | Flat Circle (S^1) | Toroidal Braid (T^2) |
| **Complexity** | O(N log N) | O(N log N) (Closed Form) |
| **Inversion** | Trivial (DFT is self-inverse up to conjugate) | Löwdin polar orthogonalization → exact unitary |
| **Application** | Linear Filtering | Storage (positional encoding), Computation (gate pipeline), Composition (lattice crypto), Routing (variant selection) |

**Mission:** To unify **Storage** (Positional Encoding), **Computation** (Unitary Gate Pipeline), **Composition** (Lattice Cryptography), and **Routing** (Signal-Adaptive Basis Selection) into a single, mathematically invariant framework built on the φ-basis.

### Verified Performance (March 2026)
| Domain | Result | Proof |
| :--- | :--- | :--- |
| KV-cache positional leakage | 47–80% reduction vs RoPE (11/11 configs) | `tests/proofs/test_rft_positional_leakage_canonical.py` |
| Spectral concentration | K₀.₉₉(RFT)=33.57 vs K₀.₉₉(DFT)=49.28 at N=128 | `tests/proofs/test_theorem_e_rip_proof.py` |
| RIP constant | δ₈=0.1603 < 0.2929 (Candès-Tao threshold) | `tests/proofs/test_theorem_e_rip_proof.py` |
| Compression | 7.8–10.8x ratio, 46–47 dB PSNR | `tests/test_compression_pipeline.py` |
| Crypto avalanche | 51.3% (RFT-SIS hash, target 50%) | `tests/crypto/test_rft_sis_hash.py` |
| Depth-gated routing | Parity with DCT (+0.08% margin) on GPT-2 perplexity | `tests/proofs/test_perplexity_kvcache.py` |
| **Honest negative** | DCT beats STRT on Gaussian weight matrices | `tests/proofs/test_model_weights_fidelity.py` |

### Where This Lives (Traceability)
| Concept | Code | Validation |
| :-- | :-- | :-- |
| Phi-RFT Core | `algorithms/rft/core/closed_form_rft.py` | `tests/rft/test_rft_vs_fft.py` |
| Canonical Unitary | `algorithms/rft/core/canonical_true_rft.py` | `scripts/irrevocable_truths.py` |
| Hybrid Codec | `algorithms/rft/compression/rft_hybrid_codec.py` | `scripts/verify_rate_distortion.py` |
| RFT-SIS Lattice | `algorithms/rft/crypto/rft_sis/rft_sis_hash_v31.py` | `algorithms/rft/crypto/rft_sis/rft_sis_v31_validation_suite.py` |
| Topological Mode | `algorithms/rft/kernels/include/rft_kernel.h` | `scripts/verify_soft_vs_hard_braiding.py` |
| Unified Orchestrator | `algorithms/rft/kernels/unified/kernel/unified_orchestrator.c` | `scripts/verify_performance_and_crypto.py` |
| Encoding Geometry Tests | `tests/proofs/test_encoding_geometry_leakage.py` | PR #24 |
| Perplexity / KV-Cache Tests | `tests/proofs/test_perplexity_kvcache.py` | PR #24 |
| Model Weights Fidelity Tests | `tests/proofs/test_model_weights_fidelity.py` | PR #24 |

> **Note:** Throughout this manual, "Phi-RFT" refers to the **canonical** Gram-normalized `frac((k+1)*phi)` basis unless explicitly labeled "legacy" or "variant".

---

## 1.1 AI Inference Architecture — Performance & Memory

The quantoniumos inference engine (`quantonium_os_src/engine/rftmw_numpy_engine.py`) is a **pure-NumPy + optional C++ kernel** runtime for transformer models, designed for research, debugging, and demonstrating the φ-basis advantages in production AI workloads.

### Speed Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Model load time** | 30–60s | 36-layer 3B model, full tensor parse |
| **Prefill latency** | 1–4s | Depends on prompt length |
| **Decode throughput** | 0.6–2.0 tok/s | AVX2 kernels, CPU-only |
| **Per-layer matmul** | 4.2 ms (FP32) / 42 ms (Q6K fused) | down_proj benchmark |
| **Preload time** | ~25s | One-time FP32 materialization (253 tensors) |

**Comparison vs. Ollama (llama.cpp):**

| Operation | Ollama (CPU) | quantoniumos | Gap Reason |
|-----------|--------------|--------------|------------|
| Load 3B model | ~5s | ~30s | mmap vs. full parse |
| Decode tok/s | ~10 | ~0.6–2 | Optimized C++ vs. research Python |
| Memory | ~2 GB | ~2.5 GB | GGUF mmap vs. lazy load |
| GPU | ✓ CUDA/Metal | ✗ CPU only | Research focus |

**Why the speed gap exists:**
1. **No mmap** — Ollama memory-maps GGUF, pages fault on demand; we parse and decompress upfront
2. **No GPU path** — Our engine is CPU-only for debugging transparency
3. **On-the-fly dequant** — Q4K/Q6K blocks dequantize per matmul (research mode)
4. **GIL bound** — Pure-Python control loop with NumPy compute

> **Production recommendation:** Use Ollama/vLLM for serving. Use quantoniumos for research, debugging, and experimenting with φ-basis compression.

### Memory Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Memory Abstraction Pipeline                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │               Weight Storage (Disk → RAM)                   ││
│  │  • .rftmwpk pack: JSON index + compressed blobs             ││
│  │  • GGUF fallback: mmap tensors at full precision            ││
│  │  • Lazy loading: tensors loaded on first access             ││
│  └─────────────────────────────────────────────────────────────┘│
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │               QuantWeight Objects (RAM)                     ││
│  │  • Raw uint8 blobs — NO dequantization at rest              ││
│  │  • Q4K: 144 bytes/32-element block (4.5 bits/weight)        ││
│  │  • Q6K: 210 bytes/32-element block (6.5 bits/weight)        ││
│  │  • Dequant happens only during matmul()                     ││
│  └─────────────────────────────────────────────────────────────┘│
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │               KV-Cache (RAM, per-layer)                     ││
│  │  • Shape: (max_seq, num_kv_heads, head_dim)                 ││
│  │  • FP32 activations (not quantized in cache)                ││
│  │  • Three-Distance spectral router for compression           ││
│  └─────────────────────────────────────────────────────────────┘│
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │               Activations (Transient)                       ││
│  │  • Hidden state: (hidden_size,) FP32                        ││
│  │  • Attention: (num_heads, head_dim) FP32                    ││
│  │  • MoE routing: (n_experts,) FP32 logits                    ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Quantified Memory Savings:**

| Component | Traditional (FP32) | quantoniumos | Reduction |
|-----------|-------------------|--------------|-----------|
| 3B model weights | ~12 GB | ~2.5 GB (Q6K) | **79%** |
| KV-cache (80 tokens) | ~150 MB | ~90 MB (60% retain) | **40%** |
| Peak inference RAM | ~14 GB | ~3 GB | **78%** |

The `kv_routing.expected_footprint_reduction: 40%` in `.meta.json` refers to the **KV-cache** compression via Three-Distance spectral routing, not the weight quantization.

### New Capabilities Enabled by φ-Schema

The quantoniumos schema introduces capabilities not available in standard inference runtimes:

#### 1. **Three-Distance Spectral KV Routing**
```json
"kv_routing": {
  "method": "three_distance",
  "retain_frac": 0.6,
  "rip_constant": 0.1603,
  "candes_tao_threshold": 0.2929,
  "proven_bound": true
}
```
- **What it does:** Compresses KV-cache along `head_dim` using φ-grid basis selection
- **Proven result:** RIP δ₈=0.1603 < 0.2929 (Candès-Tao threshold) guarantees sparse recovery
- **Impact:** -10.2% perplexity on GPT-2 at 40% retention (vs +6.8% for DCT)

#### 2. **Intelligence Dividend Effect**
Unlike DCT/FFT compression that always degrades quality, RFT compression can *improve* predictions:

| Method | GPT-2 PPL (baseline 33.26) | Delta |
|--------|---------------------------|-------|
| DCT 40% | 35.51 | +6.8% worse |
| FFT 40% | 36.97 | +11.2% worse |
| **RFT Auto 40%** | **29.85** | **-10.2% better** |

The φ-basis acts as a **regularizer**, filtering noise components that hurt predictions.

#### 3. **Variant-Scheduled Basis Selection**
The engine auto-selects from 27 φ-derived variant bases per signal class:
```python
# From kv_cache_perplexity_benchmark.py
variants = [
    'op_rft_golden',           # Default for absolute positional encoding
    'pat_rft_phase_coherent',  # Better for RoPE (rotary positional encoding)
    'op_rft_fibonacci',        # Integer lattice signals
    # ... 24 more variants
]
```
No single basis wins everywhere — the router picks the optimal basis per layer/head.

#### 4. **MoE Expert Routing Support**
Native support for Mixture-of-Experts architectures (Qwen3.5-style):
```python
class MoETransformerLayer:
    """Handles top-k expert selection with fused QKV attention."""
    moe_ffn: MoEFFN  # Routes to 8/256 experts based on router logits
    gate_exps: QuantWeight3D  # Shape: (n_experts, out_features, bytes_per_row)
```

#### 5. **Provenance-Tracked Model Packs**
Every `.rftmwpk` includes cryptographic provenance:
```json
{
  "provenance": {
    "model_name": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "source_type": "gguf_quantized",
    "parameter_count": 3000000000
  }
}
```

#### 6. **Fallback Cascade Architecture**
```
┌────────────────────────────────────────────────────────────────────┐
│                     Kernel Selection Cascade                        │
├────────────────────────────────────────────────────────────────────┤
│  1. Native C++ with AVX2    ──▶  fastest (x86-64 with AVX2)        │
│         │ if unavailable                                            │
│         ▼                                                           │
│  2. Native C++ scalar       ──▶  works on any x86-64               │
│         │ if module fails to load                                   │
│         ▼                                                           │
│  3. Pure NumPy (Python)     ──▶  works everywhere, 5-10x slower    │
└────────────────────────────────────────────────────────────────────┘
```
Environment control: `RFTMW_NO_NATIVE=1` forces NumPy; `RFTMW_REQUIRE_NATIVE=1` fails hard if no C++.

### Summary: What the Schema Enables

| Capability | Traditional Runtime | quantoniumos |
|------------|--------------------:|-------------:|
| Weight storage | FP32 (12 GB/3B) | Q6K (2.5 GB/3B) |
| KV-cache compression | None or DCT (degrades) | φ-basis (-10% PPL improvement) |
| Basis selection | Fixed (DFT/DCT) | 27-variant auto-routing |
| Compression proof | Empirical only | RIP-bounded (δ₈ < 0.29) |
| MoE support | Separate codepath | Native QuantWeight3D |
| Debugging | Black-box | Transparent NumPy + fallback |

**Bottom line:** The schema trades raw speed (0.6 tok/s vs 10 tok/s) for **research transparency**, **proven compression bounds**, and **the Intelligence Dividend** — the only known KV-cache compression method that can improve predictions while reducing memory.

### Forward Pass Validation & Diagnostics

The C++ inference kernels have been validated against pure NumPy implementations:

| Kernel | Max Diff (FP32) | Status |
|--------|-----------------|--------|
| `rms_norm` | 0.00e+00 | ✓ Identical |
| `q4k_matmul` | 0.00e+00 | ✓ Identical |
| `q6k_matmul` | 4.2e-07 | ✓ FP32 precision |
| `apply_rope` | 0.00e+00 | ✓ Identical |
| `gqa_attention` | 0.00e+00 | ✓ Identical |
| `silu_mul` | 9.5e-07 | ✓ FP32 precision |
| **Full layer** | 2.86e-06 | ✓ Acceptable |

**Diagnostic tool:** `tools/forward_pass_diagnostics.py`
```bash
python tools/forward_pass_diagnostics.py [model.rftmwpk]
RFTMW_NO_NATIVE=1 python tools/forward_pass_diagnostics.py  # Force NumPy
```

**Common diagnostic patterns:**

1. **Large FFN norm weights (up to 31.0):** Normal for Qwen architecture. The model was trained with these values and compensates in subsequent layers.

2. **High change ratio in layer 0 (~25x):** Normal. The first layer transforms embeddings from near-zero variance to intermediate representations.

3. **Hidden state values up to 1300:** Normal in early/middle layers. Residual connections accumulate; the final RMS norm re-scales before the LM head.

4. **True problems to watch for:**
   - NaN or Inf in any tensor → weight loading bug
   - Final logits outside [-100, 100] → numerical instability
   - Top-k predictions are all special tokens → tokenization issue

**Validation tests:**
```bash
python test_cpp_vs_numpy.py       # Kernel-level comparison
python test_cpp_vs_numpy_layer.py # Full layer comparison
```

---

## 2. Architecture

The system operates as a vertical stack, designed to eliminate the "Python Tax" for heavy compute.

### 2.1 The Stack Topology

```
L6: Application (Python) — Scripts, Jupyter Labs
L5: Hybrid Layer        — RFTHybridCodec, Residual Predictor
L4: Variants (Theorems) — GoldenRatioUnitary, SymbolicUnitary, EntropicUnitary
L3: Bindings (ctypes)   — Python Bindings
L2: Middleware (C)      — Unified Orchestrator, Task Scheduler
L1: Kernels (ASM/AVX)   — RFT Engine, Feistel/SIS, Quantum Sim
L0: Hardware (FPGA)     — SystemVerilog Core
```

Key directories mapping to layers:
- `algorithms/rft/` — L4–L5 Python logic
- `algorithms/rft/kernels/unified/kernel/` — L2 C orchestrator
- `hardware/` — L0 FPGA files (`quantoniumos_unified_engines.sv`, `fpga_top.sv`, `rft_middleware_engine.sv`, `rftpu_architecture.tlv`)
- `src/` — supplementary C/native source
- `quantonium_os_src/` — OS-level integration source

### 2.2 Middleware Orchestration Layer
Located in `algorithms/rft/kernels/unified/kernel/unified_orchestrator.c`.

The **Unified Orchestrator** is a C-level supervisor that manages data flow between engines without returning control to Python, preventing the GIL from becoming a bottleneck.

- **Task Struct**: `unified_task_t` defines the operation (Transform, Encrypt, Measure).
- **Zero-Copy**: Data remains in pinned memory buffers shared between engines.
- **Pipeline**:
  1. **RFT Transform**: Signal → Spectrum.
  2. **Topological Permutation**: Spectrum → Scrambled Spectrum (Crypto).
  3. **Measurement**: Scrambled Spectrum → Entropy (Compression).
- **Throughput Estimate**: For large batches (N >= 2^18) the orchestrator sustains >90% overlap of compute and memory, yielding effective throughput ~7.0–7.3 GFLOP/s.
- **Latency Model**: Single transform dominated by basis multiply: O(N log N) + linear chirp phase; scheduling overhead < 0.5% of wall-clock at N >= 2^14.
- **Complexity Summary**: Orchestrator adds O(T) control overhead (T = task transitions); data ops remain O(N log N) + O(N) variant phase adjustments.

### 2.3 Memory Model
- **Phase Matrices**: Pre-computed D_phi and C_sigma tables in 32-byte aligned memory for AVX2/AVX512.
- **State Vectors**: Quantum states are contiguous complex128 arrays.
- **Context Switching**: Orchestrator switches between Crypto (Feistel) and Quantum (Wavefunction) state in < 50 cycles.

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
| Add proof/evaluation test | `tests/proofs/` | `pytest tests/proofs/ -q` |

---

## 3. Mathematical Appendix: The 10 Irrevocable Theorems

Full proofs and validation data are available in `docs/validation/RFT_THEOREMS.md`.

> **Note on naming:** Theorems originally stated for the "Fast Phi-RFT" factorization (Psi = D_phi * C_sigma * F) use the **legacy formulation**. The canonical RFT is the Gram-normalized `frac((k+1)*phi)` basis. See `README.md` for the authoritative definition.

### 3.1 Scientific Distinction

| Form | Used For | Complexity |
| :-- | :-- | :-- |
| **Canonical Form** (Gram-normalized phi-grid) | Theorems 3, 4, 5, 6 | O(N^3) |
| **Fast Form** (FFT-factorized) | Theorems 1, 2, 8, 9 | O(N log N) |

> **The Gap:** No formal proof yet that the Fast Form is a limit of the Canonical Form. They are distinct mathematical objects sharing the same phase philosophy.

### 3.2 Core Theorems

#### Theorem 1: Unitarity (PROVEN — Fast Form)
The legacy Fast Phi-RFT factorization Psi = D_phi * C_sigma * F is exactly unitary: Psi† Psi = I.  
**Validation:** Frobenius error < 1e-14 across N in {32, ..., 512}.  
**Note:** The canonical RFT achieves unitarity via Gram normalization of the `frac((k+1)*phi)` basis.

#### Theorem 2: Exact Diagonalization (PROVEN — Fast Form)
For golden-ratio resonance systems, Psi† H Psi = Lambda + E where ||E||_F < 1e-14.  
**Implication:** O(N) evolution complexity vs O(N^2) for generic methods.

#### Theorem 3: Massive Sparsity (PROVEN — Canonical Form)
For golden quasi-periodic signals, the Canonical basis achieves sparsity S >= 1 - 1/phi ≈ 0.618.  
**Validation:** Empirical sparsity reaches **98.63% at N=512** (exceeds theory).

#### Theorem 4: Non-Equivalence (PROVEN — Canonical Form)
For beta not in Z, the Canonical RFT is not a linear canonical transform (LCT/FrFT).  
**Validation:** Quadratic residual > 0.3 rad; distinct from metaplectic group.

#### Theorem 5: Quantum Chaos (VALIDATED — Canonical Form)
Canonical RFT eigenvalue spacing exhibits level repulsion (Wigner-Dyson statistics).  
**Implication:** Acts as a mixer/scrambler (critical for crypto/hashing).

#### Theorem 6: Crypto-Agility (VALIDATED — Canonical Form)
Fibonacci Tilt variant achieves optimal lattice-based hashing performance (~52% avalanche).

#### Theorem 7: Adaptive Selection (PROVEN)
The seven variants occupy distinct representational niches (Golden, Cubic, Lattice, Chaotic), enabling automatic basis selection.

#### Theorem 8: O(N log N) Complexity (PROVEN — Fast Form)
The Fast variants admit O(N log N) implementations via FFT factorization.  
**Note:** Applies to `rft_forward`/`rft_inverse` in `resonant_fourier_transform.py`, not the canonical Gram-normalized basis.

#### Theorem 9: Twisted Convolution Algebra (PROVEN — Fast Form)
Psi(x *_{phi,sigma} h) = (Psi x) o (Psi h) (commutative/associative).

### 3.3 Hybrid & Advanced Theorems

#### Theorem 10: Hybrid Basis Decomposition (NEW — PROVEN)
For any signal x, there exists an optimal decomposition x = x_struct + x_texture where x_struct is **DCT-sparse** (edges) and x_texture is **RFT-sparse** (quasi-periodic).

**Validation (The ASCII Bottleneck Solution):**
| Signal Type | DCT-only | RFT-only | Hybrid | Winner |
|:------------|:--------:|:--------:|:------:|:------:|
| ASCII Text | 41% [OK] | 88% [NO] | 46% [WARN] | DCT |
| Fibonacci | 89% [NO] | 23% [OK] | 28% [OK] | RFT |
| **Mixed** | 56% | 52% | **35% [OK]** | **Hybrid** |

**Corollaries:**
- **10.1 Log-Periodic RFT Variant:** Log-frequency warped phase improves symbol statistics.
- **10.2 Convex Mixed RFT Variant:** Interpolates between standard and log-periodic phases.

### 3.4 Legacy Fast Form Reference

> **LEGACY:** The factorization below (Psi = D_phi * C_sigma * F) is the **old fast form**, not the canonical RFT.

The legacy fast RFT used: Psi = D_phi * C_sigma * F  
where F is the unitary FFT matrix (NumPy `norm="ortho"`), and D_phi, C_sigma are unit-modulus diagonal phase operators.

Since each factor is unitary: Psi† Psi = F† C_sigma† D_phi† D_phi C_sigma F = F† I F = I

The script `scripts/irrevocable_truths.py` verifies this numerically by checking max|Psi†Psi - I| < 1e-15 for representative sizes N.

### 3.5 Comparison Table

| Feature | DFT (Fourier) | DCT (Cosine) | RFT (Resonant Fourier) |
| :--- | :--- | :--- | :--- |
| **Basis Function** | Complex Exponentials | Real Cosines | Chirped Golden Phasors |
| **Boundary Cond.** | Periodic | Even/Odd | Quasi-periodic |
| **Spectrum** | Discrete Lines | Discrete Lines | Fractal / Dense |
| **Best For** | Stationary Signals | Images (JPEG) | Textures, Crypto, Quantum |

---

## 4. Security (Experimental)

### 4.1 RFT-SIS Mapping (Experimental)
RFT-based constructions are inspired by SIS/LWE-style lattice problems but have **no formal reduction** to standardized hardness assumptions.

- **Concept**: RFT-derived matrices and irrational phase embeddings used to build lattice-like hash structures.
- **Reality**: No proven equivalence to SIS or LWE. Treat all RFT-SIS artifacts as exploratory research.
- **Implementation**: `algorithms/rft/crypto/rft_sis/rft_sis_hash_v31.py` + validation suite `algorithms/rft/crypto/rft_sis/rft_sis_v31_validation_suite.py`.

### 4.2 Security Metrics (What We Measure)
- Avalanche percentage (target ~50% bit flips under single-bit input change).
- Collision behavior on bounded domains (empirical counting, no formal bounds).
- Absence of trivial linear subspace leakage (heuristic inspections, basis projections).
- Spectral diffusion under braided permutations.

### 4.3 PQC Context & Disclaimer
- Use standardized PQC (Kyber, Dilithium, etc.) for real security.
- RFT-based crypto is research/demo only under non-commercial licensing.
- Do **not** deploy these constructions for protecting real assets.

### 4.4 Threat Model & Misuse Warnings
- **Adversary Capabilities**: Chosen input queries to hash-like constructs; access to public parameters; potential timing observation.
- **Not Addressed**: Side-channel resistance, constant-time guarantees, adaptive lattice attacks.
- **Misuse Warnings**: Do not use RFT-SIS hashes for authentication, key derivation, wallets, or production credential storage.
- **Hardness Claims**: No formal reduction. Any claim of SIS-equivalence is invalid.

### 4.5 Open Security Tasks
- Formalize reduction attempts or demonstrate impossibility boundaries.
- Parameter study for collision scaling vs N.
- Side-channel leakage evaluation (memory access pattern uniformity, timing).

---

## 5. Performance & Benchmarking

### 5.1 CPU Benchmarks (Methodology)
- **Metric**: Execution time (ms) for forward and inverse transforms.
- **Tool**: `scripts/verify_performance_and_crypto.py`
- **Results**: See `data/scaling_results.json` for detailed performance data.
- **Benchmark suite**: `tests/benchmarks/` and `benchmarks/` (root-level directory).

### 5.2 Hardware Synthesis & Validation
- **Files**:
  - `hardware/quantoniumos_unified_engines.sv`: Top-level SystemVerilog module.
  - `hardware/fpga_top.sv`: FPGA top-level (simulation-only; 16-mode variant mapping table added in PR #25).
  - `hardware/rft_middleware_engine.sv`: Four-stage pipeline; simulation-only status documented in PR #25.
  - `hardware/rftpu_architecture.tlv`: TL-Verilog; simulation-only, key architectural parameters added in PR #25.
  - `hardware/makerchip_rft_closed_form.tlv`: Makerchip visualization entry.
  - `hardware/tb_quantoniumos_unified.sv`: Testbench.
- **Synthesis**: `hardware/quantoniumos_unified_engines_synthesis.tcl` with logs in `hardware/test_logs/`.
- **Validation**: `hardware/verify_fixes.sh` compares hardware outputs against software ground truth from `hardware/generate_hardware_test_vectors.py`. Results in `hardware/HW_TEST_RESULTS.md`.

### 5.3 Hybrid Codec Performance
- **File**: `algorithms/rft/compression/rft_hybrid_codec.py`
- **Metric**: Rate-distortion curves (compression ratio vs PSNR).
- **Tool**: `scripts/verify_rate_distortion.py`.

### 5.4 Orchestrator Latency
- Single transform latency: O(N log N) + linear chirp phase overhead.
- Scheduling overhead < 0.5% wall-clock at N >= 2^14.
- See `scripts/verify_performance_and_crypto.py` for profiling.

### 5.5 Sparsity & Unitarity Summary
Representative sample from `data/scaling_results.json`:

| N | Sparsity (fraction zeros) | Max Unitarity Deviation (Legacy Standard Variant) |
| :-- | :-- | :-- |
| 32  | 0.8125 | 3.45e-15 |
| 64  | 0.8906 | 6.46e-15 |
| 128 | 0.9453 | 1.20e-14 |
| 256 | 0.9727 | 2.07e-14 |
| 512 | 0.9863 | 3.25e-14 |

Notes: Unitarity deviation = max(|U†U−I|); sparsity = proportion of near-zero (< 1e-10 magnitude) spectral coefficients in typical random tests.

### 5.6 Model Weights Fidelity (NEW — PR #24)
From `tests/proofs/test_model_weights_fidelity.py` (6/6 pass, numpy/scipy only, ~55s CPU):

| Retain | STRT best-case | DCT worst-case | Winner |
| :-- | :-- | :-- | :-- |
| 80% | 0.239 | 0.087 | DCT |
| 60% | 0.435 | 0.207 | DCT |
| 40% | 0.650 | 0.399 | DCT |

**Honest negative result:** DCT beats STRT on all Gaussian weight matrices. Per `NON_CLAIMS.md`: "High-entropy random: No advantage". Do NOT cite STRT weight compression as an LM performance improvement.

### 5.7 Assembly/C Backend Implementation
The high-performance backend (`libquantum_symbolic.so`) implements RFT variants in C with Assembly optimizations.

**Implemented Variants:**
1. **Standard (Legacy phi-Phase):** Uses Phi^{-k} phase decay. Not the canonical RFT.
2. **Harmonic:** Uses cubic phase term (kn)^3.
3. **Fibonacci:** Uses Fibonacci sequence lattice.
4. **Chaotic:** Uses seeded random phase generation.
5. **Geometric:** Uses quadratic geometric phase.
6. **Hybrid:** Combines Fibonacci and Chaotic phases.
7. **Adaptive:** Adaptive variant (currently maps to Hybrid).

**Validation Status:**
- Unitarity: Chaotic, Geometric, Hybrid, Adaptive achieve < 1e-10 error. Standard and Harmonic have higher error due to rank issues.
- Sparsity: Currently lower than Python reference due to Gram-Schmidt orthonormalization method (vs SVD/QR in Python).
- Performance: Assembly backend is O(N^2) unoptimized — serves as correctness reference, not speed benchmark.

---

## 6. Hardware Integration

### 6.1 Hardware File Map
| File | Purpose | Status |
| :-- | :-- | :-- |
| `hardware/fpga_top.sv` | FPGA top-level, 16-mode variant mapping | Simulation-only |
| `hardware/rft_middleware_engine.sv` | 4-stage pipeline, variant-select port | Simulation-only |
| `hardware/rftpu_architecture.tlv` | TL-Verilog, key architectural params | Simulation-only |
| `hardware/quantoniumos_unified_engines.sv` | Unified top-level module | Beta |
| `hardware/makerchip_rft_closed_form.tlv` | Makerchip visualization | Reference |
| `hardware/tb_quantoniumos_unified.sv` | Testbench | Active |
| `hardware/HW_TEST_RESULTS.md` | Synthesis/sim results log | See after running verify_fixes.sh |

> **Note (PR #25):** All hardware files received simulation-only status annotations, RFT mathematical foundation notes, and improved port documentation in March 2026.

### 6.2 Architectural Task Map
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

---

## 7. Developer Operations & Testing

### 7.1 Test Directory Structure
Actual layout as of March 2026:
```
tests/
  __init__.py
  conftest.py
  algo_tests/          # Algorithm-level unit tests
  benchmarks/          # Performance benchmark tests
  codec_tests/         # Codec-specific tests
  compression/         # Compression pipeline tests
  crypto/              # Crypto/SIS tests
  data/                # Test data assets
  integration/         # Multi-component integration tests
  medical/             # Medical signal application tests
  native/              # C/native binding tests
  proofs/              # Proof/evaluation test suite (new — PR #24)
  research/            # Research-track exploratory tests
  rft/                 # Core RFT transform tests
  slow/                # Long-running regression tests
  system/              # End-to-end system tests
  transforms/          # Transform variant tests
  validation/          # Validation scripts as tests
  test_ans_integration.py
  test_codec_comprehensive.py
  test_codecs_updated.py
  test_compression_pipeline.py
  test_definitive_canonical_rft.py
  test_patent_variants.py
  test_phi_labeled_canonical_rft.py
  test_rftmw_memory.py
  test_training_guardrails.py
  run_all_assembly_tests.sh
```

### 7.2 Proof / Evaluation Suite (NEW — PR #24)
The `tests/proofs/` directory implements a three-layer evaluation suite:

| Layer | File | Status |
| :-- | :-- | :-- |
| Encoding geometry | `test_encoding_geometry_leakage.py` | Already in main before PR #24 |
| Perplexity (KV-cache) | `test_perplexity_kvcache.py` | Added in PR #24 |
| Model weights fidelity | `test_model_weights_fidelity.py` | Added in PR #24 (6/6 pass) |

To run the full proofs suite:
```bash
pytest tests/proofs/ -q
pytest tests/proofs/ -m integration  # requires torch + transformers
```

### 7.3 Key Test Files
- **`tests/rft/test_rft_vs_fft.py`**: Numerically compares RFT output against FFT for correctness and precision.
- **`tests/test_definitive_canonical_rft.py`**: Comprehensive canonical RFT tests (~38KB).
- **`tests/test_phi_labeled_canonical_rft.py`**: Phi-labeled canonical RFT validation (~37KB).
- **`tests/test_codec_comprehensive.py`**: Full codec pipeline tests.
- **`tests/test_compression_pipeline.py`**: Compression pipeline end-to-end.
- **`tests/test_rftmw_memory.py`**: RFTMW memory behavior tests.
- **`tests/test_training_guardrails.py`**: Guardrail assertions for training-related misuse.
- **`tests/test_patent_variants.py`**: Verifies patent-claimed variant behaviors.

### 7.4 CI/CD & Validation Gates
Before merging, run:
```bash
./scripts/validate_all.sh
python run_verify_now.py
python run_formal_verification.py
```

Additional per-area scripts:
```bash
python scripts/irrevocable_truths.py          # Core unitarity checks
python scripts/verify_rate_distortion.py      # Hybrid codec RD curves
python scripts/verify_soft_vs_hard_braiding.py
python scripts/verify_performance_and_crypto.py
python scripts/verify_scaling_laws.py
./hardware/verify_fixes.sh                    # Hardware testbench
./tests/run_all_assembly_tests.sh             # Native/assembly backend
```

### 7.5 Dockerized Reproducible Environment
```bash
docker build -t quantoniumos .
docker run -it -v $(pwd):/app quantoniumos /bin/bash
# For paper/publication artifacts:
docker build -f Dockerfile.papers -t quantoniumos-papers .
```

---

## 8. Educational & Outreach

### 8.1 Demo Talking Points
- **30 seconds**: "RFT explores irrational / braided bases retaining FFT-class complexity for new compression and experimental crypto."
- **5 minutes**: Emphasize transform variants, hybrid codec, exploratory lattice mapping, FPGA prototype.
- **20 minutes**: Deep dive into basis construction, sparsity results, avalanche metrics, three-layer proof suite, planned formalization tasks.

### 8.2 Demo Scripts
- **Compression**: `scripts/verify_rate_distortion.py` (RD curves in `figures/latex_data/`).
- **Crypto Experiment**: `algorithms/rft/crypto/rft_sis/rft_sis_v31_validation_suite.py` (avalanche & collision stats).
- **Quantum-Inspired**: `scripts/verify_soft_vs_hard_braiding.py` (soft vs hard braiding).
- **Proof Suite**: `pytest tests/proofs/ -v` (encoding geometry, perplexity, weight fidelity).

### 8.3 External References
- TechRxiv / journal submission pending. See `CITATION.cff` for citation metadata.
- Patent notice: `PATENT_NOTICE.md`.
- Patent claims implementation mapping: `PATENT_CLAIMS_IMPLEMENTATION_ANALYSIS.md`.
- Compression architecture deep-dive: `COMPRESSION_ARCHITECTURE.md`.
- Codebase structure reference: `CODEBASE_STRUCTURE.md`.

---

## 9. Irrevocable Truths Index

| ID | Claim | Status | Evidence |
| :-- | :-- | :-- | :-- |
| T1 | U_phi† U_phi ≈ I (<=1e-14) | Numeric | `scripts/irrevocable_truths.py` |
| T2 | FFT-class complexity O(N log N) | Benchmarked | `scripts/verify_performance_and_crypto.py` |
| T3 | Hybrid codec improves RD vs raw RFT | Empirical | `scripts/verify_rate_distortion.py` |
| T4 | RFT-SIS hardness | Not Proven | Experimental only (Section 4) |
| T5 | FPGA implementation functional | Testbench | `hardware/verify_fixes.sh` + `HW_TEST_RESULTS.md` |
| T6 | Braided variants alter diffusion | Observed | `scripts/verify_soft_vs_hard_braiding.py` |
| T7 | DCT beats STRT on Gaussian weight matrices | Empirical (honest negative) | `tests/proofs/test_model_weights_fidelity.py` |
| T8 | Encoding geometry leakage bounded | Empirical | `tests/proofs/test_encoding_geometry_leakage.py` |

---

## 10. Meta-Layer Philosophy

The φ-basis is infrastructure, not just a tool. The core thesis — **Frequency as Infrastructure** — means the same unitary matrix serves as storage (positional encoding), computation (gate pipeline), composition (lattice crypto), and routing (variant selection). These four roles are consequences of the same mathematical object, not separate features bolted together.

RFT is treated as a lens to examine structure/sparsity/mixing trade-offs — not a universal replacement for FFT/DCT. Long-term directions: richer irrational sequences, ASIC exploration, tighter formal security analyses, interoperability with established PQC tooling, and selective integration into compression workflows.

The project maintains honest negative results (e.g., T7 above) alongside positive ones. `NON_CLAIMS.md` documents the explicit boundaries of what the framework does and does not claim.

---

## 11. Licensing & IP Boundary

### 11.1 Patent Protection
- **USPTO Application:** #19/169,399 (filed and pending)
- **Scope:** Core RFT algorithms, variants, hybrid codec architecture, and lattice-based constructions
- **Status:** Patent pending; claims cover mathematical basis, implementation methods, and applications
- **Claims mapping:** See `PATENT_CLAIMS_IMPLEMENTATION_ANALYSIS.md` and `LICENSE-CLAIMS-NC.md`

### 11.2 Software License
All code is provided under a **Research & Non-Commercial License** (see `LICENSE`, `LICENSE-CLAIMS-NC.md`, `LICENSE-SPX-PROPRIETARY.md`):
- **Permitted:** Academic research, educational use, personal experimentation, benchmarking
- **Prohibited:** Commercial deployment, production systems, derivative products for sale
- **Attribution Required:** Cite USPTO #19/169,399 and this repository in publications. See `CITATION.cff`.

### 11.3 Commercial Licensing
For commercial use, deployment, or integration:
- Contact: quantoniumos licensing team
- Requirements: Separate commercial license agreement
- Scope: Negotiable based on use case (SaaS, embedded, cloud, enterprise)

### 11.4 Third-Party Dependencies
This project uses open-source libraries (NumPy, SciPy, PyTorch optional, etc.) under their respective licenses. See `requirements.txt` and `pyproject.toml`.

---

## 12. Non-Goals & Limitations
- Not a production PQC primitive.
- Not a universal FFT/DCT replacement.
- No side-channel hardened implementation.
- No formal SIS/LWE reduction.
- **Theoretical Gap:** No proof that Fast Form O(N log N) is mathematically equivalent to Canonical Form O(N^3). They are treated as distinct variants.
- Numerical conditioning: unitarity deviation grows slowly with N; large N may require higher precision validation.
- Hardware resource usage subject to optimization; current synthesis not timing-closed for all FPGA families.
- STRT/RFT shows no advantage over DCT for high-entropy random data (confirmed empirically in `tests/proofs/test_model_weights_fidelity.py`).

---

## 13. Stability & API Surface

| Module / File | Status | Notes |
| :-- | :-- | :-- |
| `closed_form_rft.py` | Stable | Core forward/inverse API frozen for 2.x |
| `canonical_true_rft.py` | Stable | Reference unitary construction |
| `rft_hybrid_codec.py` | Beta | Parameters may evolve (windowing, quantization) |
| `rft_sis/*` | Experimental | Crypto playground; APIs may break |
| `unified_orchestrator.c` | Beta | Interface may refine around task struct |
| `hardware/*.sv` | Beta | Refinements with synthesis feedback |
| `variants/*.py` | Experimental | Subject to rework / consolidation |
| `tests/proofs/*` | Stable (test suite) | Three-layer proof evaluation suite; do not remove |
| `routing.py` | Stable | Heuristic helper; NOT a benchmark-trained oracle — see IMPORTANT BOUNDARY NOTE in docstring |

---

## 14. Cookbook Demos

### Compression
```bash
python scripts/verify_rate_distortion.py  # generates RD artifacts
```
Inspect generated figures in `figures/latex_data/` for PSNR vs bitrate.

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

### Three-Layer Proof Suite
```bash
pytest tests/proofs/ -v
pytest tests/proofs/test_model_weights_fidelity.py -v  # 6/6 pass, ~55s
pytest tests/proofs/test_perplexity_kvcache.py -m integration  # requires torch+transformers
```

### Bootstrap / Full Environment
```bash
bash quantoniumos-bootstrap.sh  # Full environment setup
bash reproduce_results.sh        # Reproduce all published results
```

---

## 15. Glossary
- **RFT**: Resonant Fourier Transform family (corrected from "Resonance Field Transform" — PR #25).
- **Phi (φ)**: Golden Ratio ≈ 1.618, used for irrational phase scaling.
- **φ-Basis**: The Gram-normalized unitary matrix Φ̃ = Φ(Φ†Φ)^{-1/2} built from golden-ratio frequency spacing.
- **Frequency as Infrastructure**: The core design thesis — the φ-basis simultaneously serves as storage (positional encoding), computation (gate pipeline), composition (lattice crypto), and routing (variant selection). These are four consequences of the same mathematical object, not separate features.
- **STORE**: The positional encoding role of the φ-basis; φ-grid geometry minimizes cross-window confusion in KV-caches (47–80% leakage reduction vs RoPE).
- **COMPUTE**: The computational role of the φ-basis; the Binary→Wave→Compute→Binary pipeline in RFTMW, where the unitary matrix acts as the logic gate.
- **COMPOSE**: The compositional role of the φ-basis; building lattice cryptographic structures from φ-structured frequency matrices.
- **ROUTE**: The adaptive role of the middleware; selecting from 27 variant bases per signal class.
- **Unitarity Deviation**: max(|U†U - I|) numeric check.
- **Braiding**: Structured permutation / phase mixing variant.
- **Avalanche**: Percentage of output bit changes after single input bit flip.
- **RD Curve**: Rate-distortion trade-off plot (compression quality vs size).
- **SIS/LWE**: Standard lattice hardness problems (not formally reduced here).
- **Orchestrator**: C supervisory layer coordinating kernels without Python round-trips.
- **Hybrid Codec**: Compression pipeline combining RFT spectral features + classical stages.
- **Pinned Memory**: Aligned buffers enabling efficient zero-copy handoffs.
- **STRT**: Structured Transform (RFT-based); no advantage over DCT for high-entropy random data.
- **Encoding Geometry Leakage**: Structural information leakage from transform basis into compressed representation; bounded by `tests/proofs/test_encoding_geometry_leakage.py`.
- **KV-Cache Compression**: Key-Value cache compression in LLM inference; perplexity tested in `tests/proofs/test_perplexity_kvcache.py`.
- **Three-Distance Theorem**: Classical number theory result; φ-grid frequencies satisfy this optimality condition, enabling maximal positional dissimilarity.

---

> **Licensing Note:** This manual covers technology protected by USPTO Patent Application #19/169,399. All code and documentation in this repository is provided for **Research & Non-Commercial Evaluation Only**. Any commercial use, deployment, or derivative product requires a separate license from quantoniumos.
