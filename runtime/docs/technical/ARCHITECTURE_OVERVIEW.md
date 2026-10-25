# quantoniumos Architecture Overview

**Updated:** March 2026 | **Patent:** USPTO #19/169,399 | **License:** Research/Non-Commercial

## Executive Summary

quantoniumos is a **full-stack computational system** — from desktop environment to hardware RTL — built around the **Resonant Fourier Transform (RFT)**, a unitary transform basis that uses golden-ratio (φ) frequency spacing instead of integer harmonics.

This is **classical signal processing**. No qubits, no quantum gates, no exponential speedup. The RFT provably outperforms FFT/DFT on **golden quasi-periodic signals** and provably loses on everything else.

> **Scope boundaries (maintained from prior audits):**
> - The "quantum-inspired" modules are classical matrix operations mirroring gate definitions — NOT quantum simulation
> - The crypto subsystem is experimental research — NOT production-ready, NOT formally proven secure
> - STRT/RFT shows no advantage over DCT for high-entropy random data (confirmed empirically)

---

## The Core Thesis: Frequency as Infrastructure

Most systems use frequency transforms as a **tool** — analyze a signal, extract features, move on. quantoniumos uses the φ-basis as the **infrastructure itself**. The single unitary matrix Φ̃ = Φ(Φ†Φ)^{-1/2} simultaneously serves four roles:

```
                        ┌──────────────────────────────────┐
                        │     φ-Basis  Φ̃ = Φ(Φ†Φ)^{-1/2}  │
                        │     (One unitary matrix)         │
                        └──────┬───────┬───────┬───────────┘
                               │       │       │
                    ┌──────────┘       │       └──────────┐
                    ▼                  ▼                   ▼
            ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
            │  STORE       │  │  COMPUTE     │  │  COMPOSE     │
            │              │  │              │  │              │
            │  Positional  │  │  Binary→Wave │  │  Lattice     │
            │  encoding    │  │  →Compute    │  │  crypto      │
            │  via φ-grid  │  │  →Binary     │  │  from φ      │
            │  geometry    │  │  pipeline    │  │  structure   │
            │              │  │              │  │              │
            │  KV-cache    │  │  Quantum     │  │  Geometric   │
            │  boundaries  │  │  gate sim    │  │  waveform    │
            │              │  │              │  │  hashing     │
            └──────────────┘  └──────────────┘  └──────────────┘

     Theorem 8 + Three-    Theorem 2 + 4        Theorem 6 + SIS
     Distance Theorem      (unitarity +          (irrationality +
                            convolution)          lattice hardness)
```

A fourth role — **ROUTE** — selects from 27 φ-derived variant bases per signal class via the middleware routing layer. No single basis wins everywhere; the framework value is in adaptive selection.

---

## System Architecture

```
Layer 6  quantoniumos Desktop         PyQt5 desktop environment + golden spiral launcher
         quantonium_os_src/frontend/   [Proprietary — Claim 4]

Layer 5  Application Suite            7 integrated apps (crypto, vault, notes, validator,
         quantonium_os_src/apps/       visualizer, simulator, monitor)  [Proprietary — Claim 4]

Layer 4  RFTMW Middleware Engine      Binary↔Wave↔Compute↔Binary pipeline, compression,
         quantonium_os_src/engine/     inference, memory management  [Claims-NC Proprietary]

Layer 3  Algorithm Library            Canonical RFT, 27 variants, crypto subsystem,
         algorithms/rft/               geometric hashing, quantum-inspired modules  [Claims-NC]

Layer 2  Native Kernels (C++/ASM)     AVX2/AES-NI accelerated transforms, Jacobi eigensolver,
         src/rftmw_native/             Bluestein FFT, 13 variant kernels  [Claims-NC]

Layer 1  Hardware RTL (RFTPU)         TL-Verilog/SystemVerilog FPGA design
         hardware/                     (simulation only — no silicon)  [Claims-NC]
```

Each layer practices one or more claims of **USPTO Patent 19/169,399**:

| Claim | Title | Primary Layers | Infrastructure Role |
|-------|-------|----------------|---------------------|
| **Claim 1** | Symbolic Resonance Fourier Transform Engine | Layers 2-4 | STORE + COMPUTE |
| **Claim 2** | Resonance-Based Cryptographic Subsystem | Layer 3 | COMPOSE |
| **Claim 3** | Geometric Waveform Hashing | Layer 3 | COMPOSE |
| **Claim 4** | Hybrid Mode Integration (Unified Framework) | Layers 4-6 | ROUTE |

---

## Core Components

### 1. RFT Mathematical Kernel (STORE + COMPUTE)

The canonical RFT definition:

```
Φ_{n,k} = (1/√N) exp(j·2π·frac((k+1)·φ)·n)
Φ̃ = Φ·(Φ†Φ)^{-1/2}   (Löwdin orthogonalization)

Forward:  X = Φ̃†·x
Inverse:  x̂ = Φ̃·X
```

| Property | Status | Evidence |
|----------|--------|----------|
| Unitary (Φ̃†Φ̃ = I) | **Proven (Theorem 2)** | Unitarity error < 5e-15 |
| Not a DFT variant | **Proven (Theorem 6)** | φ irrational ⇒ frequencies never rational |
| Golden spectral concentration | **Proven (Theorem 8)** | K₀.₉₉(U)=O(log N) vs K₀.₉₉(F)∝N^0.75 |
| RIP constant | **Empirical (Theorem E)** | δ₈=0.1603 < 0.2929 |
| Twisted convolution | **Proven (Theorem 4)** | Φ̃(x ⋆_Φ h) = (Φ̃x) ⊙ (Φ̃h) |
| O(N log N) fast surrogate | **Proven (Theorem 13)** | Factorized φ-phase FFT |

**Positional encoding (STORE):** The φ-grid satisfies the Three-Distance Theorem optimality condition, making adjacent position vectors maximally dissimilar across window boundaries. Result: 47–80% cross-window leakage reduction vs RoPE (11/11 configurations).

**Gate pipeline (COMPUTE):** The RFTMW engine converts binary data to waveforms, applies the φ-basis as a computational transform, and converts back. The twisted convolution theorem enables algebraic operations directly on frequency-domain representations.

**Implementation:** `algorithms/rft/core/canonical_true_rft.py` (Python API), `src/rftmw_native/` (C++/AVX2), `algorithms/rft/kernels/` (C/ASM)

### 2. Cryptographic Subsystem (COMPOSE)

The φ-basis builds cryptographic structures through composition:

| Component | File | Mechanism |
|-----------|------|-----------|
| RFT-SIS Hash | `algorithms/rft/crypto/rft_sis_hash.py` | Lattice matrix A = A_φ + R (mod q); security from SIS hardness |
| 48-Round Feistel | `algorithms/rft/crypto/enhanced_cipher.py` | Golden-ratio-parameterized key schedule, AES S-box, AEAD mode |
| Geometric Hash | `algorithms/rft/quantum_inspired/geometric_waveform_hash.py` | φ-basis projection → lattice point quantization |

**Verified metrics:** 51.3% avalanche (target 50%), 0 collisions in 10K inputs, χ² uniformity = 243.7 (threshold 293.2).

> **Caveats:** No formal hardness reduction to SIS/LWE. No production audit. Use standard reviewed primitives (AES-GCM, SHA-3) for real security. See `docs/NON_CLAIMS.md`.

### 3. Routing & Variant System (ROUTE)

No single basis wins everywhere. The middleware routes signals to the best-matched transform:

| Signal class | Winning transform | Empirical source |
|-------------|-------------------|------------------|
| ECG (real) | `op_rft_fibonacci` | `tests/variant_shootout.py` |
| Bearing vibration (real) | `op_rft_fibonacci` | `tests/variant_shootout.py` |
| Fibonacci quasicrystal | `noise_shrink_manifold` | `tests/variant_shootout.py` |
| Quasi-periodic synthetic | `dct` | `tests/variant_shootout.py` |
| Deep V-cache (GPT-2) | STRT (depth-gated) | `tests/proofs/test_perplexity_kvcache.py` |
| Gaussian weight matrices | DCT | `tests/proofs/test_model_weights_fidelity.py` |

**Implementation:** 27 variants in `algorithms/rft/variants/registry.py`, routing in `algorithms/rft/routing.py`, dispatch in `quantonium_os_src/engine/RFTMW.py`.

**Important boundary:** The repository contains the pieces for routing, but not a benchmark-trained oracle that always reproduces the shootout winners at runtime. The Claim 4 value is the framework architecture, not a proof that one hard-coded router is globally optimal.

### 4. Desktop Environment (Layers 5-6)

| App | Description |
|-----|-------------|
| **Golden Spiral Launcher** | PyQt5 desktop with animated φ-spiral intro |
| **Symbolic Simulator** | Classical matrix transform workbench |
| **Quantum-Inspired Crypto** | AES + RFT Feistel cipher with AEAD mode |
| **Q-Vault** | AES-encrypted JSON key-value store |
| **Q-Notes** | Integrated text editor |
| **RFT Validator** | Bijection, reversibility, and entropy verification |
| **RFT Visualizer** | Coefficient distribution plots (RFT vs DFT) |
| **System Monitor** | CPU/memory/disk monitoring dashboard |

### 5. Hardware RTL (Layer 1)

**Status:** RTL simulation only — no fabricated silicon, no deployed FPGA bitstream.

RFTPU (RFT Processing Unit) implements the φ-basis transform in TL-Verilog/SystemVerilog for FPGA targets. 16-mode variant mapping verified through simulation. See `hardware/README.md`.

---

## Performance Summary

| Layer | Throughput | Status |
|-------|-----------|--------|
| Pure Python (NumPy) | ~2 GB/s | Always available |
| +Numba JIT | ~5 GB/s | Optional |
| C++ Engine (AVX2) | ~30 GB/s | Optional, pybind11 |
| C/ASM Kernels | ~50 GB/s | Optional, ctypes |

| Metric | Value | How to Reproduce |
|--------|-------|------------------|
| Unitarity error | 1.2e-14 → 8.5e-13 (N=16→1024) | `pytest tests/proofs/test_formal_proofs.py -v` |
| KV-cache leakage reduction | 47–80% vs RoPE | `pytest tests/proofs/test_rft_positional_leakage_canonical.py -v` |
| Compression ratio | 7.8–10.8x | `pytest tests/test_compression_pipeline.py -v` |
| Crypto avalanche | 51.3% | `pytest tests/crypto/test_rft_sis_hash.py -v` |
| Depth-gated perplexity | Parity with DCT (+0.08%) | `pytest tests/proofs/test_perplexity_kvcache.py -v` |
| 479 proof tests | All passing | `pytest tests/proofs/ -v` |

---

## Honest Boundaries

| Claim | Reality |
|-------|---------|
| ~~"Million-qubit quantum simulation"~~ | Classical structured state compression; O(N) for separable states is trivially expected |
| ~~"Post-quantum cryptography"~~ | Experimental, no formal hardness reduction; use NIST PQC for real security |
| ~~"Universal FFT replacement"~~ | RFT wins on golden quasi-periodic signals, loses on everything else |
| ~~"O(1) memory"~~ | O(N) for structured states |
| DCT vs STRT on random data | DCT wins — confirmed in `tests/proofs/test_model_weights_fidelity.py` |

Full non-claims inventory: [`docs/NON_CLAIMS.md`](../NON_CLAIMS.md)

---

## For Paper Submissions

**Lead with:** STRT depth-gated routing for KV-cache compression (strongest empirical result)

**Include:**
- `algorithms/rft/core/*` — Core transform implementations
- `tests/proofs/` — 479 automated proof tests
- `docs/THEOREMS_RFT_IRONCLAD.md` — Formal proofs
- `benchmarks/` — Reproducible benchmarks

**Scope carefully:**
- "Quantum-inspired" means classical matrix operations mirroring gate definitions
- All crypto is experimental research
- Honest negative results belong in the paper (DCT beats STRT on random data)

---

## See Also

- [README.md](../../README.md) — Project overview with "Frequency as Infrastructure" section
- [ENGINEER_MANUAL.md](../../ENGINEER_MANUAL.md) — Full engineering manual (v2.1.0)
- [THEOREMS_RFT_IRONCLAD.md](../THEOREMS_RFT_IRONCLAD.md) — Formal proofs
- [NON_CLAIMS.md](../NON_CLAIMS.md) — What we don't claim
- [ARCHITECTURE.md](../ARCHITECTURE.md) — Layer-by-layer implementation details
- [COMPRESSION_ARCHITECTURE.md](../../COMPRESSION_ARCHITECTURE.md) — Compression pipeline
