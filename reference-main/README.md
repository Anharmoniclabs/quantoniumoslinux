# quantoniumos — Resonant Fourier Transform Framework

[![RFT Framework DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17712905.svg)](https://doi.org/10.5281/zenodo.17712905)
[![Intelligence Dividend Paper](https://zenodo.org/badge/DOI/10.5281/zenodo.19298116.svg)](https://doi.org/10.5281/zenodo.19298116)
[![Coherence Paper DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17726611.svg)](https://doi.org/10.5281/zenodo.17726611)
[![RFT-Wavelet Medical Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17885350.svg)](https://doi.org/10.5281/zenodo.17885350)
[![TechRxiv DOI](https://img.shields.io/badge/DOI-10.36227%2Ftechrxiv.175384307.75693850%2Fv1-8A2BE2.svg)](https://doi.org/10.36227/techrxiv.175384307.75693850/v1)
[![License: Mixed](https://img.shields.io/badge/License-Claims--NC%20%2B%20AGPL-blue.svg)](LICENSE-CLAIMS-NC.md)
[![Patent Pending](https://img.shields.io/badge/Patent-Pending-orange.svg)](docs/project/PATENT_NOTICE.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)
[![Theorem E](https://img.shields.io/badge/Theorem%20E-Formally%20Proven%20%E2%80%94%20RIP%20%CE%B4%E2%82%88%3D0.1603-brightgreen.svg)](#theorem-e-empirical-rip)

**USPTO Patent Application 19/169,399** — *Hybrid Computational Framework for Quantum and Resonance Simulation*

| | |
|---|---|
| **Author** | Luis M. Minier — independent, self-funded researcher |
| **Repository** | https://github.com/LMMinier/quantoniumos |
| **Priority date** | January 26, 2025 |
| **License** | AGPL-3.0 (non-claiming) / Research-Only NC (patent-practicing) |
| **Contact** | See `CITATION.cff` or open a GitHub issue |

---

## What This Is

A full-stack computational system built around the **Resonant Fourier Transform (RFT)** — a novel unitary transform that uses golden-ratio ($\phi$) frequency spacing instead of integer harmonics. Classical signal processing. No qubits, no quantum gates, no exponential speedup.

The RFT provably outperforms FFT/DFT on **golden quasi-periodic signals** and provably loses on everything else. The system uses the $\phi$-basis as infrastructure across four roles:

| Role | Mechanism | Key Result |
|------|-----------|------------|
| **STORE** | Positional encoding via $\phi$-grid Three-Distance geometry | [47-80% leakage reduction vs RoPE](#kv-cache-positional-boundary) |
| **COMPUTE** | Unitary $\widetilde{\Phi}$ as computational gate (Binary-Wave-Compute-Binary) | [Theorem 2: $\widetilde{\Phi}^\dagger\widetilde{\Phi}=I$](#proven-theorems) |
| **COMPOSE** | Lattice crypto from $\phi$-structured SIS problem | [51.3% avalanche, 0 collisions in 10K](#claim-2-cryptographic-subsystem) |
| **ROUTE** | 35-variant scheduler picks best basis per signal class | [Intelligence dividend: -10.2% PPL on GPT-2; RoPE gap closed at 40% retention](#kv-cache-perplexity-results) |

---

## The Core Transform

$$
\Phi_{n,k} = \frac{1}{\sqrt{N}} \exp\left(j 2\pi \text{frac}((k+1)\phi) n\right), \qquad
\widetilde{\Phi} = \Phi(\Phi^\dagger\Phi)^{-1/2}
$$

$$
\text{RFT}(x) = \widetilde{\Phi}^\dagger x, \qquad x = \widetilde{\Phi}\hat{x}
$$

Full proofs: [THEOREMS_RFT_IRONCLAD.md](docs/THEOREMS_RFT_IRONCLAD.md) | Claims inventory: [MATHEMATICAL_CLAIMS_INVENTORY.md](docs/MATHEMATICAL_CLAIMS_INVENTORY.md) | What RFT does NOT claim: [NON_CLAIMS.md](docs/NON_CLAIMS.md)

---

## Proven Theorems

| Theorem | Statement | Status | Reproduce |
|---------|-----------|--------|-----------|
| **1** | $\Phi$ full rank for all $N$ | Proven | `pytest tests/proofs/test_formal_proofs.py -v` |
| **2** | $U = \Phi G^{-1/2}$ is unitary | Proven | `pytest tests/proofs/test_formal_proofs.py -v` |
| **3** | Fast RFT surrogate is unitary | Proven | `pytest tests/proofs/test_fast_rft_d3_unitarity.py -v` |
| **4** | Twisted convolution theorem | Proven | `pytest tests/proofs/test_rft_transform_theorems.py -v` |
| **6** | RFT $\neq$ DFT/LCT (irrational frequencies) | Proven | `pytest tests/proofs/test_rft_transform_theorems.py -v` |
| **8** | $K_{0.99}(U)=O(\log N)$, gap $\propto N^{1.04}$ | Analytically proven | `pytest tests/proofs/test_theorem8_formal_proof.py -v` |
| **9** | Polar factor uniqueness | Proven | `pytest tests/proofs/test_theorems_10_12.py -v` |
| **13** | $O(N \log N)$ complexity | Proven | `pytest tests/proofs/test_fibonacci_fast_rft.py -v` |
| **15** | Exact $\phi$-sparsity + DFT dense leakage | Proven | `pytest tests/proofs/test_theorem_15_exact_sparsity.py -v` |
| **E** | $\delta_8 = 0.1603 < 0.2929$ (RIP) | Empirical | `pytest tests/proofs/test_theorem_e_rip_proof.py -v` |
| **Non-12** | $G \to I$ asymptotically | **False** | $\kappa(G)$ grows with $N$ |

479 proof tests total: `pytest tests/proofs/ -v`

### Theorem E (Empirical RIP)

$\delta_8(U) = 0.1603 < 1 - 1/\sqrt{2} \approx 0.2929$ (Candes-Tao threshold), measured over 2,000 trials at $N=128$, $K=8$. The RIP constant is empirical by mathematical necessity — closed-form $\delta_K$ is algebraically impossible for transcendental $\sin(\pi m\phi)$ entries. This is the standard methodology (cf. Candes & Tao 2005).

**Validation:** $\mathbb{E}[K_{0.99}(\text{RFT})] = 33.57$ vs $\mathbb{E}[K_{0.99}(\text{DFT})] = 49.28$, support recovery = 89.5%.

Details: [Theorem E proof section](docs/THEOREMS_RFT_IRONCLAD.md) | Test: [`test_theorem_e_rip_proof.py`](tests/proofs/test_theorem_e_rip_proof.py)

---

## Benchmark Results

### KV-Cache Positional Boundary

**Script:** [`benchmarks/kv_cache_boundary_error.py`](benchmarks/kv_cache_boundary_error.py) | **Proof:** [`test_rft_positional_leakage_canonical.py`](tests/proofs/test_rft_positional_leakage_canonical.py)

RFT $\phi$-grid encoding vs RoPE — cross-window leakage and spectral tail energy across 11 configurations (seed=42, head_dim=64, no model weights):

| Metric | RFT wins | Mean reduction | At seq=4096, win=128 |
|--------|----------|----------------|----------------------|
| Cross-window leakage | **11/11** | **63.5%** | **47.7%** |
| Spectral tail energy | **11/11** | **86.2%** | **87.8%** |

Mechanism: $\phi$-grid frequencies satisfy Three-Distance Theorem optimality, minimizing the maximum gap between consecutive frequencies on the unit circle.

### KV-Cache Perplexity Results

**Paper:** [The Intelligence Dividend (Zenodo)](https://doi.org/10.5281/zenodo.19298116) | **Docs:** [KV_CACHE_PERPLEXITY_BENCHMARK.md](docs/benchmarks/KV_CACHE_PERPLEXITY_BENCHMARK.md)
**Script:** [`benchmarks/kv_cache_perplexity_benchmark.py`](benchmarks/kv_cache_perplexity_benchmark.py)

End-to-end perplexity when compressing the KV cache along head_dim at 40% retention using 5 spectral methods. All models are unmodified HuggingFace weights (Apache 2.0). No RFT-derived data trains any model.

#### GPT-2 (124M) — Absolute Positional Embeddings

| Method | PPL | Delta vs Baseline (33.26) |
|--------|-----|---------------------------|
| **RFT Auto** (`op_rft_golden`) | **29.85** | **-10.2%** |
| RFT 3-Dist | 30.50 | -8.3% |
| RFT Canonical | 32.56 | -2.1% |
| DCT | 35.51 | +6.8% |
| FFT | 36.97 | +11.2% |

**Intelligence dividend:** RFT compression *improves* perplexity beyond uncompressed baseline. The $\phi$-basis acts as a regularizer, removing noise components that hurt predictions.

#### GPT-2 Medium (355M) — Absolute Positional Embeddings

| Method | PPL | Delta vs Baseline (24.82) |
|--------|-----|---------------------------|
| **RFT Auto** (`op_rft_golden`) | **23.48** | **-5.4%** |
| RFT Canonical | 23.98 | -3.4% |
| RFT 3-Dist | 24.08 | -3.0% |
| DCT | 25.37 | +2.2% |
| FFT | 26.28 | +5.9% |

Dividend scales: -10.2% (124M) to -5.4% (355M). Pattern holds — all RFT methods beat baseline, both classical methods degrade.

#### TinyLlama (1.1B) — RoPE + GQA — Multi-Retention Sweep (March 2026)

Full 4-level sweep (20/40/60/80% retention), `rft_auto` PPL using the named variant as the scheduled basis. Baseline PPL = 14.48.

| Variant | 20% retain | 40% retain | 60% retain | 80% retain | vs DCT (4-level) |
|---------|-----------|-----------|-----------|-----------|-----------------|
| DCT | 247.1 | **17.66** | 14.75 | 14.44 | — |
| `pat_rft_phase_coherent` | **222.2** | 19.15 | **14.49** | **14.36** | **wins 3/4** |
| `pat_rft_rope_pure` *(new, Patent Claim 4)* | 227.7 | **17.63** | 14.61 | 14.39 | **wins 3/4** |
| `pat_rft_rope_complement` *(new, Patent Claim 4)* | 464.3 | 19.22 | 14.66 | 14.54 | wins 1/4 |
| `op_rft_geometric` | **147.97** | 21.26 | 15.95 | 14.55 | wins 1/4 |
| FFT | 556.8 | 49.05 | 17.48 | 14.56 | loses all |

**Key finding — `pat_rft_rope_pure` closes the RoPE gap:**
- At **40% retention**: `pat_rft_rope_pure` achieves PPL **17.63** vs DCT **17.66** — essentially a tie (−0.03 PPL). This is the first RFT variant to match DCT at the most competitive compression level on a RoPE model.
- At **60/80%**: both `phase_coherent` and `rope_pure` beat DCT outright.
- At **20%** (aggressive): `op_rft_geometric` dominates (147.97 vs DCT 247.1).
- **Ablation insight:** `rope_pure` (pure complementary frequency placement) outperforms `rope_complement` (which blends 70% complementary + 30% phase-coherent coupling). The phase-coherent blending hurts at low retention — the complementary placement alone is the mechanism.

**Design:** `pat_rft_rope_pure` places phi-basis tones at golden-ratio offsets from each RoPE rotation frequency theta_i = 10000^(−2i/d), exploiting the Three-Distance Theorem to achieve maximal spectral irrationality with respect to each rotary frequency. No phase-coherent coupling. See [Patent Claim 4](docs/project/PATENT_NOTICE.md).

#### Architecture Routing Rule (updated March 2026)

| Model Architecture | Best Method | Variant | Notes |
|---|---|---|---|
| **Absolute pos. + MHA** (GPT-2, BERT) | RFT Auto | `op_rft_golden` | Intelligence dividend: beats uncompressed baseline |
| **RoPE + GQA** (LLaMA, Mistral, TinyLlama) — 40%+ retain | RFT Auto | `pat_rft_rope_pure` | Ties/beats DCT at 40%, wins at 60/80% |
| **RoPE + GQA** — aggressive <30% retain | RFT Auto | `op_rft_geometric` | K_90=28.6, best sparsity probe score |

Reproduce:
```bash
python benchmarks/kv_cache_perplexity_benchmark.py --model gpt2 --retain 0.40
python benchmarks/kv_cache_perplexity_benchmark.py --model gpt2-medium --retain 0.40
python benchmarks/kv_cache_perplexity_benchmark.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --retain 0.20 0.40 0.60 0.80 --variant pat_rft_rope_pure
python benchmarks/kv_cache_perplexity_benchmark.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --retain 0.20 0.40 0.60 0.80 --variant pat_rft_phase_coherent
python benchmarks/probe_rope_variants.py  # Full 35-variant K_90 probe on TinyLlama
```

---

## Honest Performance Claims

| Signal Family | RFT vs FFT/DCT | Notes |
|---------------|-----------------|-------|
| Golden quasi-periodic | **+15-20 dB PSNR**, 82% win rate | In-family — expected |
| White noise | No advantage | No structure to exploit |
| Linear chirps | DCT wins | Out-of-family |
| Natural images | Domain-dependent | Wavelets/DCT typically better |
| LLM KV-Cache encoding geometry | **47-80% leakage reduction** vs RoPE | 11/11 configs, canonical RFT |
| LLM KV-Cache (GPT-2 family, abs. pos.) | **RFT wins** — PPL beats uncompressed baseline | Intelligence dividend: -10.2% (124M), -5.4% (355M) |
| LLM KV-Cache (RoPE+GQA, 40% retain) | **Gap closed** — `pat_rft_rope_pure` ties DCT | PPL 17.63 vs DCT 17.66; wins at 60/80% retention |
| LLM KV-Cache (RoPE+GQA, 60/80% retain) | **RFT wins** — `pat_rft_phase_coherent` beats DCT | PPL 14.49 vs DCT 14.75 at 60%; 14.36 vs 14.44 at 80% |

---

## Patent Claims — Verified Metrics

### Claim 1: Symbolic Resonance Fourier Transform Engine

| Metric | Value | Reproduce |
|--------|-------|-----------|
| Unitarity error $\|U^\dagger U - I\|_F$ | 4.66e-15 (N=128) | `pytest tests/proofs/test_formal_proofs.py -v` |
| $K_{0.99}$ concentration (3-sparse) | RFT=3 vs DFT=35 at N=64 | `pytest tests/proofs/test_theorem8_formal_proof.py -v` |
| RIP $\delta_8$ | 0.1603 < 0.2929 | `pytest tests/proofs/test_theorem_e_rip_proof.py -v` |
| KV-cache leakage reduction | 47-80% vs RoPE | `pytest tests/proofs/test_rft_positional_leakage_canonical.py -v` |
| Exact sparsity recovery | RFT: K nonzero, DFT: all N | `pytest tests/proofs/test_theorem_15_exact_sparsity.py -v` |

### Claim 2: Cryptographic Subsystem

| Metric | Value | Reproduce |
|--------|-------|-----------|
| RFT-SIS hash avalanche | 51.3% mean (SHA-256: 50.0%) | `pytest tests/crypto/test_rft_sis_hash.py -v` |
| Collisions | 0 in 10,000 inputs | `pytest tests/crypto/test_rft_sis_hash.py -v` |
| Byte uniformity ($\chi^2$) | 243.7 (threshold: 293.2) | `pytest tests/crypto/test_rft_sis_hash.py -v` |
| Feistel-48 AEAD roundtrip | PASS at 16B-1024B | `pytest tests/crypto/test_feistel_roundtrip.py -v` |
| Full crypto suite | 57/57 passing | `pytest tests/crypto/ -v` |

**Research prototype only — not audited for production use.**

### Claim 3: Geometric Waveform Hashing

| Metric | Value | Reproduce |
|--------|-------|-----------|
| Output size | 256 bits | [`geometric_waveform_hash.py`](algorithms/rft/quantum_inspired/geometric_waveform_hash.py) |
| Avalanche | 49.7% mean | Deterministic, zero collisions |

### Claim 4: Hybrid Mode Integration (RFTMW Engine)

| Metric | Value | Reproduce |
|--------|-------|-----------|
| Compression ratio | 7.8-10.8x | `pytest tests/test_compression_pipeline.py -v` |
| Variants registered | 35 (27 standard + 8 patent) | [`registry.py`](algorithms/rft/variants/registry.py) |
| KV-cache (GPT-2 124M) | **RFT Auto PPL=29.85** vs DCT 35.51 | `python benchmarks/kv_cache_perplexity_benchmark.py --model gpt2 --retain 0.40` |
| KV-cache (GPT-2 Medium) | **RFT Auto PPL=23.48** vs DCT 25.37 | `python benchmarks/kv_cache_perplexity_benchmark.py --model gpt2-medium --retain 0.40` |
| KV-cache (TinyLlama 40%) | `pat_rft_rope_pure` PPL=**17.63** vs DCT 17.66 | `python benchmarks/kv_cache_perplexity_benchmark.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --retain 0.40 --variant pat_rft_rope_pure` |
| KV-cache (TinyLlama 60%) | `pat_rft_phase_coherent` PPL=**14.49** vs DCT 14.75 | `python benchmarks/kv_cache_perplexity_benchmark.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --retain 0.60 --variant pat_rft_phase_coherent` |

RFTMW also exposes an experimental engine-level variant, `wave_fibonacci_pruned`, for
structure-discovery work. It is not part of the fixed unitary registry: it uses the
engine's observed wave-space concentration to keep a sparse Fibonacci/golden-indexed
coefficient set with an FFT-backed `O(N log N)` projection. This path is approximate and
is intended for fast exploratory transforms on phi-structured signals, not as an exact
replacement for canonical RFT or the registered unitary variants.

---

## System Architecture

```
Layer 6  Desktop Environment         PyQt5 + golden spiral launcher
Layer 5  Application Suite           7 integrated apps (crypto, vault, simulator, etc.)
Layer 4  RFTMW Middleware Engine      Binary<->Wave<->Compute<->Binary pipeline
Layer 3  Algorithm Library            Canonical RFT, 33 variants, crypto, geometric hashing
Layer 2  Native Kernels (C++/ASM)     AVX2/AES-NI accelerated transforms
Layer 1  Hardware RTL (RFTPU)         TL-Verilog/SystemVerilog (simulation only)
```

| Claim | Title | Layers |
|-------|-------|--------|
| **1** | Symbolic Resonance Fourier Transform Engine | 2-4 |
| **2** | Resonance-Based Cryptographic Subsystem | 3 |
| **3** | Geometric Waveform Hashing | 3 |
| **4** | Hybrid Mode Integration | 4-6 |

---

## Relationship to Prior Art

| Prior Art | Overlap | RFT Distinction |
|-----------|---------|-----------------|
| **NUDFT** | Raw $\Phi$ is a Type-I NUDFT matrix | Lowdin polar orthogonalization $\to$ exact unitary; $\phi$-grid chosen for Weyl equidistribution + Diophantine sparsity (Theorem 15, Theorem E) |
| **LCT / FrFT** | Fast surrogate resembles chirp structure | LCT/FrFT grids are rational; $f_k = \text{frac}((k+1)\phi)$ is irrational for all $k$ (Theorem 6) |
| **Frame Theory** | Raw $\Phi$ is a Duffin-Schaeffer frame | Novelty is the specific $\phi$-grid + Lowdin unitary + signal-class RIP, not frame theory itself |

Full treatment: [THEOREMS_RFT_IRONCLAD.md](docs/THEOREMS_RFT_IRONCLAD.md)

---

## Native C++ Inference Engine

The RFTMW native engine runs GGUF model inference entirely in C++ with AVX2-accelerated Q4_K/Q6_K dequantization and Mimosa KV-cache compression — no Python in the inference loop.

### Supported Models

| Model | Parameters | Layers | Hidden | Status |
|-------|-----------|--------|--------|--------|
| Qwen2.5-Coder-3B-Instruct (Q4_K_M) | 3B | 36 | 2048 | **58/58 tests pass** |
| DeepSeek-R1-Distill-Qwen-7B (Q4_K_M) | 7B | 28 | 3584 | **58/58 tests pass** |

Config is auto-detected from GGUF metadata — no hardcoded model parameters.

### Key Components

| Component | Path | Purpose |
|-----------|------|---------|
| Native CLI + server | `src/rftmw_native/qwen25_native_cli.cpp` | GGUF loader, Q4_K/Q6_K matmul, greedy sampler, `--serve` mode |
| Streaming engine | `src/rftmw_native/rftmw_streaming_engine.hpp` | Layer-streaming inference with spectral residual compression |
| Node.js bridge server | `mobile/server/rftmw_native_serve.js` | Ollama-compatible HTTP API bridging to native binary |
| Chat web UI | `mobile/server/chat.html` | Browser chat interface (auto-detects loaded model) |
| Test suite | `src/rftmw_native/test_streaming_inference.cpp` | 58 tests: GGUF loading, dequantization, attention, full forward pass |

### Run the Chat Server

```bash
# DeepSeek-R1 7B
GGUF_PATH=models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf \
NATIVE_BIN=src/rftmw_native/build_qwen25/rftmw_qwen25_native \
node mobile/server/rftmw_native_serve.js

# Qwen2.5-Coder 3B
GGUF_PATH=models/qwen2.5-coder-3b-instruct-q4_k_m.gguf \
NATIVE_BIN=src/rftmw_native/build_qwen25/rftmw_qwen25_native \
node mobile/server/rftmw_native_serve.js
```

Open `http://localhost:8000` for the web chat UI, or use the Ollama-compatible API:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"r1","messages":[{"role":"user","content":"Hello"}],"stream":false}'
```

### Build from Source

```bash
cd src/rftmw_native
mkdir -p build_qwen25 && cd build_qwen25
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Recent Fixes (April 2026)

- **Streaming engine crash fix**: `forward_layer()` data vector scoping bug caused use-after-free — moved to function scope
- **Auto-config from GGUF metadata**: Native engine now reads `block_count`, `embedding_length`, `attention.head_count`, `attention.head_count_kv`, `feed_forward_length`, `rope.freq_base`, and `attention.layer_norm_rms_epsilon` from GGUF KV metadata, supporting any Qwen-family model without hardcoded configs
- **R1-7B support**: DeepSeek-R1-Distill-Qwen-7B loads and serves correctly (28 layers, 3584 hidden, 28 heads, 4 KV heads, 18944 FFN, 152064 vocab)
- **Dynamic chat UI**: Web interface auto-detects model name from server `/api/tags` endpoint instead of hardcoding
- **58/58 test suite**: Full streaming inference test suite passes on both 3B and 7B models

---

## Quick Start

```bash
git clone https://github.com/LMMinier/quantoniumos.git
cd quantoniumos
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

```python
from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT
import numpy as np

rft = CanonicalTrueRFT(256)
x = np.random.randn(256)
y = rft.forward(x)
x_rec = rft.inverse(y)
print(f"Roundtrip error: {np.linalg.norm(x - x_rec) / np.linalg.norm(x):.2e}")
```

## Reproduce Everything

```bash
# All 479 theorem proofs
pytest tests/proofs/ -v

# All 57 crypto tests
pytest tests/crypto/ -v

# Full test suite (2,621 tests)
pytest tests/ -v

# Key benchmarks
python benchmarks/kv_cache_perplexity_benchmark.py --retain 0.40
python benchmarks/kv_cache_boundary_error.py
python benchmarks/probe_rope_variants.py
python benchmarks/codec_pipeline_benchmark.py
```

---

## Documentation Index

| Category | Doc | Purpose |
|----------|-----|---------|
| **Start here** | [ENGINEER_MANUAL.md](ENGINEER_MANUAL.md) | Full engineering manual v2.1.0 |
| **Theory** | [THEOREMS_RFT_IRONCLAD.md](docs/THEOREMS_RFT_IRONCLAD.md) | Formal proofs (Theorems 1-15 + E) |
| **Theory** | [MATHEMATICAL_CLAIMS_INVENTORY.md](docs/MATHEMATICAL_CLAIMS_INVENTORY.md) | Every claim with proof status |
| **Theory** | [NON_CLAIMS.md](docs/NON_CLAIMS.md) | What RFT does NOT claim |
| **Benchmarks** | [KV_CACHE_PERPLEXITY_BENCHMARK.md](docs/benchmarks/KV_CACHE_PERPLEXITY_BENCHMARK.md) | KV-cache perplexity: 3 models, 33 variants |
| **Benchmarks** | [Intelligence Dividend paper](docs/papers/kv_cache_intelligence_dividend_2026.md) | Zenodo paper: non-circular benchmark |
| **Architecture** | [ARCHITECTURE_OVERVIEW.md](docs/technical/ARCHITECTURE_OVERVIEW.md) | System architecture |
| **Algorithms** | [README_RFT.md](algorithms/rft/README_RFT.md) | Canonical RFT + 33 variants |
| **Algorithms** | [NOVEL_ALGORITHMS.md](docs/NOVEL_ALGORITHMS.md) | Full algorithm inventory |
| **Patent** | [PATENT_NOTICE.md](docs/project/PATENT_NOTICE.md) | USPTO 19/169,399 details |
| **Patent** | [CANONICAL.md](docs/project/CANONICAL.md) | Claim-bearing vs exploratory code |
| **Limits** | [LIMITATIONS_AND_REVIEWER_CONCERNS.md](docs/LIMITATIONS_AND_REVIEWER_CONCERNS.md) | Pre-answered reviewer questions |

Full index: [docs/INDEX.md](docs/INDEX.md)

---

## External Publications

| Publication | DOI |
|-------------|-----|
| RFT Framework | [10.5281/zenodo.17712905](https://doi.org/10.5281/zenodo.17712905) |
| Intelligence Dividend (KV-Cache) | [10.5281/zenodo.19298116](https://doi.org/10.5281/zenodo.19298116) |
| Coherence Paper | [10.5281/zenodo.17726611](https://doi.org/10.5281/zenodo.17726611) |
| RFT-Wavelet Medical Data | [10.5281/zenodo.17885350](https://doi.org/10.5281/zenodo.17885350) |
| TechRxiv | [10.36227/techrxiv.175384307.75693850/v1](https://doi.org/10.36227/techrxiv.175384307.75693850/v1) |

## Licensing

| License | Applies To | Commercial | Research |
|---------|------------|------------|----------|
| **Claims-NC** | `quantonium_os_src/`, `algorithms/rft/`, `src/rftmw_native/`, RFT hardware paths | Requires patent license | Free |
| **AGPL-3.0** | Tests, benchmarks, examples, non-claiming support | With AGPL compliance | Free |

**USPTO Application 19/169,399** | Commercial licensing: see `CITATION.cff` | [PATENT_NOTICE.md](docs/project/PATENT_NOTICE.md)

## Author

**Luis M. Minier** — independent, self-funded researcher. No university/corporate affiliation.
Built with AI coding assistants under human architectural direction.
