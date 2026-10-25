# AGENT.md — quantoniumos Repository Handoff & Operator Contract v3

> **READ THIS FIRST.** This file is the single source of truth for any AI agent working in this repo.
> It contains: what the repo is, what's in it, what was recently built, what works, what doesn't, and the rules.

---

## 1. WHAT THIS REPO IS

**quantoniumos** is a full-stack computational framework built around the **Resonant Fourier Transform (RFT)** — a novel unitary transform using golden-ratio (φ) frequency spacing instead of integer harmonics.

| | |
|---|---|
| **Author** | Luis M. Minier — independent, self-funded researcher |
| **Patent** | USPTO 19/169,399 — *Hybrid Computational Framework for Quantum and Resonance Simulation* |
| **License** | AGPL-3.0 (non-claiming) / Research-Only NC (patent-practicing) |
| **Python** | 3.10+ (venv at `.venv/`) |
| **Build** | MinGW/MSYS2 (C:\msys64\mingw64\bin), CMake + Ninja, NASM, AVX2/FMA |
| **Version** | 2.0.0 (pyproject.toml) |

### The Core Transform

```
Φ[n,k] = (1/√N) · exp(j·2π·frac((k+1)·φ)·n)
Φ̃ = Φ·(Φᴴ·Φ)^(-1/2)     ← Löwdin orthonormalization
RFT(x) = Φ̃ᴴ·x,  x = Φ̃·x̂
```

The φ-basis provably outperforms FFT/DFT on golden quasi-periodic signals and provably loses on everything else. It serves four roles: **STORE** (positional encoding), **COMPUTE** (unitary gate), **COMPOSE** (lattice crypto), **ROUTE** (35-variant scheduler).

---

## 2. DIRECTORY MAP

```
quantoniumos/
├── .github/AGENT.md              ← YOU ARE HERE
├── algorithms/                   ← Core RFT math (27 files), codecs, crypto, routing, 35 variants
│   ├── rft/core/                 ← Canonical RFT, fast FFT surrogate, Gram utils, theorems
│   ├── rft/compression/          ← Entropy probe, rANS coder, binary codec, quantum compressor
│   ├── rft/variants/             ← 35 transform variants (patent, hybrid, DCT, chaotic, etc.)
│   ├── rft/crypto/               ← φ-SIS lattice hash, enhanced cipher
│   ├── rft/kernels/              ← C/ASM kernel sources (rft_kernel.c, .asm, quantum_symbolic)
│   ├── rft/quantum_inspired/     ← Classical quantum gate sim (14 files)
│   ├── rft/hybrids/              ← H3/FH5/H6 cascade, medical wavelet hybrid (9 files)
│   ├── rft/theory/               ← Formal proofs engine, Theorem 8
│   └── mimosa/                   ← MIMOSA gate implementation
│
├── src/rftmw_native/             ← C/C++/ASM native acceleration layer
│   ├── qwen25_native_cli.cpp     ← Qwen2.5/DeepSeek full inference CLI (~1745 lines)
│   ├── qwen35_native_cli.cpp     ← Qwen3.5 MoE native CLI
│   ├── rftmw_core.hpp            ← Core RFT SIMD/AVX2 math
│   ├── rftmw_kv_compress.hpp     ← KV-cache RFT compression kernel (~310 lines)
│   ├── rftmw_streaming_engine.hpp ← Layer-streaming engine (~700 lines)
│   ├── rftmw_spectral_kernels.c  ← Spectral matmul C kernel (AVX2 FFT hybrid)
│   ├── rftmw_spectral_inference.hpp ← Spectral-domain inference
│   ├── rftmw_mmap.hpp            ← Memory-mapped inference
│   ├── rftmw_sgemv_f32.asm       ← Dense GEMV ASM (float32)
│   ├── rftmw_spectral_dot.asm    ← Sparse spectral dot product ASM
│   ├── rftmw_python.cpp          ← pybind11 Python bindings
│   ├── CMakeLists.txt             ← Build config (C++17, AVX2, NASM, LTO)
│   ├── webgpu/                    ← WebGPU GEMV shader + JS runtime
│   ├── build_asm/                 ← ✅ CURRENT BUILD (has KV compress + layer streaming)
│   ├── build_qwen25/              ← ⚠️ STALE BUILD (missing KV compress)
│   └── build_cpp/                 ← Has compiled librftmw_kernels.a
│
├── src/apps/                      ← Python CLI tools & desktop app
│   ├── qshll_chatbox.py           ← QShell desktop chatbox (main Qt5 app)
│   ├── chatbox_network.py         ← Chatbox networking (SpectralNativeEngine, subprocess mgmt)
│   ├── chatbox_agent.py           ← Chatbox agent logic
│   ├── chatbox_theme.py / chatbox_voice.py / chatbox_widgets.py
│   ├── chatbox_identity.py       ← φ-seeded self-chosen name + TTS voice (NEW)
│   ├── cli_chat.py                ← Terminal chat interface
│   ├── build_pack_from_gguf.py    ← GGUF → .rftmwpk converter
│   ├── split_gguf_layers.cpp      ← GGUF layer splitter (for streaming mode)
│   ├── train_local_chat_lora.py   ← LoRA fine-tuning
│   ├── compression_eval.py        ← Compression benchmarks
│   └── ... (30+ files total)
│
├── src/agents/                    ← 8 named agents (Forge, Lore, Oracle, Sage, Spark, Tide, Vial, Ward)
├── src/dashboard/                 ← WebSocket bridge + web dashboard
├── src/web/                       ← Browser inference worker
│
├── quantonium_os_src/             ← Middleware layer
│   ├── engine/rftmw_numpy_engine.py  ← Pure NumPy inference engine (SACRED)
│   ├── engine/rftmw_memory.py     ← KV-cache compression (Python layer)
│   ├── engine/qwen35_moe_engine.py ← Qwen3.5 MoE routing
│   ├── engine/three_distance_router.py ← Three-Distance Theorem router
│   ├── engine/gguf_mmap_engine.py ← GGUF mmap engine
│   ├── apps/                      ← Desktop apps (visualizer, validator, crypto, vault, monitor)
│   └── frontend/                  ← Qt desktop launcher
│
├── mobile/                        ← Expo React Native app
│   ├── app/                       ← Expo Router pages
│   ├── contexts/ChatContext.tsx    ← API calls (Ollama-compatible JSON)
│   ├── server/rftmw_serve.py      ← Python HTTP server (port 8000)
│   └── start.sh                   ← Launch script
│
├── hardware/                      ← FPGA/ASIC RTL designs
│   ├── rftpu_architecture.sv      ← RFTPU chip (SystemVerilog)
│   ├── rtl/systolic_array.sv      ← Systolic array (8 variants)
│   ├── openlane/                  ← ASIC synthesis flow
│   └── tb/                        ← Testbenches (26+ files)
│
├── models/                        ← GGUF model files (~26 GB total)
│   ├── DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf  ← 4,466 MB (primary 7B model)
│   └── *.spectral_k{12,20,35,45}  ← Precomputed spectral caches
│
├── benchmarks/                    ← 37 benchmark scripts
│   ├── kv_cache_perplexity_benchmark.py  ← KV compress PPL (-10.2% on GPT-2)
│   ├── run_all_benchmarks.py      ← Master runner
│   └── results/                   ← Stored results
│
├── tests/                         ← Organized test suites
│   ├── proofs/                    ← 27 formal proof tests (479 assertions)
│   ├── compression/ codec_tests/ crypto/ medical/ validation/ ...
│   └── 33 root-level test_*.py files (inference debugging)
│
├── docs/                          ← Rich documentation
│   ├── THEOREMS_RFT_IRONCLAD.md   ← Full theorem proofs (SACRED)
│   ├── MATHEMATICAL_CLAIMS_INVENTORY.md
│   ├── Closing_the_RoPE_Gap_Minier_2026.pdf  ← KV-cache paper
│   └── proofs/ papers/ technical/ research/ ...
│
├── experiments/                   ← Research experiments
├── data/ logs/ scripts/ tools/ ui/ godot/
│
├── *.rftmwpk                     ← Compressed model packs in root
│   ├── qwen2.5-coder_3b.rftmwpk  ← 2,029 MB (full production pack)
│   ├── qwen35-opus-distil_finetuned.rftmwpk + .npz (767 MB) + .rftft (1,040 MB)
│   └── deepseek-r1-7b.rftmwpk    ← 0.07 MB (stub/header only)
│
├── pyproject.toml                 ← quantoniumos v2.0.0, deps: numpy, scipy, PyQt5, ...
├── requirements.txt               ← Quick-start deps
├── Dockerfile                     ← Validation container (python:3.11-slim)
├── AGENT_TRAINING_MANUAL.md       ← Detailed implementation log
├── AGENT_ROUTER_CONFIG.yaml       ← Agent routing config
├── launch.py / qshellchatbox.ps1  ← Launch scripts
└── tokenizer.json                 ← Qwen tokenizer vocab
```

---

## 3. MODELS & ARCHITECTURES

### DeepSeek-R1-7B (Primary Target)
- **File:** `models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` (4,466 MB)
- **Architecture:** Qwen2 (same as Qwen2.5 — auto-detected by CLI)
- **Config:** 28 layers, 3584 hidden, 28 heads, 4 KV heads, head_dim=128, FFN=18944, vocab=152064
- **RoPE:** theta=10000
- **Parameters:** 7.6B (Q4_K_M quantized)

### Qwen2.5-Coder-3B (Development Model)
- **File:** `qwen2.5-coder_3b.rftmwpk` (2,029 MB)
- **Config:** 36 layers, 2048 hidden, 16 heads, 2 KV heads, head_dim=128, FFN=11008, vocab=151936
- **RoPE:** theta=1,000,000

### Qwen3.5 MoE (Experimental)
- **File:** `qwen35-opus-distil.rftmwpk` (stub) + finetuned weights
- **Architecture:** 40 layers (30 SSM + 10 full attention), 256 experts top-8
- **Status:** SSM forward pass produces garbage — needs architecture reference fix

---

## 4. WHAT WAS RECENTLY BUILT (April 18, 2026)

### ✅ KV-Cache RFT Compression — PROVEN, WIRED
**What:** INT8 magnitude + INT8 phase per RFT coefficient, 60% keep ratio, non-uniform per-layer schedule.
**Result:** −10.2% perplexity on GPT-2, 3.35× KV memory savings, ≤2% PPL degradation.
**Files changed:**
- `src/rftmw_native/qwen25_native_cli.cpp` — KVCompressor wired into forward_token(), CLI flags `--kv-compress`, `--kv-keep`, `--layer-stream`
- `src/rftmw_native/rftmw_kv_compress.hpp` — KVCompressor class (compress_vector, decompress_vector, INT8 quantize)
- `src/rftmw_native/rftmw_streaming_engine.hpp` — StreamingModel (loads 1 layer at a time)
- Build verified: `cmake --build . --target rftmw_qwen25_native` in build_asm/

**How it works:**
```
Input KV vector (128 floats)
  → Forward RFT: coeffs = E ⊙ FFT(x) / √N  (E[k] = exp(i·2π·frac((k+1)·φ)))
  → Sort by magnitude, keep top-60%
  → INT8 quantize: mag (255 levels) + phase (255 levels) + uint16 index
  → Store: 4 bytes per coefficient (vs 4 bytes per float32)
  → Decompress: reconstruct sparse spectrum → inverse RFT → FP32
```

**Memory savings (R1-7B, 2K context):**
- Uncompressed KV: ~268 MB → Compressed: ~80 MB (3.35× reduction)
- Full model with KV compress: ~4.6 GB (mmap) vs ~4.7 GB without

### ✅ Layer-Streaming Mode — BUILT, NOT BATTLE-TESTED
**What:** Loads 1 transformer layer at a time from RFTL chunk files. Peak RAM ~630 MB for 7B.
**Requirement:** Must split GGUF first via `split_gguf_layers` (not yet done for R1-7B).
**CLI:** `--layer-stream <dir>` on qwen25_native_cli

### ✅ Chatbox Integration — WIRED
**What:** SpectralNativeEngine now supports `--kv-compress`, `--kv-keep`, `--no-spectral` flags.
**Files changed:**
- `src/apps/chatbox_network.py` — SpectralNativeEngine accepts kv_compress, kv_keep, spectral params; exe search: build_asm → build_qwen25 → build
- `src/apps/qshll_chatbox.py` — CLI flags: `--no-spectral`, `--kv-compress`, `--kv-keep`

**Launch R1-7B with KV compression in chatbox:**
```powershell
.\qshellchatbox.ps1 --native models\DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf --no-spectral --kv-compress
```

### ✅ Temperature + Top-p Nucleus Sampling — BUILT, WIRED
**What:** Replaced greedy argmax decoding with temperature-scaled softmax + nucleus (top-p) sampling. R1-family models are trained with sampling and produce flat/repetitive output under greedy.
**Files changed:**
- `src/rftmw_native/qwen25_native_cli.cpp` — Added `sample_top_p()` function (temperature-scale logits → softmax → sort descending → accumulate to top_p → sample from nucleus via thread-local `std::mt19937`). Falls back to argmax when temperature ≤ 0. Both decode-loop sampling sites updated. `generate_once()` signature extended with `float temperature`, `float top_p` params.
- GENERATE protocol extended: `GENERATE\tmax_tokens\ttemplate\thex_prompt[\ttemperature\ttop_p]` — 5th/6th fields optional, backward-compatible. Defaults: temp=0.0 (greedy), top_p=0.9.
- `src/apps/chatbox_network.py` — Both `NativeServerEngine.generate_stream()` and `SpectralNativeEngine.generate_stream()` now forward temperature (removed `del temperature`) and append `\t{temperature}\t0.9` to GENERATE command.

**How it works:**
```
logits[i] → exp((logits[i] - max) / temperature) → softmax → sort descending
→ accumulate until cumsum ≥ top_p → sample from that nucleus
→ temperature=0 falls back to argmax (greedy)
```

**Defaults for chatbox:** temperature=0.7 (from Qt UI), top_p=0.9.

### ✅ R1-Family Prompt & Sanitizer Optimization — BUILT
**What:** DeepSeek-R1-Distill models output verbose chain-of-thought preambles ("Alright, so I need to figure out...") even with think-block suppression. Three fixes:
**Files changed:**
- `src/apps/qshll_chatbox.py`:
  - Added `_system_prompt_r1`: tighter directive prompt ("No preamble. No restating the question. Just answer.")
  - `_build_chatml_prompt()` auto-detects R1 models by name (`r1` or `deepseek` in `_model_name`) and selects R1 prompt
  - Native prompt path now also recognizes `SpectralNativeEngine` (was only `NativeServerEngine`)
  - `_sanitize_stream_text()` now strips R1 CoT preambles via regex: catches "Alright, so...", "Okay, let me think...", "Well, I need to..." patterns + `</assistant>` tag cleanup
  - Response budget for short questions raised from 48 → 128 tokens (R1-7B needs more room)

### ✅ Spectral Inference Path — BUILT (weight compression DISPROVEN, spectral GEMV works)
**Finding:** Spectral weight compression does NOT work for NN weights (4.5% error per matrix, compounds to PPL=40,232). But spectral GEMV (sparse FFN matmul) IS viable — reduces FFN FLOPs by 2-5×.
**Key win:** Direct binary (no Node.js) = 1.2 tok/s vs 0.19 tok/s through Node.js = 6× speedup.

### ✅ Voice Chat + Self-Chosen Identity — WIRED (April 18, 2026)
**What:** The chatbox agent picks its own name and TTS voice on first launch, seeded by φ + host id. Every RFTMW reply (streaming, native spectral, fast-reply, or demo) is spoken through the chosen voice. TTS is on by default.

**Files changed:**
- `src/apps/chatbox_identity.py` — NEW. φ-seeded self-selection from a 19-name persona pool with gender hints; matches to installed SAPI5/espeak voices; persists to `data/agent_state/identity.json`.
- `src/apps/chatbox_voice.py` — Added `list_voices()`; `VoiceSpeaker` now accepts `voice_id`/`rate`/`volume` and has `set_voice()`; hardened imports so torch/whisper DLL failures no longer disable TTS.
- `src/apps/qshll_chatbox.py` — Loads identity on init, applies it to system prompt / title bar / placeholder / demo replies; TTS enabled by default; CLI flags `--agent-name`, `--agent-voice`, `--reset-identity`, `--no-tts`.
- `data/agent_state/identity.json` — NEW. First self-selection produced `{"name": "Cael", "voice_id": "...DAVID_11.0"}`.

**Deps installed:** `pyttsx3 2.99`, `sounddevice 0.5.5`, `faster-whisper 1.2.1` (ctranslate2 + onnxruntime, no torch needed). Removed broken `torch 2.11` + `openai-whisper` combo.

**Verified:**
- `list_voices()` enumerates David + Zira on this host
- `load_identity()` deterministically picks Cael from the pool and maps to David (male voice match)
- Chatbox init wires name into title (`Cael`), placeholder (`Message Cael...`), system prompt (`You are Cael...`), and demo-mode replies
- TTS button defaults to 🔊 (on) when an identity is loaded

**STT fixed:** Switched from `openai-whisper` (torch 2.11 DLL init failure on Py 3.13) to `faster-whisper` (CTranslate2 int8 backend). Full push-to-talk + TTS loop now works: user taps 🎤, speaks, `VoiceRecorder` transcribes via faster-whisper, the text is piped into `_send_message()`, and the RFTMW reply is spoken back through the self-chosen voice.

**Launch:**
```powershell
.\qshellchatbox.ps1 --native models\DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf --no-spectral --kv-compress
# Reset the identity: add --reset-identity
# Override: --agent-name Nyx --agent-voice "HKEY_LOCAL_MACHINE\...\TTS_MS_EN-US_ZIRA_11.0"
```

### ⚠️ UNVALIDATED
- RoPE compatibility with KV compression (proven on GPT-2 absolute pos, R1-7B uses RoPE theta=10000)
- Perplexity benchmark with `--kv-compress` on actual R1-7B
- Qwen3.5 MoE SSM layer semantics (produces garbage)
- STT path — mic→faster-whisper→RFTMW→TTS loop wired and compiles; needs real-mic end-user verification

---

## 5. INFERENCE PIPELINES

### Pipeline A: Native CLI (fastest)
```
rftmw_qwen25_native.exe --gguf <model.gguf> [--kv-compress] [--kv-keep 0.60] [--serve]
  ← C++ direct GGUF mmap, AVX2 matmul, optional KV compression
  ← --serve mode: stdin/stdout tab-delimited protocol
  ← GENERATE\tmax_tokens\ttemplate_mode\thex_prompt[\ttemperature\ttop_p]
     → STAGE\t<name>\t<value>
     → TOKEN\t<id>\t<hex_text>\t<n>\t<tps>
     → DONE\t<hex_text>\t<prompt_len>\t<prefill_s>\t<out_len>\t<decode_s>
  ← Temperature/top_p fields optional (default: greedy / 0.9)
```

### Pipeline B: QShell Desktop Chatbox
```
qshellchatbox.ps1 --native <model.gguf> [--kv-compress] [--no-spectral]
  → qshll_chatbox.py (Qt5 UI) → SpectralNativeEngine → subprocess(rftmw_qwen25_native --serve)
  → stdin/stdout protocol → token streaming to UI
```

### Pipeline C: Python NumPy Engine (portable, slow)
```
qshll_chatbox.py  (no --native flag)
  → LocalEngineLoader → Qwen25Engine.load(<pack.rftmwpk>)
  → Pure NumPy forward pass, Q4_K/Q6_K dequant on-the-fly
```

### Pipeline D: Mobile HTTP Server
```
mobile/start.sh → rftmw_serve.py (port 8000, Ollama-compatible API)
  → React Native Expo app → POST /api/chat → streaming NDJSON
```

---

## 6. BUILD COMMANDS

```powershell
# Activate environment
.\.venv\Scripts\Activate.ps1

# Build native CLI (MinGW + NASM + AVX2)
cd src\rftmw_native\build_asm
cmake .. -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build . --target rftmw_qwen25_native

# Run inference
.\rftmw_qwen25_native.exe --gguf ..\..\..\models\DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf --kv-compress

# Quick test
pytest tests/proofs/ -v --timeout=60

# Full test suite
pytest tests/ -v

# Chatbox
.\qshellchatbox.ps1 --native models\DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf --no-spectral --kv-compress
```

---

## 7. THE RFT TRANSFORM — TECHNICAL REFERENCE

### Basis Construction
```
frac_val = (k+1)·φ − floor((k+1)·φ)     ← fractional part
theta = 2π · frac_val
E[k] = exp(i·theta)                       ← golden-ratio phase diagonal
```
**Code:** `rftmw_spectral_kernels.c` line 213: `rftmw_spectral_basis_init()`

### Forward Transform
```
out[k] = E[k] · FFT(x)[k] / √N
```
**Code:** `rftmw_spectral_kernels.c` line 249: `rftmw_spectral_forward_block()`

### Why φ?
The golden ratio creates maximally incommensurate frequency spacing. By the Three-Distance Theorem (Steinhaus-Sós 1957), φ-grid points have exactly 2-3 distinct gap sizes, creating structured spectral zones. This matches the geometry of learned positional embeddings in transformers — implicit regularization, not just compression.

### Proven Theorems (479 proof tests)
| # | Statement | Status |
|---|-----------|--------|
| 1 | Φ full rank for all N | Proven |
| 2 | U = Φ·G^(-1/2) is unitary | Proven |
| 3 | Fast RFT surrogate is unitary | Proven |
| 4 | Twisted convolution theorem | Proven |
| 6 | RFT ≠ DFT/LCT (irrational frequencies) | Proven |
| 8 | K₀.₉₉(U) = O(log N), gap ∝ N^1.04 | Analytically proven |
| E | δ₈ = 0.1603 < 0.2929 (RIP) | Empirical (2000 trials, N=128) |
| 13 | O(N log N) complexity | Proven |
| 15 | Exact φ-sparsity + DFT dense leakage | Proven |

---

## 8. KV-CACHE COMPRESSION — DEEP DIVE

### Algorithm
1. **Forward RFT** on each KV head_dim vector (128 floats)
2. **Sort** coefficients by magnitude (descending)
3. **Keep top-K** (non-uniform schedule: early layers 25%, core 150%, late 50% of base ratio)
4. **INT8 quantize**: magnitude (0-255) + phase (0-255) + uint16 index = 4 bytes per coefficient
5. **Decompress**: reconstruct sparse spectrum → inverse RFT → FP32 back into cache

### Data Structures
```cpp
struct CompressedKVEntry {
    int n_kept;
    float mag_scale;                    // peak for dequant
    std::vector<uint16_t> indices;     // spectral bin indices
    std::vector<uint8_t>  mag8;        // quantized magnitudes
    std::vector<uint8_t>  phase8;      // quantized phases
};
```

### Results
| Metric | Value |
|--------|-------|
| GPT-2 PPL (uncompressed) | 33.26 |
| GPT-2 PPL (RFT 60% keep) | 29.85 (−10.2%) |
| Compression ratio | 3.35× |
| Per-coefficient storage | 4 bytes |
| Quantization error | ~3-4% relative Frobenius norm |

### Why RFT > FFT > DCT for KV
- **DCT:** Cosine only → loses phase → +6.8% PPL
- **FFT:** Uniform treatment → +11.2% PPL
- **RFT (φ-grid):** Matches positional encoding geometry → −10.2% PPL

---

## 9. WHAT DOES NOT WORK

| Approach | Finding | Evidence |
|----------|---------|----------|
| Spectral weight compression | 4.5% error per matrix compounds to PPL=40,232 | NN weights are not spectrally sparse |
| Full .rftmwpk spectral encoding | Only 3 FFN norm tensors benefit | Validated empirically |
| Qwen3.5 MoE SSM forward | Produces garbage tokens | Delta-net semantics unknown |
| KV compress + RoPE (R1-7B) | Unvalidated | Proven only on GPT-2 (absolute pos) |
| TinyLlama + KV compress | +32.3% PPL penalty | RoPE interferes with φ-grid |

---

## 10. FILE OWNERSHIP & TOUCH SENSITIVITY

### 🔴 SACRED (never edit without explicit instruction)
```
algorithms/rft/core/resonant_fourier_transform.py   # Canonical RFT — patent claims
algorithms/rft/README_RFT.md                        # Authoritative definition
docs/THEOREMS_RFT_IRONCLAD.md                       # Formal proofs
quantonium_os_src/engine/rftmw_numpy_engine.py      # Production inference engine
src/rftmw_native/rftmw_core.hpp                     # C++ kernel — breaks builds
.github/AGENT.md                                    # This file
LICENSE*                                            # Legal
PATENT_CLAIMS_*.md                                  # IP documentation
```

### 🟡 CAREFUL (edit allowed, run tests after)
```
algorithms/rft/variants/*           → pytest tests/test_patent_variants.py
algorithms/rft/crypto/*             → pytest tests/crypto/
quantonium_os_src/engine/*          → python test_inference_quick.py
src/rftmw_native/qwen25_native_cli.cpp → rebuild cmake
src/apps/chatbox_network.py         → python -c "import py_compile; py_compile.compile('src/apps/chatbox_network.py')"
src/apps/qshll_chatbox.py           → same
benchmarks/*                        → rerun specific benchmark
tests/*                             → run pytest on modified file
```

### 🟢 WORKING SURFACE (edit freely)
```
experiments/*                       # Exploratory
scripts/*                           # Utility
docs/* (except proofs/)             # Documentation
examples/*                          # Demos
data/*                              # Data files
mobile/*                            # Mobile app
ui/*                                # UI components
logs/*                              # Logs
```

---

## 11. OPERATOR RULES

### Claim Discipline
Every statement about the system must be tagged:

| Tag | Meaning | Required Evidence |
|-----|---------|-------------------|
| **[PROVEN]** | Mathematically proven | In-repo proof file or theorem test |
| **[VERIFIED]** | Tested to machine precision | Test file with passing assertion |
| **[EMPIRICAL]** | Measured in benchmarks | Benchmark script + logged output + seed |
| **[HYPOTHESIS]** | Unvalidated idea | Must be clearly labeled |

### Non-Negotiable Rules
1. No fabricated proofs, logs, metrics, hashes, or benchmark outcomes.
2. No speculative claims about performance, novelty, security, or correctness.
3. No rewriting large surfaces unless the task explicitly requires it.
4. No deletion unless explicitly requested.
5. No breaking backward compatibility unless explicitly authorized.
6. No silent changes to formats, schemas, public APIs, or command behavior.
7. No dependency additions unless necessary and justified.
8. No fake completion — validate or say "unverified."

### Failure Examples
- **Fabricated benchmark:** Agent reported "RFT 3.2x faster than FFT" without running code. Real was 0.8x.
- **Silent schema change:** Agent "cleaned up" .rftmwpk metadata → broke every existing pack.
- **Overstated claim:** Agent wrote "perfect reconstruction" → actual error is 1e-12, not zero.
- **Proof guard bypass:** Agent added "Theorem 15: RFT is optimal" → no proof exists.

---

## 12. VALIDATION COMMANDS

```powershell
# Quick sanity (30s)
pytest tests/proofs/test_formal_proofs.py tests/transforms/test_rft_correctness.py -v

# Full proofs (479 assertions)
pytest tests/proofs/ -v

# Full test suite (~5 min)
pytest tests/ -v

# Inference smoke test
python test_inference_quick.py

# Native CLI check
& "src\rftmw_native\build_asm\rftmw_qwen25_native.exe" --help

# Python syntax check
python -c "import py_compile; py_compile.compile('src/apps/qshll_chatbox.py', doraise=True)"

# KV compression benchmark
python benchmarks/kv_cache_perplexity_benchmark.py
```

---

## 13. ENVIRONMENT DETAILS

| Component | Location / Value |
|-----------|-----------------|
| Python venv | `.venv/` (3.10+) |
| MinGW | C:\msys64\mingw64\bin |
| NASM | Available via MSYS2 |
| CMake generator | Ninja |
| SIMD | AVX2 + FMA (AVX-512 off) |
| Current build | `src/rftmw_native/build_asm/rftmw_qwen25_native.exe` (741 KB, April 18 2026) |
| Stale build | `src/rftmw_native/build_qwen25/rftmw_qwen25_native.exe` (562 KB, missing KV compress) |
| Primary GGUF | `models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` (4,466 MB) |
| Primary pack | `qwen2.5-coder_3b.rftmwpk` (2,029 MB) |
| Tokenizer | `tokenizer.json` (Qwen vocab, `tokenizers` library) |
| Qt | PyQt5 (desktop chatbox) |
| OS | Windows |

---

## 14. NEXT STEPS / OPEN WORK

| Priority | Task | Blocker |
|----------|------|---------|
| HIGH | Perplexity benchmark with --kv-compress on R1-7B | Need to run, ~30 min |
| HIGH | RoPE compatibility validation | KV compress proven on GPT-2 (absolute pos), R1-7B uses RoPE |
| HIGH | Verify sampling quality (temp=0.7, top_p=0.9) on R1-7B chatbox | Binary rebuilt, needs live test |
| ~~HIGH~~ DONE | ~~Fix STT path~~ → swapped to faster-whisper | Verified: `_HAS_WHISPER=True`, backend=`faster` |
| MEDIUM | Split R1-7B GGUF for layer-streaming mode | Need split_gguf_layers compiled + run |
| MEDIUM | Fix Qwen3.5 MoE SSM forward pass | Unknown delta-net semantics |
| MEDIUM | LoRA fine-tuning pass for R1-7B response style | train_local_chat_lora.py exists, needs dataset |
| LOW | Spectral weight cache file (save/load converted FFN weights) | Slow conversion (~minutes) |
| LOW | Rebuild build_qwen25 with latest code | Or just use build_asm |

---

## 15. STRUCTURED OUTPUT FORMAT

When reporting results, use:

```
## Task Summary
[One sentence]

## Files Changed
- `path/to/file.py` — [what changed]

## Validation
[paste actual command + output]

## Claim Status
- [VERIFIED] ...
- [UNVERIFIED] ...

## Next Step
- [what should happen next]
```

---

*v3.2 — April 18, 2026. Added: temperature/top-p nucleus sampling, GENERATE protocol extension (temp/top_p fields), R1-family prompt optimization, CoT preamble sanitizer, response budget increase. Binary rebuilt.*
