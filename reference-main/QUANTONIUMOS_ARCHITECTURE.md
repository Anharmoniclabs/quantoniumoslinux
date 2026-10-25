# QuantoniumOS — Full System Architecture

> **Version:** 0.1.0-alpha  
> **Runtime:** Tauri v2 (Rust) + React 18 + Ollama  
> **Generated:** April 19, 2026

---

## System Overview

QuantoniumOS is a **local-first AI operating system** that runs as a desktop application on top of the host OS (Windows/macOS/Linux). All AI inference, cryptography, and signal processing runs **on-device** — zero cloud, zero telemetry. The Rust middleware layer acts as the sole gateway between the user interface and all backend services.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HOST OS (Windows)                            │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    QuantoniumOS Process                        │  │
│  │                                                               │  │
│  │  ┌─────────────────────┐    ┌──────────────────────────────┐  │  │
│  │  │   React Frontend    │    │     Rust Middleware (12 MB)   │  │  │
│  │  │   (Tauri WebView)   │◄──►│                              │  │  │
│  │  │                     │IPC │  ┌────────────────────────┐  │  │  │
│  │  │  9 App Views        │    │  │   RFT Engine Crate     │  │  │  │
│  │  │  8 Canvas Renders   │    │  │   14 Rust Modules      │  │  │  │
│  │  │  Launchpad Shell    │    │  │   20 Unit Tests Pass   │  │  │  │
│  │  │                     │    │  └────────────────────────┘  │  │  │
│  │  │  CSP: default-src   │    │                              │  │  │
│  │  │       'self' ONLY   │    │  ┌────────────────────────┐  │  │  │
│  │  │  (no external HTTP) │    │  │   Ollama Manager       │  │  │  │
│  │  │                     │    │  │   ensure / status /     │  │  │  │
│  │  └─────────────────────┘    │  │   stream / lifecycle    │  │  │  │
│  │                             │  └───────────┬────────────┘  │  │  │
│  │                             │              │ HTTP :11434   │  │  │
│  │                             └──────────────┼──────────────┘  │  │
│  │                                            │                  │  │
│  │                             ┌──────────────▼──────────────┐  │  │
│  │                             │   Ollama (Managed Child)    │  │  │
│  │                             │   deepseek-r1:7b (4.7 GB)   │  │  │
│  │                             │   7.6B params · Q4_K_M      │  │  │
│  │                             │   100% CPU · 131K context   │  │  │
│  │                             │   Auto-unloads after 5 min  │  │  │
│  │                             └─────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  RAM: 64 GB total · ~5 GB used by OS layer when model is hot        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Boot Sequence

```
┌──────────┐     ┌──────────────┐     ┌───────────────┐     ┌──────────┐
│  main()  │────►│ensure_ollama │────►│  BootScreen    │────►│ Desktop  │
│  Rust    │     │ TCP :11434   │     │  polls         │     │ Launchpad│
│  starts  │     │ check/spawn  │     │  ollama_status │     │ ready    │
└──────────┘     └──────────────┘     │  every 800ms   │     └──────────┘
                                      │  until model   │
                                      │  responds      │
                                      └───────────────┘
```

1. **Rust `main()`** initializes sysinfo, calls `ensure_ollama()`
2. **`ensure_ollama()`** checks TCP :11434 — if Ollama is already running, does nothing; otherwise spawns `ollama serve` as a child process
3. **Tauri WebView** loads React, renders `BootScreen`
4. **BootScreen** polls `invoke("ollama_status")` every 800ms until `running: true` and a model name is returned
5. **Transition** — fade-out animation → desktop with Launchpad

---

## Chat / AI Inference Pipeline

```
┌──────────┐   Tauri IPC    ┌──────────────┐   HTTP POST   ┌──────────┐
│  React   │───────────────►│  Rust        │──────────────►│  Ollama  │
│  useChat │  invoke()      │  chat_stream │  /api/chat    │  Runner  │
│          │  + Channel     │  (async +    │  stream:true  │          │
│          │                │   spawn_     │               │  7.6B    │
│          │◄───────────────│   blocking)  │◄──────────────│  params  │
│  render  │  ChatToken     │              │  NDJSON       │  Q4_K_M  │
│  tokens  │  {token, done} │  BufReader   │  lines        │          │
└──────────┘                └──────────────┘               └──────────┘

Security: CSP blocks all direct HTTP from frontend.
The ONLY path to Ollama is through Rust IPC.
```

**Data flow for one message ("hi cael"):**

| Step | Component | Action | Overhead |
|------|-----------|--------|----------|
| 1 | React `useChat` | Builds message array (system prompt + last 20 messages), creates `Channel<ChatToken>` | ~0ms |
| 2 | Tauri IPC | Serializes request, dispatches to Rust via `invoke("chat_stream")` | ~1ms |
| 3 | Rust `chat_stream` | Spawns blocking thread, builds JSON payload, POSTs to Ollama :11434 | ~2ms |
| 4 | Ollama | Loads model if cold (4.6 GB into RAM), runs transformer forward passes, samples tokens | **50+ CPU-seconds** |
| 5 | Rust `BufReader` | Reads each NDJSON line, parses token + done flag | ~0.1ms/token |
| 6 | Tauri Channel | Sends `ChatToken { token, done }` to frontend | ~0.1ms/token |
| 7 | React | Appends token to assistant message, re-renders | ~1ms/token |

**System prompt (Cael personality):**
```
You are Cael, the local AI assistant inside QuantoniumOS.
You are helpful, concise, and friendly. You run fully on-device
via Ollama — no cloud, no telemetry. Answer directly and keep
responses focused.
```

---

## Desktop Shell

### Window Manager

The desktop uses a **floating window manager** — each app opens in a draggable, resizable window with minimize/maximize/close controls. Windows have z-index ordering (click-to-focus). Minimized apps appear in the bottom taskbar.

- **Fullscreen by default**, no native decorations
- **ESC key** toggles fullscreen + decorations
- Window positions cascade (30px offset per window)

### Launchpad

Central **Q** logo button expands 8 app icons in a **270° arch** (starting upper-left, sweeping clockwise). Staggered animation (40ms per icon). Click any icon to open its app window.

```
                    ┌─────────┐
              ┌─────│Validator│─────┐
              │     └─────────┘     │
        ┌─────────┐           ┌──────────┐
        │Visualizer│          │  AI Chat  │
        └─────────┘           └──────────┘
              │                     │
        ┌─────────┐     Q     ┌──────────┐
        │ Monitor │    ◉──►   │Simulator │
        └─────────┘           └──────────┘
              │                     │
        ┌─────────┐           ┌──────────┐
        │ Q-Vault │           │  Crypto  │
        └─────────┘           └──────────┘
              │     ┌─────────┐     │
              └─────│ Q-Notes │─────┘
                    └─────────┘
```

---

## Applications (9 Total)

### 1. Cael — AI Chat

| | |
|---|---|
| **File** | `views/ChatView.tsx` + `hooks/useChat.ts` |
| **Category** | AI |
| **Icon** | MessageCircle |
| **Purpose** | On-device AI assistant powered by deepseek-r1:7b |
| **Capabilities** | Streaming token display, conversation history (last 20 messages as context), stop generation, clear history, connection status indicator, model badge |
| **Tauri Commands** | `chat_stream` (streaming via Channel), `ollama_status` (health poll) |
| **Backend** | Rust → Ollama HTTP → deepseek-r1:7b (7.6B params, Q4_K_M) |
| **Canvas** | None |

### 2. System Monitor

| | |
|---|---|
| **File** | `views/MonitorView.tsx` |
| **Category** | SYSTEM |
| **Icon** | Activity |
| **Purpose** | Real-time system resource monitoring + RFT engine status |
| **Capabilities** | CPU/Memory/Disk usage with progress rings, sparkline history charts, animated Vertex Resonance Graph (φ-modulated sinusoid), RFT engine panel (11 engines, 16 variants), process table (top 12 by memory) |
| **Tauri Commands** | `get_system_stats` (polled every 2s), `rft_engine_status` (once on mount) |
| **Canvas** | 3 — sparkline charts (×2), vertex resonance graph (animated) |

### 3. RFT Validator

| | |
|---|---|
| **File** | `views/ValidatorView.tsx` |
| **Category** | ANALYSIS |
| **Icon** | ShieldCheck |
| **Purpose** | Run and verify all RFT engine self-tests |
| **Capabilities** | Executes 6 real engine validation tests, shows pass/warn/fail status with details, progressive reveal animation |
| **Tauri Commands** | `rft_validate` |
| **Canvas** | None |

**Tests executed:**

| # | Test | What It Validates |
|---|------|-------------------|
| 1 | Bijection Integrity | RFT forward → inverse roundtrip error < 1e-6 |
| 2 | Entropy Conservation | Compression → decompression fidelity |
| 3 | Spectral Residual | FFT forward → inverse roundtrip residual |
| 4 | Token Fidelity | RMS normalization factor ≈ 1.0, softmax sum ≈ 1.0 |
| 5 | Compression Ratio | Spectral compression achieves > 2x ratio |
| 6 | Header Checksum | Feistel-48 encrypt → decrypt roundtrip matches |

### 4. Q-Notes

| | |
|---|---|
| **File** | `views/NotesView.tsx` |
| **Category** | RESEARCH |
| **Icon** | StickyNote |
| **Purpose** | In-app note-taking for research and documentation |
| **Capabilities** | Two-panel layout (sidebar list + editor), create/delete notes, search/filter, title + body editing |
| **Tauri Commands** | None (in-memory state only) |
| **Canvas** | None |
| **Note** | No persistence — notes are lost on app restart |

### 5. Settings

| | |
|---|---|
| **File** | `views/SettingsView.tsx` |
| **Category** | SYSTEM |
| **Icon** | Settings |
| **Purpose** | System configuration and about info |
| **Capabilities** | Displays backend URL, version info (v0.1.0-alpha), built-with info |
| **Tauri Commands** | None |
| **Canvas** | None |
| **Note** | Currently read-only — no editable settings |

### 6. Q-Vault

| | |
|---|---|
| **File** | `views/VaultView.tsx` |
| **Category** | DATA |
| **Icon** | Lock |
| **Purpose** | Encrypted secret storage (passwords, keys, sensitive notes) |
| **Capabilities** | Password-locked vault, two-panel layout (item list + editor), create/delete items, search, reveal/copy secret values, "RFT Keystream Mixer" toggle |
| **Tauri Commands** | None (client-side password check, in-memory storage) |
| **Canvas** | None |
| **Note** | Password validation is client-side only (length ≥ 4). No disk persistence yet. |

### 7. Quantum Simulator

| | |
|---|---|
| **File** | `views/SimulatorView.tsx` |
| **Category** | SIMULATION |
| **Icon** | FlaskConical |
| **Purpose** | Interactive quantum circuit simulator with RFT analysis |
| **Capabilities** | |

| Tab | Features |
|-----|----------|
| **State** | Qubit count selector (1–8), gate palette (H/X/Y/Z/CNOT/T/S/Rφ), animated 3D Bloch sphere with state vector arrow, circuit diagram renderer, state vector heatmap, RFT concentration overlay |
| **Algorithms** | Grover's Search, QFT, Shor's Factorization, Quantum Walk, VQE, QAOA, QML Kernel — runs with fidelity results |
| **Scaling** | Log-scale chart: qubits vs memory/time requirements |
| **Measurements** | Configurable shot count, measurement histogram with gradient bars |

| | |
|---|---|
| **Tauri Commands** | `rft_transform` (feeds state vector into real RFT φ-basis engine) |
| **Canvas** | 4 — Bloch sphere (animated), circuit diagram, scaling chart, measurement histogram |

### 8. RFT Visualizer

| | |
|---|---|
| **File** | `views/VisualizerView.tsx` |
| **Category** | VISUAL |
| **Icon** | Waves |
| **Purpose** | Real-time visualization of RFT spectral transforms |
| **Capabilities** | Depth slider (1–10 layers), Frequency slider (0.1–5.0 Hz), play/pause animation, layered φ-RFT wave display with golden-ratio coupling, live RFT magnitude spectrum from Rust engine, real-time metrics overlay (concentration %, sparsity %, φ-coupling) |
| **Tauri Commands** | `rft_transform` (128-point spectrum computed in Rust) |
| **Canvas** | 1 full-size — dual visualization (wave field + frequency spectrum) |

### 9. Quantum Crypto

| | |
|---|---|
| **File** | `views/CryptoView.tsx` |
| **Category** | SECURITY |
| **Icon** | Cpu |
| **Purpose** | Quantum key distribution simulation + real RFT cryptography |
| **Capabilities** | |

| Tab | Features |
|-----|----------|
| **QKD** | Protocol selector (BB84/B92/SARG04), key length config, noise slider, eavesdropper toggle, animated photon channel (Alice→Bob with optional Eve interception), basis comparison grid, sifted key display, QBER metric |
| **Encryption** | Plaintext/ciphertext editing, Encrypt/Decrypt buttons using real Feistel-48 cipher (φ-RFT permutation, AES S-box, MixColumns, 48 rounds) |
| **Analysis** | Protocol analysis log with real RFT-SIS lattice hash computation |

| | |
|---|---|
| **Tauri Commands** | `rft_encrypt`, `rft_decrypt`, `rft_hash` |
| **Canvas** | 1 — quantum photon channel animation with polarization particles |

---

## Rust Engine Architecture

```
desktop-shell/src-tauri/src/
├── main.rs                    ← Tauri app + 12 commands + Ollama manager
└── engines/
    ├── mod.rs                 ← Module re-exports
    ├── types.rs               ← Shared types (Complex, RftVariant, RftError)
    ├── rft_kernel.rs          ← 16-variant RFT engine (φ-basis, Gram-Schmidt)
    ├── orchestrator.rs        ← Multi-threaded task scheduler (3 assembly types)
    ├── split_gguf.rs          ← GGUF model file splitter
    ├── crypto/
    │   ├── mod.rs
    │   ├── feistel.rs         ← 48-round Feistel cipher (AES S-box + φ-RFT)
    │   └── rft_sis.rs         ← Lattice hash (SIS, n=512, q=3329, φ-RFT diffusion)
    └── rftmw/
        ├── mod.rs
        ├── core.rs            ← φ-basis construction, forward/inverse transform
        ├── fft.rs             ← Radix-2 DIT FFT (f32)
        ├── compression.rs     ← Spectral quantize → dequantize pipeline
        ├── inference.rs       ← Q4_K/Q6_K dequant, GEMV, RoPE, GQA attention, SSM
        ├── spectral.rs        ← Spectral weight compression (RFT → top-k → 4-bit)
        ├── streaming.rs       ← Layer-streaming engine (RFTL chunks, KV cache)
        ├── kv_compress.rs     ← KV-cache compression (FFT → φ-diagonal → INT8)
        └── mmap.rs            ← Memory-mapped .rftmwpk model loader
```

### Engine Status

| Module | Used By | Status |
|--------|---------|--------|
| `rftmw/core.rs` | `rft_transform`, `rft_compress`, `rft_validate` | **ACTIVE** — real φ-basis transforms |
| `rftmw/fft.rs` | `rft_validate`, spectral.rs, kv_compress.rs | **ACTIVE** — radix-2 FFT |
| `rftmw/compression.rs` | `rft_compress`, `rft_validate` | **ACTIVE** — spectral compression |
| `rftmw/inference.rs` | `rft_validate` (rms_norm, softmax only) | **PARTIAL** — kernels correct but mostly unused |
| `crypto/feistel.rs` | `rft_encrypt`, `rft_decrypt`, `rft_validate` | **ACTIVE** — real 48-round cipher |
| `crypto/rft_sis.rs` | `rft_hash`, feistel.rs KDF | **ACTIVE** — real lattice hash |
| `rft_kernel.rs` | orchestrator.rs | **DORMANT** — 16 variants built, not exposed |
| `orchestrator.rs` | — | **DORMANT** — task scheduler not instantiated |
| `rftmw/spectral.rs` | — | **DORMANT** — spectral weight compression |
| `rftmw/streaming.rs` | — | **DORMANT** — layer-streaming inference |
| `rftmw/kv_compress.rs` | — | **DORMANT** — KV-cache compression |
| `rftmw/mmap.rs` | — | **DORMANT** — .rftmwpk model loader |
| `split_gguf.rs` | — | **DORMANT** — GGUF splitter |

---

## Tauri Command Reference

### AI / Inference

| Command | Type | Description |
|---------|------|-------------|
| `chat_stream` | `async`, streaming | Sends messages to Ollama, streams tokens back via `Channel<ChatToken>` |
| `ollama_status` | sync | Returns `{ running, model, models[], backend }` |
| `get_pack_info` | sync | Legacy compat — returns pack name + Ollama status |

### RFT Signal Processing

| Command | Type | Description |
|---------|------|-------------|
| `rft_transform` | sync | φ-basis forward transform (1–256 points) → real/imag/magnitudes/phases/concentration/sparsity |
| `rft_compress` | sync | Transform → quantize → dequantize → roundtrip error measurement |
| `rft_validate` | sync | Runs 6 self-tests, returns pass/warn/fail results |
| `rft_engine_status` | sync | Returns list of 11 engines + capability flags |

### Cryptography

| Command | Type | Description |
|---------|------|-------------|
| `rft_encrypt` | sync | Feistel-48 (φ-RFT permutation, AES S-box, MixColumns) → hex output |
| `rft_decrypt` | sync | Reverse Feistel-48 from hex → plaintext |
| `rft_hash` | sync | RFT-SIS lattice hash (n=512, q=3329, φ-diffusion) → 64-byte hex |

### System

| Command | Type | Description |
|---------|------|-------------|
| `get_system_stats` | sync | CPU%, RAM, disk, top 25 processes |
| `get_project_root` | sync | Walks up from exe to find workspace root |

---

## Security Model

```
┌──────────────────────────────────────────────────────┐
│                  TRUST BOUNDARY                       │
│                                                      │
│  ┌────────────────┐      ┌─────────────────────────┐ │
│  │   WebView       │      │   Rust Process          │ │
│  │                 │      │                         │ │
│  │  CSP enforced:  │      │  • Ollama gateway       │ │
│  │  default-src    │ IPC  │  • Crypto engine        │ │
│  │    'self'       │◄────►│  • RFT engine           │ │
│  │                 │      │  • System stats          │ │
│  │  NO fetch()     │      │  • Process lifecycle     │ │
│  │  NO XHR         │      │                         │ │
│  │  NO WebSocket   │      │  Only component that    │ │
│  │  to external    │      │  can reach network      │ │
│  │                 │      │                         │ │
│  └────────────────┘      └─────────────────────────┘ │
│                                                      │
└──────────────────────────────────────────────────────┘
```

| Layer | Enforcement |
|-------|-------------|
| **CSP** | `default-src 'self'` — frontend cannot make any HTTP/WS requests to localhost or external |
| **IPC Only** | All frontend→backend communication uses Tauri `invoke()` + `Channel` — type-checked, serialized |
| **Capability Permissions** | `default.json` explicitly whitelists: window control, shell spawn/kill, IPC default |
| **Process Isolation** | Ollama runs as a managed child process — only Rust can send it requests |
| **Lifecycle** | If QuantoniumOS started Ollama, it kills it on window destroy |

---

## Resource Footprint

| Component | RAM | CPU (idle) | CPU (inference) |
|-----------|-----|-----------|-----------------|
| Rust middleware | 12 MB | ~0% | ~0.1% (token relay) |
| React WebView | ~50 MB | ~1% | ~1% (render tokens) |
| Ollama server | ~30 MB | ~0% | ~0% |
| Ollama runner (model hot) | ~4,600 MB | ~0% | 16%+ (multi-core matmul) |
| **Total (model hot)** | **~4,700 MB** | **~1%** | **~17%** |
| **Total (model cold)** | **~100 MB** | **~1%** | **0%** |

Model auto-unloads after **5 minutes** of inactivity. On a 64 GB system, the hot model uses ~7% of total RAM.

---

## Frontend → Backend Call Map

```
  BootScreen ──────── ollama_status (poll 800ms)
       │
       ▼
  ┌─ Launchpad ────────────────────────────────────────────┐
  │                                                        │
  │  ChatView ─────── chat_stream (streaming Channel)      │
  │      │             ollama_status (poll 10s)             │
  │      │                                                  │
  │  MonitorView ──── get_system_stats (poll 2s)           │
  │      │             rft_engine_status (once)              │
  │      │                                                  │
  │  ValidatorView ── rft_validate (on click)              │
  │      │                                                  │
  │  SimulatorView ── rft_transform (on state change)      │
  │      │                                                  │
  │  VisualizerView ─ rft_transform (on param change)      │
  │      │                                                  │
  │  CryptoView ───── rft_encrypt (on click)               │
  │      │             rft_decrypt (on click)               │
  │      │             rft_hash (during analysis)           │
  │      │                                                  │
  │  NotesView ────── (no backend calls)                   │
  │  SettingsView ─── (no backend calls)                   │
  │  VaultView ────── (no backend calls)                   │
  └────────────────────────────────────────────────────────┘
```

---

## Canvas / Visualization Inventory

| View | Canvas | Type | Animation |
|------|--------|------|-----------|
| MonitorView | Sparkline (×2) | CPU + Memory history line charts | Static redraw on data update |
| MonitorView | Vertex Resonance Graph | φ-modulated sinusoid, 32 vertices | `requestAnimationFrame` continuous |
| SimulatorView | Bloch Sphere | 3D projected sphere + state vector arrow | `requestAnimationFrame` continuous |
| SimulatorView | Circuit Diagram | Quantum gate circuit with wire lines | Static redraw |
| SimulatorView | Scaling Chart | Log-scale memory/time vs qubits | Static redraw |
| SimulatorView | Measurement Histogram | Gradient bars with count labels | Static redraw |
| VisualizerView | RFT Wave + Spectrum | Dual: φ-wave field + magnitude bars | `requestAnimationFrame` (play/pause) |
| CryptoView | Photon Channel | Alice→Bob polarized particles | `requestAnimationFrame` during QKD |

**Total: 8 canvas elements, 3 continuous animation loops**

---

## Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Desktop Runtime | Tauri | 2.x |
| Backend Language | Rust | 1.95.0 |
| Frontend Framework | React | 18.3 |
| Build Tool | Vite | 5.4 |
| Type System | TypeScript | 5.5 |
| AI Backend | Ollama | latest |
| AI Model | deepseek-r1:7b | Q4_K_M, 7.6B params |
| HTTP Client (Rust) | ureq | 2.10 |
| System Info | sysinfo | 0.32 |
| Crypto | sha2, custom Feistel/SIS | 0.10 |
| Memory Mapping | memmap2 | 0.9 |
| Parallelism | rayon | 1.10 |

---

## Design Language

| Property | Value |
|----------|-------|
| Background | `#121212` |
| Foreground | `#e0e0e0` |
| Accent | `#909090` |
| Palette | Monochrome (gray scale) |
| Window chrome | Custom (no native decorations) |
| Font | System default |
| Icons | Lucide React |
| Layout | Fullscreen desktop with floating windows |

---

## Dormant Engines (Built, Not Yet Wired)

These modules are compiled, algorithmically complete, and unit-testable — but not yet connected to the inference pipeline:

| Module | Purpose | Potential Impact |
|--------|---------|-----------------|
| `spectral.rs` | Compress model weights in spectral domain (RFT → top-k → 4-bit sparse) | ~40-60% weight memory reduction |
| `kv_compress.rs` | Compress KV-cache during inference (FFT → φ-diagonal → INT8) | ~60% KV memory savings |
| `streaming.rs` | Load transformer layers on-demand from RFTL chunks instead of all-at-once | Enable larger models in less RAM |
| `mmap.rs` | Memory-mapped .rftmwpk2 model file access | Zero-copy weight loading |
| `inference.rs` | Full inference kernels: Q4_K/Q6_K dequant, GEMV, RoPE, GQA attention, SiLU, SSM | Replace Ollama with native Rust inference |
| `rft_kernel.rs` | 16 RFT transform variants (Fibonacci, Chebyshev, Hermite, Legendre, etc.) | Specialized basis selection per layer |
| `orchestrator.rs` | Multi-threaded task scheduler with 3 assembly types (Optimized/Unitary/Vertex) | Parallel engine execution |

**To activate:** Replace Ollama's inference loop with native Rust execution using these engines. The building blocks exist — they need to be composed into a complete forward pass.
