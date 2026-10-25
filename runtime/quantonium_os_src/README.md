# quantoniumos — Unified Framework (Claim 4)

> **License:** LicenseRef-quantoniumos-Claims-NC (research/education only)
> **Patent:** USPTO Application 19/169,399, Claim 4 — Hybrid Mode Integration
> **Contact:** See top-level `CITATION.cff` for commercial licensing

---

## Architecture

This directory implements the **proprietary unified framework** of Patent Claim 4 — the desktop environment, application suite, and middleware engine that integrates all RFT algorithm layers into a cohesive system.

```
quantonium_os_src/
├── frontend/
│   └── quantonium_desktop.py   # PyQt5 desktop with golden spiral launcher
├── apps/
│   ├── symbolic_simulator/     # Classical matrix transform workbench (RFT/DFT)
│   ├── quantum_inspired_crypto/ # AES + RFT 48-round Feistel cipher (AEAD)
│   ├── q_vault/                # AES-encrypted JSON key-value store
│   ├── q_notes/                # Integrated text editor
│   ├── rft_validator/          # Bijection, reversibility, entropy checks
│   ├── rft_visualizer/         # RFT vs DFT coefficient distribution plots
│   └── system_monitor/         # CPU/memory/disk monitoring
└── engine/
    ├── RFTMW.py                # Middleware Transform Engine (Claims 1 & 4)
    ├── rft_compress.py         # Unified compression pipeline
    ├── rftmw_inference.py      # RFT-based inference
    └── rftmw_memory.py         # Transform memory management
```

## Technical Notes

- This is **classical signal processing** — no qubits, no quantum hardware, no exponential speedup
- The "quantum-inspired" naming refers to mathematical structures (unitary matrices, state vectors), not quantum computing
- Crypto uses AES-128 + RFT Feistel cipher — classically secure, not post-quantum
- Desktop environment requires PyQt5

## Launch

```bash
# From repo root
pip install PyQt5 psutil matplotlib
python quantonium_os_src/frontend/quantonium_desktop.py
```

## Engine Layer

The RFTMW `MiddlewareTransformEngine` is the core of Claim 4:

- **Binary-to-Waveform:** Direct byte-to-amplitude mapping (`[0,255] -> [-1,1]`)
- **27-Variant Dispatch:** Auto-selects optimal RFT variant by signal type
- **Three Backends:** Python, C/ASM, C++ native (AVX2)
- **Compression:** Rate-distortion routing with INT8+zlib fallback

The engine also includes an opt-in experimental variant, `wave_fibonacci_pruned`.
Unlike the registry-backed unitary transforms, it uses FFT plus a sparse Fibonacci/golden
candidate-bin mask inferred from RFTMW wave-space probes. This makes it useful for
acceleration experiments on phi-structured signals, but it remains approximate and should
not be treated as an exact round-trip transform.

The algorithm library in [`algorithms/rft/`](../algorithms/rft/) is also part of the Claims-NC / proprietary RFT stack.
For formal proofs, see [`tests/proofs/`](../tests/proofs/).
