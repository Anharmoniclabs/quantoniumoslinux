# Project Name Note

> **"quantoniumos" is the canonical project name (one word).**
> **The final two letters `os` are part of the name and not an acronym.**

This repository implements the **Resonant Fourier Transform (RFT)**: a classical unitary transform that provides provable spectral concentration advantages for signals with golden-ratio quasi-periodic structure, and **no general advantage on other signals**.

---

## What This Actually Is

| Claim | True / False |
|-------|--------------|
| Classical signal-processing library | **True** |
| Implements a novel unitary transform basis (RFT) | **True** |
| RFT provably outperforms DFT on φ-grid quasi-periodic signals | **True** (Theorem 8, Theorem E) |
| RFT provides general-purpose speedup over FFT | **False** — O(N²) canonical path is *slower* than FFT |
| Quantum computing project | **False** — no qubits, no quantum gates, no quantum hardware |
| Operating system | **False** — `quantonium_os_src/` is an unmaintained PyQt5 GUI shell |
| "Quantum-inspired" algorithms | **Rejected** — this framing is dropped. The math is classical number theory (Weyl equidistribution, Hurwitz's theorem, Vandermonde rank). |

---

## Name Consistency

The repository URL and PyPI package name are both `quantoniumos`, and all documentation should preserve that exact spelling.

---

## What the Real Contributions Are

1. **RFT basis construction** — Gram-normalized golden-ratio frequency grid, proven unitary (Theorem 2)
2. **Theorem 8** — Analytically proven K₀.₉ₙ(U) = O(log N) vs K₀.ₙₙ(F) ∝ N⁰·⁷⁵ gap via Weyl + Hurwitz + Steinhaus-Sós
3. **Theorem E** — Empirical RIP constant δ₈ = 0.1603 < 0.2929 (Candès-Tao threshold) at N=128, K=8
4. **STRT depth-gated routing** — 91.7% win rate over hard-block DCT on GPT-2 deep V-cache layers (layers 9–11)
5. **C++ SIMD engine** — Custom Jacobi eigendecomposition, Bluestein FFT, 48-round Feistel cipher
6. **255/255 formal proof tests passing** — see [`docs/PROOF_RESULTS.md`](docs/PROOF_RESULTS.md)

---

*This file is linked from `README.md` and exists to give any reviewer, fork, or
aggregator the complete picture in one place.*
