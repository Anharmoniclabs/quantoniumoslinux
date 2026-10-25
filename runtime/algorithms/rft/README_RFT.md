<!-- SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC -->

# Resonant Fourier Transform (RFT) - Canonical Definition

**USPTO Patent 19/169,399: "Hybrid Computational Framework for Quantum and Resonance Simulation"**

> **Clarification:** "Quantum" in the patent title refers to mathematical structure (unitarity, phase coherence), not quantum computing. All implementations are classical signal processing.

---

## ⚡ Industrial Application: Bearing Fault Detection

RFT has been benchmarked against FFT and STFT for **industrial vibration monitoring** using CWRU SKF6205 bearing parameters. Key results:

| Method | OR Fault SNR | False Alarm Rate | Drift Stable? |
|---|---|---|---|
| **RFT** | **45–52 dB** | **0%** | ✅ Yes |
| FFT | 43–50 dB | 0% | ✅ Yes |
| STFT | **0 dB** | **33%** | ❌ No |

- RFT outperforms FFT by **+0.1 to +1.8 dB** across all fault types
- STFT — the method most industrial firmware uses — produces **0 dB SNR** on outer race faults and **33% false alarm rate** on normal bearings
- Both RFT and FFT remain **stable under ±5% shaft speed drift**

📄 **Full benchmark:** [`applications/vibration/CWRU_BENCHMARK_RESULTS.md`](applications/vibration/CWRU_BENCHMARK_RESULTS.md)  
🧪 **Reproducible code:** [`applications/vibration/cwru_benchmark.py`](applications/vibration/cwru_benchmark.py)

```bash
python algorithms/rft/applications/vibration/cwru_benchmark.py
```

---

## RFT Family Overview

The RFT is a family of unitary transforms based on golden-ratio frequency spacing. There are **three main algorithmic variants**:

| Variant | Complexity | Phase Schedule | Implementation |
|---------|------------|----------------|----------------|
| **Canonical RFT** | O(N²) | `PHI_FRAC` | `resonant_fourier_transform.py` |
| **RFTMW-Hybrid** | O(N log N) | `PHI_POST_FFT` | `rftmw_native.forward_hybrid()` |
| **Vertex RFT** | O(N log N) | `PHI_VERTEX` | `vertex_quantum_rft.py` |

### Phase Schedule Reference

Let `frac(x)` denote the fractional part of x, i.e. `x - floor(x)`.

| Schedule | Formula | Used By |
|----------|---------|---------|
| `PHI_FRAC` | `f_k = frac((k+1)·φ)` | Canonical RFT (Gram-normalized basis) |
| `PHI_POST_FFT` | `E[k] = exp(j·2π·frac((k+1)·φ))` | RFTMW-Hybrid post-FFT modulation |
| `PHI_VERTEX` | `φ·k/N + (k%7)·n/N` | Vertex RFT topological encoding |
| `PHI_LEGACY` | `θ_k = 2π·k/φ` | **DEPRECATED** — do not use |

---

## 1. Canonical RFT: Gram-Normalized Basis

In this repository, the **canonical Resonant Fourier Transform (RFT)** is strictly defined as the **Gram-normalized irrational-frequency exponential basis**. This ensures exact unitarity at finite N while capturing golden-ratio resonance structure.

### 1.1 Basis Construction (Phase Schedule: PHI_FRAC)

**Step 1 — Raw Exponential Basis (Φ):**  
Construct an N×N matrix using golden-ratio frequencies `f_k = frac((k+1)·φ)`:

```
Φ_{n,k} = (1/√N) · exp(j·2π·f_k·n)
```

**Step 2 — Gram Normalization (U_φ):**  
Apply symmetric orthogonalization (Löwdin) using the Gram matrix `G = Φ†Φ` (See Theorem 10 for proof of uniqueness):

```
U_φ = Φ · (Φ†Φ)^(-1/2)
```

### 1.2 Forward / Inverse Transforms

Because `U_φ† · U_φ = I` (unitary by Theorem 2), the forward and inverse transforms are exact:

```
x̂ = U_φ† · x        (forward)
x  = U_φ  · x̂       (inverse)
```

**Implementation:** `algorithms/rft/core/resonant_fourier_transform.py`

**Complexity:** O(N²) apply, O(N³) basis construction (one-time initialization, cacheable)

---

## 2. RFTMW-Hybrid: FFT + Phase Modulation

The hybrid algorithm achieves O(N log N) computational complexity by factoring the operation as a standard FFT followed by a golden-ratio phase diagonal matrix.

### 2.1 Algorithm (Phase Schedule: PHI_POST_FFT)

```
Y = E ⊙ FFT(x) / √N

where E[k] = exp(j·2π·frac((k+1)·φ))
```

**Implementation:** `src/rftmw_native/core/rftmw_core.hpp::forward_hybrid()` (line 156-198)

**Complexity:** O(N log N) — equivalent to FFT

**Power-of-2 Requirement:** The native FFT requires power-of-2 sizes. Non-power-of-2 inputs trigger an O(N²) fallback unless auto-padded.

### 2.2 Usage

```python
import sys; sys.path.insert(0, 'src/rftmw_native/build')
import rftmw_native
import numpy as np

x = np.random.randn(1024).astype(np.float64)
Y = rftmw_native.forward_hybrid(x)  # O(N log N)
x_rec = np.real(rftmw_native.inverse_hybrid(Y))
print(f'Error: {np.linalg.norm(x - x_rec) / np.linalg.norm(x):.2e}')
```

---

## 3. Transform Theory (Test-Backed Theorems)

The repository includes a strict set of falsifiable mathematical theorems governing the RFT's operator properties. These theorems define the golden-native operator families that the canonical RFT basis diagonalizes, and prove its spectral advantages (Slepian-type concentration) over the DFT.

- Canonical proofs: `docs/THEOREMS_RFT_IRONCLAD.md`
- Test suite (deterministic, CI-stable): `tests/proofs/test_rft_transform_theorems.py`

```bash
pytest -q tests/proofs/test_rft_transform_theorems.py
```

---

## 4. Binary-to-Wave Encoding (φ-OFDM)

> ⚠️ **Reality Check: Algorithmic Mechanics**

What previous versions claimed: "Operations work directly on the waveform without decoding."

What the code actually executes:

```python
def wave_xor(self, w1, w2):
    for k in range(self.num_bits):
        sym1 = self._get_symbol(w1, k)  # ← DECODES bit via correlation
        sym2 = self._get_symbol(w2, k)  # ← DECODES bit via correlation
        sym_xor = -sym1 * sym2          # ← CLASSICAL Python logic
        result += sym_xor * carriers[k] # ← RE-ENCODES result
```

**Conclusion:** This is standard BPSK-OFDM modulation coupled with classical intermediate computation and re-modulation. It is not a "post-binary" or purely "wave-domain" logic paradigm. The only genuinely direct wave operation implemented is `wave_not(w) = -w` (trivial matrix negation).

### 4.1 What φ-OFDM Actually Contributes

The novel contribution is substituting standard integer harmonics with golden-ratio frequency spacing to bypass rational spectral leakage (See Theorem 8A).

- **Standard OFDM:** `f_k = k/N` (integer harmonics, susceptible to Diophantine misalignment)
- **φ-OFDM:** `f_k = frac((k+1)·φ)` (irrational frequencies — no two carriers are rationally commensurate)

---

## 5. Comparison to Standard FFT

| Property | Standard FFT | Canonical RFT (U_φ) | RFTMW-Hybrid |
|---|---|---|---|
| Frequency Grid | Rational bins (k/N) | Irrational (φ-grid) | FFT + φ-phase |
| Complexity | O(N log N) | O(N²) apply | O(N log N) |
| Unitarity | Exact (F†F = I) | Exact (U_φ†U_φ = I) | Exact (Theorem 3) |
| Spectral Leakage | sinc²(ε) bounded | Zero grid mismatch | FFT leakage |

---

## 6. Vertex RFT (Classical Graph-Based Engine)

> ⚠️ **Important:** Despite historical "quantum" naming, this is NOT quantum computing. It is classical signal processing using quantum-inspired phase arrays (geometric phases).

### 6.1 Overview

The Vertex RFT (`algorithms/rft/kernels/python_bindings/vertex_quantum_rft.py`) provides:

- O(N log N) transforms via RFTMW-Hybrid internally
- Automatic power-of-2 padding for non-power-of-2 inputs
- Phase schedule PHI_VERTEX: `φ·k/N + (k%7)·n/N` for topological data analysis
- Machine precision roundtrip accuracy (< 1e-12 error)

### 6.2 Quick Test

```bash
python -c "
import sys; sys.path.insert(0, 'algorithms/rft/kernels/python_bindings')
import numpy as np
from vertex_quantum_rft import VertexQuantumRFT
for N in [64, 100, 256, 1000]:
    vrft = VertexQuantumRFT(N)
    x = np.random.randn(N) + 1j*np.random.randn(N)
    y = vrft.forward_transform(x)
    x_rec = vrft.inverse_transform(y)
    err = np.linalg.norm(x - x_rec) / np.linalg.norm(x)
    print(f'N={N}: error={err:.2e}')
"
```

---

## 7. File Index

| Purpose | File | Phase Schedule |
|---|---|---|
| Canonical RFT kernel | `algorithms/rft/core/resonant_fourier_transform.py` | PHI_FRAC |
| RFTMW-Hybrid | `src/rftmw_native/core/rftmw_core.hpp` | PHI_POST_FFT |
| Vertex RFT | `algorithms/rft/kernels/python_bindings/vertex_quantum_rft.py` | PHI_VERTEX |
| Legacy kernels | `algorithms/rft/kernels/rft_kernels.py` | PHI_LEGACY (deprecated) |
| Wave-domain hash | `algorithms/rft/core/symbolic_wave_computer.py` | — |
| Formal Math Proofs | `docs/THEOREMS_RFT_IRONCLAD.md` | — |
| **Vibration Benchmark** | **`algorithms/rft/applications/vibration/cwru_benchmark.py`** | — |
| **Benchmark Results** | **`algorithms/rft/applications/vibration/CWRU_BENCHMARK_RESULTS.md`** | — |

---

## 8. Patent Claims Context

This framework implements:

- **Claim 1:** Binary → Wave encoding via BPSK on resonant (golden) carriers
- **Claim 2:** Waveform manipulation logic (routing BPSK via classical decoding)
- **Claim 3:** Cryptographic hashing utilizing the canonical resonance structure (RFT-SIS)
- **Claim 4:** Geometric data feature preservation utilizing the golden-ratio basis limits

---

## 9. Citation

```text
@misc{rft2025,
  title   = {Resonant Fourier Transform: Golden-Ratio Multi-Carrier Wave Encoding},
  author  = {Minier, Luis M.},
  year    = {2025},
  note    = {USPTO Patent Application 19/169,399}
}
```

---
*Canonical Definition — Updated March 2026*
