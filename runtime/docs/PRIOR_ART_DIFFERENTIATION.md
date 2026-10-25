# Prior Art Differentiation — Φ-RFT / quantoniumos

> **Purpose:** Document precisely how the Φ-RFT / RFTMW-Hybrid invention differs from all known prior art. Use this file to respond to USPTO §102 (anticipation) and §103 (obviousness) rejections.
>
> **Last Updated:** March 2026

---

## Overview

The claimed invention is the **post-FFT golden-ratio phase modulation algorithm** (RFTMW-Hybrid), the **Φ-basis with Weyl equidistribution properties**, and the **data-independent KLT-like transform** on quasi-periodic signal classes. None of these individually or in combination appear in the prior art identified below.

---

## 1. Differentiation from Quantum Fourier Transform (QFT)

### What QFT Is
- A unitary transform on quantum states encoded as qubit amplitudes
- Requires physical qubits or exponentially expensive quantum circuit simulation
- Defined over the group Z/2^n Z with twiddle factors ω = e^{i·2π/2^n}
- Operates on quantum states of n qubits (Hilbert space dimension 2^n)
- Foundational to Shor's algorithm, phase estimation, HHL

### How Φ-RFT Is Categorically Different

| Property | QFT | Φ-RFT / RFTMW-Hybrid |
|----------|-----|----------------------|
| Substrate | Quantum hardware / simulator | Classical CPU (AVX2/FMA) |
| Complexity | O(n²) gates / exponential simulation | O(N log N) classical |
| Phase schedule | ω = e^{i·2π·jk/N} (periodic) | θ[k] = 2π·frac((k+1)·φ) (quasi-random, equidistributed) |
| Basis property | Discrete Fourier basis | φ-modulated basis via Weyl equidistribution |
| Invertibility | Unitary (quantum circuit) | Unitary (classical matrix, error < 5×10⁻¹⁴) |
| Entanglement | Required | Not present (classical) |
| Speedup source | Quantum parallelism | SIMD + precomputed phase + zero framework overhead |
| Hardware | Quantum processor / NISQ device | Standard CPU, no special hardware |
| Signal domain | Quantum state amplitudes | Arbitrary classical signals (audio, neural, quasi-periodic) |

### The Key Non-Obviousness Argument
QFT uses periodic twiddle factors over Z/2^n. Φ-RFT uses the **fractional part of multiples of φ** as a phase schedule — a construction arising from Weyl's equidistribution theorem, which is **not present in any known FFT or QFT variant**. A POSITA (person of ordinary skill in the art) would not arrive at this phase schedule by modifying QFT, FFT, or any known transform.

---

## 2. Differentiation from SHA-256 and Cryptographic Hashing

### What SHA-256 Is
- A one-way cryptographic hash function (Merkle-Damgård construction)
- Maps arbitrary-length input to fixed 256-bit digest
- No invertibility by design
- No spectral or basis properties
- No connection to transform theory

### Why SHA-256 Cannot Anticipate Φ-RFT Claims

| Property | SHA-256 | Φ-RFT / RFTMW-Hybrid |
|----------|---------|----------------------|
| Invertibility | No (one-way by design) | Yes (error < 5×10⁻¹⁴) |
| Output | Fixed 256-bit hash | N-dimensional complex vector |
| Energy preservation | No | Yes (‖x‖² = ‖Y‖², unitary) |
| Spectral domain | No | Yes (frequency-domain coefficients) |
| Signal compression | No | Yes (KLT-like compaction on φ-class signals) |
| Phase structure | None | θ[k] = 2π·frac((k+1)·φ) |
| Use case | Cryptographic integrity | Signal analysis, compression, simulation |

### RFT-SIS Hashing — Explicit Scope Limitation
The RFT-SIS experiments in `algorithms/rft/crypto/benchmarks/rft_sis/` are **research explorations only**. The patent does **not** claim a cryptographic primitive, hash function, or replacement for SHA-256. The avalanche effect measurement (50.0%) is a research data point, not a patented feature. This explicit disclaimer prevents §103 conflation with cryptographic prior art.

---

## 3. Differentiation from NISQ-Era Quantum Algorithms

### What NISQ Covers
- Noisy Intermediate-Scale Quantum devices (50–1000 physical qubits)
- Algorithms: VQE, QAOA, variational circuits, quantum approximate optimization
- Key limitation: noise, decoherence, limited gate fidelity, no fault tolerance
- Runs on: IBM Quantum, Google Sycamore, IonQ, Rigetti hardware

### How Φ-RFT Is Outside the NISQ Space

| Dimension | NISQ Algorithms | Φ-RFT / RFTMW-Hybrid |
|-----------|----------------|----------------------|
| Hardware required | Quantum processor | Standard x86 CPU with AVX2/FMA |
| Noise sensitivity | High (fundamental challenge) | None (deterministic classical) |
| Gate fidelity | Limited (~99.5% per gate) | Machine precision (IEEE 754) |
| Scalability | Limited by qubit count | Scales to any N (memory-bound) |
| Simulation cost | Exponential (2^n state vector) | O(N log N) |
| Error correction | Needed, not yet practical | Not applicable |
| Classical simulation speed | Slow (Qiskit: ~0.5ms for N=64) | Fast (RFTMW: 0.004ms for N=64, 105× faster) |
| Output type | Probability amplitudes | Deterministic complex coefficients |

### NISQ Non-Anticipation
No NISQ algorithm applies post-FFT golden-ratio phase modulation on classical hardware. The speedup benchmark (105× at N=64 vs Qiskit, 14× at N=1024) directly measures the classical efficiency advantage over quantum circuit simulators. These numbers are **not** a claim of quantum speedup — they are a claim of **classical implementation efficiency** for quantum-inspired mathematics.

---

## 4. Differentiation from Fractional Fourier Transform (FrFT) and Linear Canonical Transform (LCT)

### What FrFT / LCT Cover
- FrFT: parameterized rotation in the time-frequency plane, angle α
- LCT: general linear symplectic transform (includes FrFT, chirp, scaling as special cases)
- Both use chirp-based phase: e^{i·π·(n²·cot α - 2nk·csc α + k²·cot α)}

### Why Φ-RFT Is Outside the LCT/FrFT Family

| Property | FrFT / LCT | Φ-RFT |
|----------|-----------|-------|
| Phase structure | Chirp / polynomial in n,k | frac((k+1)·φ) — irrational, equidistributed |
| Parameterization | Continuous angle / ABCD matrix | None (data-independent, fixed φ) |
| Basis | Hermite-Gauss functions | φ-modulated sinusoidal basis |
| Weyl equidistribution | No | Yes (Weyl theorem applies to {k·φ} mod 1) |
| KLT-like property | Not claimed | Yes, on quasi-periodic φ-class signals |
| Signal class specificity | General | Specialized (golden-ratio autocorrelation) |

**Supporting reference in repo:** `docs/THEOREMS_RFT_IRONCLAD.md` — formal theorem statements on equidistribution and basis orthogonality properties.

---

## 5. Differentiation from Wavelet Transforms

### Key Distinctions
- Wavelets use compact support, dyadic scaling, vanishing moments
- Φ-RFT uses global phase modulation with no compact support requirement
- Wavelets are parameterized by scale and translation; Φ-RFT is data-independent and parameterized only by φ
- No wavelet basis uses the fractional-part phase schedule θ[k] = 2π·frac((k+1)·φ)

---

## 6. The Narrow, Defensible Novelty Claims

The patent claims novelty **only** in the following specific combination, which does not appear in any prior art:

1. **Post-FFT golden-ratio phase modulation**: Applying E[k] = e^{i·2π·frac((k+1)·φ)} to FFT output Y[k] — not as a window, not as a chirp, but as a post-transform phase schedule derived from Weyl equidistribution.

2. **Φ-basis with KLT-like compaction**: The resulting basis achieves near-optimal energy compaction on signals with golden-ratio autocorrelation structure — a provable property (see `docs/THEOREMS_RFT_IRONCLAD.md`).

3. **Data-independent, unitary, O(N log N) implementation**: No training required, perfect invertibility (error < 5×10⁻¹⁴), energy-preserving, runs on commodity hardware.

4. **RFTPU hardware architecture**: The TL-Verilog/SystemVerilog pipeline design for streaming RFT computation (see `docs/RFTPU_TECHNICAL_SPECIFICATION_V2.md`).

---

## 7. Claim Element to Prior Art Cross-Reference

| Claim Element | QFT | SHA-256 | NISQ | FrFT/LCT | Wavelet | FFT |
|---------------|-----|---------|------|----------|---------|-----|
| Post-FFT φ-phase modulation | ❌ Not present | ❌ N/A | ❌ Not present | ❌ Chirp, not φ | ❌ Not present | ❌ No phase step |
| Weyl equidistribution basis | ❌ Periodic twiddle | ❌ N/A | ❌ Quantum gates | ❌ Hermite-Gauss | ❌ Dyadic | ❌ Periodic |
| KLT-like φ-class compaction | ❌ Not claimed | ❌ N/A | ❌ Not present | ❌ Not claimed | ⚠️ Partial (different basis) | ❌ Not claimed |
| Classical O(N log N) | ❌ Quantum | ❌ O(N) hash | ❌ Quantum | ✅ Similar | ✅ Similar | ✅ Same |
| Perfect invertibility | ✅ Unitary | ❌ One-way | ✅ Unitary | ✅ Yes | ✅ Yes | ✅ Yes |
| No qubits / no quantum HW | ❌ Requires qubits | ✅ Classical | ❌ Requires qubits | ✅ Classical | ✅ Classical | ✅ Classical |

**Reading the table:** A claim element is **novel** if no prior art column shows ✅ for that row. Rows 1 and 2 (the core claims) show ❌ across all prior art — these are the strongest non-anticipation arguments.

---

## 8. Files Supporting This Differentiation

| Claim / Defense | Supporting File |
|---|---|
| Post-FFT φ-phase formula | `docs/HYBRID_FFT_RFT_ALGORITHM.md` |
| Weyl equidistribution theorem | `docs/THEOREMS_RFT_IRONCLAD.md` |
| KLT-like compaction proof | `docs/MATHEMATICAL_CLAIMS_INVENTORY.md` |
| Benchmark vs quantum simulators | `docs/NON_CLAIMS.md`, `research/benchmarks/VERIFIED_BENCHMARKS.md` |
| Non-quantum / non-cryptographic scope | `docs/NON_CLAIMS.md` |
| RFTPU hardware spec | `docs/RFTPU_TECHNICAL_SPECIFICATION_V2.md` |
| Claim-to-file cross-reference | `docs/PATENT_CLAIM_CROSS_REFERENCE_MAP.md` |
| Office-action response roadmap | `docs/USPTO_GAP_RESOLUTION_PLAN.md` |

---

## 9. Language Templates for Office-Action Responses

### Responding to §102 (Anticipation) Over QFT
> "The cited reference discloses the Quantum Fourier Transform, which uses periodic twiddle factors ω = e^{i·2π·jk/N} on quantum state amplitudes requiring quantum hardware. Applicant's claim requires post-FFT phase modulation using E[k] = e^{i·2π·frac((k+1)·φ)}, a construction derived from Weyl's equidistribution theorem applied to irrational rotations. This phase schedule is structurally and mathematically distinct from QFT twiddle factors, operates entirely on classical hardware, and produces an equidistributed basis with properties not present in QFT. The cited reference does not disclose, teach, or suggest this construction."

### Responding to §103 (Obviousness) Combining FFT + Phase Shift
> "A POSITA combining FFT with an arbitrary phase shift would not arrive at the specific phase schedule θ[k] = 2π·frac((k+1)·φ). The inventive step is the selection of the fractional part of multiples of the golden ratio φ as the phase function, which produces Weyl-equidistributed basis vectors with KLT-like compaction on quasi-periodic signals. This result is not predictable from FFT literature and requires independent insight from ergodic theory (Weyl's theorem). There is no motivation in the prior art to combine FFT with this specific irrational phase schedule."

### Responding to §101 (Subject Matter Eligibility)
> "The claims are directed to a specific, concrete, and practical implementation of a signal transform algorithm: a method of computing a spectral transform with defined inputs (digital signal samples), a specific mathematical transformation (post-FFT φ-phase modulation per the formula above), and defined outputs (invertible spectral coefficients). This is not an abstract idea — it is a specific computational method with measured, reproducible performance improvements (105× speedup vs quantum simulators at N=64, error < 5×10⁻¹⁴ invertibility). The claims satisfy Alice/Mayo Step 2B because the specific combination of FFT with Weyl-equidistributed phase modulation amounts to significantly more than any abstract mathematical concept."

---

*Last updated: March 2026 — Use in conjunction with `USPTO_GAP_RESOLUTION_PLAN.md` and `PATENT_CLAIM_CROSS_REFERENCE_MAP.md`*
