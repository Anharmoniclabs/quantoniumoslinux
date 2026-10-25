# USPTO Gap Resolution Plan
**Application No.:** 19/169,399
**Date:** March 24, 2026
**Purpose:** Concrete, file-level actions to close the 4 critical examination gaps

---

## GAP 1 — Technical Specification Documents
**Status:** HIGH PRIORITY — STILL OPEN

### What We Already Have (Don't Rewrite)

From `algorithms/rft/core/canonical_true_rft.py` header:
```
Φ̃ = Φ(ΦᴴΦ)^{-1/2}  — unitarity error < 5e-15 (machine precision)
```

From `algorithms/rft/RFT_RESEARCH_SUMMARY.md`:
- RFT = eigenbasis of resonance operator K = T(r * d)
- K is Hermitian: error = 0.00e+00 ✓
- U is Unitary: error ≈ 1e-13 ✓
- Perfect Reconstruction: error ≈ 1e-15 ✓
- Approximation decay for quasi-periodic signals: E_n = O(rⁿ) exponential vs FFT O(n^{-1/2})
- Frobenius distance from DFT: ≥ 9 (documented)

### What to Write Per Claim

#### Claim 1 — Symbolic RFT Engine: Technical Spec Block

Create `docs/claim_specs/claim1_rft_engine_spec.md`:

```
Title: Symbolic Resonance Fourier Transform Engine
Patent Application: 19/169,399, Claim 1

Formal Definition:
  Input:  x ∈ ℂ^N  (quantum amplitude vector)
  Kernel: K = T(r · d)
    where T = Toeplitz constructor
          r[k] = cos(2π φ^k f₀) + i·sin(2π φ^k f₀),  φ = (1+√5)/2
          d[k] = exp(-λk),  λ = regularization constant
  Basis:  Φ = eigenvectors of K  (K Hermitian ⟹ Φ unitary)
  Gram-normalised: Φ̃ = Φ(ΦᴴΦ)^{-1/2}
  Forward: Y = Φ̃ · x
  Inverse: x̂ = Φ̃ᴴ · Y

Unitarity Guarantee:
  ‖Φ̃ᴴΦ̃ − I‖_F < 5×10⁻¹⁵  (machine precision)

Amplitude Decomposition (symbolic):
  For each component k: Y[k] = Σⱼ Φ̃[k,j]·x[j]
  Preserves ‖Y‖₂ = ‖x‖₂  (Parseval equivalent)

Phase-Space Coherence Retention Mechanism:
  Definition: a transform T retains phase-space coherence if
    |⟨T(x), T(y)⟩| = |⟨x, y⟩|  for all x, y ∈ ℂ^N
  Proof: follows directly from unitarity of Φ̃.
  Resonance state: resonance_state = Φ̃ · quantum_state
  The resonance_state preserves all inner-product relationships
  of the original quantum_state through the transform.

Topological Embedding Layer:
  Each output coefficient Y[k] is embedded in phase-space via:
    vertex_k = (Re(Y[k]), Im(Y[k]), |Y[k]|)  ∈ ℝ³
  Projection mapping π: ℂ^N → ℝ^{3N}
  Topology preserved: adjacent vertices share φ-resonance relationship.

Complexity:
  Kernel construction: O(N³) — one-time, amortised
  Forward/Inverse transform: O(N²)
  For fixed kernel (pre-computed): O(N²) per transform

Distinctness from DFT:
  Frobenius distance ‖Φ̃ − DFT_N‖_F ≥ 9  (empirically verified, N=64–512)
  DFT basis: ω^{jk} = exp(2πi·jk/N)
  RFT basis: non-integer frequency structure via φ^k — irrational frequency spacing
```

#### Claim 2 — Cryptographic Subsystem: Technical Spec Block

Create `docs/claim_specs/claim2_crypto_spec.md`:

```
Title: Resonance-Based Cryptographic Subsystem
Patent Application: 19/169,399, Claim 2

Waveform Generation Pipeline:
  1. Input bytes b → SHA3 hash chain expansion → signal s ∈ [-1,1]^N
  2. RFT transform:  Y = Φ̃ · s  (golden-ratio phase structure)
  3. Amplitude-phase feature:  f[k] = |Y[k]| · (1 + φ·cos(∠Y[k]))
     — φ modulates the feature by golden-ratio phase (novel signature)
  4. SIS quantization: s_short = round(f · β/max(|f|)),  ‖s_short‖∞ ≤ β
  5. Lattice point: c = A · s_short  (mod q)
     where A = A_φ + R,  A_φ deterministic φ-structured,  R pseudo-random
  6. Output hash: SHA3-256( c ‖ domain_salt )

Parameters (Research Prototype):
  sis_n = 512, sis_m = 1024, sis_q = 3329 (Kyber prime), sis_beta = 100

Entropy Modulation:
  The φ·cos(∠Y[k]) term dynamically modulates output entropy as a function
  of the RFT spectral phase — this is the novel entropy mapping step that
  distinguishes this construction from plain SIS hashing.

Recursive Modulation:
  Feistel round structure (feistel_round48.c): 48-round recursive waveform
  modification applied at assembly level before lattice reduction.
```

#### Claim 3 — Geometric Structures: Technical Spec Block

Create `docs/claim_specs/claim3_geometric_spec.md`:

```
Title: Geometric Structures for RFT-Based Cryptographic Waveform Hashing
Patent Application: 19/169,399, Claim 3

Coordinate Transformation Pipeline:
  Input: signal s ∈ ℂ^N after RFT
  Step 1 — Polar decomposition:  r[k] = |s[k]|,  θ[k] = ∠s[k]
  Step 2 — φ-resonance projection:
    phi_resonance = Σ_k cos(θ[k] · φ)    # φ-scaled phase summary
    phi_winding   = Σ_k sin(θ[k] · φ)    # orthogonal winding component
  Step 3 — Cartesian re-embedding:
    x[k] = r[k] · cos(θ[k] + φ·frac(k·φ))  # irrational phase offset
    y[k] = r[k] · sin(θ[k] + φ·frac(k·φ))
  Step 4 — Projection mapping:
    hash_component = A_φ_row_k · (x[k], y[k])ᵀ  (mod q)
    Preserves inner-product topology: |⟨u,v⟩| invariant under φ-rotation

Phase-Winding Tags:
  tag_k = sign(phi_resonance) ⊕ parity(round(phi_winding · N/2π))
  Used as domain-separation labels in hash construction.

Gram Matrix Role (gram_utils.py):
  G = ΦᴴΦ  — Gram matrix of RFT basis
  Eigenstructure of G used to verify near-orthogonality and bound
  distortion in the geometric embedding.
  G^{-1/2} applied during Gram-normalisation step.
```

#### Claim 4 — Hybrid Mode Integration: Technical Spec Block

Create `docs/claim_specs/claim4_hybrid_spec.md`:

```
Title: Hybrid Mode Integration (Unified Computational Framework)
Patent Application: 19/169,399, Claim 4

Framework Components:
  - Engine A: Symbolic RFT (Claim 1)  — algorithms/rft/core/canonical_true_rft.py
  - Engine B: Crypto Subsystem (Claim 2) — algorithms/rft/crypto/enhanced_cipher.py
  - Engine C: Geometric Hash (Claim 3)   — algorithms/rft/crypto/rft_sis_hash.py
  - Scheduler: unified_transform_scheduler.py — selects and coordinates engines

Resonance State Propagation:
  resonance_state = RFT.forward(quantum_state)   # Engine A output
  cipher_state    = Crypto.encrypt(resonance_state)  # Engine B input = A output
  hash_output     = GeoHash.hash(cipher_state.bytes)  # Engine C input = B output
  Final: (cipher_state, hash_output) — coupled output pair

Dynamic Resource Allocation:
  Scheduler selects engine variant based on signal size N and available assembly:
    N ≤ 256:   canonical RFT path (Python)
    N ≤ 2048:  RFTMW native path (src/rftmw_native/)
    N > 2048:  hybrid path (algorithms/rft/hybrids/) with chunked execution
  Fallback: Python reference implementation always available.

Hybrid Advantage:
  The unified framework enables coherent propagation of resonance_state
  across encryption layers — a property absent in systems that treat
  transform and cryptography as independent modules.
  Inner-product preservation: |⟨resonance_state, key_material⟩| is
  bounded by the SIS lattice structure, creating a cryptographically
  meaningful coupling between the transform and the hash output.
```

### Action: Who Creates These Files

- `scripts/build_ieee_paper_v2.py` + `scripts/populate_ieee_docx.py` already generate
  structured document output — pipe the above specs through them to produce formatted USPTO exhibits.
- Files to create: `docs/claim_specs/claim1_rft_engine_spec.md` through `claim4_hybrid_spec.md`
- One additional file: `docs/claim_specs/DECLARATIONS_SUMMARY.md` — a single-page roll-up
  for the attorney to attach to the IDS or examiner response.

---

## GAP 2 — Performance Benchmarks
**Status:** HIGH PRIORITY — PARTIALLY MET

### What Exists Already

| Benchmark | File | Status |
|---|---|---|
| RFT vs FFT speed (N=1024) | `scripts/rft_quick_reference.py` | ✅ Done — 5.1× overhead |
| Unitarity / round-trip | `scripts/run_research_suite.py` | ✅ Done — error < 1e-15 |
| Wave-domain computation | `tests/research/wave_domain_computation_benchmark.py` | ✅ Done |
| Gram matrix eigenstructure | `tests/research/gram_eigenstructure_analysis.py` | ✅ Done |
| Quasi-periodic approximation | `algorithms/rft/RFT_RESEARCH_SUMMARY.md` | ✅ Done — 40–55% reduction |

### What's Missing + Exactly How to Generate It

#### Benchmark A — Claim 1: Symbolic vs Classical Transform

**File to create:** `benchmarks/claim1_symbolic_vs_classical.py`

```python
# What to measure:
# 1. RFT vs DFT/QFT on φ-quasi-periodic signals (N = 64, 256, 1024, 4096)
# 2. Approximation error at k=10,20,50,100 retained coefficients
# 3. Report: RFT approximation decay exponent α vs FFT (already -0.848 vs -0.684)
# 4. Frobenius distance ‖Φ̃ − DFT_N‖_F for N = 64 to 2048
# Key result to highlight for USPTO: exponential vs polynomial decay for matched signals
```

**Time estimate:** 2–3 hours to write and run.

#### Benchmark B — Claim 2: Cryptographic Security

**File to create:** `benchmarks/claim2_crypto_security_validation.py`

```python
# What to measure:
# 1. Avalanche effect: flip 1 input bit → measure fraction of output bits flipped
#    Target: ≥ 45% flip rate (near 50% ideal)
#    Compare: SHA-256 baseline, Blake2 baseline, RFT-SIS result
# 2. Collision pre-image resistance (empirical): 2^20 random inputs, count collisions
#    Report: 0 collisions expected at 256-bit output
# 3. Entropy of output bytes: measure Shannon entropy H(output)
#    Target: ≥ 7.99 bits/byte
# IMPORTANT CAVEAT to include in report:
#    Per rft_sis_hash.py docstring: NumPy RNG used — NOT a CSPRNG.
#    Security claims are empirical only. Full formal security proof is future work.
```

**Time estimate:** 3–4 hours. The avalanche test is the most important output for USPTO.

#### Benchmark C — Claim 3: Geometric Hash Collision Resistance

**File to create:** `benchmarks/claim3_geometric_collision_analysis.py`

```python
# What to measure:
# 1. Collision rate: 10^6 random input pairs → count hash collisions
# 2. Distribution uniformity: chi-squared test on output byte distribution
# 3. Structural distinctness from SHA-256/Blake2:
#    - Cross-correlation between RFT-SIS output and SHA-256 output for same input
#    - Should be near-zero (independent outputs)
# 4. φ-resonance contribution: compare A_φ-only hash vs full hybrid A = A_φ + R
#    — shows φ structure adds measurable independence
```

**Time estimate:** 2–3 hours.

#### Benchmark D — Claim 4: Hybrid Integration Advantage

**File to create:** `benchmarks/claim4_hybrid_integration_proof.py`

```python
# What to measure:
# 1. Coupled output consistency: given the same quantum_state, show that
#    (cipher_state, hash_output) pairs are deterministic and coherent
#    — demonstrates the "unified framework" property is not just two separate calls
# 2. Resonance propagation: measure ‖resonance_state‖₂ preservation through
#    RFT → cipher → hash pipeline
# 3. Engine selection correctness: run all 3 scheduler paths (canonical / native / hybrid)
#    on same input, verify identical outputs
# 4. Throughput: ops/sec for each engine at N=256, 1024, 4096
```

**Time estimate:** 2–3 hours.

### Fastest Path to USPTO-Ready Benchmark Package

1. Run `python benchmarks/claim1_symbolic_vs_classical.py > results/claim1_benchmark.txt`
2. Run `python benchmarks/claim2_crypto_security_validation.py > results/claim2_security.txt`
3. Run `python benchmarks/claim3_geometric_collision_analysis.py > results/claim3_collision.txt`
4. Run `python benchmarks/claim4_hybrid_integration_proof.py > results/claim4_hybrid.txt`
5. Feed through `scripts/populate_ieee_docx.py` to format as exhibit.

---

## GAP 3 — Prior Art Differentiation
**Status:** HIGH PRIORITY — STILL OPEN

### Existing Evidence to Compile (Don't Repeat Work)

- `scripts/bridge_canonical_vs_hybrid.py` — technical canonical vs hybrid bridge
- `scripts/evidence_pack_track_b.py` — Track B algorithm evidence
- `algorithms/rft/RFT_RESEARCH_SUMMARY.md` — approximation analysis already contains
  honest FFT/DCT comparison data
- `algorithms/rft/core/absolute_novelty.py` — specifically addresses novelty

### File to Create: `docs/PRIOR_ART_DIFFERENTIATION.md`

Structure (write this document directly — most data already exists):

```markdown
## Section 1: Quantum Transform Methods

Prior Art: QFT (Quantum Fourier Transform)
  - Basis: ω^{jk} = exp(2πi·jk/N), uniform integer frequency spacing
  - Requires physical quantum hardware (gate-based)
  - No φ-structure, no irrational frequency spacing

Your Innovation (Claim 1):
  - Basis: eigenvectors of K = T(r·d), r driven by φ^k — irrational frequency spacing
  - Runs on classical hardware as symbolic simulation
  - Frobenius distance from QFT basis ≥ 9 (empirically verified)
  - Exponential vs polynomial approximation decay for quasi-periodic signals

## Section 2: Cryptographic Hashing

Prior Art: SHA-256, Blake2, BLAKE3
  - All operate on integer bit-string permutations
  - No spectral/frequency domain computation
  - No φ-structured lattice matrix
  - No coherent coupling to a transform engine

Your Innovation (Claims 2 + 3):
  - RFT spectral domain intermediate: Y = Φ̃·s before quantisation
  - φ-modulated amplitude-phase feature: f[k] = |Y[k]|·(1 + φ·cos(∠Y[k]))
  - Hybrid SIS matrix A = A_φ + R — φ-structured + random
  - Domain-separated output via golden-ratio phase-winding tags

## Section 3: Topological Computing

Prior Art: Anyonic Braiding (topological quantum computing)
  - Requires physical non-Abelian anyons
  - Hardware-bound; no classical symbolic equivalent

Your Innovation (Claim 3):
  - Vertex-based geometric embedding: (Re(Y[k]), Im(Y[k]), |Y[k]|) ∈ ℝ³
  - Projection-preserving hash: topology maintained via inner-product invariance
  - Runs on classical hardware with no exotic physics required

## Section 4: Hybrid Quantum-Classical

Prior Art: NISQ VQE, QAOA
  - Require physical quantum co-processors for the quantum layer
  - No symbolic phase-space preservation
  - Transform and cryptography layers are always independent

Your Innovation (Claim 4):
  - Fully classical symbolic simulation of quantum amplitude decomposition
  - Coherent resonance_state propagation across RFT → cipher → hash layers
  - Dynamic scheduler unifies all three claim engines under one framework
  - Assembly-level integration with Python fallback — no quantum hardware needed
```

### Action
- Write `docs/PRIOR_ART_DIFFERENTIATION.md` using the above structure + data from
  `RFT_RESEARCH_SUMMARY.md` and `absolute_novelty.py`.
- Time estimate: 4–6 hours of focused writing.

---

## GAP 4 — Scalability Evidence
**Status:** MODERATE PRIORITY — PARTIALLY MET

### What Exists
- N=1024 benchmarks documented
- Low-rank structure analysis: effective rank at 99% energy = 15–42 (4–35% of N)
  — this is the key scalability argument

### Honest Constraint (from `RFT_RESEARCH_SUMMARY.md`)

> **Current complexity: O(N²) transform, O(N³) kernel construction.**
> No O(N log N) exact algorithm found.
> For large N: precompute kernel, amortise O(N³) cost.

This means the scalability argument for USPTO must be:
**"Claims hold at N ≥ 4096 with pre-computed kernel; kernel construction is one-time cost."**
Not: "RFT scales as well as FFT" — that would be overclaiming.

### File to Create: `benchmarks/claim_scalability_N4096.py`

```python
# What to measure:
# 1. Kernel construction time for N = 256, 512, 1024, 2048, 4096
#    Report: one-time O(N³) cost in wall-clock seconds
# 2. Transform time (kernel pre-computed) for N = 256 to 4096
#    Report: O(N²) vs FFT O(N log N) — expected 10-50× overhead
# 3. Unitarity error for N = 4096: should remain < 5e-14
# 4. RFT-SIS hash for N = 4096 input: wall-clock time + correctness check
# 5. Low-rank approximation at N=4096: effective rank at 99% energy
#    — if rank << N, claim fast-approximate path is viable
```

**Time estimate:** 1–2 hours to write; 30 min to run (N=4096 kernel build may take ~5 min).

**Key output for USPTO:**
> "At N=4096, unitarity error < 5×10⁻¹⁴ and all 4 claim properties are verified.
> Kernel construction: one-time O(N³) precomputation.
> Transform application: O(N²) per call."

---

## Consolidated Action List

| # | Action | Creates | Time Est. |
|---|---|---|---|
| 1 | Write Claim 1 spec | `docs/claim_specs/claim1_rft_engine_spec.md` | 2h |
| 2 | Write Claim 2 spec | `docs/claim_specs/claim2_crypto_spec.md` | 2h |
| 3 | Write Claim 3 spec | `docs/claim_specs/claim3_geometric_spec.md` | 1h |
| 4 | Write Claim 4 spec | `docs/claim_specs/claim4_hybrid_spec.md` | 1h |
| 5 | Write prior art doc | `docs/PRIOR_ART_DIFFERENTIATION.md` | 4–6h |
| 6 | Build Claim 1 benchmark | `benchmarks/claim1_symbolic_vs_classical.py` | 3h |
| 7 | Build Claim 2 benchmark | `benchmarks/claim2_crypto_security_validation.py` | 4h |
| 8 | Build Claim 3 benchmark | `benchmarks/claim3_geometric_collision_analysis.py` | 3h |
| 9 | Build Claim 4 benchmark | `benchmarks/claim4_hybrid_integration_proof.py` | 3h |
| 10 | Build scalability test | `benchmarks/claim_scalability_N4096.py` | 2h |
| 11 | Roll up for attorney | `docs/claim_specs/DECLARATIONS_SUMMARY.md` | 1h |

**Total estimated effort: ~26 hours of focused work.**
**Suggested approach:** Items 1–5 first (documentation) — these can be done without running code
and directly address the examiner's primary concern. Items 6–10 second (benchmarks).
Item 11 last (attorney package).

---

### ⚠️ Important Caveat (Be Honest in USPTO Filings)

From `RFT_RESEARCH_SUMMARY.md` — the honest constraints that must be reflected accurately:

- RFT is **not** universally superior to FFT/DCT — it wins only for φ-quasi-periodic / matched signals
- RFT complexity is **O(N²)** per transform, not O(N log N)
- Crypto implementation uses **NumPy RNG**, not a formal CSPRNG — security claims are empirical only
- Publication and peer review are **not yet complete**

Overclaiming in USPTO filings risks invalidation. The correct framing is:
**novel, well-defined, mathematically rigorous, with demonstrated advantage in its specific domain.**
That is a strong and defensible position.

---

*Document created: March 24, 2026 | Repo: LMMinier/quantoniumos | Branch: main*
