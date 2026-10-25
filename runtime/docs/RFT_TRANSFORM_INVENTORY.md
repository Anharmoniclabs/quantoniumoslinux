# RFT Transform Inventory (Full-Stack)

**Purpose:** single place to locate **every transform/basis** in this repository that is part of the RFT stack (canonical RFT, variants, fast/approximate transforms, hybrid transforms, native kernels, and hardware validation artifacts).

**Generated:** 2026-03-20

> Naming note: this repo contains multiple historical definitions. The *canonical* RFT (square, unitary at finite N) is the Gram/Löwdin-normalized phi-grid exponential basis. Some older “waveform synthesis” code paths and operator-eigenbasis variants are preserved for backward compatibility and experiments.

---

## 1) Canonical (Square) Unitary RFT

This is the mathematically “absolute” RFT used for proofs/tests: a **square N×N basis** folded into the discrete-time fundamental band and made **unitary** via Gram normalization.

**Core definition and API**
- `algorithms/rft/core/resonant_fourier_transform.py`:
  - `rft_basis_matrix(N, T=None, use_gram_normalization=True)`
    - If `T == N`: builds the square phi-grid exponential basis and (optionally) Gram-normalizes it to unitary.
    - If `T != N`: returns the legacy waveform synthesis basis (non-square).
  - `rft_forward_square(x, ...)` / `rft_inverse_square(y, ...)` (canonical N→N forward/inverse)
  - `rft_matrix_canonical(n, ...)`, `rft_unitary_error_canonical(n, ...)`, `rft_phase_vectors_canonical(n, ...)`
- `algorithms/rft/core/canonical_true_rft.py`:
  - `class CanonicalTrueRFT`: the “no fallback” pure-Python reference operator (builds the canonical Gram-normalized basis and applies it).

**Frame-correct (non-unitary) mode**
- `algorithms/rft/core/resonant_fourier_transform.py`:
  - `rft_forward_frame(x, Phi)` uses dual-frame coefficients `(Phi^H Phi)^{-1} Phi^H x`
  - `rft_inverse_frame(X, Phi)` synthesizes `Phi @ X`

---

## 2) Legacy Waveform “RFT” (Non-Square Synthesis Mode)

These paths treat the basis as a **bank of oscillators** to synthesize a longer waveform (N×T), and invert by correlation. This is explicitly labeled *not the canonical RFT* in the canonical module docs.

- `algorithms/rft/core/resonant_fourier_transform.py`:
  - `rft_forward(x, T=...)` with `T != N` → waveform synthesis (`x @ Psi`)
  - `rft_inverse(W, N=..., ...)` with `T != N` → correlation-style inverse
- `algorithms/rft/core/rft_phi_legacy.py`:
  - older locked legacy API: `rft_forward`, `rft_inverse`, `rft_basis_matrix`, etc.

---

## 3) Binary/Wave-Domain Transform (BPSK Carrier Bank)

This is a transform-like encoding/decoding system for binary symbols using RFT carriers.

- `algorithms/rft/core/resonant_fourier_transform.py`:
  - `class BinaryRFT` (`encode`, `decode`, and wave-domain ops implemented on carrier correlations)

---

## 4) Variant Catalog (Unitary Matrices / Bases)

This repo has **two major “variant” families**:
1) **Matrix-generator variants** that directly produce an N×N unitary basis (typically via QR orthonormalization).
2) **Operator-eigenbasis variants** that define a Toeplitz “resonance operator” K and use its eigenvectors as the basis.

### 4.1 Variant manifest (single source of truth for runtime ordering)

- `algorithms/rft/variants/manifest.py` (`VARIANT_MANIFEST`)

Each row below lists: `CODE` → `registry_key` → native kernel enum (if any) → generator.

**Matrix-generator variants (registry-backed)**
- `STANDARD` → `original` → `RFT_VARIANT_STANDARD` → `algorithms.rft.variants.registry.generate_original_phi_rft`
- `HARMONIC` → `harmonic_phase` → `RFT_VARIANT_HARMONIC` → `...generate_harmonic_phase`
- `FIBONACCI` → `fibonacci_tilt` → `RFT_VARIANT_FIBONACCI` → `...generate_fibonacci_tilt`
- `CHAOTIC` → `chaotic_mix` → `RFT_VARIANT_CHAOTIC` → `...generate_chaotic_mix`
- `GEOMETRIC` → `geometric_lattice` → `RFT_VARIANT_GEOMETRIC` → `...generate_geometric_lattice`
- `PHI_CHAOTIC` → `phi_chaotic_hybrid` → `RFT_VARIANT_PHI_CHAOTIC` → `...generate_phi_chaotic_hybrid`
- `HYPERBOLIC` → `hyperbolic_phase` → `RFT_VARIANT_HYPERBOLIC` → `...generate_hyperbolic_phase`
- `DCT` → `dct` → `RFT_VARIANT_DCT` → `...generate_dct_basis`
- `HYBRID_DCT` → `hybrid_dct` → `RFT_VARIANT_HYBRID_DCT` → `...generate_hybrid_dct_rft`
- `LOG_PERIODIC` → `log_periodic` → (no kernel enum) → `...generate_log_periodic_phi_rft`
- `CONVEX_MIX` → `convex_mix` → (no kernel enum) → `...generate_convex_mixed_phi_rft`
- `GOLDEN_EXACT` → `golden_ratio_exact` → (no kernel enum) → `...generate_exact_golden_ratio_unitary`
- `CASCADE` → `h3_cascade` → `RFT_VARIANT_CASCADE` → `...generate_h3_hierarchical_cascade`
- `ADAPTIVE_SPLIT` → `adaptive_split` → `RFT_VARIANT_ADAPTIVE_SPLIT` → `...generate_adaptive_split_variant`
- `ENTROPY_GUIDED` → `fh5_entropy` → `RFT_VARIANT_ENTROPY_GUIDED` → `...generate_fh5_entropy_guided`
- `DICTIONARY` → `h6_dictionary` → `RFT_VARIANT_DICTIONARY` → `...generate_h6_dictionary_learning`
- `NOISE_SHRINK_MANIFOLD` → `noise_shrink_manifold` → (no kernel enum) → `...generate_noise_shrink_manifold`
- `ROBUST_MANIFOLD_2D` → `robust_manifold_2d` → (no kernel enum) → `...generate_robust_manifold_2d`

**Operator-eigenbasis variants (manifested as entries with lambda generators)**
- `OP_RFT_GOLDEN` → `op_rft_golden` → (no kernel enum) → `algorithms.rft.variants.operator_variants.generate_rft_golden`
- `OP_RFT_FIBONACCI` → `op_rft_fibonacci` → (no kernel enum) → `...generate_rft_fibonacci`
- `OP_RFT_HARMONIC` → `op_rft_harmonic` → (no kernel enum) → `...generate_rft_harmonic`
- `OP_RFT_GEOMETRIC` → `op_rft_geometric` → (no kernel enum) → `...generate_rft_geometric`
- `OP_RFT_BEATING` → `op_rft_beating` → (no kernel enum) → `...generate_rft_beating`
- `OP_RFT_PHYLLOTAXIS` → `op_rft_phyllotaxis` → (no kernel enum) → `...generate_rft_phyllotaxis`
- `OP_RFT_CASCADE` → `op_rft_cascade` → (no kernel enum) → `...generate_rft_cascade_h3`
- `OP_RFT_HYBRID_DCT` → `op_rft_hybrid_dct` → (no kernel enum) → `...generate_rft_hybrid_dct`

### 4.2 Operator-based variant generators (explicit functions)

- `algorithms/rft/variants/operator_variants.py`:
  - `generate_rft_golden`
  - `generate_rft_fibonacci`
  - `generate_rft_harmonic`
  - `generate_rft_geometric`
  - `generate_rft_beating`
  - `generate_rft_phyllotaxis`
  - `generate_rft_cascade_h3`
  - `generate_rft_adaptive`
  - `generate_rft_hybrid_dct`

### 4.3 Patent-aligned operator variants (explicit functions)

- `algorithms/rft/variants/patent_variants.py`:
  - `generate_rft_polar_golden`
  - `generate_rft_spiral_golden`
  - `generate_rft_loxodrome`
  - `generate_rft_complex_exp`
  - `generate_rft_exp_decay_golden`
  - `generate_rft_mobius`
  - `generate_rft_winding`
  - `generate_rft_torus_parametric`
  - `generate_rft_sphere_parametric`
  - `generate_rft_klein`
  - `generate_rft_phase_coherent`
  - `generate_rft_kuramoto`
  - `generate_rft_entropy_modulated`
  - `generate_rft_bloom_hash`
  - `generate_rft_manifold_projection`
  - `generate_rft_hopf_fibration`
  - `generate_rft_hybrid_phase`
  - `generate_rft_recursive_modulation`
  - `generate_rft_trefoil_knot`

---

## 5) Fast / Approximate / Structured Transforms

These are “engineering” transforms intended to approximate or accelerate RFT-like behavior.

- `algorithms/rft/fast/fast_rft_structured.py`: `class FastRFT`
- `algorithms/rft/fast/lowrank_rft.py`: `class LowRankRFT`
- `algorithms/rft/core/fibonacci_fast_rft.py`: Fibonacci fast transform result container (`class FibonacciRFTResult`) and related fast routines
- `algorithms/rft/fast/fast_rft_exploration.py`: `class FastApproximateRFT`, `class RefinedFastRFT` (exploration/prototyping)

---

## 6) Hybrid Codecs / Transform Cascades (Compression-Facing)

These couple RFT with DCT-like structure transforms or routing policies.

- `algorithms/rft/hybrids/legacy_mca.py`:
  - `rft_forward`, `rft_inverse` used by hybrid codec experiments
- `algorithms/rft/hybrids/rft_hybrid_codec.py`: `class RFTHybridCodec`
- `algorithms/rft/hybrids/h3_arft_cascade.py`: `class H3ARFTCascade`
- `algorithms/rft/compression/rft_vertex_codec.py`: `class RFTVertex`, `class RFTVertexCodec`
- `algorithms/rft/compression/rft_quantum_sim.py`: `class RFTCompressedSimulator`

---

## 7) Crypto-Facing Transforms (Research-Only)

These modules implement hash/cipher-like systems that depend on RFT variants/bases.

- `algorithms/rft/crypto/rft_sis_hash.py`: `class RFTSISHash`
- `algorithms/rft/crypto/enhanced_cipher.py`: `class EnhancedRFTCryptoV2`
- `algorithms/rft/crypto/primitives/quantum_prng.py`: `class RFTEnhancedHMAC`, `class RFTEnhancedFeistel`
- Native crypto engine interfaces are under `algorithms/rft/kernels/engines/crypto/**`

---

## 8) Native Kernel Layer (C / ASM) + Python Bindings

This is the “full-stack” transform implementation used by native acceleration paths.

**Kernel API and variant enum**
- `algorithms/rft/kernels/include/rft_kernel.h`:
  - `rft_forward`, `rft_inverse`, `rft_init_with_variant`, `rft_set_variant`, etc.
  - `RFT_VARIANT_*` enum (STANDARD, HARMONIC, FIBONACCI, CHAOTIC, GEOMETRIC, PHI_CHAOTIC, HYPERBOLIC, DCT, HYBRID_DCT, CASCADE, ADAPTIVE_SPLIT, ENTROPY_GUIDED, DICTIONARY, CANONICAL, BINARY_WAVE, CRYPTO_SIS)

**Kernel implementations**
- `algorithms/rft/kernels/kernel/rft_kernel.c` (variant dispatch + engine)
- `algorithms/rft/kernels/kernel/rft_kernel_fixed.c` (fixed-point/alternate path)
- ASM kernels referenced from the native build system (see `src/rftmw_native/README.md` and CMake):
  - `algorithms/rft/kernels/kernel/rft_kernel_asm.asm`
  - (plus additional crypto ASM, orchestrator ASM, etc.)

**Python bindings**
- `algorithms/rft/kernels/python_bindings/unitary_rft.py`: `class UnitaryRFT`, `class RFTProcessor` (+ ctypes layout mirroring native structs)
- `algorithms/rft/kernels/python_bindings/optimized_rft.py`: `class OptimizedRFT`, `class EnhancedRFTProcessor`
- `algorithms/rft/kernels/python_bindings/vertex_quantum_rft.py`: `class VertexQuantumRFT`

---

## 9) Native C++ Engine (RFTMW)

This layer wraps kernels and provides convenience forward/inverse entry points.

- `src/rftmw_native/rftmw_core.hpp`:
  - `rftmw::rft_forward` / `rftmw::rft_inverse` (canonical)
  - `rftmw::rft_forward_hybrid` / `rftmw::rft_inverse_hybrid` (FFT approximation)
- `src/rftmw_native/rftmw_asm_kernels.hpp`: C++ wrapper around `rft_kernel.h`
- `src/rftmw_native/rftmw_python.cpp`: Python extension bindings for the native engine
- `src/rftmw_native/rftmw_transform_theorems.hpp`: native-side theorem/validation helpers

---

## 10) Hardware / Verification Artifacts

- `hardware/theorem8_concentration_test.py`: generates test vectors + tables for Theorem 8-style concentration checks

---

## 11) “Where to Look First” (Quick Index)

If you only open a few files to understand “every transform,” start here:
- Canonical definition + forward/inverse: `algorithms/rft/core/resonant_fourier_transform.py`
- Pure reference operator (no fallbacks): `algorithms/rft/core/canonical_true_rft.py`
- Variant registry + manifest (names, generators, ordering): `algorithms/rft/variants/registry.py`, `algorithms/rft/variants/manifest.py`
- Operator/patent variant families: `algorithms/rft/variants/operator_variants.py`, `algorithms/rft/variants/patent_variants.py`
- Native kernel enum + API: `algorithms/rft/kernels/include/rft_kernel.h`
- Python bindings to native kernels: `algorithms/rft/kernels/python_bindings/unitary_rft.py`
- Native C++ engine entry points: `src/rftmw_native/rftmw_core.hpp`

---

## 12) Suggested “show me every transform exists” commands

Run a targeted set of tests/validations:
- `python -m pytest tests/rft -q`
- `python scripts/validation/verify_variant_claims.py`
- `python scripts/validation/verify_scaling_laws.py`

