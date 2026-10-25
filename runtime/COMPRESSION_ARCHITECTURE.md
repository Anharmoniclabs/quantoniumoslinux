# quantoniumos — Compression Architecture

> Canonical reference for the unified RFT compression pipeline.
> Generated 2026-03-26 as part of the compression reorganisation commit.

---

## Corrected File Paths

The table below supersedes any earlier documentation that listed wrong paths.

| Prior (incorrect) | Correct path |
|---|-----------|
| `algorithms/rft/hybrids/cascade_hybrids.py` | `algorithms/rft/hybrids/rft_hybrid_codec.py` |
| `algorithms/rft/RFT_RESEARCH_SUMMARY.md` (as `rftmw_memory.py` link) | `quantonium_os_src/engine/rftmw_memory.py` |
| `algorithms/rft/RFT_RESEARCH_SUMMARY.md` (as `class_c_compression.py` link) | `benchmarks/class_c_compression.py` |
| `algorithms/rft/compression/lossless/__init__.py` | `algorithms/rft/compression/lossless/rans_stream.py` |

---

## Layer Map

```
LAYER 0 — MATH FOUNDATION
  algorithms/rft/core/canonical_true_rft.py          O(N²) exact unitary basis
  algorithms/rft/core/resonant_fourier_transform.py  rft_basis_matrix(), Löwdin ortho

LAYER 1 — ANALYSIS TOOLS
  algorithms/rft/compression/entropy.py              R-D: quantizer + bitrate estimator
  algorithms/rft/unified_transform_scheduler.py      Backend selector (Python/C-ASM/C++)
  algorithms/rft/routing.py                          Codec variant selector (0–12)

LAYER 2 — CODEC PRIMITIVES
  algorithms/rft/compression/ans.py                  rANS entropy coder
  algorithms/rft/compression/lossless/rans_stream.py Streaming rANS

LAYER 3 — CODEC IMPLEMENTATIONS
  algorithms/rft/compression/rft_vertex_codec.py     Research codec (exact roundtrip)
  algorithms/rft/compression/rft_binary_codec.py     Production bitstream codec
  algorithms/rft/compression/rft_binary_pack.py      .rftb chunk serialiser
  algorithms/rft/compression/rft_quantum_sim.py      Amplitude vector compressor
  algorithms/rft/hybrids/rft_hybrid_codec.py         h3/fh5/h6 hybrid modes
  tools/rft_hybrid_compress.py                       CLI: MLP residual compress
  tools/rft_hybrid_decode.py                         CLI: MLP residual decode

LAYER 4 — ROUTING / MIDDLEWARE  ← ENTRY POINT: quantonium_os_src/engine/rft_compress.py
  quantonium_os_src/engine/rft_compress.py           NEW unified entry point
  quantonium_os_src/engine/rftmw_memory.py           Entropy-gated tensor router
  quantonium_os_src/engine/RFTMW.py                  Python↔C++ bridge
  quantonium_os_src/engine/rftmw_inference.py        HuggingFace LLM wrapper

LAYER 5 — NATIVE ACCELERATION
  src/rftmw_native/rftmw_core.hpp                    C++ RFT math (SIMD)
  src/rftmw_native/rftmw_compression.hpp             C++ codec pipeline
  src/rftmw_native/rftmw_python.cpp                  pybind11 bindings
  algorithms/rft/kernels/kernel/quantum_symbolic_compression.c   C kernel
  algorithms/rft/kernels/kernel/quantum_symbolic_compression.asm ASM kernel
```

---

## Two-Decision Architecture

Before any compression bytes are written, two independent routing decisions are made:

```
tensor.ravel()
    │
    ├─ DECISION 1  unified_transform_scheduler.py → select_backend(n)
    │       → Python (always) | C-ASM (N≤8) | C++ Native (AVX2, if built)
    │
    ├─ DECISION 2  routing.py → detect_signal_type() + select_best_variant()
    │       → 'smooth'→12 DICTIONARY | 'edges'→11 ENTROPY_GUIDED
    │         'general'/'quantum'→8 CASCADE | 'audio'→1 HARMONIC | ...
    │
    ├─ R-D PROBE   entropy.py → calculate_rd_point(RFT) vs estimate_bitrate(INT8)
    │       → pick whichever BPP is lower AND mse ≤ mse_budget
    │
    ├─ [RFT path]  RFTHybridCodec(n, variant) or RFTVertexCodec(n)
    │                → pack_container() → .rftb bytes
    │                → ANSCoder().encode()  (optional)
    │                → if mse > budget*0.5: _apply_hybrid_residual()
    │
    └─ [INT8 path] INT8 quantise → zlib.compress(level=9)

    All paths → RFTC-magic blob (decompress() reads path byte to route back)
```

---

## Routing Variant Registry (from routing.py)

| ID | Name | BPP | PSNR | Latency | Best for |
|----|------|-----|------|---------|----------|
| 0  | STANDARD | — | — | 0.80 ms | General compression |
| 1  | HARMONIC | — | — | 1.20 ms | Audio / harmonic signals |
| 2  | FIBONACCI | — | — | 1.10 ms | Lattice crypto |
| 3  | CHAOTIC | — | — | 1.50 ms | Diffusion layers |
| 4  | PRIME | — | — | 2.00 ms | Number theory |
| 5  | ADAPTIVE | — | — | 1.30 ms | Multi-scale |
| 6  | SYMBOLIC | — | — | 1.00 ms | Quantum state |
| 7  | LOGARITHMIC | — | — | 1.10 ms | Scale-invariant |
| **8** | **CASCADE** | **0.673** | **48.5 dB** | **0.57 ms** | **Best general (η=0)** |
| 9  | BRAIDED | — | — | 2.50 ms | Heterogeneous data |
| 10 | ADAPTIVE_SPLIT | — | — | 1.80 ms | Structure/texture |
| **11** | **ENTROPY_GUIDED** | **0.406** | **42.1 dB** | **7.00 ms** | **Sharp edges/steps** |
| **12** | **DICTIONARY** | **0.751** | **49.9 dB** | **3.00 ms** | **Smooth / high quality** |

---

## Import Fixes Applied in This Commit

### 1. `unified_transform_scheduler.py` — sys.path hack removed

Replaced:
```python
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'rftmw_native', 'build'))
import rftmw_native
```
With an `importlib.util.find_spec()` probe that fails gracefully.

### 2. `quantonium_os_src/engine/__init__.py` — exports added

`compress` and `decompress` from `rft_compress.py` are now exported at
package level so callers can use the one-liner:
```python
from quantonium_os_src.engine import compress, decompress
```

### Pending fixes (to be applied in follow-up PR)

- `rftmw_memory.py`: replace inline `_spectral_entropy()` with
  `from algorithms.rft.compression.entropy import estimate_bitrate`
- `rftmw_memory.py`: replace inline routing logic with
  `from algorithms.rft.routing import detect_signal_type, select_best_variant`
- `rftmw_inference.py`: replace direct codec calls with
  `from quantonium_os_src.engine.rft_compress import compress, decompress`
- `tools/rft_hybrid_compress.py`: add
  `from quantonium_os_src.engine.rftmw_memory import RFTMWMemoryLayer`

---

## Benchmark Validation Loop

```bash
# 1. Verify codec correctness
pytest tests/compression/test_compression_engineering.py -v

# 2. Measure R-D against baselines (gzip/lzma/zstd/brotli/lz4)
python benchmarks/class_c_compression.py

# 3. Find sweet spot per signal family
python benchmarks/rft_compression_sweet_spot.py

# 4. KV-cache boundary leakage (RFT vs RoPE geometric metric)
python benchmarks/kv_cache_boundary_error.py

# 5. Smoke-test the new unified entry point
python quantonium_os_src/engine/rft_compress.py

# 6. Print routing guide (variant IDs 0-12 with BPP/PSNR/latency)
python algorithms/rft/routing.py

# 7. Probe all backends + unitarity check
python algorithms/rft/unified_transform_scheduler.py
```

---

## .rftb Wire Format (rft_binary_pack.py)

```
Header (fixed, 57+ bytes):
  4 bytes   magic        "RFTB"
  2 bytes   version      uint16 = 1
  2 bytes   flags        bit0=lossy  bit1=assembly
  1 byte    dtype_len
  16 bytes  dtype        ASCII zero-padded
  1 byte    ndim
  32 bytes  shape[8]     uint32 each, zero-padded to 8 dims
  4 bytes   chunk_size   uint32
  4 bytes   n_chunks     uint32
  8 bytes   total_len    uint64

Per chunk (variable):
  8 bytes   offset       uint64
  4 bytes   seg_len      uint32
  4 bytes   rft_size     uint32
  1 byte    backend      0=python  1=assembly
  1 byte    codec_mode   0=lossless  1=pruned  2=quantized
  4 bytes   kept_count   uint32
  1 byte    amp_bits
  1 byte    phase_bits
  8 bytes   amp_max      float64
  4+N bytes mask         uint32 length + packbits
  4+N bytes idx          uint32 length + uint32 array
  4+N bytes amp_data     uint32 length + raw bytes
  4+N bytes phase_data   uint32 length + raw bytes
```

### Codec Modes

| Byte | Name | Stored |
|------|------|--------|
| 0 | lossless | float64 amp + phase arrays + index list |
| 1 | pruned | packbits mask + quantised uint amp/phase |
| 2 | quantized | pruned + explicit amp_max for dequant scaling |
