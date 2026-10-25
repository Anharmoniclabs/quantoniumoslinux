# quantoniumos Inference Architecture

> **See also:** [ENGINEER_MANUAL.md §1.1](../ENGINEER_MANUAL.md#11-ai-inference-architecture--performance--memory) for the executive summary of speed, memory, and new capabilities.

## How This Differs from Ollama

### Executive Summary

| Aspect | **Ollama** | **quantoniumos** |
|--------|-----------|------------------|
| **Core Runtime** | llama.cpp (C++) | Pure-NumPy + C++ kernels hybrid |
| **Memory Model** | GGUF loaded to RAM/VRAM | GGUF-on-disk + lazy tensor loading |
| **Quantization** | GGUF native (Q4_K, Q6_K, etc.) | GGUF native + RFTMW compression layer |
| **GPU Support** | CUDA, Metal, ROCm | CPU only (AVX2/AVX512 SIMD) |
| **MoE Support** | llama.cpp's expert routing | Custom NumPy MoE with top-k routing |
| **Tensor Format** | GGUF blobs | `.rftmwpk` packs with GGUF fallback |
| **Dependency** | Self-contained binary | Python + NumPy + optional C++ kernels |
| **License** | MIT | AGPL-3.0 / Patent-pending (Claims NC) |

### Architecture Philosophy

**Ollama** is a polished, production-ready LLM serving platform optimized for:
- Zero-config local deployment
- GPU acceleration via llama.cpp backends
- Model registry and management
- REST API for chat completion

**quantoniumos** is a research-oriented inference framework optimized for:
- Understanding transformer internals at the tensor level
- Experimenting with novel compression (RFTMW / φ-basis)
- Pure-NumPy reference implementation for verification
- Hybrid CPU inference with SIMD-accelerated hot paths

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         quantoniumos Inference Stack                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       Application Layer                              │   │
│  │  • CLI: rftmw_numpy_engine.py --prompt "..."                        │   │
│  │  • UI: Streamlit dashboard (src/dashboard/)                         │   │
│  │  • API: HuggingFace-compatible wrapper (rftmw_inference.py)         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Qwen25Engine (Pure Python)                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │   │
│  │  │   Tokenizer │  │   Config    │  │     Layer Management        │  │   │
│  │  │ (tiktoken)  │  │ (Qwen25Cfg) │  │ Dense / MoE / Hybrid        │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │                    Forward Pass Pipeline                        │ │   │
│  │  │  embed → [layer₀..layerₙ] → output_norm → lm_head → sample     │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                 ┌──────────────────┼──────────────────┐                    │
│                 ▼                  ▼                  ▼                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐ │
│  │   QuantWeight    │  │  QuantWeight3D   │  │   Native C++ Kernels     │ │
│  │  (2D Q4K/Q6K)    │  │ (3D MoE experts) │  │  (src/rftmw_native/)     │ │
│  │                  │  │                  │  │                          │ │
│  │ • raw uint8 blob │  │ • n_experts x    │  │ • q4k_matmul()           │ │
│  │ • lazy dequant   │  │   out x bytes    │  │ • q6k_matmul()           │ │
│  │ • .matmul(x)     │  │ • .matmul_expert │  │ • rms_norm()             │ │
│  │ • optional dense │  │   (idx, x)       │  │ • silu_mul()             │ │
│  └──────────────────┘  └──────────────────┘  │ • apply_rope()           │ │
│                                              │ • gqa_attention()        │ │
│                                              │ • softmax()              │ │
│                                              └──────────────────────────┘ │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Weight Storage Layer                          │   │
│  │                                                                      │   │
│  │  ┌────────────────────┐           ┌────────────────────────────────┐│   │
│  │  │  .rftmwpk Pack     │           │  GGUF Source (Fallback)        ││   │
│  │  │  • INT8_ZLIB       │ ──────▶   │  • Direct tensor mmap          ││   │
│  │  │  • RFT compressed  │  (lossy)  │  • Q4_K / Q6_K / F16 / F32     ││   │
│  │  │  • .meta.json      │           │  • Full precision              ││   │
│  │  └────────────────────┘           └────────────────────────────────┘│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Deep Dive

### 1. Weight Loading Pipeline

```
.rftmwpk file
     │
     ├──▶ Read header (JSON index)
     │         • weight names, offsets, compression method
     │         • original GGUF path reference
     │
     ├──▶ Check for GGUF source
     │         • .meta.json contains provenance.gguf_path
     │         • If GGUF exists → load tensors directly (lossless)
     │         • If not → decompress from pack (INT8_ZLIB lossy)
     │
     └──▶ Build QuantWeight objects
               • Store raw uint8 bytes (no dequantization)
               • Compute in/out features from shape
               • Detect quant type (Q4_K: 144 bytes/block, Q6_K: 210 bytes/block)
```

**File Formats:**

| Format | Magic | Purpose |
|--------|-------|---------|
| `.rftmwpk` | `RFTMWPK1` | Compressed model pack with JSON index |
| `.rftmwpk.meta.json` | — | Sidecar with provenance, config, shapes |
| GGUF | `GGUF` | llama.cpp native tensor format |
| `.rftb` | `RFTB` | RFT binary compressed tensor chunk |

### 2. QuantWeight Classes

```python
class QuantWeight:
    """2D quantized weight matrix (rows × bytes_per_row)"""
    raw: np.ndarray          # uint8 GGUF bytes, NOT dequantized
    in_features: int         # input dimension (K)
    out_features: int        # output dimension (M)
    quant_type: str          # 'q4k' | 'q6k' | 'f32' | 'f16'
    dense: Optional[np.ndarray]  # FP32 materialized (for preload_dense)
    
    def matmul(self, x: np.ndarray) -> np.ndarray:
        """Compute x @ W^T with on-the-fly dequantization"""
        if self.dense is not None:
            return self.dense @ x  # Use pre-materialized FP32
        
        # Try native C++ kernel first
        if _native_infer and self.quant_type == 'q6k':
            return _native_infer.q6k_matmul(self.raw, x, rows, bytes_per_row)
        
        # NumPy fallback: dequant block-by-block
        blocks = self.raw.reshape(-1, block_size)
        W = _dequant_q6k_blocks(blocks).reshape(rows, in_features)
        return W @ x
```

```python
class QuantWeight3D:
    """3D expert weight tensor (n_experts × out × bytes_per_row)"""
    raw: np.ndarray          # shape: (n_experts, rows_per_expert, bytes_per_row)
    n_experts: int
    out_features: int
    in_features: int
    
    def matmul_expert(self, x: np.ndarray, expert_idx: int) -> np.ndarray:
        """Compute x @ W[expert_idx]^T"""
        expert_raw = self.raw[expert_idx]  # Select expert slice
        # Uses same q6k_matmul kernel, just on expert's rows
        return _native_infer.q6k_matmul(expert_raw, x, ...)
```

### 3. Transformer Layer Types

#### Dense Layer (TransformerLayer)
Standard Qwen2.5 architecture:
```
input ──▶ attn_norm ──▶ Q,K,V projections ──▶ RoPE ──▶ GQA ──▶ o_proj ──▶ (+residual)
                                                                              │
       ◀── down_proj ◀── SiLU×gate_proj×up_proj ◀── ffn_norm ◀────────────────┘
```

#### MoE Layer (MoETransformerLayer)
Qwen3.5 MoE architecture with fused QKV and top-k expert routing:
```
input ──▶ attn_norm ──▶ fused_qkv ──▶ split(Q,K,V) ──▶ RoPE ──▶ GQA
                                                               │
      ┌── attn_gate (sigmoid) ◀───────────────────────────────┘
      │         │
      │         ▼
      │    ssm_out projection ──▶ (+residual)
      │                                │
      │    ┌───────────────────────────┘
      │    ▼
      │  post_attn_norm ──▶ MoE Router ──▶ top-k expert selection
      │                           │
      │         ┌─────────────────┼─────────────────┐
      │         ▼                 ▼                 ▼
      │    expert_0          expert_1    ...   expert_255
      │    (gate,up,down)    (gate,up,down)    (gate,up,down)
      │         │                 │                 │
      │         └─────────────────┼─────────────────┘
      │                           ▼
      │                  weighted sum + shared_expert
      │                           │
      └───────────────────────────┘
                    ▼
             (+residual) ──▶ output
```

### 4. Native C++ Kernels

Located in `src/rftmw_native/`, providing 5-10x speedup over NumPy:

| Kernel | Function | with AVX2 | without AVX2 |
|--------|----------|-----------|---------------|
| `q4k_matmul` | Q4_K dequant + matrix-vector | SIMD | Scalar C++ |
| `q6k_matmul` | Q6_K dequant + matrix-vector | SIMD | Scalar C++ |
| `rms_norm` | RMS normalization | SIMD | Scalar C++ |
| `silu_mul` | SiLU(x) * y fused | SIMD | Scalar C++ |
| `apply_rope` | Rotary position embedding | SIMD | Scalar C++ |
| `gqa_attention` | Grouped-query attention | SIMD | Scalar C++ |
| `softmax` | Numerically stable softmax | SIMD | Scalar C++ |

**Fallback Behavior:**

```
┌────────────────────────────────────────────────────────────────────┐
│                     Kernel Selection Cascade                        │
├────────────────────────────────────────────────────────────────────┤
│  1. Native C++ with AVX2    ──▶  fastest (x86-64 with AVX2)        │
│         │ if unavailable                                            │
│         ▼                                                           │
│  2. Native C++ scalar       ──▶  works on any x86-64               │
│         │ if module fails to load                                   │
│         ▼                                                           │
│  3. Pure NumPy (Python)     ──▶  works everywhere, 5-10x slower    │
└────────────────────────────────────────────────────────────────────┘
```

**Platform Notes:**
- **x86-64 with AVX2** (most modern Intel/AMD): Full SIMD acceleration
- **x86-64 without AVX2** (pre-2013 CPUs): C++ scalar fallback, ~2-3x slower than SIMD
- **ARM64 (Apple Silicon, Raspberry Pi)**: Pure NumPy fallback (no native kernels)
- **Windows**: Requires MinGW runtime DLLs in PATH

**Environment Variables:**
- `RFTMW_NO_NATIVE=1`: Force NumPy fallback (useful for debugging)
- `RFTMW_REQUIRE_NATIVE=1`: Fail hard if C++ kernels unavailable

**Build:**
```bash
cd src/rftmw_native
mkdir build_linux && cd build_linux
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 5. Generation Pipeline

```python
def generate_stream(prompt, max_tokens, temperature, top_k):
    # 1. Tokenize
    input_ids = encode(prompt)
    
    # 2. Initialize KV caches
    k_caches = [zeros((max_seq, num_kv_heads, head_dim)) for _ in range(num_layers)]
    v_caches = [zeros((max_seq, num_kv_heads, head_dim)) for _ in range(num_layers)]
    
    # 3. Prefill: process all prompt tokens
    for pos, token_id in enumerate(input_ids):
        h = embed_token(token_id)
        for layer_idx, layer in enumerate(layers):
            h = layer.forward(h, pos, rope_freqs, k_caches[layer_idx], v_caches[layer_idx], pos)
        h = rms_norm(h, output_norm)
    
    # 4. Decode: generate new tokens autoregressively
    for step in range(max_tokens):
        logits = lm_head.matmul(h)
        next_id = sample(logits, temperature, top_k)
        
        if next_id in stop_tokens:
            break
        
        yield decode([next_id])
        
        # Forward pass for next token
        h = embed_token(next_id)
        for layer_idx, layer in enumerate(layers):
            h = layer.forward(h, pos, rope_freqs, k_caches[layer_idx], v_caches[layer_idx], pos)
        h = rms_norm(h, output_norm)
        pos += 1
```

---

## RFTMW Compression (Research Layer)

The `.rftmwpk` format adds an experimental compression layer on top of GGUF:

### Compression Methods

| Method | ID | Description | Use Case |
|--------|-----|-------------|----------|
| `INT8_ZLIB` | 1 | Group-wise INT8 quantization + zlib | Most weights |
| `RFT` | 2 | Resonant Fourier Transform (φ-basis) | FFN norm tensors (experimental) |
| `NONE` | 0 | Raw float32 passthrough | Attention biases |

### Actual Compression Performance

> ⚠️ **Honest Assessment**: The `.rftmwpk` format does NOT achieve meaningful compression over raw GGUF.

| Component | Technique | Compression vs GGUF | Notes |
|-----------|-----------|---------------------|-------|
| Q4_K/Q6_K tensors | INT8_ZLIB | ~1.0x (break-even) | Already quantized; zlib on uint8 is ~lossless |
| F32 biases | NONE | 1.0x | Passthrough |
| FFN norms | RFT (optional) | **Expands data** | See below |

**Why the pack exists**: The `.rftmwpk` format is a **research artifact**, not a production compression format. It exists to:
1. Provide a self-contained model distribution format
2. Enable experimentation with alternative tensor compression
3. Store provenance metadata for reproducibility

**The GGUF fallback is why inference works**: When a `.meta.json` sidecar points to the original GGUF, the engine loads tensors directly from GGUF at full precision. The pack's INT8_ZLIB weights are lossy and only used when the GGUF is unavailable.

### RFT (Resonant Fourier Transform) — Experimental

The φ-basis transform uses golden-ratio frequency spacing:

$$
\Phi_{n,k} = \frac{1}{\sqrt{N}} \exp\left(j 2\pi \text{frac}((k+1)\phi) n\right)
$$

Where $\phi = \frac{1+\sqrt{5}}{2} \approx 1.618$ (golden ratio).

**Properties:**
- Unitary after Löwdin orthonormalization: $\widetilde{\Phi}^\dagger \widetilde{\Phi} = I$
- Better energy concentration than DFT for *quasi-periodic signals* (e.g., bearing vibration)
- Empirical RIP constant $\delta_8 = 0.1603 < 0.2929$ (Candès-Tao threshold). See [Candès & Tao, "Decoding by Linear Programming", IEEE Trans. Info. Theory, 2005](https://doi.org/10.1109/TIT.2005.858979).

**RFT for model weights — status: UNPROVEN**

> The hypothesis that neural network weights exhibit φ-sparse structure is **under investigation**. Current evidence:
> - No published study shows Qwen (or any LLM) weights are naturally φ-sparse.
> - Internal testing shows RFT **expands** general data (>8 BPP vs zlib's ~8 BPP). See `docs/LIMITATIONS_AND_REVIEWER_CONCERNS.md` §Q11.
> - The RFT codec is useful for signal analysis research, not production compression.

**If you came here expecting compression wins**: Use GGUF directly with llama.cpp or Ollama. The RFTMW layer adds overhead, not savings.

### Pack File Structure

```
┌──────────────────────────────────────────┐
│ RFTMWPK1 (8 bytes magic)                 │
├──────────────────────────────────────────┤
│ header_length (8 bytes, uint64 LE)       │
├──────────────────────────────────────────┤
│ JSON Header                              │
│   {                                      │
│     "weights": [                         │
│       {"name": "...", "offset": N,       │
│        "length": M, "method": "...",     │
│        "original_shape": [...]}          │
│     ]                                    │
│   }                                      │
├──────────────────────────────────────────┤
│ Compressed Weight Data                   │
│   • INT8_ZLIB blobs                      │
│   • RFT binary chunks                    │
│   • Raw floats                           │
└──────────────────────────────────────────┘
```

---

## Performance Characteristics

### Benchmarks (Qwen2.5-Coder 3B, CPU only)

| Metric | Value |
|--------|-------|
| Model load time | ~30-60s (36 layers) |
| Time to first token (prefill) | ~1-4s (depends on prompt length) |
| Decode throughput | ~2 tok/s (AVX2 kernels) |
| Memory footprint | ~2.5 GB (Q6_K quantization) |

### Comparison with Ollama

| Operation | Ollama (CPU) | quantoniumos |
|-----------|--------------|--------------|
| Load 3B model | ~5s | ~30s |
| Decode tok/s | ~10 | ~2 |
| Memory | ~2 GB | ~2.5 GB |
| GPU acceleration | ✓ | ✗ |

**Note:** quantoniumos prioritizes correctness and research flexibility over raw speed. For production inference, use Ollama or vLLM.

---

## MoE-Specific Features

### Expert Routing

```python
class MoEFFN:
    def forward(self, x):
        # 1. Compute router logits
        logits = router.matmul(x)  # (n_experts,)
        
        # 2. Top-k selection with softmax normalization
        top_k_idx = argpartition(-logits, k)[:k]
        weights = softmax(logits[top_k_idx])
        
        # 3. Weighted expert computation
        out = zeros(hidden_size)
        for i in range(k):
            expert_idx = top_k_idx[i]
            # SwiGLU: silu(gate) * up, then down
            gate = gate_exps.matmul_expert(x, expert_idx)
            up = up_exps.matmul_expert(x, expert_idx)
            hidden = silu(gate) * up
            expert_out = down_exps.matmul_expert(hidden, expert_idx)
            out += weights[i] * expert_out
        
        # 4. Add shared expert (always active)
        if has_shared_expert:
            out += shared_expert_forward(x)
        
        return out
```

### Attention Gating (Hybrid Attention-SSM)

For models like Qwen3.5-A3.5B that combine attention with SSM blocks:

```python
# After GQA attention
attn_flat = attention_output.reshape(-1)

# Apply gating (sigmoid) before output projection
if attn_gate is not None:
    gate = sigmoid(attn_gate.matmul(normed))  # (attn_dim,)
    attn_flat = attn_flat * gate

# Project to hidden size
output = ssm_out.matmul(attn_flat)
```

---

## Extending the Engine

> ⚠️ **This section provides a minimal overview.** Adding new architectures involves many edge cases not covered here. For serious development, read the implementation of `MoETransformerLayer` as a worked example.

### Adding a New Model Architecture

#### Step 1: Understand the GGUF tensor naming

GGUF tensor names follow llama.cpp conventions. For a new architecture, first dump the tensor index:

```python
from rftmw_numpy_engine import _parse_gguf_index
index, _ = _parse_gguf_index('/path/to/model.gguf')
for name, (offset, size, dtype, shape) in sorted(index.items())[:20]:
    print(f'{name}: {shape} (type={dtype})')
```

Common patterns:
- `blk.{i}.attn_q.weight` → separate Q projection
- `blk.{i}.attn_qkv.weight` → fused QKV (Qwen3.5-style)
- `blk.{i}.ffn_gate_exps.weight` → MoE expert gates (3D tensor)

#### Step 2: Add config preset

```python
# In Qwen25Config._PRESETS:
_PRESETS = {
    # num_layers: (hidden, heads, kv_heads, ffn)
    40: (2048, 32, 16, 11008),  # Your new model
}

# For MoE models, also add to _MOE_PRESETS:
_MOE_PRESETS = {
    # (n_layers, hidden, n_experts): (heads, kv_heads, expert_dim, experts_per_tok)
    (40, 2048, 256): (32, 16, 512, 8),
}
```

#### Step 3: Implement layer class (if architecture differs)

If the new model has a different attention or FFN structure:

```python
class NewArchLayer:
    __slots__ = ('attn_norm', 'q_proj', ...)  # Use __slots__ for memory efficiency
    
    def __init__(self, config):
        self.config = config
        # Initialize weight slots to None
    
    def forward(self, x, pos, rope_freqs, k_cache, v_cache, cache_len):
        # Implement forward pass
        # Use _rms_norm(), _apply_rope(), _softmax() helpers
        # Check `if _ni is not None:` for native kernel path
        pass
```

#### Step 4: Add loading logic

In `Qwen25Engine.load()`, detect the architecture and load weights:

```python
# Detect architecture from tensor names
is_new_arch = any('new_arch_specific_tensor' in n for n in check_names)

if is_new_arch:
    layer = NewArchLayer(cfg)
    layer.attn_norm = _load_1d(f'blk.{i}.attn_norm.weight')
    # ... load other weights
```

#### Step 5: Validate against reference

```bash
# Run same prompt through Ollama and your engine, compare logits
ollama run your-model "Test prompt" --verbose 2>&1 | grep -A5 "logits"
python -c "from engine import ...; engine.generate('Test prompt')"  
```

### Adding Native Kernels

1. **Implement in C++** (`src/rftmw_native/rftmw_inference.hpp`):
   - Include scalar fallback for non-AVX2 builds
   - Use `#ifdef __AVX2__` guards around SIMD code

2. **Add pybind11 binding** (`src/rftmw_native/rftmw_python.cpp`):
   ```cpp
   m.def("new_kernel", &rftmw::inference::new_kernel);
   ```

3. **Call from Python** via `_get_native_infer()`:
   ```python
   _ni = _get_native_infer()
   if _ni is not None:
       result = _ni.new_kernel(args)
   else:
       result = numpy_fallback(args)  # Always provide fallback
   ```

### Common Pitfalls

| Issue | Symptom | Fix |
|-------|---------|-----|
| Wrong QKV split | Attention output NaN/garbage | Check `ssm_out.weight` shape for true Q dim |
| Missing gating | Hidden state explodes across layers | Look for `attn_gate` tensor |
| Transposed weights | Shape mismatch errors | GGUF stores `(out, in)`, not `(in, out)` |
| F32 tensor as quantized | Segfault in dequant | Check GGUF type_id; use `from_f32()` for type 0 |

---

## Files Reference

| File | Purpose |
|------|---------|
| `quantonium_os_src/engine/rftmw_numpy_engine.py` | Core inference engine |
| `src/rftmw_native/rftmw_inference.hpp` | C++ kernel implementations |
| `src/rftmw_native/rftmw_core.hpp` | RFT transform core |
| `src/rftmw_native/rftmw_compression.hpp` | RFTMW compression codec |
| `src/rftmw_native/rftmw_python.cpp` | pybind11 bindings |
| `algorithms/rft/core/resonant_fourier_transform.py` | Python RFT reference |
| `tools/rft_hybrid_compress.py` | CLI for model compression |

---

## License

- **AGPL-3.0**: Non-patent-claiming code
- **Research-Only NC**: Patent-practicing files (see `LICENSE-CLAIMS-NC.md`)
- **Patent Pending**: USPTO Application 19/169,399

© 2025-2026 Luis M. Minier
