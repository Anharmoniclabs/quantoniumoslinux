# Qwen3.5-MoE Engine Fixes & Bottlenecks

## Overview

This document details the critical bugs discovered and fixed in the RFTMW NumPy inference engine when running the **Qwen3.5-MoE 35B hybrid architecture** (qwen35-opus-distil.rftmwpk). The model uses a novel design: **GatedDeltaNet (SSM)** + **Full GQA Attention** in a 3:1 interleaved pattern.

---

## Architecture Summary

| Component | Value |
|-----------|-------|
| Layers | 40 |
| Hidden Dim | 2048 |
| Attention Heads | 16 query, 2 KV (GQA) |
| Head Dim | 256 |
| RoPE Dim | 64 (partial) |
| RoPE Theta | 10,000,000 |
| MRoPE Sections | [11, 11, 10, 0] |
| Layer Pattern | 3× DeltaNet + 1× Full Attention (repeating) |
| Full Attention Layers | 3, 7, 11, 15, 19, 23, 27, 31, 35, 39 |
| MoE Config | 256 experts, 8 active per token |
| Expert FFN Dim | 512 (per expert) |

---

## Bug #1: Wrong RoPE Pairing (CRITICAL)

### Symptom
- Attention layers produced **12.6 norm output** instead of expected ~1.5
- Model output was "Jedi Salad" — random unrelated tokens like "RoundedRectangleBorder DRIVER"

### Root Cause
The RoPE implementation used **split-half pairing**:
```python
# WRONG: pairs (dim[0:32], dim[32:64])
x0 = x[..., :d2]
x1 = x[..., d2:]
out[..., :d2] = x0 * cos - x1 * sin
out[..., d2:] = x0 * sin + x1 * cos
```

But Qwen3.5 uses **interleaved pairing**:
```python
# CORRECT: pairs (dim[0], dim[1]), (dim[2], dim[3]), etc.
x_even = x[..., 0::2]  # dims 0, 2, 4, ...
x_odd = x[..., 1::2]   # dims 1, 3, 5, ...
out[..., 0::2] = x_even * cos - x_odd * sin
out[..., 1::2] = x_even * sin + x_odd * cos
```

### Fix Location
- **Python**: `_apply_rope()` and partial RoPE in `_attention_forward()` 
- **C++**: `apply_rope()` in `rftmw_inference.hpp`

### Files Modified
- `quantonium_os_src/engine/rftmw_numpy_engine.py` (lines ~1095-1115, ~1455-1475)
- `src/rftmw_native/rftmw_inference.hpp` (lines ~589-615)

---

## Bug #2: Q-Projection Gate Not Properly Applied

### Symptom
- Q projection shape is (8192, 2048) — double expected (4096, 2048)
- Missing sigmoid gate caused 2× energy into residual stream

### Root Cause
Qwen3.5 fuses Q with an attention gate: `torch.chunk(q_proj_output, 2, dim=-1)`
- First half: Query states
- Second half: Gate (sigmoid-applied to attention output)

### Fix
```python
qg_flat = qkv[:n_heads * hd * 2]  # (8192,)
query = qg_flat[:n_heads * hd].reshape(n_heads, hd)  # (16, 256)
gate = qg_flat[n_heads * hd:]  # (4096,)

# After attention computation:
gate_sig = 1.0 / (1.0 + np.exp(-np.clip(gate, -88, 88)))
attn_flat = attn_flat * gate_sig  # Apply sigmoid gate
```

### Files Modified
- `quantonium_os_src/engine/rftmw_numpy_engine.py` (`_attention_forward()`)

---

## Bug #3: MoE Resonance Gate (Removed)

### Symptom  
- MoE output was artificially zeroing experts
- Re-normalization caused unstable expert weights

### Root Cause
A "Sigma-Threshold Gate" was applying resonance filtering that wasn't in the original Qwen3.5 architecture.

### Fix
Removed the resonance gate entirely. Standard Top-K routing with softmax normalization.

### Files Modified
- `quantonium_os_src/engine/rftmw_numpy_engine.py` (`MoEFFN.forward()`)

---

## Bug #4: Missing Shared Expert Gate

### Symptom
- Shared expert contribution not properly gated

### Root Cause
GGUF contains `ffn_gate_inp_shexp.weight` — a sigmoid gate that scales the shared expert's contribution.

### Fix
```python
if self.shared_expert_gate is not None:
    gate_scalar = float(sigmoid(self.shared_expert_gate @ x))
    shared_out = shared_out * gate_scalar
```

### Files Modified
- `quantonium_os_src/engine/rftmw_numpy_engine.py` (`MoEFFN.forward()`)

---

## Bug #5: Config Override Order

### Symptom
- `rope_dim=0` causing full RoPE instead of partial (64 dims)

### Root Cause
Sidecar JSON config was overriding GGUF config with incomplete values.

### Fix
GGUF config takes precedence when values are non-None:
```python
if gguf_cfg is not None:
    for field in ['rope_dim', 'rope_theta', ...]:
        val = getattr(gguf_cfg, field, None)
        if val is not None:
            setattr(self.config, field, val)
```

### Files Modified
- `quantonium_os_src/engine/rftmw_numpy_engine.py` (`Qwen25Config.from_gguf()`, `__init__()`)

---

## Performance: C++ Native Kernels

### Added C++ Fast Paths

| Kernel | Description | Speedup |
|--------|-------------|---------|
| `deltanet_step()` | Full GatedDeltaNet recurrence (180 lines) | ~10× |
| `gqa_attention()` | Grouped Query Attention | ~5× |
| `apply_rope()` | Interleaved RoPE (fixed) | ~3× |
| `rms_norm()` | RMS normalization | ~2× |

### Build
```bash
cd src/rftmw_native/build_linux
cmake .. && make -j4
```

---

## Remaining Investigation

### Attention Output Still Large
Even with interleaved RoPE, L3 attention output norm is ~12.6. Possible causes:
1. **MRoPE Sections**: `[11, 11, 10, 0]` may require section-aware frequency interleaving
2. **attention_scaling**: HuggingFace code shows `cos * attention_scaling` — may need scaling factor
3. **Per-head Q/K norm weights**: Mean ~1.3, may be amplifying values

### Next Steps
1. Implement full MRoPE with dimension sections
2. Add attention scaling from GGUF metadata
3. Verify V projection norm isn't exploding through attention

---

## Test Commands

```bash
# Quick 35B test
python3 -c "
from quantonium_os_src.engine.rftmw_numpy_engine import Qwen25Engine
e = Qwen25Engine('qwen35-opus-distil.rftmwpk')
e.load()
print(e.generate('What is 2+2?', max_tokens=10))
"

# Verify 3B regression
python3 -c "
from quantonium_os_src.engine.rftmw_numpy_engine import Qwen25Engine
e = Qwen25Engine('qwen2.5-coder_3b.rftmwpk')
e.load()
print(e.generate('What is 2+2?', max_tokens=20))
"
# Expected: "2+2 equals 4"
```

---

## File Change Summary

| File | Changes |
|------|---------|
| `rftmw_numpy_engine.py` | RoPE interleaving, Q-gate split, MoE fixes, config order |
| `rftmw_inference.hpp` | C++ interleaved RoPE, deltanet_step kernel |
| `rftmw_python.cpp` | Python bindings for new kernels |

---

*Last Updated: April 13, 2026*
