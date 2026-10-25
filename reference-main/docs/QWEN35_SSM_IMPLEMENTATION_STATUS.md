# Qwen3.5 MoE SSM Implementation Status

## Summary

The Qwen35MoeEngine NumPy implementation produces **garbage output** due to SSM layer explosion. The first SSM layer amplifies the residual stream by **62x**, causing cascading instability.

## Root Cause Analysis

### Traced Values (Layer 0 SSM)

| Stage | Norm | Notes |
|-------|------|-------|
| Input `h` | 0.37 | Embedding for token 1 |
| After RMSNorm | 48.8 | Expected (sqrt(2048) scale) |
| `qk` projection | 194.3 | attn_qkv output [8192] |
| `v_flat` (gate) | 54.6 | attn_gate output with SiLU |
| `conv_out_silu` | 7.3 | After depthwise conv1d |
| `y_heads` pre-norm | 12.3 | SSM outer-product output |
| `y_heads` post-norm | **30.7** | **2.5x amplification!** |
| `ssm_output` | **22.99** | Added to h(0.37) = 62x ratio |

### The Bug

The `ssm_norm` application amplifies instead of normalizing:
```python
rms = np.sqrt(np.mean(y_heads ** 2, axis=1, keepdims=True) + eps)
y_normed = (y_heads / rms) * layer.ssm_norm  # <-- amplifies 2.5x
```

After all 40 layers, final `h` norm reaches **172** (should be ~10-20).

## What's Broken

### 1. NumPy SSM (`qwen35_moe_engine.py`)
- Outer-product state accumulation may be wrong
- RMSNorm + scale is amplifying not stabilizing  
- Missing output scaling / residual gating

### 2. C++ SSM Stub (`rftmw_inference.hpp:1044-1163`)
- **NOT EXPOSED TO PYTHON** - `has_ssm=true` is a lie
- Dimensional mismatch: state is `[32]` scalars but output reads `[128]`
- B computed from raw `x`, not conv output
- Comment says "simplified - real impl would have full projection"

### 3. Missing ASM Kernels
The following C++ kernels exist and work:
- `q4k_matmul` / `q6k_matmul` ✓
- `rms_norm`, `softmax`, `silu`, `silu_mul` ✓
- `apply_rope`, `gqa_attention` ✓
- `moe_topk_route` ✓

**Missing:**
- `ssm_scan` / `mamba_ssd_scan`
- `ssm_conv1d`
- `ssm_outer_product_state`

## Correct Qwen3.5 SSM Architecture (from GGUF tensors)

```
Tensors per SSM layer:
  attn_qkv.weight    [2048, 8192]  # Projects to Q[4096] + K[4096]
  attn_gate.weight   [2048, 4096]  # V values (NOT a gate!)
  ssm_a              [32]          # log(A) per head
  ssm_alpha.weight   [2048, 32]    # dt projection
  ssm_beta.weight    [2048, 32]    # B gate projection
  ssm_conv1d.weight  [4, 8192]     # Causal depthwise conv
  ssm_dt.bias        [32]          # dt bias per head
  ssm_norm.weight    [128]         # Per-dim output scale
  ssm_out.weight     [4096, 2048]  # Output projection

Config:
  ssm_time_step_rank = 32    # n_heads
  ssm_state_size     = 128   # head_dim  
  ssm_conv_kernel    = 4
  ssm_inner_size     = 4096  # 32 * 128
```

### Correct Forward Pass (Mamba-2 SSD)

```
1. normed = RMSNorm(h, attn_norm)
2. qk = matmul(normed, attn_qkv)           # [8192]
3. v_flat = SiLU(matmul(normed, attn_gate)) # [4096] - these are VALUES not gate
4. conv_out = Conv1D(qk, ssm_conv1d)        # causal with buffer
5. conv_act = SiLU(conv_out)
6. Q, K = split(conv_act, [4096, 4096])
7. Q_norm = L2Normalize(Q.reshape(32,128), per_head)
8. K_norm = L2Normalize(K.reshape(32,128), per_head)
9. V = v_flat.reshape(32, 128)

10. dt = softplus(matmul(normed, ssm_alpha) + dt_bias)  # [32]
11. decay = exp(-exp(ssm_a) * dt)                        # [32]
12. beta_g = sigmoid(matmul(normed, ssm_beta))           # [32]
13. B_bar = dt * beta_g                                  # [32]

14. For each head hi:
      S[hi] = decay[hi] * S[hi] + B_bar[hi] * outer(K_norm[hi], V[hi])
      y[hi] = Q_norm[hi] @ S[hi]                         # [128]

15. y_normed = RMSNorm(y, ssm_norm)         # per-head norm
16. output = matmul(y_normed.flat, ssm_out) # [2048]
17. h = h + output                          # residual
```

### Key Differences from Current Implementation

1. **V is the gate output** - not a multiplicative gate
2. **State is outer-product** - `S[hi] ∈ R^{128×128}`, not diagonal
3. **Output should be scaled** - missing residual scale factor
4. **B_bar = dt * sigmoid(beta)** - fixed in current code
5. **Partial RoPE** - fixed (only 64/256 dims rotated)

## Recommended Fix

1. **Short term**: Route Qwen3.5 to Ollama backend (already done)
2. **Long term**: Implement C++ `ssm_mamba2_scan` kernel with:
   - Outer-product state accumulation
   - Fused conv1d + SiLU
   - Proper output scaling
   - AVX2/AVX-512 vectorization

## Test Command

```bash
ollama run gag0/qwen35-opus-distil "What is 2+2?" --verbose
# Works correctly via llama.cpp native SSM
```

---
*Generated: 2026-04-13*
