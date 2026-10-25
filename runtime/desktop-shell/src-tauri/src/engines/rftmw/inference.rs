// SPDX-License-Identifier: LicenseRef-SPX-Proprietary
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
//
// inference.rs — Port of rftmw_inference.hpp
//
// Quantized neural network inference kernels:
// - Q4_K / Q6_K dequantization
// - Dense/quantized GEMV
// - RMS norm, SiLU, SwiGLU, Softmax
// - RoPE, GQA attention
// - SSM selective scan, GatedDeltaNet

use rayon::prelude::*;

// ── Quantization: FP16 ──────────────────────────────────────────────────────

pub fn fp16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;

    if exp == 0 {
        // Subnormal
        let val = f32::from_bits((sign << 31) | 0) + (mant as f32) * (1.0 / (1 << 24) as f32);
        if sign == 1 { -val.abs() } else { val }
    } else if exp == 31 {
        // Inf / NaN
        f32::from_bits((sign << 31) | 0x7F800000 | (mant << 13))
    } else {
        // Normalized
        let new_exp = (exp as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (new_exp << 23) | (mant << 13))
    }
}

// ── Q4_K Dequantization ─────────────────────────────────────────────────────
// Block: 144 bytes → 256 f32 values
// Layout: 2 bytes d(FP16), 2 bytes dmin(FP16), 12 bytes sub-scales, 128 bytes q4 data

const Q4K_BLOCK_SIZE: usize = 256;
const Q4K_BYTES: usize = 144;

pub fn dequant_q4k_block(raw: &[u8], out: &mut [f32; Q4K_BLOCK_SIZE]) {
    let d = fp16_to_f32(u16::from_le_bytes([raw[0], raw[1]]));
    let dmin = fp16_to_f32(u16::from_le_bytes([raw[2], raw[3]]));
    let scales = &raw[4..16];   // 12 bytes of 6-bit scales
    let qs = &raw[16..144];     // 128 bytes of 4-bit data

    for j in 0..Q4K_BLOCK_SIZE {
        let sub_block = j / 32;
        // 6-bit scale extraction (packed in 12 bytes for 8 sub-blocks)
        let sc_byte_idx = (sub_block * 3) / 2;
        let sc_nibble = if sub_block % 2 == 0 {
            scales.get(sc_byte_idx).copied().unwrap_or(0) & 0x3F
        } else {
            ((scales.get(sc_byte_idx).copied().unwrap_or(0) >> 6)
                | (scales.get(sc_byte_idx + 1).copied().unwrap_or(0) << 2))
                & 0x3F
        };

        let byte_idx = j / 2;
        let nibble = if j % 2 == 0 {
            qs.get(byte_idx).copied().unwrap_or(0) & 0x0F
        } else {
            qs.get(byte_idx).copied().unwrap_or(0) >> 4
        };

        out[j] = d * sc_nibble as f32 * nibble as f32 - dmin * sc_nibble as f32;
    }
}

// ── Q6_K Dequantization ─────────────────────────────────────────────────────
// Block: 210 bytes → 256 f32 values

const Q6K_BLOCK_SIZE: usize = 256;
const Q6K_BYTES: usize = 210;

pub fn dequant_q6k_block(raw: &[u8], out: &mut [f32; Q6K_BLOCK_SIZE]) {
    let d = fp16_to_f32(u16::from_le_bytes([raw[0], raw[1]]));
    let ql = &raw[2..130];      // 128 bytes: low 4 bits
    let qh = &raw[130..194];    // 64 bytes: high 2 bits
    let scales = &raw[194..210]; // 16 bytes: INT8 sub-block scales

    for j in 0..Q6K_BLOCK_SIZE {
        let lo = if j % 2 == 0 {
            ql.get(j / 2).copied().unwrap_or(0) & 0x0F
        } else {
            ql.get(j / 2).copied().unwrap_or(0) >> 4
        };

        let hi_byte = qh.get(j / 4).copied().unwrap_or(0);
        let hi_shift = (j % 4) * 2;
        let hi = (hi_byte >> hi_shift) & 0x03;

        let q_val = (lo | (hi << 4)) as i8 - 32;
        let sub_block = j / 16;
        let sc = scales.get(sub_block).copied().unwrap_or(0) as i8;
        out[j] = d * sc as f32 * q_val as f32;
    }
}

// ── Dense GEMV ──────────────────────────────────────────────────────────────

pub fn dense_sgemv_f32(weights: &[f32], x: &[f32], y: &mut [f32], rows: usize, cols: usize) {
    assert!(weights.len() >= rows * cols);
    for row in 0..rows {
        let mut sum = 0.0_f32;
        let w_row = &weights[row * cols..(row + 1) * cols];
        for (w, xi) in w_row.iter().zip(x.iter()) {
            sum += w * xi;
        }
        y[row] = sum;
    }
}

// ── Q4K / Q6K Fused Matmul ─────────────────────────────────────────────────

pub fn q4k_matmul_vec(raw_weights: &[u8], x: &[f32], y: &mut [f32], out_features: usize, bytes_per_row: usize) {
    let blocks_per_row = bytes_per_row / Q4K_BYTES;
    for row in 0..out_features {
        let row_data = &raw_weights[row * bytes_per_row..];
        let mut sum = 0.0_f32;
        let mut col = 0;
        for b in 0..blocks_per_row {
            let block_data = &row_data[b * Q4K_BYTES..];
            let mut vals = [0.0_f32; Q4K_BLOCK_SIZE];
            dequant_q4k_block(block_data, &mut vals);
            for v in &vals {
                if col < x.len() {
                    sum += v * x[col];
                    col += 1;
                }
            }
        }
        y[row] = sum;
    }
}

pub fn q6k_matmul_vec(raw_weights: &[u8], x: &[f32], y: &mut [f32], out_features: usize, bytes_per_row: usize) {
    let blocks_per_row = bytes_per_row / Q6K_BYTES;
    for row in 0..out_features {
        let row_data = &raw_weights[row * bytes_per_row..];
        let mut sum = 0.0_f32;
        let mut col = 0;
        for b in 0..blocks_per_row {
            let block_data = &row_data[b * Q6K_BYTES..];
            let mut vals = [0.0_f32; Q6K_BLOCK_SIZE];
            dequant_q6k_block(block_data, &mut vals);
            for v in &vals {
                if col < x.len() {
                    sum += v * x[col];
                    col += 1;
                }
            }
        }
        y[row] = sum;
    }
}

// ── Activation Functions ────────────────────────────────────────────────────

pub fn rms_norm(x: &[f32], weight: &[f32], out: &mut [f32], eps: f32) {
    let size = x.len();
    let sq_sum: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sq_sum / size as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    for i in 0..size {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

pub fn softmax(x: &[f32], out: &mut [f32]) {
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for (i, &v) in x.iter().enumerate() {
        let e = (v - max_val).exp();
        out[i] = e;
        sum += e;
    }
    let inv_sum = 1.0 / sum;
    for v in out.iter_mut() {
        *v *= inv_sum;
    }
}

pub fn silu(x: &[f32], out: &mut [f32]) {
    for (i, &v) in x.iter().enumerate() {
        let clamped = v.clamp(-88.0, 88.0);
        out[i] = clamped / (1.0 + (-clamped).exp());
    }
}

/// Fused SwiGLU: silu(gate) * up
pub fn silu_mul(gate: &[f32], up: &[f32], out: &mut [f32]) {
    for i in 0..gate.len() {
        let g = gate[i].clamp(-88.0, 88.0);
        let s = g / (1.0 + (-g).exp());
        out[i] = s * up[i];
    }
}

// ── Rotary Position Embedding (RoPE) ────────────────────────────────────────

pub fn apply_rope(
    x: &[f32],
    cos_f: &[f32],
    sin_f: &[f32],
    out: &mut [f32],
    num_heads: usize,
    head_dim: usize,
) {
    let half = head_dim / 2;
    for h in 0..num_heads {
        let base = h * head_dim;
        for i in 0..half {
            let x0 = x[base + i];
            let x1 = x[base + half + i];
            let c = cos_f[i];
            let s = sin_f[i];
            out[base + i] = x0 * c - x1 * s;
            out[base + half + i] = x0 * s + x1 * c;
        }
    }
}

// ── Grouped Query Attention ─────────────────────────────────────────────────

pub fn gqa_attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    attn_out: &mut [f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) {
    let repeat_factor = num_heads / num_kv_heads;
    let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();

    for h in 0..num_heads {
        let kv_h = h / repeat_factor;
        let q_offset = h * head_dim;

        // Q @ K^T → logits
        let mut logits = vec![0.0_f32; seq_len];
        for pos in 0..seq_len {
            let k_offset = (pos * num_kv_heads + kv_h) * head_dim;
            let mut dot = 0.0_f32;
            for d in 0..head_dim {
                dot += q[q_offset + d] * k_cache[k_offset + d];
            }
            logits[pos] = dot * inv_sqrt_d;
        }

        // Softmax
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0_f32;
        for v in logits.iter_mut() {
            *v = (*v - max_val).exp();
            exp_sum += *v;
        }
        let inv_sum = 1.0 / exp_sum;
        for v in logits.iter_mut() {
            *v *= inv_sum;
        }

        // Weighted sum of V
        let out_offset = h * head_dim;
        for d in 0..head_dim {
            let mut sum = 0.0_f32;
            for pos in 0..seq_len {
                let v_offset = (pos * num_kv_heads + kv_h) * head_dim;
                sum += logits[pos] * v_cache[v_offset + d];
            }
            attn_out[out_offset + d] = sum;
        }
    }
}

// ── SSM Selective Scan Step ─────────────────────────────────────────────────

pub fn softplus_stable(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + (-x.abs()).exp()).ln() + x.max(0.0)
    }
}

pub fn ssm_selective_scan_step(
    x: &[f32],
    dt: &[f32],
    a_param: &[f32],
    b_param: &[f32],
    c_param: &[f32],
    d_param: f32,
    state: &mut [f32],
    out: &mut [f32],
    dim: usize,
    state_dim: usize,
) {
    for i in 0..dim {
        let dt_i = softplus_stable(dt[i]);
        // State update: h = exp(-dt·A) * h + dt·B·x
        for s in 0..state_dim {
            let idx = i * state_dim + s;
            let decay = (-dt_i * a_param[s].abs()).exp();
            state[idx] = decay * state[idx] + dt_i * b_param[s] * x[i];
        }
        // Output: y = C @ h + D * x
        let mut y = d_param * x[i];
        for s in 0..state_dim {
            y += c_param[s] * state[i * state_dim + s];
        }
        out[i] = y;
    }
}

// ── GatedDeltaNet Step ──────────────────────────────────────────────────────

pub fn deltanet_step(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: &[f32],
    s_state: &mut [f32],
    out: &mut [f32],
    head_dim: usize,
    value_dim: usize,
) {
    // L2-normalize Q and K
    let q_norm: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    let k_norm: f32 = k.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    let q_hat: Vec<f32> = q.iter().map(|x| x / q_norm).collect();
    let k_hat: Vec<f32> = k.iter().map(|x| x / k_norm).collect();

    // Delta rule: S += beta * (k ⊗ v - k ⊗ (k^T @ S))
    // First compute k^T @ S for each value dimension
    for vd in 0..value_dim {
        let mut kt_s = 0.0_f32;
        for d in 0..head_dim {
            kt_s += k_hat[d] * s_state[d * value_dim + vd];
        }
        // Update S
        for d in 0..head_dim {
            let delta = beta[0] * k_hat[d] * (v[vd] - kt_s);
            s_state[d * value_dim + vd] += delta;
        }
    }

    // Output: o = q^T @ S
    for vd in 0..value_dim {
        let mut sum = 0.0_f32;
        for d in 0..head_dim {
            sum += q_hat[d] * s_state[d * value_dim + vd];
        }
        out[vd] = sum;
    }
}

// ── Layer Keep Ratio Schedule ───────────────────────────────────────────────

pub fn layer_keep_ratio(layer_idx: usize, total_layers: usize, base_keep: f32) -> f32 {
    let frac = layer_idx as f32 / total_layers.max(1) as f32;
    let multiplier = if frac < 0.15 {
        0.25 // Early layers: aggressive compression
    } else if frac < 0.85 {
        1.5  // Core layers: preserve more
    } else {
        0.5  // Late layers: moderate compression
    };
    (base_keep * multiplier).clamp(0.02, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp16_conversion() {
        // Test 1.0 in FP16 = 0x3C00
        assert!((fp16_to_f32(0x3C00) - 1.0).abs() < 1e-6);
        // Test 0.5 = 0x3800
        assert!((fp16_to_f32(0x3800) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_rms_norm() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let w = [1.0; 4];
        let mut out = [0.0; 4];
        rms_norm(&x, &w, &mut out, 1e-5);
        let norm: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        // After RMS norm, L2 norm should be close to sqrt(dim)
        assert!((norm - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_softmax() {
        let x = [1.0, 2.0, 3.0];
        let mut out = [0.0; 3];
        softmax(&x, &mut out);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(out[2] > out[1] && out[1] > out[0]);
    }

    #[test]
    fn test_layer_keep_ratio() {
        // Early layer
        assert!(layer_keep_ratio(0, 36, 0.6) < 0.2);
        // Core layer
        assert!(layer_keep_ratio(18, 36, 0.6) > 0.5);
        // Late layer
        assert!(layer_keep_ratio(35, 36, 0.6) < 0.5);
    }
}
