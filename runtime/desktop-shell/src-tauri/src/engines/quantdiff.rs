// SPDX-License-Identifier: LicenseRef-SPX-Proprietary
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
//
// quantdiff.rs — QuantoniumOS proprietary diffusion inference engine.
//
// Fully native Rust implementation — zero Python subprocess, zero HuggingFace:
//   - DDIM noise scheduler (scaled-linear beta, deterministic sampling)
//   - Custom lightweight U-Net (6.6M params, pixel-space, 64×64 native)
//       • Conv2d / ConvTranspose2d via parallel im2col
//       • GroupNorm, SiLU, Linear, Multi-head Self-Attention
//       • Time-embedding (sinusoidal → MLP)
//       • Prompt-conditioning (adaGN scale+shift)
//   - Weight loading from custom binary format (models/quantdiff/unet/)
//   - Warm-start: Resonart generates structured base → low-strength DDIM refine
//   - Full DDIM: pure noise → trained U-Net → novel output
//   - 4K LANCZOS upscale post-step
//
// Weight format (models/quantdiff/unet/):
//   manifest.json  — [{name, shape}] tensor list
//   weights/<name>.f32  — raw little-endian f32 bytes
//
// Python training exports with src.resonart.quantdiff.train --save-rust-weights

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use image::imageops::{resize, FilterType};
use image::{DynamicImage, ImageFormat, Rgba, RgbaImage};
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use super::resonart::{self, ResonartRenderRequest};

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct QuantDiffRequest {
    pub prompt: String,
    pub negative_prompt: String,
    pub steps: u32,
    pub style: String,
    pub width: u32,
    pub height: u32,
    pub quality: String,
    pub scene_hint: String,
    pub palette_hint: String,
    pub spectral_complexity: f64,
    pub seed: Option<i64>,
    pub output_dir: String,
}

#[derive(Debug, Clone)]
pub struct QuantDiffResult {
    pub path: String,
    pub prompt: String,
    pub style: String,
    pub seed: i64,
    pub width: u32,
    pub height: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// DDIM Scheduler
// ─────────────────────────────────────────────────────────────────────────────

/// Pure-Rust DDIM noise scheduler.  Scaled-linear beta schedule, ε-prediction,
/// eta=0 (fully deterministic).
struct DDIMScheduler {
    num_train: usize,
    alphas_cumprod: Vec<f32>,
    sqrt_ac: Vec<f32>,
    sqrt_one_minus_ac: Vec<f32>,
    timesteps: Vec<usize>,
    step_size: usize,
}

impl DDIMScheduler {
    fn new(num_train: usize) -> Self {
        // Scaled-linear beta schedule (SD default)
        let beta_start = 0.00085f32;
        let beta_end = 0.012f32;
        let betas: Vec<f32> = (0..num_train)
            .map(|i| {
                let t = i as f32 / (num_train - 1) as f32;
                let s = beta_start.sqrt() + t * (beta_end.sqrt() - beta_start.sqrt());
                s * s
            })
            .collect();

        let mut ac = Vec::with_capacity(num_train);
        let mut running = 1.0f32;
        for b in &betas {
            running *= 1.0 - b;
            ac.push(running);
        }

        let sqrt_ac: Vec<f32> = ac.iter().map(|a| a.sqrt()).collect();
        let sqrt_oac: Vec<f32> = ac.iter().map(|a| (1.0 - a).sqrt()).collect();

        Self {
            num_train,
            alphas_cumprod: ac,
            sqrt_ac,
            sqrt_one_minus_ac: sqrt_oac,
            timesteps: Vec::new(),
            step_size: 1,
        }
    }

    fn set_timesteps(&mut self, num_inference: usize) {
        let step = self.num_train / num_inference;
        self.step_size = step;
        // Descending: T-step, T-2*step, ..., 0
        self.timesteps = (0..num_inference)
            .map(|i| {
                let t = (num_inference - 1 - i) * step;
                t.min(self.num_train - 1)
            })
            .collect();
    }

    /// Forward process: add noise to x0 at timestep t.
    fn add_noise(&self, x0: &[f32], noise: &[f32], t: usize) -> Vec<f32> {
        let sa = self.sqrt_ac[t];
        let sb = self.sqrt_one_minus_ac[t];
        x0.iter().zip(noise).map(|(x, n)| sa * x + sb * n).collect()
    }

    /// DDIM reverse step: from x_t + model ε prediction → x_{t-prev}.
    fn step(&self, model_output: &[f32], t: usize, sample: &[f32]) -> Vec<f32> {
        let prev_t = t.saturating_sub(self.step_size);
        let alpha_t = self.alphas_cumprod[t];
        let alpha_prev = if prev_t == 0 { 1.0f32 } else { self.alphas_cumprod[prev_t] };
        let beta_t = 1.0 - alpha_t;

        // Predict x_0
        let sqrt_at = alpha_t.sqrt();
        let sqrt_bt = beta_t.sqrt();

        let mut prev = Vec::with_capacity(sample.len());
        for (i, &s) in sample.iter().enumerate() {
            let eps = model_output[i];
            // x_0 prediction
            let pred_x0 = ((s - sqrt_bt * eps) / sqrt_at).clamp(-1.0, 1.0);
            // DDIM direction (eta=0)
            let dir = (1.0 - alpha_prev).max(0.0).sqrt() * eps;
            prev.push(alpha_prev.sqrt() * pred_x0 + dir);
        }
        prev
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tensor helpers
// ─────────────────────────────────────────────────────────────────────────────

/// 4D tensor (N, C, H, W) backed by a flat Vec<f32>.
#[derive(Clone)]
struct T4 {
    data: Vec<f32>,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
}

impl T4 {
    fn zeros(n: usize, c: usize, h: usize, w: usize) -> Self {
        Self { data: vec![0.0; n * c * h * w], n, c, h, w }
    }

    fn from_data(data: Vec<f32>, n: usize, c: usize, h: usize, w: usize) -> Self {
        debug_assert_eq!(data.len(), n * c * h * w);
        Self { data, n, c, h, w }
    }

    #[inline]
    fn idx(&self, n: usize, c: usize, h: usize, w: usize) -> usize {
        n * (self.c * self.h * self.w) + c * (self.h * self.w) + h * self.w + w
    }

    #[inline]
    fn at(&self, n: usize, c: usize, h: usize, w: usize) -> f32 {
        self.data[self.idx(n, c, h, w)]
    }

    #[inline]
    fn set(&mut self, n: usize, c: usize, h: usize, w: usize, v: f32) {
        let i = self.idx(n, c, h, w);
        self.data[i] = v;
    }

    fn shape_str(&self) -> String {
        format!("({},{},{},{})", self.n, self.c, self.h, self.w)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Activation
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn apply_silu(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = silu(*x);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sinusoidal time embedding
// ─────────────────────────────────────────────────────────────────────────────

fn sinusoidal_embedding(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0.0f32; dim];
    for i in 0..half {
        let freq = (-10000.0f32.ln() * i as f32 / (half - 1).max(1) as f32).exp();
        let arg = t * freq;
        out[i] = arg.cos();
        out[i + half] = arg.sin();
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Linear layer
// ─────────────────────────────────────────────────────────────────────────────

/// y = x W^T + b     (B, in_f) → (B, out_f)
fn linear_op(input: &[f32], weight: &[f32], bias: &[f32], in_f: usize, out_f: usize) -> Vec<f32> {
    let b = input.len() / in_f;
    let mut out = vec![0.0f32; b * out_f];
    for bi in 0..b {
        for o in 0..out_f {
            let mut s = bias[o];
            for i in 0..in_f {
                s += input[bi * in_f + i] * weight[o * in_f + i];
            }
            out[bi * out_f + o] = s;
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Conv2d (parallel over out_ch × batch)
// ─────────────────────────────────────────────────────────────────────────────

/// Standard 2D convolution.  Weight layout: [out_c, in_c, kH, kW].
fn conv2d_op(
    input: &T4,
    weight: &[f32],
    bias: &[f32],
    out_c: usize,
    kh: usize,
    kw: usize,
    padding: usize,
    stride: usize,
) -> T4 {
    let (n, ic, ih, iw) = (input.n, input.c, input.h, input.w);
    let oh = (ih + 2 * padding - kh) / stride + 1;
    let ow = (iw + 2 * padding - kw) / stride + 1;
    let mut out = T4::zeros(n, out_c, oh, ow);

    // Parallel over (batch, out_channel) pairs
    let results: Vec<(usize, usize, Vec<f32>)> = (0..n)
        .flat_map(|b| (0..out_c).map(move |oc| (b, oc)))
        .collect::<Vec<_>>()
        .par_iter()
        .map(|&(b, oc)| {
            let mut patch = vec![0.0f32; oh * ow];
            let b_bias = bias[oc];
            for h_out in 0..oh {
                for w_out in 0..ow {
                    let mut s = b_bias;
                    for c in 0..ic {
                        for fh in 0..kh {
                            for fw in 0..kw {
                                let ih_src = h_out * stride + fh;
                                let iw_src = w_out * stride + fw;
                                // Wrapping sub handles padding boundary without signed int
                                let ih_real = ih_src.wrapping_sub(padding);
                                let iw_real = iw_src.wrapping_sub(padding);
                                if ih_real < ih && iw_real < iw {
                                    let in_i = b * ic * ih * iw + c * ih * iw + ih_real * iw + iw_real;
                                    let w_i = oc * ic * kh * kw + c * kh * kw + fh * kw + fw;
                                    s += input.data[in_i] * weight[w_i];
                                }
                            }
                        }
                    }
                    patch[h_out * ow + w_out] = s;
                }
            }
            (b, oc, patch)
        })
        .collect();

    for (b, oc, patch) in results {
        for (i, &v) in patch.iter().enumerate() {
            let h_out = i / ow;
            let w_out = i % ow;
            out.set(b, oc, h_out, w_out, v);
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// ConvTranspose2d (2× upsampling, stride=2)
// ─────────────────────────────────────────────────────────────────────────────

/// Transposed convolution for 2× upsampling.  Weight layout: [in_c, out_c, kH, kW].
fn conv_transpose2d_op(
    input: &T4,
    weight: &[f32],
    bias: &[f32],
    in_c: usize,
    out_c: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
) -> T4 {
    let (n, _ic, ih, iw) = (input.n, input.c, input.h, input.w);
    let oh = (ih - 1) * stride - 2 * padding + kh;
    let ow = (iw - 1) * stride - 2 * padding + kw;
    let mut out_data = vec![0.0f32; n * out_c * oh * ow];

    for b in 0..n {
        // Bias init
        for oc in 0..out_c {
            let b_val = bias[oc];
            for h in 0..oh {
                for w in 0..ow {
                    out_data[b * out_c * oh * ow + oc * oh * ow + h * ow + w] = b_val;
                }
            }
        }
        // Scatter
        for ic in 0..in_c {
            for h_in in 0..ih {
                for w_in in 0..iw {
                    let in_val = input.at(b, ic, h_in, w_in);
                    for fh in 0..kh {
                        for fw in 0..kw {
                            let h_out = h_in * stride + fh;
                            let w_out = w_in * stride + fw;
                            let h_out_pad = h_out.wrapping_sub(padding);
                            let w_out_pad = w_out.wrapping_sub(padding);
                            if h_out_pad < oh && w_out_pad < ow {
                                for oc in 0..out_c {
                                    // Weight: [in_c, out_c, kH, kW]
                                    let w_i = ic * out_c * kh * kw + oc * kh * kw + fh * kw + fw;
                                    let out_i = b * out_c * oh * ow + oc * oh * ow + h_out_pad * ow + w_out_pad;
                                    out_data[out_i] += in_val * weight[w_i];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    T4::from_data(out_data, n, out_c, oh, ow)
}

// ─────────────────────────────────────────────────────────────────────────────
// GroupNorm
// ─────────────────────────────────────────────────────────────────────────────

/// GroupNorm(num_groups, channels).  Weight/bias shape: [channels].
fn group_norm_op(input: &T4, weight: &[f32], bias: &[f32], num_groups: usize) -> T4 {
    let (n, c, h, w) = (input.n, input.c, input.h, input.w);
    let group_c = c / num_groups;
    let eps = 1e-5f32;
    let mut out = input.clone();

    for b in 0..n {
        for g in 0..num_groups {
            let c_start = g * group_c;
            let c_end = c_start + group_c;
            let count = group_c * h * w;

            // Mean
            let mut mean = 0.0f32;
            for ch in c_start..c_end {
                for hi in 0..h {
                    for wi in 0..w {
                        mean += input.at(b, ch, hi, wi);
                    }
                }
            }
            mean /= count as f32;

            // Variance
            let mut var = 0.0f32;
            for ch in c_start..c_end {
                for hi in 0..h {
                    for wi in 0..w {
                        let d = input.at(b, ch, hi, wi) - mean;
                        var += d * d;
                    }
                }
            }
            var /= count as f32;
            let inv_std = 1.0 / (var + eps).sqrt();

            // Normalize + scale + shift
            for ch in c_start..c_end {
                let scale = weight[ch];
                let shift = bias[ch];
                for hi in 0..h {
                    for wi in 0..w {
                        let normed = (input.at(b, ch, hi, wi) - mean) * inv_std;
                        out.set(b, ch, hi, wi, normed * scale + shift);
                    }
                }
            }
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Self-Attention
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-head self-attention over (B, C, H, W).
/// QKV weight: [3C, C, 1, 1] (no bias).  Proj weight: [C, C, 1, 1].
fn self_attention_op(
    input: &T4,
    norm_w: &[f32],
    norm_b: &[f32],
    qkv_w: &[f32],
    proj_w: &[f32],
    proj_b: &[f32],
    heads: usize,
) -> T4 {
    let (bn, c, h, w) = (input.n, input.c, input.h, input.w);
    let num_groups = (c / 2).min(32).max(1);
    // GroupNorm
    let normed = group_norm_op(input, norm_w, norm_b, num_groups);

    // QKV projection (1×1 conv)
    let qkv_out = conv2d_op(&normed, qkv_w, &vec![0.0f32; c * 3], c * 3, 1, 1, 0, 1);
    // qkv_out: (B, 3C, H, W)

    let hw = h * w;
    let head_dim = c / heads;
    let scale = (head_dim as f32).sqrt().recip();

    let mut out_data = vec![0.0f32; bn * c * hw];

    for b in 0..bn {
        // Slice Q, K, V: each (C, HW)
        let q: Vec<f32> = (0..c * hw).map(|i| {
            let ch = i / hw; let pos = i % hw;
            qkv_out.at(b, ch, pos / w, pos % w)
        }).collect();
        let k: Vec<f32> = (0..c * hw).map(|i| {
            let ch = i / hw; let pos = i % hw;
            qkv_out.at(b, c + ch, pos / w, pos % w)
        }).collect();
        let v: Vec<f32> = (0..c * hw).map(|i| {
            let ch = i / hw; let pos = i % hw;
            qkv_out.at(b, 2 * c + ch, pos / w, pos % w)
        }).collect();

        // Multi-head attention
        let mut out_b = vec![0.0f32; c * hw];
        for head in 0..heads {
            let hd_start = head * head_dim;
            // Attention scores: (head_dim, HW) × (head_dim, HW)^T → (HW, HW)
            let mut attn = vec![0.0f32; hw * hw];
            for qi in 0..hw {
                for ki in 0..hw {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[(hd_start + d) * hw + qi] * k[(hd_start + d) * hw + ki];
                    }
                    attn[qi * hw + ki] = dot * scale;
                }
            }
            // Softmax over last dim (keys)
            for qi in 0..hw {
                let row = &mut attn[qi * hw..(qi + 1) * hw];
                let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for x in row.iter_mut() {
                    *x = (*x - max).exp();
                    sum += *x;
                }
                for x in row.iter_mut() {
                    *x /= sum.max(1e-8);
                }
            }
            // Output: attn (HW, HW) × V (head_dim, HW)^T → (HW, head_dim)
            for qi in 0..hw {
                for d in 0..head_dim {
                    let mut s = 0.0f32;
                    for ki in 0..hw {
                        s += attn[qi * hw + ki] * v[(hd_start + d) * hw + ki];
                    }
                    out_b[(hd_start + d) * hw + qi] = s;
                }
            }
        }

        // Project (1×1 conv on out_b reshaped as T4)
        let out_t = T4::from_data(out_b, 1, c, h, w);
        let proj_out = conv2d_op(&out_t, proj_w, proj_b, c, 1, 1, 0, 1);

        // Residual
        for ch in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    let i = b * c * hw + ch * hw + hi * w + wi;
                    out_data[i] = input.at(b, ch, hi, wi) + proj_out.at(0, ch, hi, wi);
                }
            }
        }
    }
    T4::from_data(out_data, bn, c, h, w)
}

// ─────────────────────────────────────────────────────────────────────────────
// Weight store
// ─────────────────────────────────────────────────────────────────────────────

struct WeightStore(HashMap<String, Vec<f32>>);

impl WeightStore {
    fn get(&self, name: &str) -> Result<&[f32], String> {
        self.0.get(name).map(|v| v.as_slice()).ok_or_else(|| format!("Missing weight: {name}"))
    }

    fn get_or_zeros(&self, name: &str, size: usize) -> Vec<f32> {
        self.0.get(name).cloned().unwrap_or_else(|| vec![0.0; size])
    }

    fn get_or_ones(&self, name: &str, size: usize) -> Vec<f32> {
        self.0.get(name).cloned().unwrap_or_else(|| vec![1.0; size])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// U-Net layers (weight structs)
// ─────────────────────────────────────────────────────────────────────────────

struct LinearLayer {
    weight: Vec<f32>,
    bias: Vec<f32>,
    in_f: usize,
    out_f: usize,
}

impl LinearLayer {
    fn from_store(ws: &WeightStore, prefix: &str, in_f: usize, out_f: usize) -> Self {
        Self {
            weight: ws.get_or_zeros(&format!("{prefix}.weight"), out_f * in_f),
            bias: ws.get_or_zeros(&format!("{prefix}.bias"), out_f),
            in_f,
            out_f,
        }
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        linear_op(x, &self.weight, &self.bias, self.in_f, self.out_f)
    }
}

struct Conv2dLayer {
    weight: Vec<f32>,
    bias: Vec<f32>,
    out_c: usize,
    kh: usize,
    kw: usize,
    padding: usize,
    stride: usize,
}

impl Conv2dLayer {
    fn from_store(ws: &WeightStore, prefix: &str, in_c: usize, out_c: usize, k: usize, padding: usize, stride: usize) -> Self {
        Self {
            weight: ws.get_or_zeros(&format!("{prefix}.weight"), out_c * in_c * k * k),
            bias: ws.get_or_zeros(&format!("{prefix}.bias"), out_c),
            out_c,
            kh: k,
            kw: k,
            padding,
            stride,
        }
    }

    fn forward(&self, x: &T4) -> T4 {
        conv2d_op(x, &self.weight, &self.bias, self.out_c, self.kh, self.kw, self.padding, self.stride)
    }
}

struct ConvTranspose2dLayer {
    weight: Vec<f32>,
    bias: Vec<f32>,
    in_c: usize,
    out_c: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
}

impl ConvTranspose2dLayer {
    fn from_store(ws: &WeightStore, prefix: &str, in_c: usize, out_c: usize, k: usize, stride: usize, padding: usize) -> Self {
        Self {
            weight: ws.get_or_zeros(&format!("{prefix}.weight"), in_c * out_c * k * k),
            bias: ws.get_or_zeros(&format!("{prefix}.bias"), out_c),
            in_c,
            out_c,
            kh: k,
            kw: k,
            stride,
            padding,
        }
    }

    fn forward(&self, x: &T4) -> T4 {
        conv_transpose2d_op(x, &self.weight, &self.bias, self.in_c, self.out_c, self.kh, self.kw, self.stride, self.padding)
    }
}

struct GroupNormLayer {
    weight: Vec<f32>,
    bias: Vec<f32>,
    num_groups: usize,
}

impl GroupNormLayer {
    fn from_store(ws: &WeightStore, prefix: &str, channels: usize) -> Self {
        let num_groups = channels.min(32).max(1);
        Self {
            weight: ws.get_or_ones(&format!("{prefix}.weight"), channels),
            bias: ws.get_or_zeros(&format!("{prefix}.bias"), channels),
            num_groups,
        }
    }

    fn forward(&self, x: &T4) -> T4 {
        group_norm_op(x, &self.weight, &self.bias, self.num_groups)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ResBlock
// ─────────────────────────────────────────────────────────────────────────────

struct ResBlock {
    norm1: GroupNormLayer,
    conv1: Conv2dLayer,
    time_proj: LinearLayer,
    norm2: GroupNormLayer,
    conv2: Conv2dLayer,
    skip: Option<Conv2dLayer>, // None = Identity
    in_c: usize,
    out_c: usize,
    time_dim: usize,
}

impl ResBlock {
    fn from_store(ws: &WeightStore, prefix: &str, in_c: usize, out_c: usize, time_dim: usize) -> Self {
        let skip = if in_c != out_c {
            Some(Conv2dLayer::from_store(ws, &format!("{prefix}.skip"), in_c, out_c, 1, 0, 1))
        } else {
            None
        };
        Self {
            norm1: GroupNormLayer::from_store(ws, &format!("{prefix}.norm1"), in_c),
            conv1: Conv2dLayer::from_store(ws, &format!("{prefix}.conv1"), in_c, out_c, 3, 1, 1),
            time_proj: LinearLayer::from_store(ws, &format!("{prefix}.time_proj"), time_dim, out_c),
            norm2: GroupNormLayer::from_store(ws, &format!("{prefix}.norm2"), out_c),
            conv2: Conv2dLayer::from_store(ws, &format!("{prefix}.conv2"), out_c, out_c, 3, 1, 1),
            skip,
            in_c,
            out_c,
            time_dim,
        }
    }

    /// x: T4(B,in_c,H,W)   t_emb: Vec<f32> (B*time_dim)  → T4(B,out_c,H,W)
    fn forward(&self, x: &T4, t_emb: &[f32]) -> T4 {
        let b = x.n;
        // norm1 → silu → conv1
        let mut h = self.norm1.forward(x);
        for v in h.data.iter_mut() { *v = silu(*v); }
        let mut h = self.conv1.forward(&h);

        // Time conditioning: project t_emb → (B, out_c) → broadcast add
        let t_proj = self.time_proj.forward(t_emb); // (B * out_c)
        for bi in 0..b {
            for oc in 0..self.out_c {
                let t_val = silu(t_proj[bi * self.out_c + oc]);
                for hi in 0..h.h {
                    for wi in 0..h.w {
                        let i = h.idx(bi, oc, hi, wi);
                        h.data[i] += t_val;
                    }
                }
            }
        }

        // norm2 → silu → conv2
        let mut h = self.norm2.forward(&h);
        for v in h.data.iter_mut() { *v = silu(*v); }
        let h = self.conv2.forward(&h);

        // Skip connection
        let skip = match &self.skip {
            Some(conv) => conv.forward(x),
            None => x.clone(),
        };

        // Add
        let mut out = T4::zeros(b, self.out_c, h.h, h.w);
        for i in 0..out.data.len() {
            out.data[i] = h.data[i] + skip.data[i];
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CondBlock (adaGN conditioning)
// ─────────────────────────────────────────────────────────────────────────────

struct CondBlock {
    norm: GroupNormLayer,
    proj: LinearLayer,
    channels: usize,
    cond_dim: usize,
}

impl CondBlock {
    fn from_store(ws: &WeightStore, prefix: &str, channels: usize, cond_dim: usize) -> Self {
        Self {
            norm: GroupNormLayer::from_store(ws, &format!("{prefix}.norm"), channels),
            proj: LinearLayer::from_store(ws, &format!("{prefix}.proj"), cond_dim, channels * 2),
            channels,
            cond_dim,
        }
    }

    /// x: T4(B,C,H,W)   cond: Vec<f32> (B * cond_dim) → T4(B,C,H,W)
    fn forward(&self, x: &T4, cond: &[f32]) -> T4 {
        let h = self.norm.forward(x);
        let params = self.proj.forward(cond); // (B * 2C)
        let b = x.n;
        let mut out = T4::zeros(b, self.channels, h.h, h.w);
        for bi in 0..b {
            for ch in 0..self.channels {
                let scale = 1.0 + params[bi * self.channels * 2 + ch];
                let shift = params[bi * self.channels * 2 + self.channels + ch];
                for hi in 0..h.h {
                    for wi in 0..h.w {
                        let v = h.at(bi, ch, hi, wi) * scale + shift;
                        out.set(bi, ch, hi, wi, v);
                    }
                }
            }
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SelfAttentionLayer
// ─────────────────────────────────────────────────────────────────────────────

struct SelfAttentionLayer {
    norm_w: Vec<f32>,
    norm_b: Vec<f32>,
    qkv_w: Vec<f32>,
    proj_w: Vec<f32>,
    proj_b: Vec<f32>,
    channels: usize,
    heads: usize,
}

impl SelfAttentionLayer {
    fn from_store(ws: &WeightStore, prefix: &str, channels: usize, heads: usize) -> Self {
        let num_groups = channels.min(32).max(1);
        Self {
            norm_w: ws.get_or_ones(&format!("{prefix}.norm.weight"), channels),
            norm_b: ws.get_or_zeros(&format!("{prefix}.norm.bias"), channels),
            qkv_w: ws.get_or_zeros(&format!("{prefix}.qkv.weight"), channels * 3 * channels),
            proj_w: ws.get_or_zeros(&format!("{prefix}.proj.weight"), channels * channels),
            proj_b: ws.get_or_zeros(&format!("{prefix}.proj.bias"), channels),
            channels,
            heads,
        }
    }

    fn forward(&self, x: &T4) -> T4 {
        self_attention_op(x, &self.norm_w, &self.norm_b, &self.qkv_w, &self.proj_w, &self.proj_b, self.heads)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Time MLP
// ─────────────────────────────────────────────────────────────────────────────

struct TimeMlp {
    fc1: LinearLayer,
    fc2: LinearLayer,
    time_dim: usize,
}

impl TimeMlp {
    fn from_store(ws: &WeightStore, time_dim: usize) -> Self {
        Self {
            fc1: LinearLayer::from_store(ws, "time_mlp.0", time_dim, time_dim * 4),
            fc2: LinearLayer::from_store(ws, "time_mlp.2", time_dim * 4, time_dim),
            time_dim,
        }
    }

    fn forward(&self, t: f32) -> Vec<f32> {
        let emb = sinusoidal_embedding(t, self.time_dim);
        let mut h = self.fc1.forward(&emb);
        apply_silu(&mut h);
        self.fc2.forward(&h)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QuantDiff U-Net
// ─────────────────────────────────────────────────────────────────────────────

const BASE_CH: usize = 64;
const MID_CH: usize = 128;
const DEEP_CH: usize = 256;
const TIME_DIM: usize = 256;
const COND_DIM: usize = 128;

struct QuantDiffUNet {
    time_mlp: TimeMlp,
    init_conv: Conv2dLayer,
    // Encoder
    enc0: ResBlock,
    cond0: CondBlock,
    down0: Conv2dLayer,
    enc1: ResBlock,
    cond1: CondBlock,
    down1: Conv2dLayer,
    enc2: ResBlock,
    cond2: CondBlock,
    // Bottleneck
    mid1: ResBlock,
    mid_attn: SelfAttentionLayer,
    mid2: ResBlock,
    // Decoder
    dec2: ResBlock,
    up2: ConvTranspose2dLayer,
    dec1: ResBlock,
    up1: ConvTranspose2dLayer,
    dec0: ResBlock,
    // Output
    out_norm: GroupNormLayer,
    out_conv: Conv2dLayer,
}

impl QuantDiffUNet {
    fn from_store(ws: &WeightStore) -> Self {
        Self {
            time_mlp: TimeMlp::from_store(ws, TIME_DIM),
            init_conv: Conv2dLayer::from_store(ws, "init_conv", 3, BASE_CH, 3, 1, 1),
            enc0: ResBlock::from_store(ws, "enc0", BASE_CH, BASE_CH, TIME_DIM),
            cond0: CondBlock::from_store(ws, "cond0", BASE_CH, COND_DIM),
            down0: Conv2dLayer::from_store(ws, "down0", BASE_CH, BASE_CH, 4, 1, 2),
            enc1: ResBlock::from_store(ws, "enc1", BASE_CH, MID_CH, TIME_DIM),
            cond1: CondBlock::from_store(ws, "cond1", MID_CH, COND_DIM),
            down1: Conv2dLayer::from_store(ws, "down1", MID_CH, MID_CH, 4, 1, 2),
            enc2: ResBlock::from_store(ws, "enc2", MID_CH, DEEP_CH, TIME_DIM),
            cond2: CondBlock::from_store(ws, "cond2", DEEP_CH, COND_DIM),
            mid1: ResBlock::from_store(ws, "mid1", DEEP_CH, DEEP_CH, TIME_DIM),
            mid_attn: SelfAttentionLayer::from_store(ws, "mid_attn", DEEP_CH, 4),
            mid2: ResBlock::from_store(ws, "mid2", DEEP_CH, DEEP_CH, TIME_DIM),
            dec2: ResBlock::from_store(ws, "dec2", DEEP_CH + DEEP_CH, MID_CH, TIME_DIM),
            up2: ConvTranspose2dLayer::from_store(ws, "up2", MID_CH, MID_CH, 4, 2, 1),
            dec1: ResBlock::from_store(ws, "dec1", MID_CH + MID_CH, BASE_CH, TIME_DIM),
            up1: ConvTranspose2dLayer::from_store(ws, "up1", BASE_CH, BASE_CH, 4, 2, 1),
            dec0: ResBlock::from_store(ws, "dec0", BASE_CH + BASE_CH, BASE_CH, TIME_DIM),
            out_norm: GroupNormLayer::from_store(ws, "out_norm", BASE_CH),
            out_conv: Conv2dLayer::from_store(ws, "out_conv", BASE_CH, 3, 3, 1, 1),
        }
    }

    /// Forward pass.
    /// x:    (1, 3, H, W) in [-1, 1]
    /// t:    f32 timestep (0..999)
    /// cond: Vec<f32> of len COND_DIM
    /// Returns predicted noise: (1, 3, H, W)
    fn forward(&self, x: &T4, t: f32, cond: &[f32]) -> T4 {
        // Time embedding
        let t_emb = self.time_mlp.forward(t);

        // Initial conv
        let h0 = self.init_conv.forward(x);
        let h0 = self.enc0.forward(&h0, &t_emb);
        let h0 = self.cond0.forward(&h0, cond);
        let h0_ds = self.down0.forward(&h0);

        let h1 = self.enc1.forward(&h0_ds, &t_emb);
        let h1 = self.cond1.forward(&h1, cond);
        let h1_ds = self.down1.forward(&h1);

        let h2 = self.enc2.forward(&h1_ds, &t_emb);
        let h2 = self.cond2.forward(&h2, cond);

        // Bottleneck
        let hm = self.mid1.forward(&h2, &t_emb);
        let hm = self.mid_attn.forward(&hm);
        let hm = self.mid2.forward(&hm, &t_emb);

        // Decoder
        let cat2 = concat_channels(&hm, &h2);
        let hd2 = self.dec2.forward(&cat2, &t_emb);
        let hd2 = self.up2.forward(&hd2);

        let cat1 = concat_channels(&hd2, &h1);
        let hd1 = self.dec1.forward(&cat1, &t_emb);
        let hd1 = self.up1.forward(&hd1);

        let cat0 = concat_channels(&hd1, &h0);
        let hd0 = self.dec0.forward(&cat0, &t_emb);

        // Output
        let mut out = self.out_norm.forward(&hd0);
        apply_silu(&mut out.data);
        self.out_conv.forward(&out)
    }
}

fn concat_channels(a: &T4, b: &T4) -> T4 {
    debug_assert_eq!((a.n, a.h, a.w), (b.n, b.h, b.w));
    let n = a.n;
    let c = a.c + b.c;
    let h = a.h;
    let w = a.w;
    let mut out = T4::zeros(n, c, h, w);
    for bi in 0..n {
        for ch in 0..a.c {
            for hi in 0..h {
                for wi in 0..w {
                    out.set(bi, ch, hi, wi, a.at(bi, ch, hi, wi));
                }
            }
        }
        for ch in 0..b.c {
            for hi in 0..h {
                for wi in 0..w {
                    out.set(bi, a.c + ch, hi, wi, b.at(bi, ch, hi, wi));
                }
            }
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Weight loading (custom binary format)
// ─────────────────────────────────────────────────────────────────────────────

/// Load weights from `models/quantdiff/unet/`.
///
/// Format:
///   manifest.json  — array of {name: String, shape: [usize]}
///   weights/<name>.f32  — raw little-endian float32 bytes
fn load_weights(dir: &Path) -> Result<WeightStore, String> {
    let manifest_path = dir.join("manifest.json");
    if !manifest_path.exists() {
        return Err(format!("No manifest.json at {}", manifest_path.display()));
    }
    let manifest_str = std::fs::read_to_string(&manifest_path)
        .map_err(|e| format!("manifest read error: {e}"))?;

    // Parse: [{name: str, shape: [usize...]}]
    let parsed: serde_json::Value = serde_json::from_str(&manifest_str)
        .map_err(|e| format!("manifest JSON error: {e}"))?;

    let entries = parsed.as_array().ok_or("manifest must be a JSON array")?;
    let mut store = HashMap::new();

    for entry in entries {
        let name = entry["name"].as_str().ok_or("manifest entry missing 'name'")?.to_string();
        let shape: Vec<usize> = entry["shape"]
            .as_array()
            .ok_or("manifest entry missing 'shape'")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(1) as usize)
            .collect();
        let numel: usize = shape.iter().product();

        let w_path = dir.join("weights").join(format!("{}.f32", name.replace('.', "/")));
        if !w_path.exists() {
            // Try flat name (replace / with _)
            let flat = dir.join("weights").join(format!("{}.f32", name.replace('.', "_")));
            if !flat.exists() {
                eprintln!("[quantdiff] warning: weight file missing: {}", w_path.display());
                store.insert(name, vec![0.0f32; numel]);
                continue;
            }
            let bytes = std::fs::read(&flat).map_err(|e| format!("weight read error: {e}"))?;
            let floats: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            store.insert(name, floats);
        } else {
            let bytes = std::fs::read(&w_path).map_err(|e| format!("weight read error: {e}"))?;
            let floats: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            store.insert(name, floats);
        }
    }

    Ok(WeightStore(store))
}

// ─────────────────────────────────────────────────────────────────────────────
// Prompt conditioning encoder
// ─────────────────────────────────────────────────────────────────────────────

const STYLE_VOCAB: &[&str] = &["photo", "anime", "oil_paint", "neon", "abstract", "sketch", "pixel"];
const SCENE_VOCAB: &[&str] = &[
    "sportscar", "dog", "cat", "tree", "landscape",
    "portrait", "city", "space", "building", "vehicle",
];
const PALETTE_VOCAB: &[&str] = &[
    "forest", "ocean", "desert", "snow", "volcano", "neon", "cyberpunk",
    "city", "space", "sunset", "racing", "dog", "golden", "cat", "default",
];

fn encode_prompt(prompt: &str, style: &str, scene_hint: &str, palette_hint: &str) -> Vec<f32> {
    let mut vec = vec![0.0f32; COND_DIM];
    let words: std::collections::BTreeSet<String> = prompt
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_ascii_lowercase())
        .collect();

    if let Some(i) = STYLE_VOCAB.iter().position(|&s| s == style) {
        vec[i] = 1.0;
    }
    let base = STYLE_VOCAB.len();
    if let Some(i) = SCENE_VOCAB.iter().position(|&s| s == scene_hint) {
        vec[base + i] = 1.0;
    }
    let base2 = base + SCENE_VOCAB.len();
    if let Some(i) = PALETTE_VOCAB.iter().position(|&s| s == palette_hint) {
        vec[base2 + i] = 1.0;
    }
    // Soft word overlap for remaining dims
    let all_vocab: Vec<&str> = STYLE_VOCAB.iter().chain(SCENE_VOCAB.iter()).chain(PALETTE_VOCAB.iter()).copied().collect();
    let offset = base2 + PALETTE_VOCAB.len();
    for (i, &term) in all_vocab.iter().enumerate().take(COND_DIM.saturating_sub(offset)) {
        if words.iter().any(|w| w.contains(term) || term.contains(w.as_str())) {
            vec[offset + i] = 0.5;
        }
    }
    vec
}

// ─────────────────────────────────────────────────────────────────────────────
// Image ↔ Tensor conversion
// ─────────────────────────────────────────────────────────────────────────────

/// RgbaImage → T4 (1, 3, H, W) in [-1.0, 1.0]
fn image_to_tensor(img: &RgbaImage) -> T4 {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut data = vec![0.0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                data[c * h * w + y * w + x] = p[c] as f32 / 127.5 - 1.0;
            }
        }
    }
    T4::from_data(data, 1, 3, h, w)
}

/// T4 (1, 3, H, W) in [-1.0, 1.0] → RgbaImage
fn tensor_to_image(t: &T4) -> RgbaImage {
    let (h, w) = (t.h, t.w);
    let mut img = RgbaImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let r = ((t.at(0, 0, y, x) + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
            let g = ((t.at(0, 1, y, x) + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
            let b = ((t.at(0, 2, y, x) + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
            img.put_pixel(x as u32, y as u32, Rgba([r, g, b, 255]));
        }
    }
    img
}

/// Add Gaussian noise to a tensor (for warm-start)
fn add_noise_rng(x: &T4, scheduler: &DDIMScheduler, t: usize, rng: &mut rand::rngs::StdRng) -> T4 {
    use rand::Rng;
    let noise: Vec<f32> = (0..x.data.len()).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
    let noisy = scheduler.add_noise(&x.data, &noise, t);
    T4::from_data(noisy, x.n, x.c, x.h, x.w)
}

// ─────────────────────────────────────────────────────────────────────────────
// DDIM loop helpers
// ─────────────────────────────────────────────────────────────────────────────

fn ddim_loop(
    mut x: T4,
    unet: &QuantDiffUNet,
    scheduler: &mut DDIMScheduler,
    steps: usize,
    cond: &[f32],
    start_t: Option<usize>,
) -> T4 {
    scheduler.set_timesteps(steps);
    let timesteps: Vec<usize> = scheduler.timesteps.clone();
    let timesteps = match start_t {
        Some(st) => timesteps.into_iter().filter(|&t| t <= st).collect::<Vec<_>>(),
        None => timesteps,
    };
    for t in timesteps {
        let noise_pred = unet.forward(&x, t as f32, cond);
        let next = scheduler.step(&noise_pred.data, t, &x.data);
        x = T4::from_data(next, x.n, x.c, x.h, x.w);
    }
    x
}

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

fn unix_ts() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

fn slugify(s: &str) -> String {
    let slug: String = s
        .chars()
        .map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
        .collect::<String>()
        .trim_matches('_')
        .chars()
        .take(40)
        .collect();
    if slug.is_empty() { "qdiff".to_string() } else { slug }
}

fn find_model_dir() -> Option<PathBuf> {
    // Look relative to exe, then relative to repo root
    let candidates = [
        PathBuf::from("models/quantdiff/unet"),
        PathBuf::from("../models/quantdiff/unet"),
        PathBuf::from("../../models/quantdiff/unet"),
    ];
    for p in &candidates {
        if p.join("manifest.json").exists() {
            return Some(p.clone());
        }
    }
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Generate an image using the QuantDiff engine.
///
/// - **Warm-start (no weights)**: Resonart generates a structured base image;
///   we add minimal noise + a short DDIM reverse pass with random-init weights
///   (output ≈ base image with slight random variation, until trained).
///
/// - **Full DDIM (trained weights)**: starts from Gaussian noise, runs the
///   full trained U-Net DDIM loop, then LANCZOS-upscales to target resolution.
pub fn has_trained_weights() -> bool {
    find_model_dir().is_some()
}

pub fn generate_image(req: QuantDiffRequest) -> Result<QuantDiffResult, String> {
    let seed = req.seed.unwrap_or_else(|| unix_ts() as i64);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed as u64);
    let steps = req.steps.clamp(4, 50) as usize;
    let target_w = req.width.clamp(64, 4096);
    let target_h = req.height.clamp(64, 4096);

    // Native resolution for the U-Net (64×64)
    const NATIVE: u32 = 64;

    let cond = encode_prompt(&req.prompt, &req.style, &req.scene_hint, &req.palette_hint);

    // Attempt to load trained weights
    let model_dir = find_model_dir();
    let weights_loaded = model_dir.is_some();
    let ws = match &model_dir {
        Some(dir) => load_weights(dir).unwrap_or_else(|e| {
            eprintln!("[quantdiff] weight load error (using defaults): {e}");
            WeightStore(HashMap::new())
        }),
        None => WeightStore(HashMap::new()),
    };
    let unet = QuantDiffUNet::from_store(&ws);
    let mut scheduler = DDIMScheduler::new(1000);

    // Determine start image
    let (x_start, start_t): (T4, Option<usize>) = if weights_loaded {
        // Full DDIM: start from pure Gaussian noise
        use rand::Rng;
        let noise: Vec<f32> = (0..3 * NATIVE as usize * NATIVE as usize)
            .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
            .collect();
        let x = T4::from_data(noise, 1, 3, NATIVE as usize, NATIVE as usize);
        (x, None)
    } else {
        // Warm-start: use Resonart output as structured base
        let resonart_req = ResonartRenderRequest {
            prompt: req.prompt.clone(),
            negative_prompt: req.negative_prompt.clone(),
            steps: req.steps.clamp(8, 30),
            style: req.style.clone(),
            width: NATIVE,
            height: NATIVE,
            quality: req.quality.clone(),
            scene_hint: req.scene_hint.clone(),
            palette_hint: req.palette_hint.clone(),
            spectral_complexity: req.spectral_complexity,
            seed: Some(seed),
            output_dir: req.output_dir.clone(),
        };
        let base = resonart::generate_image(resonart_req)
            .map_err(|e| format!("resonart warm-start failed: {e}"))?;
        let base_img = image::open(&base.path)
            .map_err(|e| format!("warm-start image open failed: {e}"))?
            .resize_exact(NATIVE, NATIVE, FilterType::Lanczos3)
            .to_rgba8();
        let x0 = image_to_tensor(&base_img);

        // Add 5% noise strength
        let warm_t = 50usize; // ~5% of 1000 steps
        let x_noisy = add_noise_rng(&x0, &scheduler, warm_t, &mut rng);
        (x_noisy, Some(warm_t))
    };

    // Run DDIM loop
    let x_out = ddim_loop(x_start, &unet, &mut scheduler, steps, &cond, start_t);

    // Convert to image
    let small_img = tensor_to_image(&x_out);

    // LANCZOS upscale to target resolution
    let final_img = if target_w != NATIVE || target_h != NATIVE {
        resize(&small_img, target_w, target_h, FilterType::Lanczos3)
    } else {
        small_img
    };

    // Save
    let out_dir = PathBuf::from(&req.output_dir);
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("output dir error: {e}"))?;
    let filename = format!("qdiff_{}_{}.png", unix_ts(), slugify(&req.prompt));
    let out_path = out_dir.join(filename);
    DynamicImage::ImageRgba8(final_img)
        .save_with_format(&out_path, ImageFormat::Png)
        .map_err(|e| format!("PNG save error: {e}"))?;

    Ok(QuantDiffResult {
        path: out_path.to_string_lossy().to_string(),
        prompt: req.prompt,
        style: req.style,
        seed,
        width: target_w,
        height: target_h,
    })
}
