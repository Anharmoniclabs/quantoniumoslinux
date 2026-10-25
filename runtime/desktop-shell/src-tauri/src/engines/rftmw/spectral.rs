// SPDX-License-Identifier: LicenseRef-SPX-Proprietary
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
//
// spectral.rs — Port of rftmw_spectral_inference.hpp + rftmw_spectral_kernels.c
//
// Spectral-domain weight representation: dense → RFT → top-k → quantized sparse
// Quantized sparse GEMV with zero-trig LUT.

use super::fft::fft_f32;
use super::inference::{dequant_q4k_block, dequant_q6k_block, layer_keep_ratio};
use crate::engines::types::PHI;
use std::f64::consts::PI;

const BLOCK_SIZE: usize = 256;

// ── Phase LUT ────────────────────────────────────────────────────────────────

pub struct PhaseLut {
    pub cos_lut: [f32; 256],
    pub sin_lut: [f32; 256],
}

impl PhaseLut {
    pub fn new() -> Self {
        let mut cos_lut = [0.0_f32; 256];
        let mut sin_lut = [0.0_f32; 256];
        for i in 0..256 {
            let phase = (i as f64 / 255.0) * 2.0 * PI - PI;
            cos_lut[i] = phase.cos() as f32;
            sin_lut[i] = phase.sin() as f32;
        }
        Self { cos_lut, sin_lut }
    }
}

// ── Spectral Basis ───────────────────────────────────────────────────────────

pub struct SpectralBasis {
    pub cos_table: Vec<f32>,
    pub sin_table: Vec<f32>,
    pub block_size: usize,
    pub inv_sqrt_n: f32,
}

impl SpectralBasis {
    pub fn new(block_size: usize) -> Self {
        let mut cos_table = vec![0.0_f32; block_size];
        let mut sin_table = vec![0.0_f32; block_size];
        for k in 0..block_size {
            let freq = (k + 1) as f64 * PHI;
            let frac = freq - freq.floor();
            let angle = 2.0 * PI * frac;
            cos_table[k] = angle.cos() as f32;
            sin_table[k] = angle.sin() as f32;
        }
        Self {
            cos_table,
            sin_table,
            block_size,
            inv_sqrt_n: 1.0 / (block_size as f32).sqrt(),
        }
    }

    /// Hybrid RFT forward: FFT(x) · diag(exp(i θ_φ)) · (1/√N)
    pub fn forward_block(&self, x_real: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = self.block_size;
        let mut re = vec![0.0_f32; n];
        let mut im = vec![0.0_f32; n];
        re[..x_real.len().min(n)].copy_from_slice(&x_real[..x_real.len().min(n)]);

        fft_f32(&mut re, &mut im);

        // Apply Φ-RFT diagonal
        for k in 0..n {
            let r = re[k];
            let i = im[k];
            re[k] = (r * self.cos_table[k] - i * self.sin_table[k]) * self.inv_sqrt_n;
            im[k] = (r * self.sin_table[k] + i * self.cos_table[k]) * self.inv_sqrt_n;
        }

        (re, im)
    }
}

// ── Quantized Sparse Row ─────────────────────────────────────────────────────

pub struct QSparseBlock {
    pub n_nonzero: usize,
    pub indices: Vec<u16>,
    pub mag4: Vec<u8>,     // 4-bit packed magnitudes (2 per byte)
    pub phase8: Vec<u8>,
    pub peak_mag: f32,
}

pub struct QSparseRow {
    pub blocks: Vec<QSparseBlock>,
}

// ── Quantized Sparse Dot Product (Hot Inner Loop) ────────────────────────────

fn qspectral_sparse_dot(
    block: &QSparseBlock,
    cos_lut: &[f32; 256],
    sin_lut: &[f32; 256],
    x_spec_re: &[f32],
    x_spec_im: &[f32],
) -> f32 {
    let mut dot = 0.0_f32;
    for i in 0..block.n_nonzero {
        let idx = block.indices[i] as usize;
        // 4-bit magnitude extraction
        let byte = block.mag4.get(i >> 1).copied().unwrap_or(0);
        let nibble = if i & 1 == 0 { byte & 0x0F } else { byte >> 4 };
        let mag = (nibble as f32 / 15.0) * block.peak_mag;
        let phase_idx = block.phase8[i] as usize;
        let re = mag * cos_lut[phase_idx];
        let im = mag * sin_lut[phase_idx];
        dot += re * x_spec_re[idx] + im * x_spec_im[idx];
    }
    dot
}

// ── Spectral Weight ──────────────────────────────────────────────────────────

pub struct SpectralWeight {
    pub out_features: usize,
    pub in_features: usize,
    pub block_size: usize,
    pub n_blocks: usize,
    pub rows: Vec<QSparseRow>,
    basis: SpectralBasis,
    phase_lut: PhaseLut,
}

impl SpectralWeight {
    /// Compress dense weights → spectral quantized.
    pub fn from_dense(
        weights: &[f32],
        out_features: usize,
        in_features: usize,
        block_size: usize,
        keep_ratio: f32,
    ) -> Self {
        let basis = SpectralBasis::new(block_size);
        let phase_lut = PhaseLut::new();
        let n_blocks = (in_features + block_size - 1) / block_size;

        let mut rows = Vec::with_capacity(out_features);
        for row_idx in 0..out_features {
            let row_data = &weights[row_idx * in_features..(row_idx + 1) * in_features];
            let mut blocks = Vec::with_capacity(n_blocks);

            for b in 0..n_blocks {
                let start = b * block_size;
                let end = (start + block_size).min(in_features);
                let chunk = &row_data[start..end];
                let block = compress_block(&basis, &phase_lut, chunk, block_size, keep_ratio);
                blocks.push(block);
            }
            rows.push(QSparseRow { blocks });
        }

        Self {
            out_features,
            in_features,
            block_size,
            n_blocks,
            rows,
            basis,
            phase_lut,
        }
    }

    /// Spectral GEMV: y = W_spectral @ x
    pub fn matmul_vec(&self, x: &[f32], y: &mut [f32]) {
        let n_blocks = self.n_blocks;
        let bs = self.block_size;

        // Transform x into spectral domain (per block)
        let mut x_spec_re = vec![0.0_f32; n_blocks * bs];
        let mut x_spec_im = vec![0.0_f32; n_blocks * bs];

        for b in 0..n_blocks {
            let start = b * bs;
            let end = (start + bs).min(self.in_features);
            let mut chunk = vec![0.0_f32; bs];
            chunk[..end - start].copy_from_slice(&x[start..end]);

            let (re, im) = self.basis.forward_block(&chunk);
            x_spec_re[b * bs..(b + 1) * bs].copy_from_slice(&re);
            x_spec_im[b * bs..(b + 1) * bs].copy_from_slice(&im);
        }

        // Compute dot products
        for (row_idx, row) in self.rows.iter().enumerate() {
            let mut sum = 0.0_f32;
            for (b, block) in row.blocks.iter().enumerate() {
                let off = b * bs;
                sum += qspectral_sparse_dot(
                    block,
                    &self.phase_lut.cos_lut,
                    &self.phase_lut.sin_lut,
                    &x_spec_re[off..off + bs],
                    &x_spec_im[off..off + bs],
                );
            }
            y[row_idx] = sum;
        }
    }

    pub fn memory_bytes(&self) -> usize {
        self.rows
            .iter()
            .flat_map(|r| &r.blocks)
            .map(|b| b.indices.len() * 2 + b.mag4.len() + b.phase8.len() + 4)
            .sum()
    }

    pub fn avg_sparsity(&self) -> f32 {
        let total: usize = self.rows.iter().flat_map(|r| &r.blocks).map(|b| b.n_nonzero).sum();
        let capacity = self.out_features * self.n_blocks * self.block_size;
        if capacity == 0 { 0.0 } else { total as f32 / capacity as f32 }
    }
}

fn compress_block(
    basis: &SpectralBasis,
    _phase_lut: &PhaseLut,
    input: &[f32],
    block_size: usize,
    keep_ratio: f32,
) -> QSparseBlock {
    let mut padded = vec![0.0_f32; block_size];
    padded[..input.len()].copy_from_slice(input);

    let (re, im) = basis.forward_block(&padded);

    // Compute magnitudes
    let mut mags: Vec<(usize, f32)> = (0..block_size)
        .map(|k| (k, (re[k] * re[k] + im[k] * im[k]).sqrt()))
        .collect();
    mags.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let n_keep = ((block_size as f32 * keep_ratio) as usize).max(1).min(block_size);
    let peak = mags[0].1.max(1e-15);

    let mut indices = Vec::with_capacity(n_keep);
    let mut phase8 = Vec::with_capacity(n_keep);
    let mut mag4_nibbles = Vec::with_capacity(n_keep);

    for &(idx, mag) in mags.iter().take(n_keep) {
        indices.push(idx as u16);
        // 4-bit magnitude
        let m4 = ((mag / peak) * 15.0).round().min(15.0) as u8;
        mag4_nibbles.push(m4);
        // 8-bit phase
        let phase = im[idx].atan2(re[idx]);
        let p8 = (((phase + PI as f32) / (2.0 * PI as f32)) * 255.0).round().min(255.0) as u8;
        phase8.push(p8);
    }

    // Pack 4-bit magnitudes (2 per byte)
    let mag4_len = (n_keep + 1) / 2;
    let mut mag4 = vec![0u8; mag4_len];
    for (i, &nib) in mag4_nibbles.iter().enumerate() {
        if i & 1 == 0 {
            mag4[i >> 1] |= nib;
        } else {
            mag4[i >> 1] |= nib << 4;
        }
    }

    QSparseBlock {
        n_nonzero: n_keep,
        indices,
        mag4,
        phase8,
        peak_mag: peak,
    }
}

pub struct SpectralConfig {
    pub block_size: usize,
    pub keep_ratio: f32,
    pub nonuniform: bool,
    pub total_layers: usize,
}

impl Default for SpectralConfig {
    fn default() -> Self {
        Self {
            block_size: 256,
            keep_ratio: 0.60,
            nonuniform: true,
            total_layers: 36,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_weight_matmul() {
        let rows = 4;
        let cols = 256;
        let weights: Vec<f32> = (0..rows * cols).map(|i| ((i as f32) * 0.01).sin()).collect();
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.02).cos()).collect();

        // Dense reference
        let mut y_ref = vec![0.0_f32; rows];
        for r in 0..rows {
            for c in 0..cols {
                y_ref[r] += weights[r * cols + c] * x[c];
            }
        }

        // Spectral (high keep ratio for accuracy)
        let sw = SpectralWeight::from_dense(&weights, rows, cols, 256, 0.90);
        let mut y_spec = vec![0.0_f32; rows];
        sw.matmul_vec(&x, &mut y_spec);

        for (r, (yr, ys)) in y_ref.iter().zip(y_spec.iter()).enumerate() {
            let err = (yr - ys).abs();
            let tol = yr.abs() * 0.3 + 0.1;
            assert!(err < tol, "row {} mismatch: ref={}, spectral={}", r, yr, ys);
        }
    }
}
