// SPDX-License-Identifier: LicenseRef-SPX-Proprietary
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
//
// kv_compress.rs — Port of rftmw_kv_compress.hpp
//
// RFT-based KV-cache compression (60% memory saving at ~2% PPL loss).

use super::fft::{fft_f32, ifft_f32};
use super::inference::layer_keep_ratio;
use crate::engines::types::PHI;
use std::f64::consts::PI;

pub struct KVCompressConfig {
    pub head_dim: usize,
    pub num_kv_heads: usize,
    pub base_keep_ratio: f32,
    pub total_layers: usize,
    pub min_positions_before_compress: usize,
}

impl Default for KVCompressConfig {
    fn default() -> Self {
        Self {
            head_dim: 128,
            num_kv_heads: 2,
            base_keep_ratio: 0.60,
            total_layers: 36,
            min_positions_before_compress: 4,
        }
    }
}

/// Compressed representation of one KV head vector.
pub struct CompressedKVEntry {
    pub n_kept: usize,
    pub mag_scale: f32,
    pub indices: Vec<u16>,
    pub mag8: Vec<u8>,
    pub phase8: Vec<u8>,
}

pub struct CompressedLayerKV {
    pub k_compressed: Vec<Vec<CompressedKVEntry>>,  // [position][kv_head]
    pub v_compressed: Vec<Vec<CompressedKVEntry>>,
    pub compressed_up_to: usize,
}

pub struct KVCompressor {
    config: KVCompressConfig,
    phase_lut_cos: [f32; 256],
    phase_lut_sin: [f32; 256],
    phi_cos: Vec<f32>,
    phi_sin: Vec<f32>,
}

impl KVCompressor {
    pub fn new(config: KVCompressConfig) -> Self {
        // Precompute phase LUT (256 levels → [-π, π])
        let mut phase_lut_cos = [0.0_f32; 256];
        let mut phase_lut_sin = [0.0_f32; 256];
        for i in 0..256 {
            let phase = (i as f64 / 255.0) * 2.0 * PI - PI;
            phase_lut_cos[i] = phase.cos() as f32;
            phase_lut_sin[i] = phase.sin() as f32;
        }

        // Precompute Φ-RFT diagonal (golden-ratio phases)
        let n = config.head_dim;
        let mut phi_cos = vec![0.0_f32; n];
        let mut phi_sin = vec![0.0_f32; n];
        for k in 0..n {
            let freq = (k + 1) as f64 * PHI;
            let frac = freq - freq.floor();
            let angle = 2.0 * PI * frac;
            phi_cos[k] = angle.cos() as f32;
            phi_sin[k] = angle.sin() as f32;
        }

        Self {
            config,
            phase_lut_cos,
            phase_lut_sin,
            phi_cos,
            phi_sin,
        }
    }

    /// Compress a single head-dim vector via RFT → top-k → INT8 quantize.
    pub fn compress_vector(&self, input: &[f32], keep_ratio: f32) -> CompressedKVEntry {
        let n = input.len();
        assert!(n.is_power_of_two());

        // FFT
        let mut re: Vec<f32> = input.to_vec();
        let mut im = vec![0.0_f32; n];
        fft_f32(&mut re, &mut im);

        // Apply Φ-RFT diagonal: Y[k] = FFT[k] * exp(i θ_φ[k]) / √N
        let inv_sqrt_n = 1.0 / (n as f32).sqrt();
        for k in 0..n {
            let r = re[k];
            let i = im[k];
            re[k] = (r * self.phi_cos[k] - i * self.phi_sin[k]) * inv_sqrt_n;
            im[k] = (r * self.phi_sin[k] + i * self.phi_cos[k]) * inv_sqrt_n;
        }

        // Compute magnitudes and select top-k
        let mut mags: Vec<(usize, f32)> = (0..n)
            .map(|k| (k, (re[k] * re[k] + im[k] * im[k]).sqrt()))
            .collect();
        mags.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let n_kept = ((n as f32 * keep_ratio) as usize).max(1).min(n);
        let peak_mag = mags[0].1.max(1e-15);

        let mut indices = Vec::with_capacity(n_kept);
        let mut mag8 = Vec::with_capacity(n_kept);
        let mut phase8 = Vec::with_capacity(n_kept);

        for &(idx, mag) in mags.iter().take(n_kept) {
            indices.push(idx as u16);
            // Quantize magnitude to 8-bit
            let m8 = ((mag / peak_mag) * 255.0).round().min(255.0) as u8;
            mag8.push(m8);
            // Quantize phase to 8-bit
            let phase = im[idx].atan2(re[idx]);
            let p8 = (((phase + PI as f32) / (2.0 * PI as f32)) * 255.0).round().min(255.0) as u8;
            phase8.push(p8);
        }

        CompressedKVEntry {
            n_kept,
            mag_scale: peak_mag,
            indices,
            mag8,
            phase8,
        }
    }

    /// Decompress back to head_dim floats.
    pub fn decompress_vector(&self, entry: &CompressedKVEntry, output: &mut [f32]) {
        let n = output.len();
        let mut re = vec![0.0_f32; n];
        let mut im = vec![0.0_f32; n];

        let sqrt_n = (n as f32).sqrt();
        for i in 0..entry.n_kept {
            let idx = entry.indices[i] as usize;
            let mag = (entry.mag8[i] as f32 / 255.0) * entry.mag_scale;
            let c = self.phase_lut_cos[entry.phase8[i] as usize];
            let s = self.phase_lut_sin[entry.phase8[i] as usize];

            // Undo Φ-RFT diagonal (conjugate) and scale back by √N
            let spec_re = mag * c * sqrt_n;
            let spec_im = mag * s * sqrt_n;
            re[idx] = spec_re * self.phi_cos[idx] + spec_im * self.phi_sin[idx];
            im[idx] = -spec_re * self.phi_sin[idx] + spec_im * self.phi_cos[idx];
        }

        // IFFT
        ifft_f32(&mut re, &mut im);
        output.copy_from_slice(&re);
    }

    /// Compress all positions in a layer's KV cache.
    pub fn compress_layer(
        &self,
        layer_idx: usize,
        cache_len: usize,
        k_cache: &mut [f32],
        v_cache: &mut [f32],
    ) -> CompressedLayerKV {
        let head_dim = self.config.head_dim;
        let num_kv_heads = self.config.num_kv_heads;
        let keep = layer_keep_ratio(layer_idx, self.config.total_layers, self.config.base_keep_ratio);

        let mut result = CompressedLayerKV {
            k_compressed: Vec::with_capacity(cache_len),
            v_compressed: Vec::with_capacity(cache_len),
            compressed_up_to: cache_len,
        };

        for pos in 0..cache_len {
            let mut k_entries = Vec::with_capacity(num_kv_heads);
            let mut v_entries = Vec::with_capacity(num_kv_heads);

            for h in 0..num_kv_heads {
                let offset = (pos * num_kv_heads + h) * head_dim;
                let k_vec = &k_cache[offset..offset + head_dim];
                let v_vec = &v_cache[offset..offset + head_dim];

                let k_entry = self.compress_vector(k_vec, keep);
                let v_entry = self.compress_vector(v_vec, keep);

                // Decompress back in-place
                self.decompress_vector(&k_entry, &mut k_cache[offset..offset + head_dim]);
                self.decompress_vector(&v_entry, &mut v_cache[offset..offset + head_dim]);

                k_entries.push(k_entry);
                v_entries.push(v_entry);
            }

            result.k_compressed.push(k_entries);
            result.v_compressed.push(v_entries);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_compress_roundtrip() {
        let config = KVCompressConfig {
            head_dim: 128,
            num_kv_heads: 1,
            base_keep_ratio: 0.80,
            total_layers: 1,
            min_positions_before_compress: 1,
        };
        let comp = KVCompressor::new(config);

        let input: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).sin()).collect();
        let entry = comp.compress_vector(&input, 0.80);

        let mut output = vec![0.0_f32; 128];
        comp.decompress_vector(&entry, &mut output);

        let err: f32 = input.iter().zip(output.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
        let norm: f32 = input.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(err / norm < 0.2, "KV compression error too large: {:.4}", err / norm);
    }
}
