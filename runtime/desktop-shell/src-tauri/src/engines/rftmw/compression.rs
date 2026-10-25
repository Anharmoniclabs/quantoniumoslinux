// SPDX-License-Identifier: LicenseRef-SPX-Proprietary
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
//
// compression.rs — Port of rftmw_compression.hpp
//
// RFT + quantization + entropy-coding compression pipeline.

use super::core::{RFTMWEngine, ComplexVec, RealVec};
use crate::engines::types::Complex;

pub struct QuantizationParams {
    pub precision_bits: u32,
    pub dead_zone: f64,
    pub use_log_scale: bool,
}

impl Default for QuantizationParams {
    fn default() -> Self {
        Self {
            precision_bits: 16,
            dead_zone: 0.0,
            use_log_scale: false,
        }
    }
}

pub struct CompressionResult {
    pub data: Vec<i32>,
    pub original_size: usize,
    pub scale_factor: f64,
    pub max_value: f64,
    pub kept_coefficients: usize,
    pub top_k_energy_ratio: f64,
}

fn quantize_coefficients(coeffs: &ComplexVec, params: &QuantizationParams) -> (Vec<i32>, f64, f64) {
    let max_val = coeffs
        .iter()
        .flat_map(|c| [c.re.abs(), c.im.abs()])
        .fold(1e-15_f64, f64::max);

    let levels = (1u64 << (params.precision_bits - 1)) as f64;
    let scale = levels / max_val;

    let mut quantized = Vec::with_capacity(coeffs.len() * 2);
    for c in coeffs {
        let re = if c.re.abs() < params.dead_zone { 0 } else { (c.re * scale).round() as i32 };
        let im = if c.im.abs() < params.dead_zone { 0 } else { (c.im * scale).round() as i32 };
        quantized.push(re);
        quantized.push(im);
    }

    (quantized, scale, max_val)
}

fn dequantize_coefficients(quantized: &[i32], scale_factor: f64) -> ComplexVec {
    let inv_scale = 1.0 / scale_factor;
    quantized
        .chunks_exact(2)
        .map(|pair| Complex {
            re: pair[0] as f64 * inv_scale,
            im: pair[1] as f64 * inv_scale,
        })
        .collect()
}

pub struct RFTMWCompressor {
    engine: RFTMWEngine,
    params: QuantizationParams,
}

impl RFTMWCompressor {
    pub fn new(n: usize, params: QuantizationParams) -> Self {
        Self {
            engine: RFTMWEngine::new(n),
            params,
        }
    }

    pub fn compress(&self, input: &RealVec) -> CompressionResult {
        let coeffs = self.engine.forward(input);
        let (data, scale_factor, max_value) = quantize_coefficients(&coeffs, &self.params);

        // Count nonzero
        let kept = data.iter().filter(|&&v| v != 0).count() / 2;
        let total_energy: f64 = coeffs.iter().map(|c| c.norm_sq()).sum();
        let kept_energy: f64 = {
            let mut mags: Vec<f64> = coeffs.iter().map(|c| c.norm_sq()).collect();
            mags.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            mags.iter().take(kept).sum()
        };

        CompressionResult {
            data,
            original_size: input.len(),
            scale_factor,
            max_value,
            kept_coefficients: kept,
            top_k_energy_ratio: if total_energy > 0.0 { kept_energy / total_energy } else { 0.0 },
        }
    }

    pub fn decompress(&self, result: &CompressionResult) -> RealVec {
        let coeffs = dequantize_coefficients(&result.data, result.scale_factor);
        self.engine.inverse(&coeffs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_decompress() {
        let n = 64;
        let comp = RFTMWCompressor::new(n, QuantizationParams::default());
        let input: RealVec = (0..n).map(|i| (i as f64 * 0.2).sin()).collect();
        let result = comp.compress(&input);
        let recovered = comp.decompress(&result);

        let err: f64 = input.iter().zip(recovered.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let norm: f64 = input.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(err / norm < 0.01, "compression roundtrip error: {}", err / norm);
    }
}
