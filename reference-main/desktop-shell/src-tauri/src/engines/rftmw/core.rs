// SPDX-License-Identifier: LicenseRef-SPX-Proprietary
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
//
// core.rs — Port of rftmw_core.hpp
//
// RFTMWEngine: canonical RFT transform with Gram-matrix normalization,
// Jacobi eigendecomposition, and IHT sparse recovery.

use crate::engines::types::{Complex, PHI, PI, TWO_PI};

pub type ComplexVec = Vec<Complex>;
pub type RealVec = Vec<f64>;

const CANONICAL_MAX_N: usize = 256;

// ── Concentration Metrics ────────────────────────────────────────────────────

pub struct ConcentrationMetrics {
    pub top_k: usize,
    pub energy_ratio: f64,
    pub sparsity_ratio: f64,
    pub coherence_proxy: f64,
    pub use_sparse_route: bool,
}

// ── RFT Basis Helpers ────────────────────────────────────────────────────────

/// Raw φ-basis entry: (1/√N) exp(i 2π frac((k+1)φ) n)
fn raw_phi_entry(n: usize, k: usize, big_n: usize) -> Complex {
    let freq = (k + 1) as f64 * PHI;
    let frac = freq - freq.floor();
    let angle = TWO_PI * frac * n as f64;
    let scale = 1.0 / (big_n as f64).sqrt();
    Complex {
        re: angle.cos() * scale,
        im: angle.sin() * scale,
    }
}

/// Build full raw Φ matrix (column-major: index = n * N + k).
fn compute_raw_phi_basis(n: usize) -> ComplexVec {
    let mut phi = vec![Complex { re: 0.0, im: 0.0 }; n * n];
    for row in 0..n {
        for col in 0..n {
            phi[row * n + col] = raw_phi_entry(row, col, n);
        }
    }
    phi
}

/// Forward: coeffs = Φ^H @ input (adjoint applied to real input)
fn apply_raw_phi_adjoint(input: &RealVec, phi: &ComplexVec, n: usize) -> ComplexVec {
    let mut coeffs = vec![Complex { re: 0.0, im: 0.0 }; n];
    for k in 0..n {
        let mut sum = Complex { re: 0.0, im: 0.0 };
        for row in 0..n {
            let basis = phi[row * n + k].conj();
            sum = sum + Complex {
                re: basis.re * input[row],
                im: basis.im * input[row],
            };
        }
        coeffs[k] = sum;
    }
    coeffs
}

/// Inverse: output = Re(Φ @ coeffs)
fn apply_raw_phi_synthesis(coeffs: &ComplexVec, phi: &ComplexVec, n: usize) -> RealVec {
    let mut output = vec![0.0; n];
    for row in 0..n {
        let mut sum = Complex { re: 0.0, im: 0.0 };
        for k in 0..n {
            sum = sum + phi[row * n + k] * coeffs[k];
        }
        output[row] = sum.re;
    }
    output
}

fn l2_norm_sq(v: &RealVec) -> f64 {
    v.iter().map(|x| x * x).sum()
}

fn count_significant_coeffs(coeffs: &ComplexVec, rel_tol: f64) -> usize {
    let max_mag = coeffs.iter().map(|c| c.abs()).fold(0.0_f64, f64::max);
    let threshold = max_mag * rel_tol;
    coeffs.iter().filter(|c| c.abs() > threshold).count()
}

fn hard_threshold_inplace(coeffs: &mut ComplexVec, keep_k: usize) {
    let mut mags: Vec<f64> = coeffs.iter().map(|c| c.abs()).collect();
    mags.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let cutoff = if keep_k < mags.len() { mags[keep_k] } else { 0.0 };
    for c in coeffs.iter_mut() {
        if c.abs() < cutoff {
            *c = Complex { re: 0.0, im: 0.0 };
        }
    }
}

// ── Signal Concentration Analysis ────────────────────────────────────────────

fn analyze_signal_concentration(
    input: &RealVec,
    energy_threshold: f64,
    phi_tilde: &ComplexVec,
    n: usize,
) -> ConcentrationMetrics {
    let coeffs = apply_raw_phi_adjoint(input, phi_tilde, n);
    let mut mags: Vec<(usize, f64)> = coeffs.iter().enumerate().map(|(i, c)| (i, c.abs())).collect();
    mags.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let total_energy: f64 = mags.iter().map(|(_, m)| m * m).sum();
    let mut cumul = 0.0;
    let mut top_k = 0;
    for &(_, m) in &mags {
        cumul += m * m;
        top_k += 1;
        if cumul >= energy_threshold * total_energy {
            break;
        }
    }

    let energy_ratio = if total_energy > 0.0 { cumul / total_energy } else { 0.0 };
    let sparsity_ratio = top_k as f64 / n as f64;
    let coherence_proxy = 1.0 / (n as f64).sqrt();
    let use_sparse_route = sparsity_ratio < 0.5 && energy_ratio > 0.85;

    ConcentrationMetrics {
        top_k,
        energy_ratio,
        sparsity_ratio,
        coherence_proxy,
        use_sparse_route,
    }
}

// ── Jacobi Eigendecomposition ────────────────────────────────────────────────

/// Hermitian dot product for complex vectors.
fn hermitian_dot(a: &[Complex], b: &[Complex]) -> Complex {
    let mut sum = Complex { re: 0.0, im: 0.0 };
    for (ai, bi) in a.iter().zip(b.iter()) {
        sum = sum + ai.conj() * *bi;
    }
    sum
}

/// Jacobi eigendecomposition for Hermitian N×N matrix (column-major).
/// Returns (eigenvectors column-major, eigenvalues sorted ascending).
fn jacobi_eigen_hermitian(a: &mut ComplexVec, n: usize, max_sweeps: usize) -> (ComplexVec, RealVec) {
    let mut v = vec![Complex { re: 0.0, im: 0.0 }; n * n];
    for i in 0..n {
        v[i * n + i] = Complex { re: 1.0, im: 0.0 };
    }

    for _sweep in 0..max_sweeps {
        let mut off_diag = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off_diag += a[i * n + j].norm_sq();
            }
        }
        if off_diag < 1e-30 {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.norm_sq() < 1e-30 {
                    continue;
                }
                let app = a[p * n + p].re;
                let aqq = a[q * n + q].re;
                let tau = (aqq - app) / (2.0 * apq.abs());
                let t = if tau.abs() < 1e15 {
                    let sign = if tau >= 0.0 { 1.0 } else { -1.0 };
                    sign / (tau.abs() + (1.0 + tau * tau).sqrt())
                } else {
                    0.0
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let phase = if apq.abs() > 0.0 {
                    Complex {
                        re: apq.re / apq.abs(),
                        im: -apq.im / apq.abs(),
                    }
                } else {
                    Complex { re: 1.0, im: 0.0 }
                };

                // Apply Givens rotation to A
                for r in 0..n {
                    let arp = a[r * n + p];
                    let arq = a[r * n + q];
                    a[r * n + p] = Complex {
                        re: c * arp.re + s * (phase * arq).re,
                        im: c * arp.im + s * (phase * arq).im,
                    };
                    a[r * n + q] = Complex {
                        re: -s * (phase.conj() * arp).re + c * arq.re,
                        im: -s * (phase.conj() * arp).im + c * arq.im,
                    };
                }
                for r in 0..n {
                    let apr = a[p * n + r];
                    let aqr = a[q * n + r];
                    a[p * n + r] = Complex {
                        re: c * apr.re + s * (phase.conj() * aqr).re,
                        im: c * apr.im + s * (phase.conj() * aqr).im,
                    };
                    a[q * n + r] = Complex {
                        re: -s * (phase * apr).re + c * aqr.re,
                        im: -s * (phase * apr).im + c * aqr.im,
                    };
                }

                // Accumulate eigenvectors
                for r in 0..n {
                    let vp = v[r * n + p];
                    let vq = v[r * n + q];
                    v[r * n + p] = Complex {
                        re: c * vp.re + s * (phase * vq).re,
                        im: c * vp.im + s * (phase * vq).im,
                    };
                    v[r * n + q] = Complex {
                        re: -s * (phase.conj() * vp).re + c * vq.re,
                        im: -s * (phase.conj() * vp).im + c * vq.im,
                    };
                }
            }
        }
    }

    let eigenvals: RealVec = (0..n).map(|i| a[i * n + i].re).collect();
    (v, eigenvals)
}

// ── Gram-Normalized Basis ────────────────────────────────────────────────────

/// Orthonormalize columns of Φ via Modified Gram-Schmidt, producing a unitary basis.
fn compute_gram_normalised_basis(n: usize) -> ComplexVec {
    let mut phi = compute_raw_phi_basis(n);

    // Modified Gram-Schmidt orthonormalization (column-wise)
    for j in 0..n {
        // Normalize column j
        let norm: f64 = (0..n).map(|r| phi[r * n + j].norm_sq()).sum::<f64>().sqrt();
        if norm > 1e-15 {
            let inv_norm = 1.0 / norm;
            for r in 0..n {
                phi[r * n + j] = phi[r * n + j].scale(inv_norm);
            }
        }
        // Subtract projection of column j from all subsequent columns
        for k in (j + 1)..n {
            let mut dot = Complex { re: 0.0, im: 0.0 };
            for r in 0..n {
                dot = dot + phi[r * n + j].conj() * phi[r * n + k];
            }
            for r in 0..n {
                let proj = phi[r * n + j] * dot;
                phi[r * n + k] = phi[r * n + k] - proj;
            }
        }
    }

    phi
}

// ── IHT Sparse Recovery ─────────────────────────────────────────────────────

fn recover_sparse_coeffs_iht(
    input: &RealVec,
    phi_tilde: &ComplexVec,
    n: usize,
    keep_k: usize,
    max_iter: usize,
    step_size: f64,
    tol: f64,
) -> ComplexVec {
    let mut coeffs = vec![Complex { re: 0.0, im: 0.0 }; n];
    let input_norm = l2_norm_sq(input).sqrt();
    if input_norm < 1e-15 {
        return coeffs;
    }

    for _iter in 0..max_iter {
        // Synthesize current estimate
        let estimate = apply_raw_phi_synthesis(&coeffs, phi_tilde, n);
        // Residual
        let residual: RealVec = input.iter().zip(estimate.iter()).map(|(a, b)| a - b).collect();
        let res_norm = l2_norm_sq(&residual).sqrt();
        if res_norm / input_norm < tol {
            break;
        }
        // Gradient = Φ^H @ residual
        let gradient = apply_raw_phi_adjoint(&residual, phi_tilde, n);
        // Update
        for (c, g) in coeffs.iter_mut().zip(gradient.iter()) {
            *c = *c + g.scale(step_size);
        }
        // Hard threshold
        hard_threshold_inplace(&mut coeffs, keep_k);
    }
    coeffs
}

// ── RFTMWEngine ──────────────────────────────────────────────────────────────

pub struct RFTMWEngine {
    n: usize,
    phi_tilde: ComplexVec,
}

impl RFTMWEngine {
    pub fn new(n: usize) -> Self {
        let phi_tilde = if n <= CANONICAL_MAX_N {
            compute_gram_normalised_basis(n)
        } else {
            compute_raw_phi_basis(n)
        };
        Self { n, phi_tilde }
    }

    /// Forward transform: coeffs = Φ_tilde^H @ input
    pub fn forward(&self, input: &RealVec) -> ComplexVec {
        assert_eq!(input.len(), self.n);
        apply_raw_phi_adjoint(input, &self.phi_tilde, self.n)
    }

    /// Inverse transform: output = Re(Φ_tilde @ coeffs)
    pub fn inverse(&self, coeffs: &ComplexVec) -> RealVec {
        assert_eq!(coeffs.len(), self.n);
        apply_raw_phi_synthesis(coeffs, &self.phi_tilde, self.n)
    }

    /// Forward with IHT sparse recovery.
    pub fn forward_sparse(&self, input: &RealVec, keep_k: usize, max_iter: usize) -> ComplexVec {
        recover_sparse_coeffs_iht(input, &self.phi_tilde, self.n, keep_k, max_iter, 0.85, 1e-7)
    }

    /// Inverse from sparse coefficients (same as `inverse`).
    pub fn inverse_sparse(&self, coeffs: &ComplexVec) -> RealVec {
        self.inverse(coeffs)
    }

    /// Signal concentration analysis.
    pub fn analyze_signal(&self, input: &RealVec, energy_threshold: f64) -> ConcentrationMetrics {
        analyze_signal_concentration(input, energy_threshold, &self.phi_tilde, self.n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_small() {
        let n = 32;
        let engine = RFTMWEngine::new(n);
        let input: RealVec = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
        let coeffs = engine.forward(&input);
        let recovered = engine.inverse(&coeffs);
        let err: f64 = input.iter().zip(recovered.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        let norm: f64 = l2_norm_sq(&input).sqrt();
        assert!(err / norm < 1e-6, "roundtrip relative error too large: {}", err / norm);
    }

    #[test]
    fn test_sparsity_analysis() {
        let n = 64;
        let engine = RFTMWEngine::new(n);
        // Sparse signal — only a few frequencies
        let mut input = vec![0.0; n];
        for i in 0..n {
            input[i] = (i as f64 * TWO_PI / n as f64 * 3.0).sin();
        }
        let metrics = engine.analyze_signal(&input, 0.85);
        assert!(metrics.top_k < n, "signal should be compressible");
    }
}
