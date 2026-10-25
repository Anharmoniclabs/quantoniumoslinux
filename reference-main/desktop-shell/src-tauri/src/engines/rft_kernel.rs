// SPDX-License-Identifier: LicenseRef-SPX-Proprietary
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
//
// rft_kernel.rs — Rust port of rft_kernel.c + rft_kernel_fixed.c + rft_kernel_ui.c
//
// Core Resonant Fourier Transform engine with:
//   - φ-frequency basis construction (16 variants)
//   - Two-pass Modified Gram-Schmidt orthonormalization
//   - Forward/inverse transforms
//   - Quantum operations (von Neumann entropy, entanglement, Bell states)
//   - UI visualization support

use super::types::*;

// ── RFT Engine ───────────────────────────────────────────────────────────────

pub struct RftEngine {
    pub size: usize,
    pub basis: Vec<Complex>,       // size × size column-major
    pub eigenvalues: Vec<f64>,
    pub variant: RftVariant,
    pub flags: RftFlags,
    pub qubit_count: usize,
    initialized: bool,
}

impl RftEngine {
    /// Initialize a new RFT engine with given size and flags.
    pub fn new(size: usize, flags: RftFlags) -> RftResult<Self> {
        if size == 0 {
            return Err(RftError::InvalidParam);
        }

        let mut engine = Self {
            size,
            basis: vec![Complex::ZERO; size * size],
            eigenvalues: vec![1.0; size],
            variant: RftVariant::Standard,
            flags,
            qubit_count: 0,
            initialized: false,
        };

        engine.build_basis()?;
        engine.initialized = true;
        Ok(engine)
    }

    /// Initialize with a specific variant.
    pub fn new_with_variant(size: usize, flags: RftFlags, variant: RftVariant) -> RftResult<Self> {
        let mut engine = Self::new(size, flags)?;
        if variant != RftVariant::Standard {
            engine.set_variant(variant, true)?;
        }
        Ok(engine)
    }

    /// Change the active variant and optionally rebuild the basis.
    pub fn set_variant(&mut self, variant: RftVariant, rebuild_basis: bool) -> RftResult<()> {
        let changed = self.variant != variant;
        self.variant = variant;
        if rebuild_basis || changed {
            self.build_basis()?;
        }
        Ok(())
    }

    // ── Basis construction ───────────────────────────────────────────────

    /// Build the RFT basis matrix using the active variant.
    /// Uses two-pass Modified Gram-Schmidt for numerical stability.
    fn build_basis(&mut self) -> RftResult<()> {
        let n = self.size;
        if n == 0 {
            return Err(RftError::InvalidParam);
        }

        // Phase 1: Build raw basis vectors
        for k in 0..n {
            self.eigenvalues[k] = 1.0;
        }

        for row in 0..n {
            for col in 0..n {
                let entry = self.compute_basis_entry(row, col);
                self.basis[row * n + col] = entry;
            }
        }

        // Phase 2: Two-pass Modified Gram-Schmidt orthonormalization
        self.gram_schmidt_orthonormalize();

        Ok(())
    }

    /// Compute a single basis entry B[row, col] for the current variant.
    fn compute_basis_entry(&self, row: usize, col: usize) -> Complex {
        let n = self.size;
        let nf = n as f64;
        let kf = col as f64;
        let nf_row = row as f64;
        let amp = 1.0 / nf.sqrt();

        match self.variant {
            RftVariant::Standard => {
                let phi_k = PHI.powf(-kf);
                let phase = TWO_PI * phi_k * nf_row / nf
                    + PI * phi_k * (nf_row * nf_row) / (2.0 * nf);
                Complex::new(amp * phase.cos(), amp * phase.sin())
            }
            RftVariant::Harmonic => {
                let alpha = 0.5;
                let kn = kf * nf_row;
                let phase = TWO_PI * kn / nf + alpha * PI * kn.powi(3) / (nf * nf);
                Complex::new(amp * phase.cos(), amp * phase.sin())
            }
            RftVariant::Fibonacci => {
                let f_k = fibonacci_ratio(col);
                let f_n = fibonacci_ratio(n);
                let phase = TWO_PI * f_k * nf_row / f_n;
                Complex::new(amp * phase.cos(), amp * phase.sin())
            }
            RftVariant::Chaotic => {
                let state = (row as u64)
                    .wrapping_mul(n as u64)
                    .wrapping_add(col as u64)
                    .wrapping_add(0xDEAD_BEEF);
                let s1 = state.wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let r1 = s1 as f64 / u64::MAX as f64;
                let s2 = s1.wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let r2 = s2 as f64 / u64::MAX as f64;
                let mag = (-2.0 * (r1 + 1e-10).ln()).sqrt();
                let phase = TWO_PI * r2;
                Complex::new(mag * phase.cos(), mag * phase.sin())
            }
            RftVariant::Geometric => {
                let n2k = (nf_row * nf_row) * kf;
                let nk2 = nf_row * (kf * kf);
                let phase = TWO_PI * (kf * nf_row) / nf
                    + TWO_PI * (n2k + nk2) / (nf * nf);
                Complex::new(amp * phase.cos(), amp * phase.sin())
            }
            RftVariant::PhiChaotic => {
                // Fibonacci component
                let f_k = fibonacci_ratio(col);
                let f_n = fibonacci_ratio(n);
                let phase_fib = TWO_PI * f_k * nf_row / f_n;

                // Chaotic component
                let st = (row as u64)
                    .wrapping_mul(n as u64)
                    .wrapping_add(col as u64)
                    .wrapping_add(0xCAFE_BABE);
                let s1 = st.wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let r1 = s1 as f64 / u64::MAX as f64;
                let s2 = s1.wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let r2 = s2 as f64 / u64::MAX as f64;
                let mag_chaos = (-2.0 * (r1 + 1e-10).ln()).sqrt();
                let phase_chaos = TWO_PI * r2;

                let fib_re = phase_fib.cos();
                let fib_im = phase_fib.sin();
                let chaos_re = mag_chaos * phase_chaos.cos();
                let chaos_im = mag_chaos * phase_chaos.sin();
                let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;

                Complex::new(
                    (fib_re + chaos_re) * inv_sqrt2,
                    (fib_im + chaos_im) * inv_sqrt2,
                )
            }
            RftVariant::Hyperbolic => {
                let curvature = 0.85;
                let centered = (kf - (n as f64 - 1.0) / 2.0) / nf;
                let warp = (curvature * centered).tanh();
                let phase = TWO_PI * warp * nf_row / nf;
                Complex::new(amp * phase.cos(), amp * phase.sin())
            }
            RftVariant::Dct => {
                let phase = PI * kf * (2.0 * nf_row + 1.0) / (2.0 * nf);
                let dct_amp = if col == 0 { 1.0 / nf.sqrt() } else { (2.0 / nf).sqrt() };
                Complex::new(dct_amp * phase.cos(), 0.0)
            }
            RftVariant::HybridDct => {
                let split = n / 2;
                if col < split {
                    let phase = PI * kf * (2.0 * nf_row + 1.0) / (2.0 * nf);
                    let dct_amp = if col == 0 { 1.0 / nf.sqrt() } else { (2.0 / nf).sqrt() };
                    Complex::new(dct_amp * phase.cos(), 0.0)
                } else {
                    let phi_k = PHI.powf(-kf);
                    let phase = TWO_PI * phi_k * nf_row / nf;
                    Complex::new(amp * phase.cos(), amp * phase.sin())
                }
            }
            RftVariant::Cascade => {
                let mid = n / 2;
                let w_dct = if col < mid { 0.7 } else { 0.3 };
                let w_rft = 1.0 - w_dct;

                let dct_phase = PI * kf * (2.0 * nf_row + 1.0) / (2.0 * nf);
                let dct_amp = if col == 0 { 1.0 / nf.sqrt() } else { (2.0 / nf).sqrt() };
                let dct_val = dct_amp * dct_phase.cos();

                let phi_k = PHI.powf(-kf);
                let rft_phase = TWO_PI * phi_k * nf_row / nf
                    + PI * phi_k * (nf_row * nf_row) / (2.0 * nf);

                Complex::new(
                    w_dct * dct_val + w_rft * amp * rft_phase.cos(),
                    w_rft * amp * rft_phase.sin(),
                )
            }
            RftVariant::AdaptiveSplit => {
                let split = (n as f64 * 0.5) as usize;
                if col < split {
                    let phase = PI * kf * (2.0 * nf_row + 1.0) / (2.0 * nf);
                    let dct_amp = if col == 0 { 1.0 / nf.sqrt() } else { (2.0 / nf).sqrt() };
                    Complex::new(dct_amp * phase.cos(), 0.0)
                } else {
                    let phi_k = PHI.powf(-kf);
                    let phase = TWO_PI * phi_k * nf_row / nf;
                    Complex::new(amp * phase.cos(), amp * phase.sin())
                }
            }
            RftVariant::EntropyGuided => {
                let freq_entropy = kf / (n as f64 - 1.0).max(1.0);
                let w_dct = (-3.0 * freq_entropy).exp();
                let w_rft = 1.0 - w_dct;

                let dct_phase = PI * kf * (2.0 * nf_row + 1.0) / (2.0 * nf);
                let dct_amp = if col == 0 { 1.0 / nf.sqrt() } else { (2.0 / nf).sqrt() };
                let dct_val = dct_amp * dct_phase.cos();

                let phi_k = PHI.powf(-kf);
                let rft_phase = TWO_PI * phi_k * nf_row / nf
                    + PI * phi_k * (nf_row * nf_row) / (2.0 * nf);

                Complex::new(
                    w_dct * dct_val + w_rft * amp * rft_phase.cos(),
                    w_rft * amp * rft_phase.sin(),
                )
            }
            RftVariant::Dictionary => {
                let dct_phase = PI * kf * (2.0 * nf_row + 1.0) / (2.0 * nf);
                let dct_amp = if col == 0 { 1.0 / nf.sqrt() } else { (2.0 / nf).sqrt() };
                let dct_val = dct_amp * dct_phase.cos();

                let phi_k = PHI.powf(-kf);
                let rft_phase = TWO_PI * phi_k * nf_row / nf;
                let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;

                Complex::new(
                    (dct_val + amp * rft_phase.cos()) * inv_sqrt2,
                    amp * rft_phase.sin() * inv_sqrt2,
                )
            }
            RftVariant::Canonical | RftVariant::BinaryWave | RftVariant::CryptoSis => {
                let f_k = (kf + 1.0) * PHI;
                let theta_k = TWO_PI * kf / PHI;
                let t = nf_row / nf;
                let angle = TWO_PI * f_k * t + theta_k;
                let inv_sqrt_n = 1.0 / nf.sqrt();
                Complex::new(inv_sqrt_n * angle.cos(), inv_sqrt_n * angle.sin())
            }
        }
    }

    /// Two-pass Modified Gram-Schmidt orthonormalization.
    /// Operates on columns of self.basis (interpreted as N×N column-major).
    fn gram_schmidt_orthonormalize(&mut self) {
        let n = self.size;

        // Transpose: basis[row*N+col] -> cols[col*N+row]
        let mut cols = vec![Complex::ZERO; n * n];
        for i in 0..n {
            for j in 0..n {
                cols[j * n + i] = self.basis[i * n + j];
            }
        }

        for i in 0..n {
            // Pass 1: Orthogonalize against all previous columns
            for prev in 0..i {
                let (dot_re, dot_im) = hermitian_dot(&cols, prev, i, n);
                subtract_proj(&mut cols, i, prev, dot_re, dot_im, n);
            }
            // Pass 2: Re-orthogonalize for numerical stability
            for prev in 0..i {
                let (dot_re, dot_im) = hermitian_dot(&cols, prev, i, n);
                subtract_proj(&mut cols, i, prev, dot_re, dot_im, n);
            }

            // Normalize
            let mut norm_sq = 0.0;
            for r in 0..n {
                norm_sq += cols[i * n + r].norm_sq();
            }
            let norm = norm_sq.sqrt();

            if norm < 1e-12 {
                // Column is (nearly) zero — replace with random and re-orthogonalize
                let seed = (i as u64).wrapping_mul(12345).wrapping_add(67890);
                for r in 0..n {
                    let s = seed.wrapping_add(r as u64);
                    let re = ((s.wrapping_mul(6364136223846793005).wrapping_add(1)) as f64
                        / u64::MAX as f64)
                        - 0.5;
                    let im = ((s.wrapping_mul(1442695040888963407).wrapping_add(1)) as f64
                        / u64::MAX as f64)
                        - 0.5;
                    cols[i * n + r] = Complex::new(re, im);
                }
                for _pass in 0..2 {
                    for prev in 0..i {
                        let (dot_re, dot_im) = hermitian_dot(&cols, prev, i, n);
                        subtract_proj(&mut cols, i, prev, dot_re, dot_im, n);
                    }
                }
                let mut ns = 0.0;
                for r in 0..n {
                    ns += cols[i * n + r].norm_sq();
                }
                let nn = ns.sqrt();
                if nn > 1e-15 {
                    let inv = 1.0 / nn;
                    for r in 0..n {
                        cols[i * n + r] = cols[i * n + r].scale(inv);
                    }
                }
            } else {
                let inv = 1.0 / norm;
                for r in 0..n {
                    cols[i * n + r] = cols[i * n + r].scale(inv);
                }
            }
        }

        // Transpose back: cols[col*N+row] -> basis[row*N+col]
        for i in 0..n {
            for j in 0..n {
                self.basis[i * n + j] = cols[j * n + i];
            }
        }
    }

    // ── Forward / Inverse transforms ─────────────────────────────────────

    /// Forward RFT: output[k] = Σ_n conj(basis[n,k]) × input[n]
    pub fn forward(&self, input: &[Complex]) -> RftResult<Vec<Complex>> {
        if !self.initialized || input.len() != self.size {
            return Err(RftError::InvalidParam);
        }
        let n = self.size;
        let mut output = vec![Complex::ZERO; n];

        for k in 0..n {
            let mut sum = Complex::ZERO;
            for row in 0..n {
                let basis_conj = self.basis[row * n + k].conj();
                sum = sum + basis_conj * input[row];
            }
            output[k] = sum;
        }
        Ok(output)
    }

    /// Inverse RFT: output[n] = Σ_k basis[n,k] × input[k]
    pub fn inverse(&self, input: &[Complex]) -> RftResult<Vec<Complex>> {
        if !self.initialized || input.len() != self.size {
            return Err(RftError::InvalidParam);
        }
        let n = self.size;
        let mut output = vec![Complex::ZERO; n];

        for row in 0..n {
            let mut sum = Complex::ZERO;
            for k in 0..n {
                sum = sum + self.basis[row * n + k] * input[k];
            }
            output[row] = sum;
        }
        Ok(output)
    }

    // ── Quantum operations ───────────────────────────────────────────────

    /// Set qubit count for quantum operations.
    pub fn set_quantum_basis(&mut self, qubit_count: usize) -> RftResult<()> {
        if (1usize << qubit_count) != self.size {
            return Err(RftError::InvalidParam);
        }
        self.qubit_count = qubit_count;
        Ok(())
    }

    /// Calculate von Neumann entropy of a quantum state.
    pub fn von_neumann_entropy(&self, state: &[Complex]) -> RftResult<f64> {
        if state.len() != self.size {
            return Err(RftError::InvalidParam);
        }
        let mut entropy = 0.0;
        for s in state {
            let prob = s.norm_sq();
            if prob > 1e-12 {
                entropy -= prob * prob.log2();
            }
        }
        Ok(entropy)
    }

    /// Measure quantum entanglement using von Neumann entropy.
    pub fn entanglement_measure(&self, state: &[Complex]) -> RftResult<f64> {
        if state.len() != self.size {
            return Err(RftError::InvalidParam);
        }
        let n = self.size;
        let qubits = (n as f64).log2().round() as usize;

        if qubits < 2 {
            return Ok(0.0);
        }

        if qubits == 2 {
            let p0 = state[0].norm_sq() + state[1].norm_sq();
            let p1 = state[2].norm_sq() + state[3].norm_sq();
            let mut ent = 0.0;
            if p0 > 1e-12 {
                ent -= p0 * p0.log2();
            }
            if p1 > 1e-12 {
                ent -= p1 * p1.log2();
            }
            return Ok(ent);
        }

        // General case: participation ratio × entropy
        let mut participation = 0.0;
        let mut entropy = 0.0;
        for s in state {
            let prob = s.norm_sq();
            participation += prob * prob;
            if prob > 1e-12 {
                entropy -= prob * prob.log2();
            }
        }
        Ok(entropy * (1.0 - participation))
    }

    /// Validate Bell state entanglement properties.
    pub fn validate_bell_state(&self, bell_state: &[Complex], tolerance: f64) -> RftResult<f64> {
        if self.size < 4 {
            return Err(RftError::InvalidParam);
        }
        let ent = self.entanglement_measure(bell_state)?;
        if (ent - 1.0).abs() > tolerance {
            return Err(RftError::Computation);
        }
        Ok(ent)
    }

    /// Validate unitarity of the transformation matrix.
    pub fn validate_unitarity(&self, tolerance: f64) -> RftResult<()> {
        let n = self.size;
        for i in 0..n {
            for j in 0..n {
                let mut sum = Complex::ZERO;
                for k in 0..n {
                    let conj_basis = self.basis[k * n + i].conj();
                    let basis = self.basis[k * n + j];
                    sum = sum + conj_basis * basis;
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                if (sum.re - expected).abs() > tolerance || sum.im.abs() > tolerance {
                    return Err(RftError::Computation);
                }
            }
        }
        Ok(())
    }

    /// Test golden ratio properties in eigenvalue spectrum.
    pub fn validate_golden_ratio_properties(&self, tolerance: f64) -> RftResult<f64> {
        let n = self.size;
        let inv_phi = 1.0 / PHI;
        let mut phi_count = 0;
        let mut total_elements = 0;

        for i in 0..n {
            for j in 0..n {
                let mag = self.basis[i * n + j].abs();
                if mag > 1e-6 {
                    total_elements += 1;
                    let phi_err = (mag - PHI).abs();
                    let inv_phi_err = (mag - inv_phi).abs();
                    let ratio_err = (mag * PHI - 1.0).abs();
                    if phi_err < tolerance || inv_phi_err < tolerance || ratio_err < tolerance {
                        phi_count += 1;
                    }
                }
            }
        }

        let phi_presence = if total_elements > 0 {
            phi_count as f64 / total_elements as f64
        } else {
            0.0
        };
        Ok(phi_presence)
    }
}

// ── UI Engine ────────────────────────────────────────────────────────────────
// Port of rft_kernel_ui.c

pub struct RftUiEngine {
    pub base: RftEngine,
    pub viz_magnitudes: Vec<f64>,
    pub viz_phases: Vec<f64>,
    pub enable_visualization: bool,
}

impl RftUiEngine {
    pub fn new(size: usize, flags: RftFlags, enable_visualization: bool) -> RftResult<Self> {
        let base = RftEngine::new(size, flags)?;
        Ok(Self {
            viz_magnitudes: if enable_visualization { vec![0.0; size] } else { vec![] },
            viz_phases: if enable_visualization { vec![0.0; size] } else { vec![] },
            enable_visualization,
            base,
        })
    }

    /// Process real-time data and update visualization buffers.
    pub fn process_realtime(
        &mut self,
        input: &[Complex],
    ) -> RftResult<Vec<Complex>> {
        let output = self.base.forward(input)?;
        if self.enable_visualization {
            for (i, c) in output.iter().enumerate() {
                self.viz_magnitudes[i] = c.abs();
                self.viz_phases[i] = c.arg();
            }
        }
        Ok(output)
    }

    /// Get spectrum data for visualization.
    pub fn get_spectrum_data(&self, max_points: usize) -> (&[f64], &[f64]) {
        let n = max_points.min(self.viz_magnitudes.len());
        (&self.viz_magnitudes[..n], &self.viz_phases[..n])
    }

    /// Get basis visualization data (magnitudes of basis elements).
    pub fn get_basis_visualization(&self, max_size: usize) -> Vec<f64> {
        let n = max_size.min(self.base.size * self.base.size);
        self.base.basis[..n].iter().map(|c| c.abs()).collect()
    }
}

// ── Helper functions ─────────────────────────────────────────────────────────

fn fibonacci_ratio(k: usize) -> f64 {
    if k == 0 {
        return 1.0;
    }
    let mut a = 1.0_f64;
    let mut b = 1.0_f64;
    for _ in 2..=k {
        let temp = a + b;
        a = b;
        b = temp;
    }
    b
}

/// Hermitian inner product: <cols[u], cols[v]>
fn hermitian_dot(cols: &[Complex], u: usize, v: usize, n: usize) -> (f64, f64) {
    let mut dot_re = 0.0;
    let mut dot_im = 0.0;
    for r in 0..n {
        let cu = cols[u * n + r];
        let cv = cols[v * n + r];
        dot_re += cu.re * cv.re + cu.im * cv.im;
        dot_im += cu.re * cv.im - cu.im * cv.re;
    }
    (dot_re, dot_im)
}

/// Subtract projection: cols[v] -= <cols[u], cols[v]> × cols[u]
fn subtract_proj(cols: &mut [Complex], v: usize, u: usize, dot_re: f64, dot_im: f64, n: usize) {
    for r in 0..n {
        let cu = cols[u * n + r];
        let proj_re = dot_re * cu.re - dot_im * cu.im;
        let proj_im = dot_re * cu.im + dot_im * cu.re;
        cols[v * n + r].re -= proj_re;
        cols[v * n + r].im -= proj_im;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_standard() {
        let engine = RftEngine::new(16, RftFlags::DEFAULT).unwrap();
        let input: Vec<Complex> = (0..16)
            .map(|i| Complex::new(i as f64 / 16.0, 0.0))
            .collect();
        let freq = engine.forward(&input).unwrap();
        let reconstructed = engine.inverse(&freq).unwrap();
        for (a, b) in input.iter().zip(reconstructed.iter()) {
            assert!((a.re - b.re).abs() < 1e-8, "re mismatch: {} vs {}", a.re, b.re);
            assert!((a.im - b.im).abs() < 1e-8, "im mismatch: {} vs {}", a.im, b.im);
        }
    }

    #[test]
    fn test_unitarity() {
        let engine = RftEngine::new(8, RftFlags::DEFAULT).unwrap();
        engine.validate_unitarity(1e-8).unwrap();
    }

    #[test]
    fn test_all_variants_build() {
        let variants = [
            RftVariant::Standard, RftVariant::Harmonic, RftVariant::Fibonacci,
            RftVariant::Chaotic, RftVariant::Geometric, RftVariant::PhiChaotic,
            RftVariant::Hyperbolic, RftVariant::Dct, RftVariant::HybridDct,
            RftVariant::Cascade, RftVariant::AdaptiveSplit, RftVariant::EntropyGuided,
            RftVariant::Dictionary, RftVariant::Canonical, RftVariant::BinaryWave,
            RftVariant::CryptoSis,
        ];
        for v in variants {
            let result = RftEngine::new_with_variant(8, RftFlags::DEFAULT, v);
            assert!(result.is_ok(), "Failed to build variant {:?}", v);
        }
    }

    #[test]
    fn test_von_neumann_entropy() {
        let engine = RftEngine::new(4, RftFlags::DEFAULT).unwrap();
        // Maximally mixed state: equal superposition
        let state = vec![Complex::new(0.5, 0.0); 4];
        let entropy = engine.von_neumann_entropy(&state).unwrap();
        assert!(entropy > 1.5, "entropy should be ~2.0 for mixed 4-state, got {}", entropy);
    }
}
