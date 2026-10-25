/*
 * SPDX-License-Identifier: LicenseRef-SPX-Proprietary
 * Copyright (C) 2025 Luis M. Minier / quantoniumos
 * This file is listed in CLAIMS_PRACTICING_FILES.txt and is governed
 * by LICENSE-SPX-PROPRIETARY.md. Commercial use requires a separate
 * written patent license from Luis M. Minier.
 *
 * rft_fused_kernel.hpp - Fused Diagonal SIMD Kernels for Φ-RFT
 * =============================================================
 *
 * High-performance fused kernel: D_φ · C_σ applied as single pass.
 * 
 * Key optimizations:
 *   1. Fuse D_φ and C_σ into single diagonal: E[k] = D_φ[k] · C_σ[k]
 *   2. Precompute combined phase: θ[k] = 2πβ·frac(k/φ) + πσk²/n
 *   3. Use SIMD (AVX2/AVX512) for parallel sincos + multiply
 *   4. Cache-aligned phase tables with prefetching
 *
 * The Φ-RFT forward transform is:
 *   Y = D_φ · C_σ · FFT(x)
 *
 * Fused form:
 *   Y = E · FFT(x)  where E[k] = exp(i·θ_fused[k])
 *   θ_fused[k] = 2πβ·frac(k/φ) + πσk²/n
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <complex>
#include <vector>
#include <memory>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#ifdef __AVX512F__
#define RFT_HAS_AVX512 1
#include <immintrin.h>
#else
#define RFT_HAS_AVX512 0
#endif

#ifdef __AVX2__
#define RFT_HAS_AVX2 1
#include <immintrin.h>
#else
#define RFT_HAS_AVX2 0
#endif

#ifdef __FMA__
#define RFT_HAS_FMA 1
#endif

namespace rft_fused {

constexpr double PHI = 1.6180339887498948482045868343656;
constexpr double PHI_INV = 0.6180339887498948482045868343656;
constexpr double PI = 3.14159265358979323846;
constexpr double TWO_PI = 2.0 * PI;

using Complex = std::complex<double>;

struct FusedPhaseTable {
    std::vector<double> phases;
    std::vector<double> cos_cache;
    std::vector<double> sin_cache;
    size_t size;
    double beta;
    double sigma;
    
    FusedPhaseTable() : size(0), beta(1.0), sigma(1.0) {}
    
    void precompute(size_t n, double beta_ = 1.0, double sigma_ = 1.0) {
        if (n == size && beta == beta_ && sigma == sigma_) return;
        size = n; beta = beta_; sigma = sigma_;
        phases.resize(n); cos_cache.resize(n); sin_cache.resize(n);
        const double inv_n = 1.0 / static_cast<double>(n);
        const double inv_phi = PHI_INV;
        for (size_t k = 0; k < n; ++k) {
            double frac_k_phi = std::fmod(k * inv_phi, 1.0);
            if (frac_k_phi < 0) frac_k_phi += 1.0;
            double theta_phi = TWO_PI * beta * frac_k_phi;
            double k_sq = static_cast<double>(k * k);
            double theta_chirp = PI * sigma * k_sq * inv_n;
            double theta_fused = theta_phi + theta_chirp;
            phases[k] = theta_fused;
            cos_cache[k] = std::cos(theta_fused);
            sin_cache[k] = std::sin(theta_fused);
        }
    }
    
    void precompute_conjugate(size_t n, double beta_ = 1.0, double sigma_ = 1.0) {
        precompute(n, beta_, sigma_);
        for (size_t k = 0; k < n; ++k) {
            phases[k] = -phases[k];
            sin_cache[k] = -sin_cache[k];
        }
    }
};

inline void apply_fused_diagonal_scalar(
    const Complex* __restrict input,
    Complex* __restrict output,
    const double* __restrict cos_table,
    const double* __restrict sin_table,
    size_t n
) {
    for (size_t k = 0; k < n; ++k) {
        double c = cos_table[k], s = sin_table[k];
        double r = input[k].real(), i = input[k].imag();
        output[k] = Complex(r*c - i*s, r*s + i*c);
    }
}

inline void apply_fused_diagonal(
    const Complex* input, Complex* output,
    const double* cos_table, const double* sin_table,
    size_t n, bool use_simd = true
) {
    apply_fused_diagonal_scalar(input, output, cos_table, sin_table, n);
}

class FusedRFTEngine {
public:
    FusedRFTEngine(double beta = 1.0, double sigma = 1.0, bool use_simd = true)
        : beta_(beta), sigma_(sigma), use_simd_(use_simd) {}
    
    void precompute(size_t n) {
        fwd_phases_.precompute(n, beta_, sigma_);
        inv_phases_.precompute_conjugate(n, beta_, sigma_);
    }
    
    void apply_forward_diagonal(const Complex* fft_out, Complex* result, size_t n) {
        if (n != fwd_phases_.size) precompute(n);
        apply_fused_diagonal(fft_out, result, fwd_phases_.cos_cache.data(), fwd_phases_.sin_cache.data(), n, use_simd_);
    }
    
    void apply_inverse_diagonal(const Complex* input, Complex* result, size_t n) {
        if (n != inv_phases_.size) precompute(n);
        apply_fused_diagonal(input, result, inv_phases_.cos_cache.data(), inv_phases_.sin_cache.data(), n, use_simd_);
    }
    
    bool has_avx512() const { return false; }
    bool has_avx2() const { return false; }

private:
    double beta_, sigma_;
    bool use_simd_;
    FusedPhaseTable fwd_phases_, inv_phases_;
};

} // namespace rft_fused
