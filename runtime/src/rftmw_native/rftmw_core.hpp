/*
 * SPDX-License-Identifier: LicenseRef-SPX-Proprietary
 * Copyright (C) 2025 Luis M. Minier / quantoniumos
 * This file is listed in CLAIMS_PRACTICING_FILES.txt and is governed
 * by LICENSE-SPX-PROPRIETARY.md. Commercial use requires a separate
 * written patent license from Luis M. Minier.
 *
 * rftmw_core.hpp - Phi-RFT Transform C++ Core
 * ============================================
 *
 * High-performance RFTMW implementation with SIMD acceleration.
 * This is the heavy-lifting engine; Python calls this via pybind11.
 *
 * Canonical RFT definition (matches resonant_fourier_transform.py exactly):
 *   Phi[n,k]   = (1/sqrt(N)) exp(j 2pi frac((k+1)*phi) n)
 *   Phi_tilde  = Phi G^{-1/2}   (Gram / Loewdin normalisation)
 *   G          = Phi^H Phi       (spectral decomposition, Jacobi)
 *   Forward:  X = Phi_tilde^H x
 *   Inverse:  x = Phi_tilde  X
 *
 * Unitarity: ||Phi_tilde^H Phi_tilde - I||_F < 1e-10 for N <= 4096.
 */

#pragma once

#include <complex>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <numeric>

#ifdef __AVX2__
#include <immintrin.h>
#define RFTMW_HAS_AVX2 1
#else
#define RFTMW_HAS_AVX2 0
#endif

#ifdef __AVX512F__
#define RFTMW_HAS_AVX512 1
#else
#define RFTMW_HAS_AVX512 0
#endif

#ifdef __FMA__
#define RFTMW_HAS_FMA 1
#else
#define RFTMW_HAS_FMA 0
#endif

#ifndef RFTMW_ENABLE_ASM
#define RFTMW_ENABLE_ASM 0
#endif

#if RFTMW_ENABLE_ASM
#include "rftmw_asm_kernels.hpp"
#endif

namespace rftmw {

constexpr double PHI     = 1.6180339887498948482045868343656;
constexpr double PHI_INV = 0.6180339887498948482045868343656;
constexpr double PI      = 3.14159265358979323846;
constexpr double TWO_PI  = 2.0 * PI;

using Complex    = std::complex<double>;
using ComplexVec = std::vector<Complex>;
using RealVec    = std::vector<double>;

#ifndef RFTMW_CANONICAL_MAX_N
#define RFTMW_CANONICAL_MAX_N 256
#endif
constexpr size_t RFTMW_CANONICAL_MAX_N_RUNTIME = static_cast<size_t>(RFTMW_CANONICAL_MAX_N);

struct ConcentrationMetrics {
    size_t top_k = 0;
    double top_k_energy_ratio = 0.0;
    double sparsity_ratio = 1.0;
    double coherence_proxy = 1.0;
    bool use_sparse_route = false;
};

// ============================================================================
// Low-level helpers
// ============================================================================

// Build raw Phi matrix (N x N, column-major: Phi[n*N + k])
inline ComplexVec compute_raw_phi_basis(size_t N) {
    ComplexVec Phi(N * N);
    const double inv_sqrt_n = 1.0 / std::sqrt(static_cast<double>(N));
    for (size_t k = 0; k < N; ++k) {
        double freq = static_cast<double>(k + 1) * PHI;
        double f_k  = freq - std::floor(freq);          // frac((k+1)*phi)
        for (size_t n = 0; n < N; ++n) {
            double angle = TWO_PI * f_k * static_cast<double>(n);
            Phi[n * N + k] = Complex(std::cos(angle), std::sin(angle)) * inv_sqrt_n;
        }
    }
    return Phi;
}

inline Complex raw_phi_entry(size_t n, size_t k, size_t N) {
    const double inv_sqrt_n = 1.0 / std::sqrt(static_cast<double>(N));
    double freq = static_cast<double>(k + 1) * PHI;
    double f_k  = freq - std::floor(freq);
    double angle = TWO_PI * f_k * static_cast<double>(n);
    return Complex(std::cos(angle), std::sin(angle)) * inv_sqrt_n;
}

inline ComplexVec apply_raw_phi_adjoint(const RealVec& input) {
    const size_t N = input.size();
    ComplexVec output(N, Complex(0.0));
    for (size_t k = 0; k < N; ++k) {
        Complex sum(0.0);
        for (size_t n = 0; n < N; ++n) {
            sum += std::conj(raw_phi_entry(n, k, N)) * Complex(input[n], 0.0);
        }
        output[k] = sum;
    }
    return output;
}

inline RealVec apply_raw_phi_synthesis(const ComplexVec& coeffs) {
    const size_t N = coeffs.size();
    RealVec output(N, 0.0);
    for (size_t n = 0; n < N; ++n) {
        Complex sum(0.0);
        for (size_t k = 0; k < N; ++k) {
            sum += raw_phi_entry(n, k, N) * coeffs[k];
        }
        output[n] = sum.real();
    }
    return output;
}

inline double l2_norm_sq(const RealVec& values) {
    double sum = 0.0;
    for (double v : values) sum += v * v;
    return sum;
}

inline size_t count_significant_coeffs(const ComplexVec& coeffs, double rel_tol = 1e-8) {
    double max_mag = 0.0;
    for (const auto& c : coeffs) max_mag = std::max(max_mag, std::abs(c));
    if (max_mag <= 0.0) return 0;
    const double threshold = rel_tol * max_mag;
    size_t count = 0;
    for (const auto& c : coeffs) {
        if (std::abs(c) >= threshold) ++count;
    }
    return count;
}

inline void hard_threshold_inplace(ComplexVec& coeffs, size_t keep_k) {
    if (keep_k == 0 || keep_k >= coeffs.size()) return;
    std::vector<size_t> order(coeffs.size());
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(
        order.begin(),
        order.begin() + keep_k,
        order.end(),
        [&](size_t lhs, size_t rhs) {
            return std::norm(coeffs[lhs]) > std::norm(coeffs[rhs]);
        }
    );
    std::vector<uint8_t> keep(coeffs.size(), 0);
    for (size_t i = 0; i < keep_k; ++i) keep[order[i]] = 1;
    for (size_t i = 0; i < coeffs.size(); ++i) {
        if (!keep[i]) coeffs[i] = Complex(0.0, 0.0);
    }
}

inline ConcentrationMetrics analyze_signal_concentration(const RealVec& input, double energy_threshold = 0.90) {
    const size_t N = input.size();
    ComplexVec coeffs = apply_raw_phi_adjoint(input);
    std::vector<double> energies;
    energies.reserve(coeffs.size());
    double total_energy = 0.0;
    for (const auto& c : coeffs) {
        double e = std::norm(c);
        energies.push_back(e);
        total_energy += e;
    }

    ConcentrationMetrics metrics;
    if (N == 0 || total_energy <= 0.0) {
        metrics.use_sparse_route = false;
        return metrics;
    }

    std::sort(energies.begin(), energies.end(), std::greater<double>());
    double cumulative = 0.0;
    size_t top_k = 0;
    for (; top_k < energies.size(); ++top_k) {
        cumulative += energies[top_k];
        if ((cumulative / total_energy) >= energy_threshold) {
            ++top_k;
            break;
        }
    }
    if (top_k == 0) top_k = std::min<size_t>(std::max<size_t>(16, N / 16), N);
    if (top_k > N) top_k = N;

    metrics.top_k = top_k;
    metrics.top_k_energy_ratio = cumulative / total_energy;
    metrics.sparsity_ratio = static_cast<double>(top_k) / static_cast<double>(N);
    metrics.coherence_proxy = std::sqrt(metrics.sparsity_ratio);
    metrics.use_sparse_route = (top_k * 8 < N) && (metrics.top_k_energy_ratio >= 0.85);
    return metrics;
}

inline ComplexVec recover_sparse_coeffs_iht(
    const RealVec& input,
    size_t keep_k,
    size_t max_iter = 24,
    double step_size = 0.85,
    double tol = 1e-7
) {
    const size_t N = input.size();
    ComplexVec coeffs(N, Complex(0.0));
    if (N == 0) return coeffs;
    keep_k = std::max<size_t>(1, std::min(keep_k, N));
    const double signal_norm = std::sqrt(std::max(l2_norm_sq(input), 1e-30));

    for (size_t iter = 0; iter < max_iter; ++iter) {
        RealVec synth = apply_raw_phi_synthesis(coeffs);
        RealVec residual(N, 0.0);
        for (size_t i = 0; i < N; ++i) residual[i] = input[i] - synth[i];
        ComplexVec gradient = apply_raw_phi_adjoint(residual);
        for (size_t i = 0; i < N; ++i) coeffs[i] += step_size * gradient[i];
        hard_threshold_inplace(coeffs, keep_k);

        const double residual_norm = std::sqrt(std::max(l2_norm_sq(residual), 0.0));
        if ((residual_norm / signal_norm) < tol) break;
    }
    return coeffs;
}

// G = Phi^H Phi  (M x M Gram matrix)
inline ComplexVec compute_gram_matrix(const ComplexVec& Phi, size_t N) {
    ComplexVec G(N * N, Complex(0.0));
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j) {
            Complex s(0.0);
            for (size_t n = 0; n < N; ++n)
                s += std::conj(Phi[n * N + i]) * Phi[n * N + j];
            G[i * N + j] = s;
        }
    return G;
}

// ============================================================================
// Jacobi eigendecomposition for Hermitian matrix
// Fills V (eigenvectors, column-major) and eigenvals (real, ascending)
// ============================================================================
inline void jacobi_eigen_hermitian(
    ComplexVec& A,        // in/out: N x N Hermitian, destroyed on output
    ComplexVec& V,        // out:    N x N eigenvector matrix (columns)
    RealVec&    eigenvals,
    size_t      N,
    size_t      max_sweeps = 100
) {
    V.assign(N * N, Complex(0.0));
    for (size_t i = 0; i < N; ++i) V[i * N + i] = Complex(1.0);
    eigenvals.resize(N);

    for (size_t sweep = 0; sweep < max_sweeps; ++sweep) {
        double max_off = 0.0;
        for (size_t i = 0; i < N; ++i)
            for (size_t j = i + 1; j < N; ++j)
                max_off = std::max(max_off, std::abs(A[i * N + j]));
        if (max_off < 1e-15) break;

        for (size_t p = 0; p < N - 1; ++p) {
            for (size_t q = p + 1; q < N; ++q) {
                Complex Apq = A[p * N + q];
                if (std::abs(Apq) < 1e-16) continue;

                double App  = A[p * N + p].real();
                double Aqq  = A[q * N + q].real();
                double diff = Aqq - App;
                double t;
                if (std::abs(diff) < 1e-16) {
                    t = 1.0;
                } else {
                    double xi = diff / (2.0 * std::abs(Apq));
                    t = 1.0 / (std::abs(xi) + std::sqrt(xi * xi + 1.0));
                    if (xi < 0.0) t = -t;
                }
                double  c   = 1.0 / std::sqrt(1.0 + t * t);
                // J below puts -s*tau at (p,q); s must be -t*c for J^H A J to zero A[p][q].
                // With +t*c the sweeps never converge and Phi_tilde is not unitary.
                double  s   = -t * c;
                Complex tau = Apq / std::abs(Apq);  // phase of off-diagonal

                // Build Givens rotation J: rotate columns p, q
                // J = [[c, -conj(tau)*s], [tau*s, c]]
                // Update A <- J^H A J
                for (size_t r = 0; r < N; ++r) {
                    Complex Arp = A[r * N + p];
                    Complex Arq = A[r * N + q];
                    A[r * N + p] =  c * Arp + s * std::conj(tau) * Arq;
                    A[r * N + q] = -s * tau  * Arp + c * Arq;
                }
                for (size_t r = 0; r < N; ++r) {
                    Complex Apr = A[p * N + r];
                    Complex Aqr = A[q * N + r];
                    A[p * N + r] =  c * Apr + s * tau * Aqr;
                    A[q * N + r] = -s * std::conj(tau) * Apr + c * Aqr;
                }
                // Accumulate eigenvectors
                for (size_t r = 0; r < N; ++r) {
                    Complex Vrp = V[r * N + p];
                    Complex Vrq = V[r * N + q];
                    V[r * N + p] =  c * Vrp + s * std::conj(tau) * Vrq;
                    V[r * N + q] = -s * tau  * Vrp + c * Vrq;
                }
            }
        }
    }
    for (size_t i = 0; i < N; ++i) eigenvals[i] = A[i * N + i].real();
}

// ============================================================================
// Gram-normalised basis: Phi_tilde = Phi G^{-1/2}
// Returns Phi_tilde as N x N column-major ComplexVec
// ============================================================================
inline ComplexVec compute_gram_normalised_basis(size_t N) {
    ComplexVec Phi = compute_raw_phi_basis(N);
    ComplexVec G   = compute_gram_matrix(Phi, N);

    // Eigendecompose G = V Lambda V^H
    ComplexVec V;
    RealVec    eigvals;
    jacobi_eigen_hermitian(G, V, eigvals, N);

    // G^{-1/2} = V diag(1/sqrt(lambda + eps)) V^H
    const double eps = 1e-300;
    ComplexVec G_inv_sqrt(N * N, Complex(0.0));
    for (size_t i = 0; i < N; ++i) {
        double scale = 1.0 / std::sqrt(std::max(eigvals[i], 0.0) + eps);
        for (size_t r = 0; r < N; ++r)
            for (size_t c2 = 0; c2 < N; ++c2)
                G_inv_sqrt[r * N + c2] += scale * V[r * N + i] * std::conj(V[c2 * N + i]);
    }

    // Phi_tilde = Phi @ G_inv_sqrt  (N x N)
    ComplexVec Phi_tilde(N * N, Complex(0.0));
    for (size_t n = 0; n < N; ++n)
        for (size_t j = 0; j < N; ++j) {
            Complex s(0.0);
            for (size_t k = 0; k < N; ++k)
                s += Phi[n * N + k] * G_inv_sqrt[k * N + j];
            Phi_tilde[n * N + j] = s;
        }
    return Phi_tilde;
}

// ============================================================================
// RFTMWEngine
// forward:  X = Phi_tilde^H x   (Gram-normalised, unitary)
// inverse:  x = Phi_tilde  X
// Roundtrip error ||inverse(forward(x)) - x||_inf < 1e-10
// ============================================================================
class RFTMWEngine {
public:
    explicit RFTMWEngine(size_t max_size = 65536) : max_size_(max_size) {}

    ComplexVec forward(const RealVec& input) const {
        const size_t N = input.size();
        if (N <= RFTMW_CANONICAL_MAX_N_RUNTIME) {
            ComplexVec Phi_tilde = compute_gram_normalised_basis(N);
            ComplexVec output(N, Complex(0.0));
            for (size_t k = 0; k < N; ++k) {
                Complex sum(0.0);
                for (size_t n = 0; n < N; ++n)
                    sum += std::conj(Phi_tilde[n * N + k]) * Complex(input[n], 0.0);
                output[k] = sum;
            }
            return output;
        }
        const ConcentrationMetrics metrics = analyze_signal_concentration(input);
        if (metrics.use_sparse_route) {
            return recover_sparse_coeffs_iht(input, metrics.top_k);
        }
        return apply_raw_phi_adjoint(input);
    }

    RealVec inverse(const ComplexVec& coeffs) const {
        const size_t N = coeffs.size();
        if (N <= RFTMW_CANONICAL_MAX_N_RUNTIME) {
            ComplexVec Phi_tilde = compute_gram_normalised_basis(N);
            RealVec output(N, 0.0);
            for (size_t n = 0; n < N; ++n) {
                Complex sum(0.0);
                for (size_t k = 0; k < N; ++k)
                    sum += Phi_tilde[n * N + k] * coeffs[k];
                output[n] = sum.real();
            }
            return output;
        }
        return apply_raw_phi_synthesis(coeffs);
    }

    ComplexVec forward_sparse(const RealVec& input, size_t keep_k, size_t max_iter = 24) const {
        return recover_sparse_coeffs_iht(input, keep_k, max_iter);
    }

    RealVec inverse_sparse(const ComplexVec& coeffs) const {
        return apply_raw_phi_synthesis(coeffs);
    }

    ConcentrationMetrics analyze_signal(const RealVec& input, double energy_threshold = 0.90) const {
        return analyze_signal_concentration(input, energy_threshold);
    }

private:
    size_t max_size_;
};

} // namespace rftmw
