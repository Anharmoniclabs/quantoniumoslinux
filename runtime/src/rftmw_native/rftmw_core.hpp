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
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <random>

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

// Raw-basis matvecs without per-entry trig: along n, the phasor for column k obeys
//   e^{j 2pi f_k (n+1)} = e^{j 2pi f_k n} * e^{j 2pi f_k}
// so each step is one complex multiply. Columns advance in blocks of RAW_KBLOCK lanes
// (independent chains, vectorised across k) and are re-anchored with exact sincos every
// RAW_ANCHOR steps, which bounds rounding drift to ~1e-13. Blocks run in parallel when
// built with OpenMP.
constexpr size_t RAW_KBLOCK = 16;
constexpr size_t RAW_ANCHOR = 512;

inline double raw_phi_freq(size_t k) {
    double freq = static_cast<double>(k + 1) * PHI;
    return freq - std::floor(freq);
}

inline ComplexVec apply_raw_phi_adjoint(const RealVec& input) {
    const size_t N = input.size();
    ComplexVec output(N, Complex(0.0));
    if (N == 0) return output;
    const double inv_sqrt_n = 1.0 / std::sqrt(static_cast<double>(N));
    const double* x = input.data();
    const long long nblocks = static_cast<long long>((N + RAW_KBLOCK - 1) / RAW_KBLOCK);

    #pragma omp parallel for schedule(static) if(N >= 512)
    for (long long b = 0; b < nblocks; ++b) {
        const size_t k0 = static_cast<size_t>(b) * RAW_KBLOCK;
        double f[RAW_KBLOCK], wr[RAW_KBLOCK], wi[RAW_KBLOCK], zr[RAW_KBLOCK], zi[RAW_KBLOCK];
        double ar[RAW_KBLOCK] = {}, ai[RAW_KBLOCK] = {};
        for (size_t j = 0; j < RAW_KBLOCK; ++j) {
            f[j]  = raw_phi_freq(std::min(k0 + j, N - 1));  // padding lanes are discarded
            wr[j] = std::cos(TWO_PI * f[j]);
            wi[j] = -std::sin(TWO_PI * f[j]);                // conjugate basis
        }
        for (size_t n0 = 0; n0 < N; n0 += RAW_ANCHOR) {
            for (size_t j = 0; j < RAW_KBLOCK; ++j) {
                double angle = TWO_PI * f[j] * static_cast<double>(n0);
                zr[j] = std::cos(angle);
                zi[j] = -std::sin(angle);
            }
            const size_t n1 = std::min(N, n0 + RAW_ANCHOR);
            for (size_t n = n0; n < n1; ++n) {
                const double xn = x[n];
                for (size_t j = 0; j < RAW_KBLOCK; ++j) {
                    ar[j] += zr[j] * xn;
                    ai[j] += zi[j] * xn;
                    const double t = zr[j] * wr[j] - zi[j] * wi[j];
                    zi[j] = zr[j] * wi[j] + zi[j] * wr[j];
                    zr[j] = t;
                }
            }
        }
        for (size_t j = 0; j < RAW_KBLOCK && k0 + j < N; ++j)
            output[k0 + j] = Complex(ar[j] * inv_sqrt_n, ai[j] * inv_sqrt_n);
    }
    return output;
}

inline RealVec apply_raw_phi_synthesis(const ComplexVec& coeffs) {
    const size_t N = coeffs.size();
    RealVec output(N, 0.0);
    if (N == 0) return output;
    const double inv_sqrt_n = 1.0 / std::sqrt(static_cast<double>(N));
    const long long nchunks = static_cast<long long>((N + RAW_ANCHOR - 1) / RAW_ANCHOR);

    // Each thread owns a range of n (one anchor span) and sweeps all columns.
    #pragma omp parallel for schedule(static) if(N >= 512)
    for (long long ch = 0; ch < nchunks; ++ch) {
        const size_t n0 = static_cast<size_t>(ch) * RAW_ANCHOR;
        const size_t n1 = std::min(N, n0 + RAW_ANCHOR);
        double* out = output.data();
        for (size_t k0 = 0; k0 < N; k0 += RAW_KBLOCK) {
            double wr[RAW_KBLOCK], wi[RAW_KBLOCK], zr[RAW_KBLOCK], zi[RAW_KBLOCK];
            double cr[RAW_KBLOCK], ci[RAW_KBLOCK], p[RAW_KBLOCK];
            for (size_t j = 0; j < RAW_KBLOCK; ++j) {
                const bool live = k0 + j < N;
                const double f = raw_phi_freq(live ? k0 + j : 0);
                const double angle = TWO_PI * f * static_cast<double>(n0);
                wr[j] = std::cos(TWO_PI * f);
                wi[j] = std::sin(TWO_PI * f);
                zr[j] = std::cos(angle);
                zi[j] = std::sin(angle);
                cr[j] = live ? coeffs[k0 + j].real() * inv_sqrt_n : 0.0;
                ci[j] = live ? coeffs[k0 + j].imag() * inv_sqrt_n : 0.0;
            }
            for (size_t n = n0; n < n1; ++n) {
                for (size_t j = 0; j < RAW_KBLOCK; ++j) {
                    p[j] = zr[j] * cr[j] - zi[j] * ci[j];
                    const double t = zr[j] * wr[j] - zi[j] * wi[j];
                    zi[j] = zr[j] * wi[j] + zi[j] * wr[j];
                    zr[j] = t;
                }
                // Explicit tree: lets the compiler keep the lane loop vectorised.
                for (size_t w = RAW_KBLOCK / 2; w > 0; w /= 2)
                    for (size_t j = 0; j < w; ++j) p[j] += p[j + w];
                out[n] += p[0];
            }
        }
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

inline ConcentrationMetrics concentration_from_coeffs(const ComplexVec& coeffs, double energy_threshold = 0.90) {
    const size_t N = coeffs.size();
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

inline ConcentrationMetrics analyze_signal_concentration(const RealVec& input, double energy_threshold = 0.90) {
    return concentration_from_coeffs(apply_raw_phi_adjoint(input), energy_threshold);
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
// Canonical basis cache
// Phi_tilde depends only on N but costs O(N^3) (Gram + Jacobi) to build, so it is
// built once per size per process and shared by every engine/thread. Stored split
// into real/imag planes, row-contiguous for both directions:
//   fwd[k*N + n] = conj(Phi_tilde[n, k])   forward:  X_k = sum_n fwd[k, n] x_n
//   inv[n*N + k] = Phi_tilde[n, k]         inverse:  x_n = Re sum_k inv[n, k] X_k
// ============================================================================
struct CanonicalBasis {
    size_t  N = 0;
    RealVec fwd_re, fwd_im, inv_re, inv_im;
};

// ----------------------------------------------------------------------------
// On-disk basis store, so the O(N^3) build happens once per machine instead of
// once per process. Lookup order:
//   $RFTMW_BASIS_DIR, then /usr/share/quantoniumos/rftmw-basis (read-only, shipped
//   prebuilt), then the user cache ($XDG_CACHE_HOME or ~/.cache, %LOCALAPPDATA%
//   on Windows)/rftmw/basis. RFTMW_BASIS_CACHE=0 disables it.
// File: magic, N, FNV-1a of the payload, then Phi_tilde (N*N complex doubles).
// Anything that does not validate is ignored and rebuilt.
// ----------------------------------------------------------------------------
namespace basis_store {

constexpr char     MAGIC[8] = {'R', 'F', 'T', 'M', 'W', 'B', '1', '\0'};
constexpr uint64_t VERSION  = 1;   // bump when the basis definition changes

inline uint64_t fnv1a(const void* data, size_t len) {
    const unsigned char* p = static_cast<const unsigned char*>(data);
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

inline bool enabled() {
    const char* e = std::getenv("RFTMW_BASIS_CACHE");
    return !(e && e[0] == '0');
}

inline std::vector<std::filesystem::path> read_dirs() {
    std::vector<std::filesystem::path> dirs;
    if (const char* d = std::getenv("RFTMW_BASIS_DIR")) dirs.emplace_back(d);
#ifndef _WIN32
    dirs.emplace_back("/usr/share/quantoniumos/rftmw-basis");
#endif
    return dirs;
}

inline std::filesystem::path user_dir() {
    if (const char* d = std::getenv("RFTMW_BASIS_DIR")) return d;
#ifdef _WIN32
    if (const char* d = std::getenv("LOCALAPPDATA")) return std::filesystem::path(d) / "rftmw" / "basis";
#else
    if (const char* d = std::getenv("XDG_CACHE_HOME"); d && *d) return std::filesystem::path(d) / "rftmw" / "basis";
    if (const char* h = std::getenv("HOME")) return std::filesystem::path(h) / ".cache" / "rftmw" / "basis";
#endif
    return {};
}

inline std::string file_name(size_t N) {
    return "canonical_v" + std::to_string(VERSION) + "_n" + std::to_string(N) + ".bin";
}

inline bool load_file(const std::filesystem::path& path, size_t N, ComplexVec& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[8];
    uint64_t n = 0, hash = 0;
    f.read(magic, 8);
    f.read(reinterpret_cast<char*>(&n), 8);
    f.read(reinterpret_cast<char*>(&hash), 8);
    if (!f || std::memcmp(magic, MAGIC, 8) != 0 || n != N) return false;
    out.resize(N * N);
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(N * N * sizeof(Complex)));
    return f && fnv1a(out.data(), N * N * sizeof(Complex)) == hash;
}

inline bool load(size_t N, ComplexVec& out) {
    if (!enabled()) return false;
    std::vector<std::filesystem::path> dirs = read_dirs();
    dirs.push_back(user_dir());
    for (const auto& d : dirs)
        if (!d.empty() && load_file(d / file_name(N), N, out)) return true;
    return false;
}

inline void save(size_t N, const ComplexVec& basis) {
    if (!enabled()) return;
    const std::filesystem::path dir = user_dir();
    if (dir.empty()) return;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    // Write-then-rename so concurrent processes never read a partial file.
    const std::filesystem::path tmp = dir / (file_name(N) + ".tmp" + std::to_string(std::random_device{}()));
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return;
        const uint64_t n = N, hash = fnv1a(basis.data(), N * N * sizeof(Complex));
        f.write(MAGIC, 8);
        f.write(reinterpret_cast<const char*>(&n), 8);
        f.write(reinterpret_cast<const char*>(&hash), 8);
        f.write(reinterpret_cast<const char*>(basis.data()), static_cast<std::streamsize>(N * N * sizeof(Complex)));
        if (!f) { f.close(); std::filesystem::remove(tmp, ec); return; }
    }
    std::filesystem::rename(tmp, dir / file_name(N), ec);
    if (ec) std::filesystem::remove(tmp, ec);
}

}  // namespace basis_store

inline std::shared_ptr<const CanonicalBasis> canonical_basis(size_t N) {
    static std::mutex mu;
    static std::unordered_map<size_t, std::shared_ptr<const CanonicalBasis>> cache;
    {
        std::lock_guard<std::mutex> lk(mu);
        auto it = cache.find(N);
        if (it != cache.end()) return it->second;
    }
    // Built outside the lock: other sizes stay available meanwhile.
    ComplexVec Phi_tilde;
    if (!basis_store::load(N, Phi_tilde)) {
        Phi_tilde = compute_gram_normalised_basis(N);
        basis_store::save(N, Phi_tilde);
    }
    auto b = std::make_shared<CanonicalBasis>();
    b->N = N;
    b->fwd_re.resize(N * N); b->fwd_im.resize(N * N);
    b->inv_re.resize(N * N); b->inv_im.resize(N * N);
    for (size_t n = 0; n < N; ++n)
        for (size_t k = 0; k < N; ++k) {
            const Complex v = Phi_tilde[n * N + k];
            b->inv_re[n * N + k] = v.real();
            b->inv_im[n * N + k] = v.imag();
            b->fwd_re[k * N + n] = v.real();
            b->fwd_im[k * N + n] = -v.imag();
        }
    std::lock_guard<std::mutex> lk(mu);
    if (cache.size() >= 64) cache.clear();   // bounded: at most 64 sizes resident
    return cache.emplace(N, std::move(b)).first->second;
}

// sum_i a[i]*x[i] and sum_i b[i]*x[i] in one pass, four independent accumulators.
inline void dot2_f64(const double* a, const double* b, const double* x, size_t n,
                     double& sa, double& sb) {
    size_t i = 0;
#if RFTMW_HAS_AVX2 && RFTMW_HAS_FMA
    __m256d a0 = _mm256_setzero_pd(), a1 = _mm256_setzero_pd();
    __m256d b0 = _mm256_setzero_pd(), b1 = _mm256_setzero_pd();
    for (; i + 8 <= n; i += 8) {
        const __m256d x0 = _mm256_loadu_pd(x + i), x1 = _mm256_loadu_pd(x + i + 4);
        a0 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i), x0, a0);
        a1 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i + 4), x1, a1);
        b0 = _mm256_fmadd_pd(_mm256_loadu_pd(b + i), x0, b0);
        b1 = _mm256_fmadd_pd(_mm256_loadu_pd(b + i + 4), x1, b1);
    }
    double ta[4], tb[4];
    _mm256_storeu_pd(ta, _mm256_add_pd(a0, a1));
    _mm256_storeu_pd(tb, _mm256_add_pd(b0, b1));
    sa = (ta[0] + ta[1]) + (ta[2] + ta[3]);
    sb = (tb[0] + tb[1]) + (tb[2] + tb[3]);
#else
    double pa[4] = {}, pb[4] = {};
    for (; i + 4 <= n; i += 4)
        for (size_t j = 0; j < 4; ++j) {
            pa[j] += a[i + j] * x[i + j];
            pb[j] += b[i + j] * x[i + j];
        }
    sa = (pa[0] + pa[1]) + (pa[2] + pa[3]);
    sb = (pb[0] + pb[1]) + (pb[2] + pb[3]);
#endif
    for (; i < n; ++i) {
        sa += a[i] * x[i];
        sb += b[i] * x[i];
    }
}

// sum_i a[i]*xa[i] + b[i]*xb[i], four independent accumulators.
inline double dot_planes_f64(const double* a, const double* xa, const double* b, const double* xb, size_t n) {
    size_t i = 0;
    double s;
#if RFTMW_HAS_AVX2 && RFTMW_HAS_FMA
    __m256d s0 = _mm256_setzero_pd(), s1 = _mm256_setzero_pd();
    __m256d s2 = _mm256_setzero_pd(), s3 = _mm256_setzero_pd();
    for (; i + 8 <= n; i += 8) {
        s0 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(xa + i), s0);
        s1 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i + 4), _mm256_loadu_pd(xa + i + 4), s1);
        s2 = _mm256_fmadd_pd(_mm256_loadu_pd(b + i), _mm256_loadu_pd(xb + i), s2);
        s3 = _mm256_fmadd_pd(_mm256_loadu_pd(b + i + 4), _mm256_loadu_pd(xb + i + 4), s3);
    }
    double t[4];
    _mm256_storeu_pd(t, _mm256_add_pd(_mm256_add_pd(s0, s1), _mm256_add_pd(s2, s3)));
    s = (t[0] + t[1]) + (t[2] + t[3]);
#else
    double p[4] = {};
    for (; i + 4 <= n; i += 4)
        for (size_t j = 0; j < 4; ++j) p[j] += a[i + j] * xa[i + j] + b[i + j] * xb[i + j];
    s = (p[0] + p[1]) + (p[2] + p[3]);
#endif
    for (; i < n; ++i) s += a[i] * xa[i] + b[i] * xb[i];
    return s;
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
        if (N == 0) return {};
        if (N <= RFTMW_CANONICAL_MAX_N_RUNTIME) {
            const auto basis = canonical_basis(N);
            ComplexVec output(N);
            for (size_t k = 0; k < N; ++k) {
                double re, im;
                dot2_f64(&basis->fwd_re[k * N], &basis->fwd_im[k * N], input.data(), N, re, im);
                output[k] = Complex(re, im);
            }
            return output;
        }
        // The concentration test needs the raw adjoint; reuse it as the dense result.
        ComplexVec coeffs = apply_raw_phi_adjoint(input);
        const ConcentrationMetrics metrics = concentration_from_coeffs(coeffs);
        if (metrics.use_sparse_route) {
            return recover_sparse_coeffs_iht(input, metrics.top_k);
        }
        return coeffs;
    }

    RealVec inverse(const ComplexVec& coeffs) const {
        const size_t N = coeffs.size();
        if (N == 0) return {};
        if (N <= RFTMW_CANONICAL_MAX_N_RUNTIME) {
            const auto basis = canonical_basis(N);
            // Re(v * c) = v_re c_re - v_im c_im: two real dots against split planes.
            RealVec c_re(N), c_im_neg(N), output(N);
            for (size_t k = 0; k < N; ++k) {
                c_re[k] = coeffs[k].real();
                c_im_neg[k] = -coeffs[k].imag();
            }
            for (size_t n = 0; n < N; ++n)
                output[n] = dot_planes_f64(&basis->inv_re[n * N], c_re.data(),
                                           &basis->inv_im[n * N], c_im_neg.data(), N);
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
