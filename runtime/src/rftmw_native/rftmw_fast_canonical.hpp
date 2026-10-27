/*
 * SPDX-License-Identifier: LicenseRef-SPX-Proprietary
 * Copyright (C) 2026 Luis M. Minier / quantoniumos
 *
 * Matrix-free canonical phi-grid RFT using FFTW Bluestein transforms and a
 * Toeplitz-Gram rational inverse-square-root action.
 */
#pragma once

#ifndef RFTMW_HAVE_FFTW3
#error "rftmw_fast_canonical.hpp requires FFTW3"
#endif

#include <fftw3.h>

#include <array>
#include <climits>
#include <cmath>
#include <complex>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rftmw::fast_canonical {

constexpr size_t MAX_N = 1024;
constexpr size_t RATIONAL_TERMS = 64;
constexpr size_t MAX_CG_ITERATIONS = 256;
constexpr double SPECTRAL_MIN = 5e-4;
constexpr double SPECTRAL_MAX = 2.5;
constexpr double SOLVE_TOLERANCE = 1e-12;

inline Complex cis(double angle) {
    return Complex(std::cos(angle), std::sin(angle));
}

inline size_t fft_size_for(size_t n) {
    if (n == 0 || n > MAX_N) throw std::invalid_argument("fast canonical RFT supports 1..1024 samples");
    const size_t required = 3 * n - 1;
    size_t size = 1;
    while (size < required) size <<= 1;
    if (size > static_cast<size_t>(INT_MAX)) throw std::length_error("FFTW transform is too large");
    return size;
}

inline std::mutex& planner_mutex() {
    static std::mutex mutex;
    return mutex;
}

class FftwWorkspace {
public:
    explicit FftwWorkspace(size_t size) : size_(size) {
        data_ = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * size_));
        if (!data_) throw std::bad_alloc();
        std::lock_guard<std::mutex> lock(planner_mutex());
        forward_ = fftw_plan_dft_1d(static_cast<int>(size_), data_, data_, FFTW_FORWARD, FFTW_ESTIMATE);
        inverse_ = fftw_plan_dft_1d(static_cast<int>(size_), data_, data_, FFTW_BACKWARD, FFTW_ESTIMATE);
        if (!forward_ || !inverse_) {
            if (forward_) fftw_destroy_plan(forward_);
            if (inverse_) fftw_destroy_plan(inverse_);
            fftw_free(data_);
            data_ = nullptr;
            throw std::runtime_error("FFTW plan creation failed");
        }
    }

    ~FftwWorkspace() {
        std::lock_guard<std::mutex> lock(planner_mutex());
        if (forward_) fftw_destroy_plan(forward_);
        if (inverse_) fftw_destroy_plan(inverse_);
        if (data_) fftw_free(data_);
    }

    FftwWorkspace(const FftwWorkspace&) = delete;
    FftwWorkspace& operator=(const FftwWorkspace&) = delete;

    std::vector<Complex> spectrum(const std::vector<Complex>& input) {
        std::lock_guard<std::mutex> lock(work_mutex_);
        load(input);
        fftw_execute(forward_);
        return store(1.0);
    }

    std::vector<Complex> convolve(
        const std::vector<Complex>& input,
        const std::vector<Complex>& kernel_spectrum
    ) {
        if (input.size() > size_ || kernel_spectrum.size() != size_)
            throw std::invalid_argument("FFTW convolution size mismatch");
        std::lock_guard<std::mutex> lock(work_mutex_);
        load(input);
        fftw_execute(forward_);
        for (size_t i = 0; i < size_; ++i) {
            const double re = data_[i][0] * kernel_spectrum[i].real()
                            - data_[i][1] * kernel_spectrum[i].imag();
            const double im = data_[i][0] * kernel_spectrum[i].imag()
                            + data_[i][1] * kernel_spectrum[i].real();
            data_[i][0] = re;
            data_[i][1] = im;
        }
        fftw_execute(inverse_);
        return store(1.0 / static_cast<double>(size_));
    }

private:
    void load(const std::vector<Complex>& input) {
        for (size_t i = 0; i < size_; ++i) {
            data_[i][0] = i < input.size() ? input[i].real() : 0.0;
            data_[i][1] = i < input.size() ? input[i].imag() : 0.0;
        }
    }

    std::vector<Complex> store(double scale) const {
        std::vector<Complex> output(size_);
        for (size_t i = 0; i < size_; ++i)
            output[i] = Complex(data_[i][0] * scale, data_[i][1] * scale);
        return output;
    }

    size_t size_;
    fftw_complex* data_ = nullptr;
    fftw_plan forward_ = nullptr;
    fftw_plan inverse_ = nullptr;
    std::mutex work_mutex_;
};

class Plan {
public:
    explicit Plan(size_t n)
        : n_(n), fft_size_(fft_size_for(n)), fft_(fft_size_), gram_fft_(2 * n) {
        build_kernels();
        build_rational_poles();
    }

    ComplexVec forward(const RealVec& input) {
        if (input.size() != n_) throw std::invalid_argument("fast canonical RFT input size mismatch");
        ComplexVec complex_input(n_);
        for (size_t i = 0; i < n_; ++i) complex_input[i] = Complex(input[i], 0.0);
        return apply_inverse_sqrt(phi_adjoint(complex_input));
    }

    RealVec inverse(const ComplexVec& coeffs) {
        if (coeffs.size() != n_) throw std::invalid_argument("fast canonical RFT coefficient size mismatch");
        const ComplexVec normalized = apply_inverse_sqrt(coeffs);
        const ComplexVec result = phi_synthesis(normalized);
        RealVec output(n_);
        for (size_t i = 0; i < n_; ++i) output[i] = result[i].real();
        return output;
    }

    ComplexVec raw_adjoint(const ComplexVec& input) { return phi_adjoint(input); }
    ComplexVec raw_synthesis(const ComplexVec& coeffs) { return phi_synthesis(coeffs); }
    ComplexVec gram_apply(const ComplexVec& vector) { return gram_matvec(vector); }

private:
    void build_kernels() {
        std::vector<Complex> gram_embedding(2 * n_, Complex(0.0));
        gram_embedding[0] = Complex(1.0, 0.0);
        for (size_t d = 1; d < n_; ++d) {
            const double phase = PI * static_cast<double>(d) * PHI;
            const double denominator = static_cast<double>(n_) * std::sin(phase);
            const double angle = static_cast<double>(n_ - 1) * phase;
            const Complex first_row = cis(angle) * (std::sin(static_cast<double>(n_) * phase) / denominator);
            gram_embedding[d] = std::conj(first_row);
            gram_embedding[2 * n_ - d] = first_row;
        }
        gram_spectrum_ = gram_fft_.spectrum(gram_embedding);

        std::vector<Complex> synthesis_kernel(fft_size_, Complex(0.0));
        for (long long offset = -static_cast<long long>(n_);
             offset <= static_cast<long long>(n_) - 2; ++offset) {
            const size_t index = static_cast<size_t>(offset + static_cast<long long>(n_));
            const double value = static_cast<double>(offset);
            synthesis_kernel[index] = cis(-PI * PHI * value * value);
        }
        synthesis_spectrum_ = fft_.spectrum(synthesis_kernel);

        std::vector<Complex> adjoint_kernel(fft_size_, Complex(0.0));
        for (long long offset = 1 - static_cast<long long>(n_);
             offset <= static_cast<long long>(n_); ++offset) {
            const size_t index = static_cast<size_t>(offset + static_cast<long long>(n_) - 1);
            const double value = static_cast<double>(offset);
            adjoint_kernel[index] = cis(PI * PHI * value * value);
        }
        adjoint_spectrum_ = fft_.spectrum(adjoint_kernel);
    }

    void build_rational_poles() {
        const double scale = std::sqrt(SPECTRAL_MIN * SPECTRAL_MAX);
        poles_.resize(RATIONAL_TERMS);
        weights_.resize(RATIONAL_TERMS);
        for (size_t i = 0; i < RATIONAL_TERMS; ++i) {
            double root = std::cos(PI * (static_cast<double>(i) + 0.75)
                                   / (static_cast<double>(RATIONAL_TERMS) + 0.5));
            double derivative = 0.0;
            for (size_t iteration = 0; iteration < 32; ++iteration) {
                double p0 = 1.0;
                double p1 = root;
                for (size_t degree = 2; degree <= RATIONAL_TERMS; ++degree) {
                    const double p2 = ((2.0 * degree - 1.0) * root * p1
                                       - (degree - 1.0) * p0) / degree;
                    p0 = p1;
                    p1 = p2;
                }
                derivative = RATIONAL_TERMS * (root * p1 - p0) / (root * root - 1.0);
                const double next = root - p1 / derivative;
                if (std::abs(next - root) < 2e-15) {
                    root = next;
                    break;
                }
                root = next;
            }
            const double quadrature_weight = 2.0
                / ((1.0 - root * root) * derivative * derivative);
            const double angle = (root + 1.0) * PI / 4.0;
            const double tangent = std::tan(angle);
            const double cosine = std::cos(angle);
            poles_[i] = scale * scale * tangent * tangent;
            weights_[i] = scale * quadrature_weight / (2.0 * cosine * cosine);
        }
    }

    ComplexVec gram_matvec(const ComplexVec& vector) {
        std::vector<Complex> padded(2 * n_, Complex(0.0));
        std::copy(vector.begin(), vector.end(), padded.begin());
        const auto convolution = gram_fft_.convolve(padded, gram_spectrum_);
        return ComplexVec(convolution.begin(), convolution.begin() + static_cast<std::ptrdiff_t>(n_));
    }

    ComplexVec phi_adjoint(const ComplexVec& input) {
        std::vector<Complex> source(fft_size_, Complex(0.0));
        for (size_t k = 0; k < n_; ++k) {
            const double index = static_cast<double>(k);
            source[k] = input[k] * cis(-PI * PHI * index * index);
        }
        const auto convolution = fft_.convolve(source, adjoint_spectrum_);
        const double inv_sqrt_n = 1.0 / std::sqrt(static_cast<double>(n_));
        ComplexVec output(n_);
        for (size_t k = 0; k < n_; ++k) {
            const double frequency = static_cast<double>(k + 1);
            output[k] = convolution[n_ + k]
                      * cis(-PI * PHI * frequency * frequency) * inv_sqrt_n;
        }
        return output;
    }

    ComplexVec phi_synthesis(const ComplexVec& coeffs) {
        std::vector<Complex> source(fft_size_, Complex(0.0));
        for (size_t k = 0; k < n_; ++k) {
            const double frequency = static_cast<double>(k + 1);
            source[k] = coeffs[k] * cis(PI * PHI * frequency * frequency);
        }
        const auto convolution = fft_.convolve(source, synthesis_spectrum_);
        const double inv_sqrt_n = 1.0 / std::sqrt(static_cast<double>(n_));
        ComplexVec output(n_);
        for (size_t sample = 0; sample < n_; ++sample) {
            const double index = static_cast<double>(sample);
            output[sample] = convolution[n_ - 1 + sample]
                           * cis(PI * PHI * index * index) * inv_sqrt_n;
        }
        return output;
    }

    ComplexVec apply_inverse_sqrt(const ComplexVec& vector) {
        const size_t count = poles_.size();
        const double base = *std::min_element(poles_.begin(), poles_.end());
        std::vector<double> deltas(count);
        for (size_t i = 0; i < count; ++i) deltas[i] = poles_[i] - base;

        std::vector<ComplexVec> solutions(count, ComplexVec(n_, Complex(0.0)));
        std::vector<ComplexVec> shifted_directions(count, vector);
        std::vector<double> z_previous(count, 1.0), z_current(count, 1.0);
        std::vector<uint8_t> active(count, 1);
        ComplexVec residual = vector;
        ComplexVec direction = vector;
        double residual_norm_sq = 0.0;
        for (const Complex& value : residual) residual_norm_sq += std::norm(value);
        if (residual_norm_sq == 0.0) return ComplexVec(n_, Complex(0.0));
        const double vector_norm = std::sqrt(residual_norm_sq);
        double alpha_previous = 1.0;
        double beta_previous = 0.0;
        const double tolerance = SOLVE_TOLERANCE * vector_norm;

        for (size_t iteration = 0; iteration < MAX_CG_ITERATIONS; ++iteration) {
            ComplexVec gram_direction = gram_matvec(direction);
            for (size_t i = 0; i < n_; ++i) gram_direction[i] += base * direction[i];
            double denominator_cg = 0.0;
            for (size_t i = 0; i < n_; ++i)
                denominator_cg += std::real(std::conj(direction[i]) * gram_direction[i]);
            if (!(denominator_cg > 0.0) || !std::isfinite(denominator_cg))
                throw std::runtime_error("fast canonical RFT Gram solve lost positive definiteness");

            const double alpha = residual_norm_sq / denominator_cg;
            ComplexVec residual_next(n_);
            double residual_next_norm_sq = 0.0;
            for (size_t i = 0; i < n_; ++i) {
                residual_next[i] = residual[i] - alpha * gram_direction[i];
                residual_next_norm_sq += std::norm(residual_next[i]);
            }
            const double beta = residual_next_norm_sq / residual_norm_sq;
            std::vector<double> z_next = z_current;
            for (size_t j = 0; j < count; ++j) {
                if (!active[j]) continue;
                const double denominator_z = alpha * beta_previous * (z_previous[j] - z_current[j])
                    + z_previous[j] * alpha_previous * (1.0 + deltas[j] * alpha);
                z_next[j] = z_current[j] * z_previous[j] * alpha_previous / denominator_z;
                if (!std::isfinite(z_next[j]) || z_current[j] == 0.0)
                    throw std::runtime_error("fast canonical RFT multi-shift recurrence became non-finite");
                const double ratio = z_next[j] / z_current[j];
                if (!std::isfinite(ratio))
                    throw std::runtime_error("fast canonical RFT shift ratio became non-finite");
                for (size_t i = 0; i < n_; ++i) {
                    solutions[j][i] += alpha * ratio * shifted_directions[j][i];
                    shifted_directions[j][i] = z_next[j] * residual_next[i]
                        + beta * ratio * ratio * shifted_directions[j][i];
                }
                if (std::abs(z_next[j]) * std::sqrt(residual_next_norm_sq) <= tolerance)
                    active[j] = 0;
            }

            bool any_active = false;
            for (uint8_t value : active) any_active = any_active || value != 0;
            if (!any_active) {
                ComplexVec output(n_, Complex(0.0));
                for (size_t j = 0; j < count; ++j)
                    for (size_t i = 0; i < n_; ++i) output[i] += weights_[j] * solutions[j][i];
                return output;
            }
            if (std::sqrt(residual_next_norm_sq) <= tolerance)
                throw std::runtime_error("fast canonical RFT base CG converged before all shifts");

            for (size_t i = 0; i < n_; ++i) direction[i] = residual_next[i] + beta * direction[i];
            residual = std::move(residual_next);
            residual_norm_sq = residual_next_norm_sq;
            for (size_t j = 0; j < count; ++j) {
                if (!active[j]) continue;
                z_previous[j] = z_current[j];
                z_current[j] = z_next[j];
            }
            alpha_previous = alpha;
            beta_previous = beta;
        }
        throw std::runtime_error("fast canonical RFT multi-shift CG reached its fixed iteration cap");
    }

    size_t n_;
    size_t fft_size_;
    FftwWorkspace fft_;
    FftwWorkspace gram_fft_;
    std::vector<Complex> gram_spectrum_;
    std::vector<Complex> synthesis_spectrum_;
    std::vector<Complex> adjoint_spectrum_;
    std::vector<double> poles_;
    std::vector<double> weights_;
};

inline std::shared_ptr<Plan> plan_for(size_t n) {
    static std::mutex mutex;
    static std::unordered_map<size_t, std::shared_ptr<Plan>> plans;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = plans.find(n);
    if (it != plans.end()) return it->second;
    auto plan = std::make_shared<Plan>(n);
    return plans.emplace(n, std::move(plan)).first->second;
}

inline ComplexVec forward(const RealVec& input) {
    if (input.empty()) return {};
    return plan_for(input.size())->forward(input);
}

inline RealVec inverse(const ComplexVec& coeffs) {
    if (coeffs.empty()) return {};
    return plan_for(coeffs.size())->inverse(coeffs);
}

}  // namespace rftmw::fast_canonical