#include "rftmw_core.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

int main() {
    std::mt19937_64 random(20261027);
    std::normal_distribution<double> normal;
    rftmw::RFTMWEngine engine;

    for (size_t n : {16, 32, 64, 128, 256, 512, 1024}) {
        std::cerr << "testing N=" << n << '\n';
        rftmw::RealVec input(n);
        for (double& value : input) value = normal(random);
        if (n == 16) {
            rftmw::fast_canonical::Plan plan(n);
            rftmw::ComplexVec input_complex(n);
            rftmw::ComplexVec coeffs(n);
            for (size_t i = 0; i < n; ++i) {
                input_complex[i] = rftmw::Complex(input[i], 0.0);
                coeffs[i] = rftmw::Complex(normal(random), normal(random));
            }
            const auto dense_phi = rftmw::compute_raw_phi_basis(n);
            const auto dense_gram = rftmw::compute_gram_matrix(dense_phi, n);
            const auto raw_adjoint = plan.raw_adjoint(input_complex);
            const auto raw_synthesis = plan.raw_synthesis(coeffs);
            const auto gram_product = plan.gram_apply(coeffs);
            double adjoint_error = 0.0;
            double synthesis_error = 0.0;
            double gram_error = 0.0;
            for (size_t i = 0; i < n; ++i) {
                rftmw::Complex expected_adjoint(0.0), expected_synthesis(0.0), expected_gram(0.0);
                for (size_t j = 0; j < n; ++j) {
                    expected_adjoint += std::conj(dense_phi[j * n + i]) * input_complex[j];
                    expected_synthesis += dense_phi[i * n + j] * coeffs[j];
                    expected_gram += dense_gram[i * n + j] * coeffs[j];
                }
                adjoint_error = std::max(adjoint_error, std::abs(raw_adjoint[i] - expected_adjoint));
                synthesis_error = std::max(synthesis_error, std::abs(raw_synthesis[i] - expected_synthesis));
                gram_error = std::max(gram_error, std::abs(gram_product[i] - expected_gram));
            }
            std::cerr << "raw adjoint/synthesis/gram errors: " << adjoint_error << "/"
                      << synthesis_error << "/" << gram_error << '\n';
            if (std::max({adjoint_error, synthesis_error, gram_error}) > 1e-10) return 3;
        }
        const auto coefficients = engine.forward_fast_canonical(input);
        const auto reconstructed = engine.inverse_fast_canonical(coefficients);
        double error = 0.0;
        double scale = 0.0;
        for (size_t i = 0; i < n; ++i) {
            error = std::max(error, std::abs(input[i] - reconstructed[i]));
            scale = std::max(scale, std::abs(input[i]));
        }
        const double relative_error = error / std::max(scale, 1e-300);
        std::cout << "N=" << n << " relative_roundtrip=" << relative_error << '\n';
        if (!std::isfinite(relative_error) || relative_error > 1e-6) return 1;

        if (n <= 32) {
            const auto dense_basis = rftmw::compute_gram_normalised_basis(n);
            rftmw::ComplexVec reference(n, rftmw::Complex(0.0));
            for (size_t k = 0; k < n; ++k)
                for (size_t sample = 0; sample < n; ++sample)
                    reference[k] += std::conj(dense_basis[sample * n + k]) * input[sample];
            double difference = 0.0;
            double reference_scale = 0.0;
            for (size_t k = 0; k < n; ++k) {
                difference = std::max(difference, std::abs(reference[k] - coefficients[k]));
                reference_scale = std::max(reference_scale, std::abs(reference[k]));
            }
            if (difference / std::max(reference_scale, 1e-300) > 1e-6) return 2;
        }
    }
    return 0;
}