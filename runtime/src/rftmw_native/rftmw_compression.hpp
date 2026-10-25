/*
 * SPDX-License-Identifier: LicenseRef-SPX-Proprietary
 * Copyright (C) 2025 Luis M. Minier / quantoniumos
 * This file is listed in CLAIMS_PRACTICING_FILES.txt and is governed
 * by LICENSE-SPX-PROPRIETARY.md. Commercial use requires a separate
 * written patent license from Luis M. Minier.
 *
 * rftmw_compression.hpp - RFTMW Compression Pipeline
 * ===================================================
 *
 * High-performance compression using RFTMW + quantization + entropy coding.
 */

#pragma once

#include "rftmw_core.hpp"
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <numeric>

namespace rftmw {

struct QuantizationParams {
    int precision_bits = 16;
    double dead_zone = 0.0;
    bool use_log_scale = false;
};

inline std::vector<int32_t> quantize_coefficients(
    const ComplexVec& coeffs,
    const QuantizationParams& params,
    double& scale_factor,
    double& max_value
) {
    size_t n = coeffs.size();
    std::vector<int32_t> quantized(n * 2);
    max_value = 0.0;
    for (const auto& c : coeffs) {
        max_value = std::max(max_value, std::abs(c.real()));
        max_value = std::max(max_value, std::abs(c.imag()));
    }
    if (max_value < 1e-30) max_value = 1.0;
    int32_t max_int = (1 << (params.precision_bits - 1)) - 1;
    scale_factor = static_cast<double>(max_int) / max_value;
    for (size_t i = 0; i < n; ++i) {
        double re = coeffs[i].real(), im = coeffs[i].imag();
        if (std::abs(re) < params.dead_zone) re = 0.0;
        if (std::abs(im) < params.dead_zone) im = 0.0;
        quantized[2*i]     = std::clamp(static_cast<int32_t>(std::round(re * scale_factor)), -max_int, max_int);
        quantized[2*i + 1] = std::clamp(static_cast<int32_t>(std::round(im * scale_factor)), -max_int, max_int);
    }
    return quantized;
}

inline ComplexVec dequantize_coefficients(const std::vector<int32_t>& quantized, double scale_factor) {
    size_t n = quantized.size() / 2;
    ComplexVec coeffs(n);
    double inv_scale = 1.0 / scale_factor;
    for (size_t i = 0; i < n; ++i) {
        coeffs[i] = Complex(static_cast<double>(quantized[2*i]) * inv_scale,
                            static_cast<double>(quantized[2*i + 1]) * inv_scale);
    }
    return coeffs;
}

struct CompressionResult {
    std::vector<uint8_t> data;
    size_t original_size;
    double scale_factor;
    double max_value;
    QuantizationParams quant_params;
    std::vector<uint32_t> frequencies;
    size_t kept_coefficients = 0;
    double top_k_energy_ratio = 0.0;
    bool used_sparse_route = false;
};

class RFTMWCompressor {
private:
    RFTMWEngine engine_;
    QuantizationParams quant_params_;
public:
    explicit RFTMWCompressor(size_t max_size = 65536, QuantizationParams params = {})
        : engine_(max_size), quant_params_(params) {}
    
    CompressionResult compress(const RealVec& input) {
        CompressionResult result;
        result.original_size = input.size();
        result.quant_params = quant_params_;
        const auto metrics = engine_.analyze_signal(input);
        ComplexVec coeffs = metrics.use_sparse_route
            ? engine_.forward_sparse(input, metrics.top_k)
            : engine_.forward(input);
        result.kept_coefficients = metrics.top_k;
        result.top_k_energy_ratio = metrics.top_k_energy_ratio;
        result.used_sparse_route = metrics.use_sparse_route;
        std::vector<int32_t> quantized = quantize_coefficients(coeffs, quant_params_, result.scale_factor, result.max_value);
        result.data.resize(quantized.size() * sizeof(int32_t));
        std::memcpy(result.data.data(), quantized.data(), result.data.size());
        int32_t offset = 1 << (quant_params_.precision_bits - 1);
        std::vector<uint32_t> symbols(quantized.size());
        for (size_t i = 0; i < quantized.size(); ++i) {
            symbols[i] = static_cast<uint32_t>(quantized[i] + offset);
        }
        size_t alphabet = 1 << quant_params_.precision_bits;
        result.frequencies.resize(alphabet, 0);
        for (uint32_t s : symbols) {
            if (s < alphabet) result.frequencies[s]++;
        }
        for (auto& f : result.frequencies) {
            if (f == 0) f = 1;
        }
        return result;
    }
    
    RealVec decompress(const CompressionResult& compressed) {
        int32_t offset = 1 << (compressed.quant_params.precision_bits - 1);
        std::vector<int32_t> quantized(compressed.data.size() / sizeof(int32_t));
        std::memcpy(quantized.data(), compressed.data.data(), compressed.data.size());
        ComplexVec coeffs = dequantize_coefficients(quantized, compressed.scale_factor);
        if (compressed.used_sparse_route || count_significant_coeffs(coeffs) * 8 < coeffs.size()) {
            return engine_.inverse_sparse(coeffs);
        }
        return engine_.inverse(coeffs);
    }
    
    const QuantizationParams& quantization_params() const { return quant_params_; }
    void set_quantization_params(const QuantizationParams& params) { quant_params_ = params; }
};

} // namespace rftmw
