/*
 * SPDX-License-Identifier: LicenseRef-SPX-Proprietary
 * Copyright (C) 2025 Luis M. Minier / quantoniumos
 * This file is listed in CLAIMS_PRACTICING_FILES.txt and is governed
 * by LICENSE-SPX-PROPRIETARY.md. Commercial use requires a separate
 * written patent license from Luis M. Minier.
 *
 * rftmw_asm_kernels.hpp - Assembly Kernel Integration
 * =====================================================
 *
 * C++ wrapper declarations for the existing x64 assembly kernels located at:
 *   algorithms/rft/kernels/kernel/rft_kernel_asm.asm
 *   algorithms/rft/kernels/kernel/quantum_symbolic_compression.asm
 *   algorithms/rft/kernels/engines/crypto/asm/feistel_round48.asm
 *   algorithms/rft/kernels/engines/orchestrator/asm/rft_transform.asm
 *
 * And their corresponding C interfaces:
 *   algorithms/rft/kernels/include/rft_kernel.h
 *   algorithms/rft/kernels/kernel/quantum_symbolic_compression.h
 *   algorithms/rft/kernels/engines/crypto/include/feistel_round48.h
 *
 * These routines use System V AMD64 ABI calling convention.
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <complex>
#include <vector>
#include <stdexcept>
#include <cstring>

#ifdef RFTMW_ENABLE_ASM
extern "C" {
#include "rft_kernel.h"
#include "quantum_symbolic_compression.h"
#include "feistel_round48.h"
}
#endif // RFTMW_ENABLE_ASM

namespace rftmw {
namespace asm_kernels {

using Complex = std::complex<double>;
using ComplexVec = std::vector<Complex>;
using RealVec = std::vector<double>;

inline bool asm_kernels_available() {
#ifdef RFTMW_ENABLE_ASM
    return true;
#else
    return false;
#endif
}

inline bool has_avx2_crypto() {
#ifdef RFTMW_ENABLE_ASM
    return feistel_has_avx2();
#else
    return false;
#endif
}

inline bool has_aes_ni() {
#ifdef RFTMW_ENABLE_ASM
    return feistel_has_aes_ni();
#else
    return false;
#endif
}

class RFTKernelEngine {
private:
#ifdef RFTMW_ENABLE_ASM
    rft_engine_t engine_;
    bool initialized_ = false;
#endif

public:
    enum class Variant {
        STANDARD = 0, HARMONIC = 1, FIBONACCI = 2, CHAOTIC = 3,
        GEOMETRIC = 4, PHI_CHAOTIC = 5, HYPERBOLIC = 6,
        DCT = 7, HYBRID_DCT = 8, CASCADE = 9,
        ADAPTIVE_SPLIT = 10, ENTROPY_GUIDED = 11, DICTIONARY = 12
    };

    RFTKernelEngine(size_t size, uint32_t flags = 0, Variant variant = Variant::STANDARD) {
#ifdef RFTMW_ENABLE_ASM
        rft_error_t err = rft_init_with_variant(&engine_, size, flags, static_cast<rft_variant_t>(variant));
        if (err != RFT_SUCCESS) throw std::runtime_error("Failed to initialize RFT kernel engine");
        initialized_ = true;
#else
        throw std::runtime_error("ASM kernels not enabled. Rebuild with -DRFTMW_ENABLE_ASM=ON");
#endif
    }

    ~RFTKernelEngine() {
#ifdef RFTMW_ENABLE_ASM
        if (initialized_) rft_cleanup(&engine_);
#endif
    }

    RFTKernelEngine(const RFTKernelEngine&) = delete;
    RFTKernelEngine& operator=(const RFTKernelEngine&) = delete;

    ComplexVec forward(const ComplexVec& input) {
#ifdef RFTMW_ENABLE_ASM
        ComplexVec output(input.size());
        rft_error_t err = rft_forward(&engine_, reinterpret_cast<const rft_complex_t*>(input.data()), reinterpret_cast<rft_complex_t*>(output.data()), input.size());
        if (err != RFT_SUCCESS) throw std::runtime_error("RFT forward transform failed");
        return output;
#else
        throw std::runtime_error("ASM kernels not enabled");
#endif
    }

    ComplexVec inverse(const ComplexVec& input) {
#ifdef RFTMW_ENABLE_ASM
        ComplexVec output(input.size());
        rft_error_t err = rft_inverse(&engine_, reinterpret_cast<const rft_complex_t*>(input.data()), reinterpret_cast<rft_complex_t*>(output.data()), input.size());
        if (err != RFT_SUCCESS) throw std::runtime_error("RFT inverse transform failed");
        return output;
#else
        throw std::runtime_error("ASM kernels not enabled");
#endif
    }

    bool validate_unitarity(double tolerance = 1e-10) {
#ifdef RFTMW_ENABLE_ASM
        return rft_validate_unitarity(&engine_, tolerance) == RFT_SUCCESS;
#else
        return false;
#endif
    }

    double von_neumann_entropy(const ComplexVec& state) {
#ifdef RFTMW_ENABLE_ASM
        double entropy = 0.0;
        rft_error_t err = rft_von_neumann_entropy(&engine_, reinterpret_cast<const rft_complex_t*>(state.data()), &entropy, state.size());
        if (err != RFT_SUCCESS) throw std::runtime_error("Failed to calculate entropy");
        return entropy;
#else
        throw std::runtime_error("ASM kernels not enabled");
#endif
    }

    double measure_entanglement(const ComplexVec& state) {
#ifdef RFTMW_ENABLE_ASM
        double entanglement = 0.0;
        rft_error_t err = rft_entanglement_measure(&engine_, reinterpret_cast<const rft_complex_t*>(state.data()), &entanglement, state.size());
        if (err != RFT_SUCCESS) throw std::runtime_error("Failed to measure entanglement");
        return entanglement;
#else
        throw std::runtime_error("ASM kernels not enabled");
#endif
    }
};

} // namespace asm_kernels
} // namespace rftmw
