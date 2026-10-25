/*
 * SPDX-License-Identifier: LicenseRef-SPX-Proprietary
 * Copyright (C) 2025 Luis M. Minier / quantoniumos
 * This file is listed in CLAIMS_PRACTICING_FILES.txt and is governed
 * by LICENSE-SPX-PROPRIETARY.md. Commercial use requires a separate
 * written patent license from Luis M. Minier.
 *
 * rftmw_python.cpp - Python bindings for RFTMW
 * =============================================
 *
 * pybind11 bindings exposing the C++ RFTMW engine to Python/NumPy.
 *
 * Architecture:
 *   Python/NumPy <-> pybind11 <-> C++ Engine <-> ASM Kernels
 *
 * ASM Kernels integrated from algorithms/rft/kernels/:
 *   - rft_kernel_asm.asm  (unitary RFT transform)
 *   - quantum_symbolic_compression.asm
 *   - feistel_round48.asm
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <chrono>
#include <array>
#include <random>
#include <cstring>

#include "rftmw_core.hpp"
#include "rftmw_compression.hpp"
#include "rftmw_transform_theorems.hpp"
#include "rftmw_inference.hpp"

#if RFTMW_ENABLE_ASM
#include "rftmw_asm_kernels.hpp"
#endif

namespace py = pybind11;
using namespace rftmw;

template <typename T>
py::array_t<T> make_1d_array(py::ssize_t size) {
    return py::array_t<T>({size}, {static_cast<py::ssize_t>(sizeof(T))});
}

// ---------------------------------------------------------------------------
// NumPy <-> std::vector helpers
// ---------------------------------------------------------------------------

RealVec numpy_to_realvec(py::array_t<double> arr) {
    auto buf = arr.request();
    if (buf.ndim != 1) throw std::runtime_error("Input must be 1-dimensional");
    double* ptr = static_cast<double*>(buf.ptr);
    return RealVec(ptr, ptr + buf.size);
}

py::array_t<double> realvec_to_numpy(const RealVec& vec) {
    py::array_t<double> result = make_1d_array<double>(static_cast<py::ssize_t>(vec.size()));
    auto buf = result.request();
    std::memcpy(buf.ptr, vec.data(), vec.size() * sizeof(double));
    return result;
}

ComplexVec numpy_to_complexvec(py::array_t<std::complex<double>> arr) {
    auto buf = arr.request();
    if (buf.ndim != 1) throw std::runtime_error("Input must be 1-dimensional");
    std::complex<double>* ptr = static_cast<std::complex<double>*>(buf.ptr);
    return ComplexVec(ptr, ptr + buf.size);
}

py::array_t<std::complex<double>> complexvec_to_numpy(const ComplexVec& vec) {
    py::array_t<std::complex<double>> result = make_1d_array<std::complex<double>>(static_cast<py::ssize_t>(vec.size()));
    auto buf = result.request();
    std::memcpy(buf.ptr, vec.data(), vec.size() * sizeof(std::complex<double>));
    return result;
}

py::array_t<std::complex<double>> dense_to_numpy(const rftmw::theorems::Dense& mat) {
    py::array_t<std::complex<double>> result({mat.n, mat.n});
    auto buf = result.request();
    std::memcpy(buf.ptr, mat.a.data(), mat.a.size() * sizeof(std::complex<double>));
    return result;
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(rftmw_native, m) {
    m.doc() = "RFTMW Native Extension -- high-performance Phi-RFT transform and compression.";
    m.attr("__version__") = "1.1.0";
    m.attr("HAS_AVX2")       = RFTMW_HAS_AVX2;
    m.attr("HAS_AVX512")     = RFTMW_HAS_AVX512;
    m.attr("HAS_FMA")        = RFTMW_HAS_FMA;
    m.attr("PHI")            = PHI;
    m.attr("HAS_ASM_KERNELS") = static_cast<int>(RFTMW_ENABLE_ASM);

    m.def("asm_kernels_available", []() -> bool {
#if RFTMW_ENABLE_ASM
        return true;
#else
        return false;
#endif
    }, "Returns True if the module was compiled with ASM kernel support.");

    // -----------------------------------------------------------------------
    // RFTMWEngine -- the core Phi-RFT forward/inverse engine
    // -----------------------------------------------------------------------
    py::class_<RFTMWEngine>(m, "RFTMWEngine",
        "High-performance Phi-RFT engine. Provides forward() and inverse() "
        "transforms using AVX2-accelerated C++ with optional ASM kernels.")
        .def(py::init<size_t>(),
             py::arg("max_size") = 65536,
             "Construct an RFTMWEngine. max_size is a capacity hint (unused "
             "in current implementation but reserved for future pooling).")
        // forward: numpy float64 array -> numpy complex128 array
        .def("forward",
             [](RFTMWEngine& self, py::array_t<double> arr) {
                 return complexvec_to_numpy(self.forward(numpy_to_realvec(arr)));
             },
             py::arg("signal"),
             "Apply forward Phi-RFT to a real 1-D signal.\n"
             "Returns complex128 NumPy array of coefficients.")
        // inverse: numpy complex128 array -> numpy float64 array
        .def("inverse",
             [](RFTMWEngine& self, py::array_t<std::complex<double>> arr) {
                 return realvec_to_numpy(self.inverse(numpy_to_complexvec(arr)));
             },
             py::arg("coeffs"),
             "Apply inverse Phi-RFT to complex coefficients.\n"
             "Returns real float64 NumPy array.")
        .def("forward_sparse",
             [](RFTMWEngine& self, py::array_t<double> arr, size_t keep_k, size_t max_iter) {
                 return complexvec_to_numpy(self.forward_sparse(numpy_to_realvec(arr), keep_k, max_iter));
             },
             py::arg("signal"),
             py::arg("keep_k"),
             py::arg("max_iter") = 24,
             "Recover a sparse coefficient vector using operator-only IHT.")
        .def("inverse_sparse",
             [](RFTMWEngine& self, py::array_t<std::complex<double>> arr) {
                 return realvec_to_numpy(self.inverse_sparse(numpy_to_complexvec(arr)));
             },
             py::arg("coeffs"),
             "Synthesize a real signal from sparse Phi coefficients.")
        .def("analyze_signal_concentration",
             [](RFTMWEngine& self, py::array_t<double> arr, double energy_threshold) {
                 const auto metrics = self.analyze_signal(numpy_to_realvec(arr), energy_threshold);
                 py::dict out;
                 out["top_k"] = py::int_(metrics.top_k);
                 out["top_k_energy_ratio"] = py::float_(metrics.top_k_energy_ratio);
                 out["sparsity_ratio"] = py::float_(metrics.sparsity_ratio);
                 out["coherence_proxy"] = py::float_(metrics.coherence_proxy);
                 out["use_sparse_route"] = py::bool_(metrics.use_sparse_route);
                 return out;
             },
             py::arg("signal"),
             py::arg("energy_threshold") = 0.90,
             "Estimate whether the signal is concentrated enough for the sparse route.");

    // -----------------------------------------------------------------------
    // Smoke-test helper: roundtrip a random signal and return max abs error
    // -----------------------------------------------------------------------
    m.def("roundtrip_error",
          [](size_t n) -> double {
              RFTMWEngine eng(n);
              RealVec signal(n);
              // deterministic pseudo-random using phi seed
              double phi = PHI;
              for (size_t i = 0; i < n; ++i) {
                  phi = phi * PHI - std::floor(phi * PHI);
                  signal[i] = phi * 2.0 - 1.0;
              }
              ComplexVec coeffs  = eng.forward(signal);
              RealVec   recovered = eng.inverse(coeffs);
              double max_err = 0.0;
              for (size_t i = 0; i < n; ++i)
                  max_err = std::max(max_err, std::abs(recovered[i] - signal[i]));
              return max_err;
          },
          py::arg("n") = 64,
          "Run a forward+inverse roundtrip on a pseudo-random signal of length n.\n"
          "Returns max absolute reconstruction error (should be < 1e-10).");

    // -----------------------------------------------------------------------
    // Inference kernels — the heavy lifting in C++ instead of NumPy
    // -----------------------------------------------------------------------

    auto infer = m.def_submodule("infer", "C++ inference acceleration kernels");

    // --- Fused dequant + matmul: y = QuantWeight @ x ---
    infer.def("q4k_matmul",
        [](py::array_t<uint8_t> raw_weights,
           py::array_t<float> x,
           int out_features,
           int bytes_per_row) -> py::array_t<float>
        {
            auto w_buf = raw_weights.request();
            auto x_buf = x.request();
            if (x_buf.ndim != 1) throw std::runtime_error("x must be 1-D");

            py::array_t<float> result = make_1d_array<float>(out_features);
            auto r_buf = result.request();

            inference::q4k_matmul_vec(
                static_cast<const uint8_t*>(w_buf.ptr),
                static_cast<const float*>(x_buf.ptr),
                static_cast<float*>(r_buf.ptr),
                out_features,
                bytes_per_row
            );
            return result;
        },
        py::arg("raw_weights"), py::arg("x"),
        py::arg("out_features"), py::arg("bytes_per_row"),
        "Fused Q4_K dequant + matrix-vector multiply with AVX2.");

    infer.def("q6k_matmul",
        [](py::array_t<uint8_t> raw_weights,
           py::array_t<float> x,
           int out_features,
           int bytes_per_row) -> py::array_t<float>
        {
            auto w_buf = raw_weights.request();
            auto x_buf = x.request();
            if (x_buf.ndim != 1) throw std::runtime_error("x must be 1-D");

            py::array_t<float> result = make_1d_array<float>(out_features);
            auto r_buf = result.request();

            inference::q6k_matmul_vec(
                static_cast<const uint8_t*>(w_buf.ptr),
                static_cast<const float*>(x_buf.ptr),
                static_cast<float*>(r_buf.ptr),
                out_features,
                bytes_per_row
            );
            return result;
        },
        py::arg("raw_weights"), py::arg("x"),
        py::arg("out_features"), py::arg("bytes_per_row"),
        "Fused Q6_K dequant + matrix-vector multiply with AVX2.");

    infer.def("sgemv_f32",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> weights_flat,
           py::array_t<float, py::array::c_style | py::array::forcecast> x,
           int out_features,
           int in_features) -> py::array_t<float>
        {
            if (weights_flat.ndim() != 1) throw std::runtime_error("weights_flat must be 1-D");
            if (x.ndim() != 1) throw std::runtime_error("x must be 1-D");
            if (x.shape(0) != in_features) {
                throw std::runtime_error("x.shape[0] must equal in_features");
            }
            if (weights_flat.shape(0) != static_cast<py::ssize_t>(out_features) * static_cast<py::ssize_t>(in_features)) {
                throw std::runtime_error("weights_flat size must equal out_features * in_features");
            }

            py::array_t<float> result = make_1d_array<float>(out_features);
            auto w_buf = weights_flat.request();
            auto r_buf = result.request();
            auto x_buf = x.request();
            inference::dense_sgemv_f32_bridge(
                static_cast<const float*>(w_buf.ptr),
                static_cast<const float*>(x_buf.ptr),
                static_cast<float*>(r_buf.ptr),
                out_features,
                in_features
            );
            return result;
        },
        py::arg("weights_flat"), py::arg("x"),
        py::arg("out_features"), py::arg("in_features"),
        "Dense float32 matrix-vector multiply backed by the native AVX2 ASM kernel.");

    // --- RMS Norm ---
    infer.def("rms_norm",
        [](py::array_t<float> x, py::array_t<float> weight, float eps) -> py::array_t<float>
        {
            auto x_buf = x.request();
            auto w_buf = weight.request();
            int size = static_cast<int>(x_buf.size);

            py::array_t<float> result = make_1d_array<float>(size);
            auto r_buf = result.request();

            inference::rms_norm(
                static_cast<const float*>(x_buf.ptr),
                static_cast<const float*>(w_buf.ptr),
                static_cast<float*>(r_buf.ptr),
                size,
                eps
            );
            return result;
        },
        py::arg("x"), py::arg("weight"), py::arg("eps") = 1e-6f,
        "RMS normalization with AVX2.");

    // --- Softmax ---
    infer.def("softmax",
        [](py::array_t<float> x) -> py::array_t<float>
        {
            auto buf = x.request();
            int size = static_cast<int>(buf.size);
            py::array_t<float> result = make_1d_array<float>(size);
            auto r_buf = result.request();
            inference::softmax(
                static_cast<const float*>(buf.ptr),
                static_cast<float*>(r_buf.ptr),
                size
            );
            return result;
        },
        py::arg("x"),
        "Softmax (numerically stable).");

    // --- SiLU ---
    infer.def("silu",
        [](py::array_t<float> x) -> py::array_t<float>
        {
            auto buf = x.request();
            int size = static_cast<int>(buf.size);
            py::array_t<float> result = make_1d_array<float>(size);
            auto r_buf = result.request();
            inference::silu(
                static_cast<const float*>(buf.ptr),
                static_cast<float*>(r_buf.ptr),
                size
            );
            return result;
        },
        py::arg("x"),
        "SiLU activation.");

    // --- Fused SiLU * elementwise mul (for SwiGLU FFN) ---
    infer.def("silu_mul",
        [](py::array_t<float> gate, py::array_t<float> up) -> py::array_t<float>
        {
            auto g_buf = gate.request();
            auto u_buf = up.request();
            int size = static_cast<int>(g_buf.size);
            py::array_t<float> result = make_1d_array<float>(size);
            auto r_buf = result.request();
            inference::silu_mul(
                static_cast<const float*>(g_buf.ptr),
                static_cast<const float*>(u_buf.ptr),
                static_cast<float*>(r_buf.ptr),
                size
            );
            return result;
        },
        py::arg("gate"), py::arg("up"),
        "Fused SiLU(gate) * up for SwiGLU.");

    // --- RoPE ---
    infer.def("apply_rope",
        [](py::array_t<float> x,
           py::array_t<float> cos_f,
           py::array_t<float> sin_f,
           int num_heads,
           int head_dim) -> py::array_t<float>
        {
            auto x_buf = x.request();
            auto c_buf = cos_f.request();
            auto s_buf = sin_f.request();

            py::array_t<float> result = make_1d_array<float>(static_cast<py::ssize_t>(x_buf.size));
            auto r_buf = result.request();

            inference::apply_rope(
                static_cast<const float*>(x_buf.ptr),
                static_cast<const float*>(c_buf.ptr),
                static_cast<const float*>(s_buf.ptr),
                static_cast<float*>(r_buf.ptr),
                num_heads,
                head_dim
            );
            // Reshape to match input
            result.resize({num_heads, head_dim});
            return result;
        },
        py::arg("x"), py::arg("cos_f"), py::arg("sin_f"),
        py::arg("num_heads"), py::arg("head_dim"),
        "Apply rotary positional embeddings.");

    // --- GQA Attention ---
    infer.def("gqa_attention",
        [](py::array_t<float> q,
           py::array_t<float> k_cache,
           py::array_t<float> v_cache,
           int num_heads,
           int num_kv_heads,
           int head_dim,
           int seq_len) -> py::array_t<float>
        {
            auto q_buf = q.request();
            auto k_buf = k_cache.request();
            auto v_buf = v_cache.request();

            py::array_t<float> result({num_heads, head_dim});
            auto r_buf = result.request();

            inference::gqa_attention(
                static_cast<const float*>(q_buf.ptr),
                static_cast<const float*>(k_buf.ptr),
                static_cast<const float*>(v_buf.ptr),
                static_cast<float*>(r_buf.ptr),
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len
            );
            return result;
        },
        py::arg("q"), py::arg("k_cache"), py::arg("v_cache"),
        py::arg("num_heads"), py::arg("num_kv_heads"),
        py::arg("head_dim"), py::arg("seq_len"),
        "Grouped Query Attention with AVX2-accelerated dot products.");

    // --- Hybrid SSM selective scan step ---
    infer.def("ssm_selective_scan",
        [](py::array_t<float, py::array::c_style> x,
           py::array_t<float, py::array::c_style> conv_kernel,
           py::array_t<float, py::array::c_style> ssm_a,
           py::array_t<float, py::array::c_style> ssm_dt_bias,
           py::array_t<float, py::array::c_style> dt_proj,
           py::array_t<float, py::array::c_style> b_proj,
           py::array_t<float, py::array::c_style> ssm_norm,
           py::array_t<float, py::array::c_style> ssm_state,
           py::array_t<float, py::array::c_style> conv_state,
           int q_dim) -> py::array_t<float>
        {
            auto x_buf = x.request();
            auto ck_buf = conv_kernel.request();
            auto a_buf = ssm_a.request();
            auto dtb_buf = ssm_dt_bias.request();
            auto dtp_buf = dt_proj.request();
            auto bp_buf = b_proj.request();
            auto n_buf = ssm_norm.request();
            auto st_buf = ssm_state.request();
            auto cv_buf = conv_state.request();

            if (x_buf.ndim != 1) throw std::runtime_error("x must be 1-D");
            if (ck_buf.ndim != 2) throw std::runtime_error("conv_kernel must be 2-D");
            if (a_buf.ndim != 1 || dtb_buf.ndim != 1 || dtp_buf.ndim != 1 || bp_buf.ndim != 1)
                throw std::runtime_error("ssm_a/ssm_dt_bias/dt_proj/b_proj must be 1-D");
            if (st_buf.ndim != 1) throw std::runtime_error("ssm_state must be 1-D");
            if (cv_buf.ndim != 2) throw std::runtime_error("conv_state must be 2-D");

            int hidden_size = static_cast<int>(x_buf.shape[0]);
            int conv_dim = static_cast<int>(ck_buf.shape[0]);
            int conv_k = static_cast<int>(ck_buf.shape[1]);
            int s_dim = static_cast<int>(a_buf.shape[0]);
            int norm_dim = (n_buf.ndim == 1) ? static_cast<int>(n_buf.shape[0]) : 0;

            if (static_cast<int>(cv_buf.shape[0]) != conv_dim || static_cast<int>(cv_buf.shape[1]) != conv_k)
                throw std::runtime_error("conv_state shape must match conv_kernel");
            if (static_cast<int>(st_buf.shape[0]) != s_dim)
                throw std::runtime_error("ssm_state shape must match ssm_a");
            if (static_cast<int>(dtb_buf.shape[0]) != s_dim || static_cast<int>(dtp_buf.shape[0]) != s_dim || static_cast<int>(bp_buf.shape[0]) != s_dim)
                throw std::runtime_error("dt/b projection vectors must match ssm_a size");

            py::array_t<float> out = make_1d_array<float>(q_dim);
            auto out_buf = out.request();

            inference::ssm_selective_scan_step(
                static_cast<const float*>(x_buf.ptr),
                hidden_size,
                static_cast<const float*>(ck_buf.ptr),
                conv_dim,
                conv_k,
                static_cast<const float*>(a_buf.ptr),
                static_cast<const float*>(dtb_buf.ptr),
                static_cast<const float*>(dtp_buf.ptr),
                static_cast<const float*>(bp_buf.ptr),
                norm_dim > 0 ? static_cast<const float*>(n_buf.ptr) : nullptr,
                norm_dim,
                static_cast<float*>(st_buf.ptr),
                static_cast<float*>(cv_buf.ptr),
                s_dim,
                q_dim,
                static_cast<float*>(out_buf.ptr)
            );
            return out;
        },
        py::arg("x"), py::arg("conv_kernel"), py::arg("ssm_a"), py::arg("ssm_dt_bias"),
        py::arg("dt_proj"), py::arg("b_proj"), py::arg("ssm_norm"),
        py::arg("ssm_state"), py::arg("conv_state"), py::arg("q_dim"),
        "Hybrid SSM selective scan step with in-place state update.");

    // --- GatedDeltaNet recurrent step (Qwen3.5 linear attention) ---
    infer.def("deltanet_step",
        [](py::array_t<float, py::array::c_style> qkv,
           py::array_t<float, py::array::c_style> z,
           py::array_t<float, py::array::c_style> a,
           py::array_t<float, py::array::c_style> b,
           py::array_t<float, py::array::c_style> conv_kernel,
           py::array_t<float, py::array::c_style> a_log,
           py::array_t<float, py::array::c_style> dt_bias,
           py::array_t<float, py::array::c_style> norm_weight,
           float norm_eps,
           py::array_t<float, py::array::c_style> conv_state,
           py::array_t<float, py::array::c_style> recurrent_state,
           int n_k_heads,
           int n_v_heads,
           int head_k_dim,
           int head_v_dim,
           int conv_k) -> py::array_t<float>
        {
            auto qkv_buf = qkv.request();
            auto z_buf = z.request();
            auto a_buf = a.request();
            auto b_buf = b.request();
            auto ck_buf = conv_kernel.request();
            auto al_buf = a_log.request();
            auto dt_buf = dt_bias.request();
            auto nw_buf = norm_weight.request();
            auto cs_buf = conv_state.request();
            auto rs_buf = recurrent_state.request();

            int value_dim = n_v_heads * head_v_dim;

            py::array_t<float> out = make_1d_array<float>(value_dim);
            auto out_buf = out.request();

            inference::deltanet_step(
                static_cast<const float*>(qkv_buf.ptr),
                static_cast<const float*>(z_buf.ptr),
                static_cast<const float*>(a_buf.ptr),
                static_cast<const float*>(b_buf.ptr),
                static_cast<const float*>(ck_buf.ptr),
                static_cast<const float*>(al_buf.ptr),
                static_cast<const float*>(dt_buf.ptr),
                nw_buf.size > 0 ? static_cast<const float*>(nw_buf.ptr) : nullptr,
                norm_eps,
                static_cast<float*>(cs_buf.ptr),
                static_cast<float*>(rs_buf.ptr),
                n_k_heads, n_v_heads,
                head_k_dim, head_v_dim,
                conv_k,
                static_cast<float*>(out_buf.ptr)
            );
            return out;
        },
        py::arg("qkv"), py::arg("z"), py::arg("a"), py::arg("b"),
        py::arg("conv_kernel"), py::arg("a_log"), py::arg("dt_bias"),
        py::arg("norm_weight"), py::arg("norm_eps"),
        py::arg("conv_state"), py::arg("recurrent_state"),
        py::arg("n_k_heads"), py::arg("n_v_heads"),
        py::arg("head_k_dim"), py::arg("head_v_dim"), py::arg("conv_k"),
        "GatedDeltaNet single-step recurrence for Qwen3.5 linear attention layers.");

    // --- Flag for inference kernel availability ---
    infer.attr("available") = true;

    m.def("analyze_signal_concentration",
          [](py::array_t<double> arr, double energy_threshold) {
              RFTMWEngine eng(static_cast<size_t>(arr.size()));
              const auto metrics = eng.analyze_signal(numpy_to_realvec(arr), energy_threshold);
              py::dict out;
              out["top_k"] = py::int_(metrics.top_k);
              out["top_k_energy_ratio"] = py::float_(metrics.top_k_energy_ratio);
              out["sparsity_ratio"] = py::float_(metrics.sparsity_ratio);
              out["coherence_proxy"] = py::float_(metrics.coherence_proxy);
              out["use_sparse_route"] = py::bool_(metrics.use_sparse_route);
              return out;
          },
          py::arg("signal"),
          py::arg("energy_threshold") = 0.90,
          "Module-level concentration analysis helper.");
}
