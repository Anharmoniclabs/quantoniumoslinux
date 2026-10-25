/*
 * SPDX-License-Identifier: LicenseRef-SPX-Proprietary
 * Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
 *
 * rftmw_spectral_inference.hpp - Spectral-Domain Inference Integration v2
 * =======================================================================
 *
 * C++ wrapper for quantized spectral kernels (4-bit mag + 8-bit phase):
 *
 *   1. SpectralWeight: manages quantized sparse spectral weight matrices
 *   2. from_dense() / from_quantized(): convert GGUF weights at load time
 *   3. matmul_vec(): drop-in replacement for quant_matmul_vec
 *   4. Non-uniform sparsity via rftmw_layer_keep_ratio()
 */

#pragma once

#include "rftmw_spectral_kernels.h"
#include "rftmw_inference.hpp"
#include <cstdint>
#include <cstring>
#include <vector>
#include <memory>
#include <cmath>
#include <cassert>
#include <cstdio>
#include <thread>

namespace rftmw {
namespace spectral {

/* ================================================================
 * Configuration
 * ================================================================ */

struct SpectralConfig {
    int   block_size  = RFTMW_SPECTRAL_DEFAULT_BLOCK;
    float keep_ratio  = RFTMW_SPECTRAL_DEFAULT_KEEP;
    bool  enabled     = false;
    bool  nonuniform  = true;  /* use per-layer sparsity schedule */
    int   total_layers = 36;   /* for non-uniform schedule */
};

/* ================================================================
 * SpectralWeight: owns a quantized sparse spectral weight matrix
 * ================================================================ */

class SpectralWeight {
public:
    SpectralWeight() : initialized_(false) {
        std::memset(&w_, 0, sizeof(w_));
    }

    ~SpectralWeight() {
        if (initialized_) {
            rftmw_spectral_weight_free(&w_);
            initialized_ = false;
        }
    }

    SpectralWeight(const SpectralWeight&) = delete;
    SpectralWeight& operator=(const SpectralWeight&) = delete;

    SpectralWeight(SpectralWeight&& other) noexcept : w_(other.w_), initialized_(other.initialized_) {
        other.initialized_ = false;
        std::memset(&other.w_, 0, sizeof(other.w_));
    }

    SpectralWeight& operator=(SpectralWeight&& other) noexcept {
        if (this != &other) {
            if (initialized_) rftmw_spectral_weight_free(&w_);
            w_ = other.w_;
            initialized_ = other.initialized_;
            other.initialized_ = false;
            std::memset(&other.w_, 0, sizeof(other.w_));
        }
        return *this;
    }

    void from_dense(
        const float* weights,
        int out_features,
        int in_features,
        int block_size = RFTMW_SPECTRAL_DEFAULT_BLOCK,
        float keep_ratio = RFTMW_SPECTRAL_DEFAULT_KEEP
    ) {
        if (initialized_) rftmw_spectral_weight_free(&w_);
        rftmw_spectral_weight_init(&w_, weights, out_features, in_features,
                                    block_size, keep_ratio);
        initialized_ = true;
    }

    void from_quantized(
        const uint8_t* raw_weights,
        int out_features,
        int in_features,
        int bytes_per_row,
        const char* quant_type,
        int block_size = RFTMW_SPECTRAL_DEFAULT_BLOCK,
        float keep_ratio = RFTMW_SPECTRAL_DEFAULT_KEEP
    ) {
        std::vector<float> dense((size_t)out_features * (size_t)in_features);

        for (int row = 0; row < out_features; ++row) {
            const uint8_t* row_data = raw_weights + (size_t)row * (size_t)bytes_per_row;
            float* out = dense.data() + (size_t)row * (size_t)in_features;

            if (std::strcmp(quant_type, "q4k") == 0) {
                int n_blocks_q = in_features / 256;
                for (int b = 0; b < n_blocks_q; ++b) {
                    inference::dequant_q4k_block(row_data + b * 144, out + b * 256);
                }
            } else {
                int n_blocks_q = in_features / 256;
                for (int b = 0; b < n_blocks_q; ++b) {
                    inference::dequant_q6k_block(row_data + b * 210, out + b * 256);
                }
            }
        }

        from_dense(dense.data(), out_features, in_features, block_size, keep_ratio);
    }

    void matmul_vec(const float* x, float* y) {
        assert(initialized_);
        rftmw_spectral_gemv(&w_, x, y);
    }

    /* Threaded spectral GEMV:  transform x once, process output rows in parallel */
    void matmul_vec_parallel(const float* x, float* y, int n_threads = 0) {
        assert(initialized_);
        if (n_threads <= 0) {
            n_threads = static_cast<int>(std::thread::hardware_concurrency());
            if (n_threads < 1) n_threads = 1;
            if (n_threads > 8) n_threads = 8;
        }
        int out = w_.out_features;
        int bs  = w_.block_size;
        int nb  = w_.n_blocks;

        /* Step 1: transform all blocks of x into spectral domain (single-thread, fast) */
        for (int b = 0; b < nb; ++b) {
            int start = b * bs;
            int end   = start + bs;
            if (end > w_.in_features) {
                int actual = w_.in_features - start;
                std::memcpy(w_.scratch_re, &x[start],
                            actual > 0 ? (size_t)actual * sizeof(float) : 0);
                std::memset(w_.scratch_re + (actual > 0 ? actual : 0), 0,
                            (size_t)(bs - (actual > 0 ? actual : 0)) * sizeof(float));
                rftmw_spectral_forward_block(&w_.basis, w_.scratch_re,
                    &w_.x_spec_re[b * bs], &w_.x_spec_im[b * bs]);
            } else {
                rftmw_spectral_forward_block(&w_.basis, &x[start],
                    &w_.x_spec_re[b * bs], &w_.x_spec_im[b * bs]);
            }
        }

        /* Step 2: parallel sparse dot products for output rows */
        const float* cos_lut = w_.phase_lut.cos_lut;
        const float* sin_lut = w_.phase_lut.sin_lut;

        auto worker = [&](int row_start, int row_end) {
            for (int row = row_start; row < row_end; ++row) {
                float sum = 0.0f;
                const rftmw_qsparse_row_t* srow = &w_.rows[row];
                for (int b = 0; b < srow->n_blocks; ++b) {
                    const rftmw_qsparse_block_t* blk = &srow->blocks[b];
                    if (blk->n_nonzero == 0) continue;
                    sum += rftmw_qspectral_sparse_dot(
                        blk->indices, blk->mag4, blk->phase8, blk->peak_mag,
                        cos_lut, sin_lut,
                        &w_.x_spec_re[(size_t)b * bs],
                        &w_.x_spec_im[(size_t)b * bs],
                        (int)blk->n_nonzero);
                }
                y[row] = sum;
            }
        };

        if (out < 256 || n_threads <= 1) {
            worker(0, out);
        } else {
            std::vector<std::thread> threads;
            threads.reserve((size_t)n_threads);
            int per = (out + n_threads - 1) / n_threads;
            for (int t = 0; t < n_threads; ++t) {
                int s = t * per;
                int e = std::min(s + per, out);
                if (s < e) threads.emplace_back(worker, s, e);
            }
            for (auto& th : threads) th.join();
        }
    }

    /* ── Streaming threaded conversion from quantized weights ────
     * Dequants + FFT-compresses one row at a time, across n_threads.
     * No big dense matrix allocation (saves ~260MB per FFN matrix). */
    void from_quantized_threaded(
        const uint8_t* raw_weights,
        int out_feat, int in_feat,
        int bytes_per_row, const char* quant_type,
        int block_sz = RFTMW_SPECTRAL_DEFAULT_BLOCK,
        float keep_ratio = RFTMW_SPECTRAL_DEFAULT_KEEP,
        int n_threads = 0
    ) {
        if (initialized_) rftmw_spectral_weight_free(&w_);
        std::memset(&w_, 0, sizeof(w_));
        w_.out_features = out_feat;
        w_.in_features  = in_feat;
        w_.block_size   = block_sz;
        w_.n_blocks     = (in_feat + block_sz - 1) / block_sz;

        rftmw_spectral_basis_init(&w_.basis, block_sz);
        rftmw_phase_lut_init(&w_.phase_lut);

        auto alloc32 = [](size_t n) -> float* {
            size_t bytes = n * sizeof(float);
            if (bytes == 0) bytes = sizeof(float);
#if defined(_WIN32)
            void* p = _aligned_malloc(bytes, 32);
#else
            void* p = nullptr; posix_memalign(&p, 32, bytes);
#endif
            if (p) std::memset(p, 0, bytes);
            return (float*)p;
        };
        w_.scratch_re = alloc32((size_t)block_sz);
        w_.scratch_im = alloc32((size_t)block_sz);
        w_.x_spec_re  = alloc32((size_t)w_.n_blocks * (size_t)block_sz);
        w_.x_spec_im  = alloc32((size_t)w_.n_blocks * (size_t)block_sz);

        w_.rows = (rftmw_qsparse_row_t*)std::calloc(
            (size_t)out_feat, sizeof(rftmw_qsparse_row_t));

        if (n_threads <= 0) {
            n_threads = static_cast<int>(std::thread::hardware_concurrency());
            if (n_threads < 1) n_threads = 1;
            if (n_threads > 16) n_threads = 16;
        }

        bool is_q4k = (std::strcmp(quant_type, "q4k") == 0);
        int bpr = bytes_per_row;
        int n_blocks_q = in_feat / 256;

        auto worker = [&](int row_start, int row_end) {
            std::vector<float> row_dense((size_t)in_feat);
            for (int r = row_start; r < row_end; ++r) {
                const uint8_t* rd = raw_weights + (size_t)r * (size_t)bpr;
                if (is_q4k) {
                    for (int b = 0; b < n_blocks_q; ++b)
                        inference::dequant_q4k_block(rd + b * 144,
                            row_dense.data() + b * 256);
                } else {
                    for (int b = 0; b < n_blocks_q; ++b)
                        inference::dequant_q6k_block(rd + b * 210,
                            row_dense.data() + b * 256);
                }
                rftmw_spectral_compress_row(
                    &w_.basis, &w_.phase_lut,
                    row_dense.data(), in_feat, keep_ratio,
                    &w_.rows[r]);
            }
        };

        if (out_feat < 100 || n_threads <= 1) {
            worker(0, out_feat);
        } else {
            std::vector<std::thread> threads;
            threads.reserve((size_t)n_threads);
            int per = (out_feat + n_threads - 1) / n_threads;
            for (int t = 0; t < n_threads; ++t) {
                int s = t * per;
                int e = std::min(s + per, out_feat);
                if (s < e) threads.emplace_back(worker, s, e);
            }
            for (auto& th : threads) th.join();
        }
        initialized_ = true;
    }

    /* ── Binary serialize / deserialize for disk cache ──────── */
    bool save_to_stream(std::FILE* fp) const {
        if (!initialized_ || !fp) return false;
        auto w4 = [&](const void* p) { std::fwrite(p, 4, 1, fp); };
        auto w2 = [&](const void* p) { std::fwrite(p, 2, 1, fp); };
        w4(&w_.out_features);
        w4(&w_.in_features);
        w4(&w_.block_size);
        w4(&w_.n_blocks);
        for (int r = 0; r < w_.out_features; ++r) {
            const auto& row = w_.rows[r];
            w4(&row.n_blocks);
            for (int b = 0; b < row.n_blocks; ++b) {
                const auto& blk = row.blocks[b];
                w2(&blk.n_nonzero);
                w4(&blk.peak_mag);
                if (blk.n_nonzero > 0) {
                    std::fwrite(blk.indices, 2, blk.n_nonzero, fp);
                    std::fwrite(blk.mag4, 1, ((size_t)blk.n_nonzero + 1) / 2, fp);
                    std::fwrite(blk.phase8, 1, blk.n_nonzero, fp);
                }
            }
        }
        return true;
    }

    bool load_from_stream(std::FILE* fp) {
        if (!fp) return false;
        if (initialized_) rftmw_spectral_weight_free(&w_);
        std::memset(&w_, 0, sizeof(w_));

        auto r4 = [&](void* p) { return std::fread(p, 4, 1, fp); };
        auto r2 = [&](void* p) { return std::fread(p, 2, 1, fp); };
        r4(&w_.out_features); r4(&w_.in_features);
        r4(&w_.block_size);   r4(&w_.n_blocks);

        rftmw_spectral_basis_init(&w_.basis, w_.block_size);
        rftmw_phase_lut_init(&w_.phase_lut);

        auto alloc32 = [](size_t n) -> float* {
            size_t bytes = n * sizeof(float);
            if (bytes == 0) bytes = sizeof(float);
#if defined(_WIN32)
            void* p = _aligned_malloc(bytes, 32);
#else
            void* p = nullptr; posix_memalign(&p, 32, bytes);
#endif
            if (p) std::memset(p, 0, bytes);
            return (float*)p;
        };
        w_.scratch_re = alloc32((size_t)w_.block_size);
        w_.scratch_im = alloc32((size_t)w_.block_size);
        w_.x_spec_re  = alloc32((size_t)w_.n_blocks * (size_t)w_.block_size);
        w_.x_spec_im  = alloc32((size_t)w_.n_blocks * (size_t)w_.block_size);

        w_.rows = (rftmw_qsparse_row_t*)std::calloc(
            (size_t)w_.out_features, sizeof(rftmw_qsparse_row_t));
        for (int r = 0; r < w_.out_features; ++r) {
            int nb = 0;
            r4(&nb);
            w_.rows[r].n_blocks = nb;
            w_.rows[r].blocks = (rftmw_qsparse_block_t*)std::calloc(
                (size_t)nb, sizeof(rftmw_qsparse_block_t));
            for (int b = 0; b < nb; ++b) {
                auto& blk = w_.rows[r].blocks[b];
                r2(&blk.n_nonzero);
                r4(&blk.peak_mag);
                if (blk.n_nonzero > 0) {
                    blk.indices = (uint16_t*)std::calloc(blk.n_nonzero, 2);
                    blk.mag4    = (uint8_t*)std::calloc(
                        ((size_t)blk.n_nonzero + 1) / 2, 1);
                    blk.phase8  = (uint8_t*)std::calloc(blk.n_nonzero, 1);
                    std::fread(blk.indices, 2, blk.n_nonzero, fp);
                    std::fread(blk.mag4, 1,
                        ((size_t)blk.n_nonzero + 1) / 2, fp);
                    std::fread(blk.phase8, 1, blk.n_nonzero, fp);
                }
            }
        }
        initialized_ = true;
        return true;
    }

    bool is_initialized() const { return initialized_; }
    int out_features() const { return w_.out_features; }
    int in_features()  const { return w_.in_features; }

    size_t memory_bytes() const {
        if (!initialized_) return 0;
        size_t total = sizeof(w_);
        total += (size_t)w_.block_size * 2 * sizeof(float); /* cos+sin basis */
        total += (size_t)w_.block_size * 2 * sizeof(float); /* scratch */
        total += (size_t)w_.n_blocks * (size_t)w_.block_size * 2 * sizeof(float); /* x_spec */
        /* Quantized sparse coefficients */
        for (int r = 0; r < w_.out_features; ++r) {
            for (int b = 0; b < w_.rows[r].n_blocks; ++b) {
                const rftmw_qsparse_block_t& blk = w_.rows[r].blocks[b];
                /* indices: 2 bytes, mag4: 0.5 bytes, phase8: 1 byte per coeff */
                total += (size_t)blk.n_nonzero * sizeof(uint16_t);           /* indices */
                total += ((size_t)blk.n_nonzero + 1) / 2;                     /* mag4 packed */
                total += (size_t)blk.n_nonzero;                                /* phase8 */
                total += sizeof(rftmw_qsparse_block_t);
            }
            total += sizeof(rftmw_qsparse_row_t);
        }
        return total;
    }

    float avg_sparsity() const {
        if (!initialized_ || w_.out_features == 0) return 0.0f;
        size_t total_nz = 0, total_entries = 0;
        for (int r = 0; r < w_.out_features; ++r) {
            for (int b = 0; b < w_.rows[r].n_blocks; ++b) {
                total_nz += w_.rows[r].blocks[b].n_nonzero;
                total_entries += (size_t)w_.block_size;
            }
        }
        return total_entries > 0 ? (float)total_nz / (float)total_entries : 0.0f;
    }

private:
    rftmw_spectral_weight_t w_;
    bool initialized_;
};

/* ================================================================
 * SpectralLayerWeights
 * ================================================================ */

struct SpectralLayerWeights {
    SpectralWeight attn_q;
    SpectralWeight attn_k;
    SpectralWeight attn_v;
    SpectralWeight attn_o;
    SpectralWeight ffn_gate;
    SpectralWeight ffn_up;
    SpectralWeight ffn_down;

    bool any_initialized() const {
        return attn_q.is_initialized() || attn_k.is_initialized() ||
               attn_v.is_initialized() || attn_o.is_initialized() ||
               ffn_gate.is_initialized() || ffn_up.is_initialized() ||
               ffn_down.is_initialized();
    }

    size_t total_memory_bytes() const {
        return attn_q.memory_bytes() + attn_k.memory_bytes() +
               attn_v.memory_bytes() + attn_o.memory_bytes() +
               ffn_gate.memory_bytes() + ffn_up.memory_bytes() +
               ffn_down.memory_bytes();
    }
};

/* ================================================================
 * Batch conversion helper with non-uniform sparsity
 * ================================================================ */

inline void convert_layer_weight(
    SpectralWeight& sw,
    const uint8_t* raw,
    int out_features,
    int in_features,
    int bytes_per_row,
    const char* quant_type,
    const SpectralConfig& cfg,
    int layer_idx = -1,
    const char* name = nullptr
) {
    float keep = cfg.keep_ratio;
    if (cfg.nonuniform && layer_idx >= 0) {
        keep = rftmw_layer_keep_ratio(layer_idx, cfg.total_layers, cfg.keep_ratio);
    }

    sw.from_quantized(raw, out_features, in_features,
                      bytes_per_row, quant_type,
                      cfg.block_size, keep);
    if (name) {
        fprintf(stderr, "[spectral] %s: %dx%d keep=%.0f%% -> %.2f MB (q4+phase)\n",
                name, out_features, in_features,
                sw.avg_sparsity() * 100.0f,
                (float)sw.memory_bytes() / (1024.0f * 1024.0f));
    }
}

} /* namespace spectral */
} /* namespace rftmw */
