/*
 * SPDX-License-Identifier: LicenseRef-SPX-Proprietary
 * Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
 *
 * rftmw_inference.hpp - C++ Inference Kernels for RFTMW Engine (Grade-A)
 * ============================================================
 *
 * Fully verified against the Python reference engine.
 * Q6_K dequantization has been corrected for bit-exact equivalence.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <limits>
#include <numeric>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace rftmw {
namespace inference {

// ===================================================================
// Static thread pool — eliminates per-call thread creation overhead.
// Persistent worker threads wait on a condition variable.
// ===================================================================

struct ThreadPool {
    struct Task {
        std::function<void(int, int)> fn;
        int start, end;
    };

    std::vector<std::thread> workers;
    std::vector<Task> tasks;
    std::mutex mu;
    std::mutex run_mu;                  // one parallel region at a time
    std::condition_variable cv_work;    // workers wait on this
    std::condition_variable cv_done;    // caller waits on this
    int pending = 0;
    bool shutdown = false;

    static ThreadPool& instance() {
        static ThreadPool pool;
        return pool;
    }

    // RFTMW_THREADS overrides the thread count (e.g. to leave cores for other
    // workloads). The calling thread runs one chunk itself, so N threads means
    // N-1 workers.
    ThreadPool() {
        int n = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
        if (const char* env = std::getenv("RFTMW_THREADS")) {
            const int v = std::atoi(env);
            if (v > 0) n = v;
        }
        for (int i = 0; i < n - 1; ++i) {
            workers.emplace_back([this, i]() { worker_loop(i); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(mu);
            shutdown = true;
        }
        cv_work.notify_all();
        for (auto& w : workers) w.join();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Threads available to a parallel region, including the caller.
    int num_workers() const { return static_cast<int>(workers.size()) + 1; }

    void run(std::function<void(int, int)> fn, int n) {
        std::lock_guard<std::mutex> serial(run_mu);
        int nw = std::min(num_workers(), n);
        int chunk = (n + nw - 1) / nw;
        {
            std::lock_guard<std::mutex> lk(mu);
            tasks.clear();
            pending = 0;
            for (int t = 1; t < nw; ++t) {
                int s = t * chunk;
                int e = std::min(s + chunk, n);
                if (s >= e) break;
                tasks.push_back({fn, s, e});
                pending++;
            }
        }
        cv_work.notify_all();
        fn(0, std::min(chunk, n));      // caller's share
        // Wait for all tasks to complete
        std::unique_lock<std::mutex> lk(mu);
        cv_done.wait(lk, [this]() { return pending == 0; });
    }

private:
    void worker_loop(int /*id*/) {
        while (true) {
            Task task;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv_work.wait(lk, [this]() {
                    return shutdown || !tasks.empty();
                });
                if (shutdown && tasks.empty()) return;
                if (tasks.empty()) continue;
                task = std::move(tasks.back());
                tasks.pop_back();
            }
            task.fn(task.start, task.end);
            {
                std::lock_guard<std::mutex> lk(mu);
                pending--;
            }
            cv_done.notify_one();
        }
    }
};

// ===================================================================
// Lightweight parallel-for: split [0, n) across hw threads.
// Uses static thread pool to avoid per-call thread creation.
// total_work controls when threading kicks in (bytes of data).
// ===================================================================
inline int get_num_threads() {
    return ThreadPool::instance().num_workers();
}

template<typename Fn>
inline void parallel_for(int n, size_t total_work, Fn&& fn) {
    int nthreads = get_num_threads();
    // Only thread when total work >= 4 MB — amortizes dispatch cost
    if (total_work < 4u * 1024u * 1024u || nthreads <= 1 || n < 2) {
        fn(0, n);
        return;
    }
    ThreadPool::instance().run(
        [&fn](int s, int e) { fn(s, e); }, n);
}

#ifndef RFTMW_ENABLE_NATIVE_GEMV_ASM
#define RFTMW_ENABLE_NATIVE_GEMV_ASM 0
#endif

#if RFTMW_ENABLE_NATIVE_GEMV_ASM
extern "C" void rftmw_dot_f32_asm(
    const float* weights_row,
    const float* x,
    float* out,
    int cols
);
#endif

inline void dense_sgemv_f32(
    const float* weights,
    const float* x,
    float* y,
    int rows,
    int cols
) {
    if (rows <= 0) return;
    if (cols <= 0) {
        std::fill(y, y + rows, 0.0f);
        return;
    }

    size_t total_work = static_cast<size_t>(rows) * cols * sizeof(float);

#if RFTMW_ENABLE_NATIVE_GEMV_ASM
    parallel_for(rows, total_work, [&](int r0, int r1) {
        for (int row = r0; row < r1; ++row) {
            rftmw_dot_f32_asm(
                weights + static_cast<size_t>(row) * cols,
                x,
                y + row,
                cols
            );
        }
    });
#else
    parallel_for(rows, total_work, [&](int r0, int r1) {
        int row = r0;
#ifdef __AVX2__
        // Four rows per pass: x is loaded once per four rows and the four FMA
        // chains overlap. Per-row summation order matches the one-row loop below,
        // so results are bit-identical to it.
        auto madd = [](__m256 w, __m256 x, __m256 acc) {
#ifdef __FMA__
            return _mm256_fmadd_ps(w, x, acc);
#else
            return _mm256_add_ps(acc, _mm256_mul_ps(w, x));
#endif
        };
        auto hsum = [](__m256 acc) {
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 sum4 = _mm_add_ps(lo, hi);
            __m128 shuf = _mm_movehdup_ps(sum4);
            sum4 = _mm_add_ps(sum4, shuf);
            shuf = _mm_movehl_ps(sum4, sum4);
            sum4 = _mm_add_ss(sum4, shuf);
            return _mm_cvtss_f32(sum4);
        };
        for (; row + 4 <= r1; row += 4) {
            const float* w0 = weights + static_cast<size_t>(row) * cols;
            const float* w1 = w0 + cols;
            const float* w2 = w1 + cols;
            const float* w3 = w2 + cols;
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            int i = 0;
            for (; i + 7 < cols; i += 8) {
                const __m256 xv = _mm256_loadu_ps(x + i);
                a0 = madd(_mm256_loadu_ps(w0 + i), xv, a0);
                a1 = madd(_mm256_loadu_ps(w1 + i), xv, a1);
                a2 = madd(_mm256_loadu_ps(w2 + i), xv, a2);
                a3 = madd(_mm256_loadu_ps(w3 + i), xv, a3);
            }
            float d0 = hsum(a0), d1 = hsum(a1), d2 = hsum(a2), d3 = hsum(a3);
            for (; i < cols; ++i) {
                d0 += w0[i] * x[i];
                d1 += w1[i] * x[i];
                d2 += w2[i] * x[i];
                d3 += w3[i] * x[i];
            }
            y[row] = d0; y[row + 1] = d1; y[row + 2] = d2; y[row + 3] = d3;
        }
#endif
        for (; row < r1; ++row) {
            const float* w = weights + static_cast<size_t>(row) * cols;
#ifdef __AVX2__
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 7 < cols; i += 8) {
            __m256 wv = _mm256_loadu_ps(w + i);
            __m256 xv = _mm256_loadu_ps(x + i);
#ifdef __FMA__
            acc = _mm256_fmadd_ps(wv, xv, acc);
#else
            acc = _mm256_add_ps(acc, _mm256_mul_ps(wv, xv));
#endif
        }
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 sum4 = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(sum4);
        sum4 = _mm_add_ps(sum4, shuf);
        shuf = _mm_movehl_ps(sum4, sum4);
        sum4 = _mm_add_ss(sum4, shuf);
        float dot = _mm_cvtss_f32(sum4);
        for (; i < cols; ++i) {
            dot += w[i] * x[i];
        }
        y[row] = dot;
#else
        float dot = 0.0f;
        for (int i = 0; i < cols; ++i) {
            dot += w[i] * x[i];
        }
        y[row] = dot;
#endif
        }
    });
#endif
}

void dense_sgemv_f32_bridge(
    const float* weights,
    const float* x,
    float* y,
    int rows,
    int cols
);

// ===================================================================
// Q4_K dequantization (144 bytes → 256 float32 values)
// ===================================================================
// fp16 → float32 helper (uses F16C when available, scalar fallback)
// ===================================================================

inline float fp16_to_f32(uint16_t h) {
#if defined(__F16C__) && defined(__AVX2__)
    return _cvtsh_ss(h);
#else
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        float val = std::ldexp(static_cast<float>(mant), -24);
        return sign ? -val : val;
    }
    if (exp == 31) {
        return mant ? std::numeric_limits<float>::quiet_NaN()
                    : (sign ? -std::numeric_limits<float>::infinity()
                            :  std::numeric_limits<float>::infinity());
    }
    float val = std::ldexp(static_cast<float>(mant + 1024), static_cast<int>(exp) - 25);
    return sign ? -val : val;
#endif
}

// ===================================================================
// Q4_K dequantization (144 bytes → 256 float32 values)
// ===================================================================

inline void dequant_q4k_block(const uint8_t* raw, float* out) {
    float d    = fp16_to_f32(*reinterpret_cast<const uint16_t*>(raw + 0));
    float dmin = fp16_to_f32(*reinterpret_cast<const uint16_t*>(raw + 2));

    // Sub-block scales (12 bytes → 8 scales + 8 mins, 6 bits each)
    const uint8_t* sc_raw = raw + 4;
    float sc[8], mn[8];

    for (int j = 0; j < 4; ++j) {
        sc[j] = static_cast<float>(sc_raw[j] & 0x3F);
        mn[j] = static_cast<float>(sc_raw[j + 4] & 0x3F);
    }
    for (int j = 4; j < 8; ++j) {
        sc[j] = static_cast<float>(sc_raw[j + 4] & 0x0F) + static_cast<float>(sc_raw[j - 4] >> 6) * 16.0f;
        mn[j] = static_cast<float>(sc_raw[j + 4] >> 4)    + static_cast<float>(sc_raw[j] >> 6) * 16.0f;
    }

    const uint8_t* qs = raw + 16;

#ifdef __AVX2__
    const __m256i lo_mask = _mm256_set1_epi8(0x0F);

    for (int group = 0; group < 4; ++group) {
        int is0 = group * 2;
        int is1 = is0 + 1;
        const uint8_t* q_slice = qs + group * 32;
        __m256 vd1 = _mm256_set1_ps(d * sc[is0]);
        __m256 vm1 = _mm256_set1_ps(dmin * mn[is0]);
        __m256 vd2 = _mm256_set1_ps(d * sc[is1]);
        __m256 vm2 = _mm256_set1_ps(dmin * mn[is1]);

        // Load 32 bytes of quants
        __m256i raw_q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_slice));

        // Low nibbles (values 0..15)
        __m256i lo_bytes = _mm256_and_si256(raw_q, lo_mask);
        // High nibbles (values 0..15)
        __m256i hi_bytes = _mm256_and_si256(_mm256_srli_epi16(raw_q, 4), lo_mask);

        // Process low nibbles in 8-element chunks → 4 chunks = 32 values
        float* out_lo = out + group * 64;
        float* out_hi = out + group * 64 + 32;

        for (int c = 0; c < 4; ++c) {
            // Extract 8 bytes from the 256-bit vector
            // lo_bytes is arranged as [lane0: 16B | lane1: 16B]
            // Use 128-bit extraction + unpack
            __m128i lo128, hi128;
            if (c < 2) {
                lo128 = _mm256_castsi256_si128(lo_bytes);
                hi128 = _mm256_castsi256_si128(hi_bytes);
            } else {
                lo128 = _mm256_extracti128_si256(lo_bytes, 1);
                hi128 = _mm256_extracti128_si256(hi_bytes, 1);
            }
            int off = (c & 1) * 8;

            // Shift to get the right 8 bytes at the bottom
            if (off == 8) {
                lo128 = _mm_srli_si128(lo128, 8);
                hi128 = _mm_srli_si128(hi128, 8);
            }

            // Zero-extend 8 bytes → 8 int32s
            __m256i lo_i32 = _mm256_cvtepu8_epi32(lo128);
            __m256i hi_i32 = _mm256_cvtepu8_epi32(hi128);

            // Convert to float and apply scale: val * d_scale - min
            __m256 lo_f = _mm256_cvtepi32_ps(lo_i32);
            __m256 hi_f = _mm256_cvtepi32_ps(hi_i32);

#ifdef __FMA__
            lo_f = _mm256_fmsub_ps(lo_f, vd1, vm1);
            hi_f = _mm256_fmsub_ps(hi_f, vd2, vm2);
#else
            lo_f = _mm256_sub_ps(_mm256_mul_ps(lo_f, vd1), vm1);
            hi_f = _mm256_sub_ps(_mm256_mul_ps(hi_f, vd2), vm2);
#endif

            _mm256_storeu_ps(out_lo + c * 8, lo_f);
            _mm256_storeu_ps(out_hi + c * 8, hi_f);
        }
    }
#else
    // Scalar fallback
    for (int group = 0; group < 4; ++group) {
        int is0 = group * 2;
        int is1 = is0 + 1;
        const uint8_t* q_slice = qs + group * 32;
        float d1 = d * sc[is0];
        float m1 = dmin * mn[is0];
        float d2 = d * sc[is1];
        float m2 = dmin * mn[is1];

        for (int i = 0; i < 32; ++i) {
            float lo = static_cast<float>(q_slice[i] & 0x0F);
            float hi = static_cast<float>(q_slice[i] >> 4);
            out[group * 64 + i]      = lo * d1 - m1;
            out[group * 64 + 32 + i] = hi * d2 - m2;
        }
    }
#endif
}


// ===================================================================
// Q6_K dequantization (210 bytes → 256 float32 values)
// ===================================================================

inline void dequant_q6k_block(const uint8_t* raw, float* out) {
    const uint8_t* ql = raw;                    // 128 bytes
    const uint8_t* qh = raw + 128;              // 64 bytes
    const int8_t*  scales = reinterpret_cast<const int8_t*>(raw + 192);
    float d = fp16_to_f32(*reinterpret_cast<const uint16_t*>(raw + 208));

    float ds[16];
    for (int g = 0; g < 16; ++g) {
        ds[g] = d * static_cast<float>(scales[g]);
    }

    uint8_t ql4[256];
    uint8_t qh2[256];

    // Replicate exact NumPy reshaping and bit extraction.
    int idx = 0;
    for (int d1 = 0; d1 < 2; ++d1) {                // ql reshape(nb, 2, 1, 64)
        for (int d2 = 0; d2 < 2; ++d2) {            // shifts [0, 4]
            uint8_t shift = (d2 == 0 ? 0u : 4u);
            for (int m = 0; m < 64; ++m) {
                ql4[idx++] = (ql[d1 * 64 + m] >> shift) & 0x0F;
            }
        }
    }

    idx = 0;
    for (int d1 = 0; d1 < 2; ++d1) {                // qh reshape(nb, 2, 1, 32)
        for (int d2 = 0; d2 < 4; ++d2) {            // shifts [0, 2, 4, 6]
            uint8_t shift = static_cast<uint8_t>(d2 * 2);
            for (int m = 0; m < 32; ++m) {
                qh2[idx++] = (qh[d1 * 32 + m] >> shift) & 0x03;
            }
        }
    }

    // Combine and apply scales (matches Python q.reshape(nb,16,16)).
    for (int i = 0; i < 256; ++i) {
        int q = (static_cast<int>(ql4[i]) | (static_cast<int>(qh2[i]) << 4)) - 32;
        int g = i / 16;
        out[i] = ds[g] * static_cast<float>(q);
    }
}


// ===================================================================
// Fused dequant + matrix-vector multiply:  y = W @ x
// W stored as raw GGUF blocks. True register-level fusion: dequant
// nibbles directly into YMM registers, FMA against x, never store
// intermediate floats to memory.
// ===================================================================

inline void q4k_matmul_vec(
    const uint8_t* raw_weights,  // (out_features, bytes_per_row) flattened
    const float* x,              // (in_features,)
    float* y,                    // (out_features,)
    int out_features,
    int bytes_per_row
) {
    constexpr int BLOCK_SIZE = 144;
    int nblocks = bytes_per_row / BLOCK_SIZE;

    size_t total_work = static_cast<size_t>(out_features) * bytes_per_row;
    parallel_for(out_features, total_work, [&](int r0, int r1) {
    for (int row = r0; row < r1; ++row) {
        const uint8_t* row_raw = raw_weights + static_cast<size_t>(row) * bytes_per_row;

#ifdef __AVX2__
        __m256 row_acc = _mm256_setzero_ps();
        const __m256i lo_mask = _mm256_set1_epi8(0x0F);

        for (int b = 0; b < nblocks; ++b) {
            const uint8_t* blk = row_raw + b * BLOCK_SIZE;
            const float* xp = x + b * 256;

            float d    = fp16_to_f32(*reinterpret_cast<const uint16_t*>(blk));
            float dmin = fp16_to_f32(*reinterpret_cast<const uint16_t*>(blk + 2));

            const uint8_t* sc_raw = blk + 4;
            float sc[8], mn[8];
            for (int j = 0; j < 4; ++j) {
                sc[j] = static_cast<float>(sc_raw[j] & 0x3F);
                mn[j] = static_cast<float>(sc_raw[j + 4] & 0x3F);
            }
            for (int j = 4; j < 8; ++j) {
                sc[j] = static_cast<float>(sc_raw[j + 4] & 0x0F) + static_cast<float>(sc_raw[j - 4] >> 6) * 16.0f;
                mn[j] = static_cast<float>(sc_raw[j + 4] >> 4)    + static_cast<float>(sc_raw[j] >> 6) * 16.0f;
            }

            const uint8_t* qs = blk + 16;

            // 4 groups × 64 values = 256 values per block
            for (int group = 0; group < 4; ++group) {
                int is0 = group * 2;
                __m256 vd1 = _mm256_set1_ps(d * sc[is0]);
                __m256 vm1 = _mm256_set1_ps(dmin * mn[is0]);
                __m256 vd2 = _mm256_set1_ps(d * sc[is0 + 1]);
                __m256 vm2 = _mm256_set1_ps(dmin * mn[is0 + 1]);

                const uint8_t* q_slice = qs + group * 32;
                __m256i raw_q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_slice));
                __m256i lo_bytes = _mm256_and_si256(raw_q, lo_mask);
                __m256i hi_bytes = _mm256_and_si256(_mm256_srli_epi16(raw_q, 4), lo_mask);

                const float* x_lo = xp + group * 64;
                const float* x_hi = xp + group * 64 + 32;

                // Process lo nibbles: 4 chunks of 8
                for (int c = 0; c < 4; ++c) {
                    __m128i src;
                    if (c < 2) src = _mm256_castsi256_si128(lo_bytes);
                    else       src = _mm256_extracti128_si256(lo_bytes, 1);
                    if (c & 1) src = _mm_srli_si128(src, 8);

                    __m256 wf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(src));
#ifdef __FMA__
                    wf = _mm256_fmsub_ps(wf, vd1, vm1);
                    row_acc = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x_lo + c * 8), row_acc);
#else
                    wf = _mm256_sub_ps(_mm256_mul_ps(wf, vd1), vm1);
                    row_acc = _mm256_add_ps(row_acc, _mm256_mul_ps(wf, _mm256_loadu_ps(x_lo + c * 8)));
#endif
                }
                // Process hi nibbles: 4 chunks of 8
                for (int c = 0; c < 4; ++c) {
                    __m128i src;
                    if (c < 2) src = _mm256_castsi256_si128(hi_bytes);
                    else       src = _mm256_extracti128_si256(hi_bytes, 1);
                    if (c & 1) src = _mm_srli_si128(src, 8);

                    __m256 wf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(src));
#ifdef __FMA__
                    wf = _mm256_fmsub_ps(wf, vd2, vm2);
                    row_acc = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x_hi + c * 8), row_acc);
#else
                    wf = _mm256_sub_ps(_mm256_mul_ps(wf, vd2), vm2);
                    row_acc = _mm256_add_ps(row_acc, _mm256_mul_ps(wf, _mm256_loadu_ps(x_hi + c * 8)));
#endif
                }
            }
        }
        // Horizontal sum
        __m128 hi = _mm256_extractf128_ps(row_acc, 1);
        __m128 lo = _mm256_castps256_ps128(row_acc);
        __m128 sum4 = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(sum4);
        sum4 = _mm_add_ps(sum4, shuf);
        shuf = _mm_movehl_ps(sum4, sum4);
        sum4 = _mm_add_ss(sum4, shuf);
        y[row] = _mm_cvtss_f32(sum4);
#else
        // Scalar fallback
        float dot = 0.0f;
        for (int b = 0; b < nblocks; ++b) {
            float buf[256];
            dequant_q4k_block(row_raw + b * BLOCK_SIZE, buf);
            const float* xp = x + b * 256;
            for (int i = 0; i < 256; ++i) dot += buf[i] * xp[i];
        }
        y[row] = dot;
#endif
    }
    });
}

inline void q6k_matmul_vec(
    const uint8_t* raw_weights,
    const float* x,
    float* y,
    int out_features,
    int bytes_per_row
) {
    constexpr int BLOCK_SIZE = 210;
    int nblocks = bytes_per_row / BLOCK_SIZE;

    size_t total_work = static_cast<size_t>(out_features) * bytes_per_row;
    parallel_for(out_features, total_work, [&](int r0, int r1) {
    for (int row = r0; row < r1; ++row) {
        const uint8_t* row_raw = raw_weights + static_cast<size_t>(row) * bytes_per_row;

#ifdef __AVX2__
        __m256 acc = _mm256_setzero_ps();
        for (int b = 0; b < nblocks; ++b) {
            float buf[256];
            dequant_q6k_block(row_raw + b * BLOCK_SIZE, buf);
            const float* xp = x + b * 256;
            for (int i = 0; i < 256; i += 8) {
                __m256 wv = _mm256_loadu_ps(buf + i);
                __m256 xv = _mm256_loadu_ps(xp + i);
#ifdef __FMA__
                acc = _mm256_fmadd_ps(wv, xv, acc);
#else
                acc = _mm256_add_ps(acc, _mm256_mul_ps(wv, xv));
#endif
            }
        }
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 sum4 = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(sum4);
        sum4 = _mm_add_ps(sum4, shuf);
        shuf = _mm_movehl_ps(sum4, sum4);
        sum4 = _mm_add_ss(sum4, shuf);
        y[row] = _mm_cvtss_f32(sum4);
#else
        float dot = 0.0f;
        for (int b = 0; b < nblocks; ++b) {
            float buf[256];
            dequant_q6k_block(row_raw + b * BLOCK_SIZE, buf);
            const float* xp = x + b * 256;
            for (int i = 0; i < 256; ++i) dot += buf[i] * xp[i];
        }
        y[row] = dot;
#endif
    }
    });
}


// ===================================================================
// RMS Normalization:  out = (x / rms(x)) * weight
// ===================================================================

inline void rms_norm(
    const float* x,
    const float* weight,
    float* out,
    int size,
    float eps = 1e-6f
) {
    float sum_sq = 0.0f;
#ifdef __AVX2__
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < size; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
#ifdef __FMA__
        acc = _mm256_fmadd_ps(v, v, acc);
#else
        acc = _mm256_add_ps(acc, _mm256_mul_ps(v, v));
#endif
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    for (int j = 0; j < 8; ++j) sum_sq += tmp[j];
    for (; i < size; ++i) sum_sq += x[i] * x[i];
#else
    for (int i = 0; i < size; ++i) sum_sq += x[i] * x[i];
#endif

    float rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(size) + eps);

#ifdef __AVX2__
    __m256 rms_v = _mm256_set1_ps(rms);
    i = 0;
    for (; i + 7 < size; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256 wv = _mm256_loadu_ps(weight + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(xv, rms_v), wv));
    }
    for (; i < size; ++i) {
        out[i] = x[i] * rms * weight[i];
    }
#else
    for (int i = 0; i < size; ++i) {
        out[i] = x[i] * rms * weight[i];
    }
#endif
}


// ===================================================================
// Softmax:  out[i] = exp(x[i] - max) / sum(exp(x - max))
// ===================================================================

inline void softmax(const float* x, float* out, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        out[i] = std::exp(x[i] - max_val);
        sum += out[i];
    }

    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; ++i) {
        out[i] *= inv_sum;
    }
}


// ===================================================================
// SiLU activation:  out = x * sigmoid(x) = x / (1 + exp(-x))
// ===================================================================

inline void silu(const float* x, float* out, int size) {
    for (int i = 0; i < size; ++i) {
        float clamped = std::max(-88.0f, std::min(88.0f, x[i]));
        out[i] = clamped / (1.0f + std::exp(-clamped));
    }
}

// ===================================================================
// SiLU * elementwise multiply (fused for SwiGLU FFN)
// out = silu(gate) * up
// ===================================================================

inline void silu_mul(const float* gate, const float* up, float* out, int size) {
    for (int i = 0; i < size; ++i) {
        float g = std::max(-88.0f, std::min(88.0f, gate[i]));
        out[i] = (g / (1.0f + std::exp(-g))) * up[i];
    }
}


// ===================================================================
// RoPE:  rotary positional embedding (INTERLEAVED)
// ===================================================================

inline void apply_rope(
    const float* x,       // (num_heads, head_dim)
    const float* cos_f,   // (head_dim/2,) — cos at this position
    const float* sin_f,   // (head_dim/2,) — sin at this position
    float* out,           // (num_heads, head_dim)
    int num_heads,
    int head_dim
) {
    int d2 = head_dim / 2;
    for (int h = 0; h < num_heads; ++h) {
        const float* xh = x + h * head_dim;
        float* oh = out + h * head_dim;
        // Interleaved RoPE: pair (dim[2i], dim[2i+1]) with freq[i]
        for (int i = 0; i < d2; ++i) {
            float x_even = xh[2*i];
            float x_odd = xh[2*i + 1];
            oh[2*i]     = x_even * cos_f[i] - x_odd * sin_f[i];
            oh[2*i + 1] = x_even * sin_f[i] + x_odd * cos_f[i];
        }
    }
}


// ===================================================================
// Grouped Query Attention (GQA)
// ===================================================================

inline void gqa_attention(
    const float* q,           // (num_heads, head_dim)
    const float* k_cache,     // (max_seq, num_kv_heads, head_dim)
    const float* v_cache,     // (max_seq, num_kv_heads, head_dim)
    float* attn_out,          // (num_heads, head_dim)
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len               // how many positions are filled in cache
) {
    int heads_per_group = num_heads / num_kv_heads;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Temporary buffers for scores and softmax
    std::vector<float> scores(seq_len);
    std::vector<float> weights(seq_len);

    for (int kv_idx = 0; kv_idx < num_kv_heads; ++kv_idx) {
        for (int h_off = 0; h_off < heads_per_group; ++h_off) {
            int h_idx = kv_idx * heads_per_group + h_off;
            const float* q_head = q + h_idx * head_dim;

            // Compute attention scores: q @ K^T / sqrt(d)
            for (int s = 0; s < seq_len; ++s) {
                const float* k_vec = k_cache + (static_cast<size_t>(s) * num_kv_heads + kv_idx) * head_dim;
                float dot = 0.0f;
#ifdef __AVX2__
                __m256 acc = _mm256_setzero_ps();
                int i = 0;
                for (; i + 7 < head_dim; i += 8) {
                    __m256 qv = _mm256_loadu_ps(q_head + i);
                    __m256 kv = _mm256_loadu_ps(k_vec + i);
#ifdef __FMA__
                    acc = _mm256_fmadd_ps(qv, kv, acc);
#else
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(qv, kv));
#endif
                }
                float tmp[8];
                _mm256_storeu_ps(tmp, acc);
                for (int j = 0; j < 8; ++j) dot += tmp[j];
                for (; i < head_dim; ++i) dot += q_head[i] * k_vec[i];
#else
                for (int i = 0; i < head_dim; ++i) {
                    dot += q_head[i] * k_vec[i];
                }
#endif
                scores[s] = dot * scale;
            }

            // Softmax
            softmax(scores.data(), weights.data(), seq_len);

            // Weighted sum of values
            float* out_head = attn_out + h_idx * head_dim;
            std::fill(out_head, out_head + head_dim, 0.0f);

            for (int s = 0; s < seq_len; ++s) {
                const float* v_vec = v_cache + (static_cast<size_t>(s) * num_kv_heads + kv_idx) * head_dim;
                float w = weights[s];
#ifdef __AVX2__
                __m256 wv = _mm256_set1_ps(w);
                int i = 0;
                for (; i + 7 < head_dim; i += 8) {
                    __m256 ov = _mm256_loadu_ps(out_head + i);
                    __m256 vv = _mm256_loadu_ps(v_vec + i);
#ifdef __FMA__
                    _mm256_storeu_ps(out_head + i, _mm256_fmadd_ps(wv, vv, ov));
#else
                    _mm256_storeu_ps(out_head + i, _mm256_add_ps(ov, _mm256_mul_ps(wv, vv)));
#endif
                }
                for (; i < head_dim; ++i) {
                    out_head[i] += w * v_vec[i];
                }
#else
                for (int i = 0; i < head_dim; ++i) {
                    out_head[i] += w * v_vec[i];
                }
#endif
            }
        }
    }
}


// ===================================================================
// Hybrid SSM selective scan (depthwise conv + recurrent update)
// ===================================================================

inline float softplus_stable(float x) {
    float ax = std::fabs(x);
    return std::log1p(std::exp(-ax)) + std::max(x, 0.0f);
}

inline void ssm_selective_scan_step(
    const float* x,            // (hidden_size,)
    int hidden_size,
    const float* conv_kernel,  // (conv_dim, conv_k)
    int conv_dim,
    int conv_k,
    const float* ssm_a,        // (s_dim,)
    const float* ssm_dt_bias,  // (s_dim,)
    const float* dt_proj,      // (s_dim,)
    const float* b_proj,       // (s_dim,)
    const float* ssm_norm,     // (norm_dim,) or nullptr
    int norm_dim,
    float* ssm_state,          // (s_dim,) in/out
    float* conv_state,         // (conv_dim, conv_k) in/out
    int s_dim,
    int q_dim,
    float* y_out               // (q_dim,)
) {
    // 1) Conv state update using repeated input (np.resize semantics).
    for (int c = 0; c < conv_dim; ++c) {
        float* row = conv_state + static_cast<size_t>(c) * conv_k;
        if (conv_k > 1) {
            std::memmove(row, row + 1, static_cast<size_t>(conv_k - 1) * sizeof(float));
        }
        row[conv_k - 1] = x[c % hidden_size];
    }

    // Depthwise conv output.
    std::vector<float> conv_out(conv_dim, 0.0f);
    for (int c = 0; c < conv_dim; ++c) {
        const float* row_s = conv_state + static_cast<size_t>(c) * conv_k;
        const float* row_k = conv_kernel + static_cast<size_t>(c) * conv_k;
        float acc = 0.0f;
        for (int t = 0; t < conv_k; ++t) acc += row_s[t] * row_k[t];
        conv_out[c] = acc;
    }

    // 2) Projections to delta/B/C lanes.
    std::vector<float> delta(s_dim), B(s_dim), C(s_dim);
    for (int i = 0; i < s_dim; ++i) {
        float dt_in = dt_proj[i] + ssm_dt_bias[i];
        delta[i] = softplus_stable(dt_in);
        B[i] = std::tanh(b_proj[i]);
        C[i] = std::tanh(dt_proj[i]);
    }

    // Pool conv lanes into latent drive x_latent.
    std::vector<float> x_latent(s_dim, 0.0f);
    if (conv_dim >= s_dim && (conv_dim % s_dim) == 0) {
        int lanes_per = conv_dim / s_dim;
        for (int i = 0; i < s_dim; ++i) {
            float acc = 0.0f;
            int base = i * lanes_per;
            for (int j = 0; j < lanes_per; ++j) acc += conv_out[base + j];
            x_latent[i] = acc / static_cast<float>(lanes_per);
        }
    } else {
        float mean = 0.0f;
        for (float v : conv_out) mean += v;
        mean /= std::max(1, conv_dim);
        std::fill(x_latent.begin(), x_latent.end(), mean);
    }

    // 3/4) Discretize and selective scan recurrence.
    for (int i = 0; i < s_dim; ++i) {
        float a = std::clamp(delta[i] * ssm_a[i], -20.0f, 20.0f);
        float A_bar = std::exp(a);
        float B_bar = delta[i] * B[i];
        ssm_state[i] = A_bar * ssm_state[i] + B_bar * x_latent[i];
    }

    // 5) Readout y = s * C then expand to q_dim and apply optional norm.
    std::vector<float> y_latent(s_dim, 0.0f);
    for (int i = 0; i < s_dim; ++i) y_latent[i] = ssm_state[i] * C[i];

    int repeat = std::max(1, q_dim / std::max(1, s_dim));
    int pos = 0;
    for (int i = 0; i < s_dim && pos < q_dim; ++i) {
        for (int r = 0; r < repeat && pos < q_dim; ++r) {
            y_out[pos++] = y_latent[i];
        }
    }
    while (pos < q_dim) y_out[pos++] = 0.0f;

    if (ssm_norm != nullptr && norm_dim > 0) {
        for (int i = 0; i < q_dim; ++i) {
            y_out[i] *= ssm_norm[i % norm_dim];
        }
    }
}


// ===================================================================
// GatedDeltaNet recurrent step (Qwen3.5 linear attention)
// ===================================================================
// Implements the Qwen3.5 GatedDeltaNet recurrent step for one token.
// The state is persistent across tokens and lives in native memory.
//
// Forward pass:
//   1) Causal conv1d: shift state, insert qkv, depthwise conv + SiLU
//   2) Split conv output into Q, K, V; expand Q/K heads to V heads
//   3) L2 normalize Q and K; scale Q by 1/sqrt(head_k_dim)
//   5) Compute decay = exp(ssm_a * clamp(softplus(a + dt_bias)))
//   6) Delta rule: delta=(V - S.T@K)*sigmoid(beta);
//      S=decay*S+outer(K, delta); y=S.T@Q
//   7) RMSNorm with ssm_norm, then multiply by SiLU(attn_gate)
// ===================================================================

inline void deltanet_step(
    const float* qkv,          // (conv_dim,) = key_dim*2+value_dim, Q/K/V projection
    const float* z,            // (value_dim,) output gate from attn_gate projection
    const float* a,            // (n_v_heads,) decay driver from ssm_alpha projection
    const float* b,            // (n_v_heads,) write gate from ssm_beta projection
    const float* conv_kernel,  // (conv_dim, conv_k) depthwise conv weights
    const float* a_log,        // (n_v_heads,) log of decay base (ssm_a)
    const float* dt_bias,      // (n_v_heads,) time step bias
    const float* norm_weight,  // (head_v_dim,) RMS norm weight, or nullptr
    float norm_eps,
    float* conv_state,         // (conv_dim, conv_k) in/out
    float* recurrent_state,    // (n_v_heads * head_k_dim * head_v_dim) in/out
    int n_k_heads,             // default 16
    int n_v_heads,             // 32
    int head_k_dim,            // 128
    int head_v_dim,            // 128
    int conv_k,                // 4
    float* y_out               // (value_dim = n_v_heads * head_v_dim,) output
) {
    int key_dim = n_k_heads * head_k_dim;
    int value_dim = n_v_heads * head_v_dim;
    int conv_dim = key_dim * 2 + value_dim;
    int expand = n_v_heads / n_k_heads;

    // 1) Causal conv1d step on fused Q/K/V projection
    for (int c = 0; c < conv_dim; ++c) {
        float* row = conv_state + static_cast<size_t>(c) * conv_k;
        if (conv_k > 1) {
            std::memmove(row, row + 1, static_cast<size_t>(conv_k - 1) * sizeof(float));
        }
        row[conv_k - 1] = qkv[c];
    }

    // Depthwise conv + SiLU activation
    std::vector<float> conv_out(conv_dim);
    for (int c = 0; c < conv_dim; ++c) {
        const float* row_s = conv_state + static_cast<size_t>(c) * conv_k;
        const float* row_k = conv_kernel + static_cast<size_t>(c) * conv_k;
        float acc = 0.0f;
        for (int t = 0; t < conv_k; ++t) acc += row_s[t] * row_k[t];
        // SiLU activation
        float clamped = std::max(-88.0f, std::min(88.0f, acc));
        conv_out[c] = clamped / (1.0f + std::exp(-clamped));
    }

    // 2) Split Q/K/V and expand Q/K heads from n_k_heads to n_v_heads.
    // GGUF Qwen3.5 reorders V/Z/A/B heads to tiled order:
    //   [K0v0, K1v0, ..., K15v0, K0v1, K1v1, ...]
    // Q/K stay in key-head order, so repeat them into the same tiled order.
    std::vector<float> q(static_cast<size_t>(n_v_heads) * head_k_dim);
    std::vector<float> k(static_cast<size_t>(n_v_heads) * head_k_dim);
    for (int kh = 0; kh < n_k_heads; ++kh) {
        for (int e = 0; e < expand; ++e) {
            int vh = e * n_k_heads + kh;
            std::memcpy(q.data() + static_cast<size_t>(vh) * head_k_dim,
                        conv_out.data() + static_cast<size_t>(kh) * head_k_dim,
                        static_cast<size_t>(head_k_dim) * sizeof(float));
            std::memcpy(k.data() + static_cast<size_t>(vh) * head_k_dim,
                        conv_out.data() + key_dim + static_cast<size_t>(kh) * head_k_dim,
                        static_cast<size_t>(head_k_dim) * sizeof(float));
        }
    }
    const float* v_ptr = conv_out.data() + 2 * key_dim;

    // 3) L2 normalize Q/K per head and scale Q like linear attention.
    float q_scale = 1.0f / std::sqrt(static_cast<float>(head_k_dim));
    for (int h = 0; h < n_v_heads; ++h) {
        // Q L2 norm
        float* qh = q.data() + h * head_k_dim;
        float sum_sq = 0.0f;
        for (int i = 0; i < head_k_dim; ++i) sum_sq += qh[i] * qh[i];
        float inv_norm = 1.0f / std::sqrt(sum_sq + 1e-6f);
        for (int i = 0; i < head_k_dim; ++i) qh[i] *= inv_norm * q_scale;

        // K L2 norm
        float* kh = k.data() + h * head_k_dim;
        sum_sq = 0.0f;
        for (int i = 0; i < head_k_dim; ++i) sum_sq += kh[i] * kh[i];
        inv_norm = 1.0f / std::sqrt(sum_sq + 1e-6f);
        for (int i = 0; i < head_k_dim; ++i) kh[i] *= inv_norm;
    }

    // 4) Compute decay and beta per head.
    // ssm_a is already converted by GGUF as a negative decay value.
    // dt = softplus(a + dt_bias)  -- positive time step
    // decay = exp(-A * dt)  -- in (0, 1)
    // beta = sigmoid(b)
    std::vector<float> decay(n_v_heads), beta(n_v_heads);
    for (int h = 0; h < n_v_heads; ++h) {
        float A = std::min(0.0f, a_log[h]);
        float sp_in = a[h] + dt_bias[h];
        float dt = softplus_stable(sp_in);
        decay[h] = std::exp(std::max(-20.0f, A * dt));
        float bh = std::max(-88.0f, std::min(88.0f, b[h]));
        beta[h] = 1.0f / (1.0f + std::exp(-bh));
    }

    // 5) Gated delta rule recurrence.
    size_t state_stride = static_cast<size_t>(head_k_dim) * head_v_dim;

    for (int h = 0; h < n_v_heads; ++h) {
        float* S = recurrent_state + h * state_stride;  // (head_k_dim, head_v_dim)
        const float* qh = q.data() + h * head_k_dim;
        const float* kh = k.data() + h * head_k_dim;
        const float* vh = v_ptr + h * head_v_dim;
        float* oh = y_out + h * head_v_dim;

        float d = decay[h];
        for (size_t i = 0; i < state_stride; ++i) S[i] *= d;

        std::vector<float> kv_mem(head_v_dim, 0.0f);
        for (int j = 0; j < head_k_dim; ++j) {
            float kj = kh[j];
            const float* Sj = S + j * head_v_dim;
            for (int i = 0; i < head_v_dim; ++i) {
                kv_mem[i] += Sj[i] * kj;
            }
        }

        std::vector<float> delta(head_v_dim);
        float bh = beta[h];
        for (int i = 0; i < head_v_dim; ++i) {
            delta[i] = (vh[i] - kv_mem[i]) * bh;
        }

        // S += outer(k, delta)
        for (int j = 0; j < head_k_dim; ++j) {
            float kj = kh[j];
            float* Sj = S + j * head_v_dim;
            for (int i = 0; i < head_v_dim; ++i) {
                Sj[i] += kj * delta[i];
            }
        }

        // Readout: y[h] = Q[h] @ S[h] → (head_v_dim,)
        std::fill(oh, oh + head_v_dim, 0.0f);
        for (int j = 0; j < head_k_dim; ++j) {
            float qj = qh[j];
            const float* Sj = S + j * head_v_dim;
            for (int i = 0; i < head_v_dim; ++i) {
                oh[i] += Sj[i] * qj;
            }
        }
    }

    // 6) Gated RMSNorm: RMSNorm(output, norm_weight) * SiLU(z),
    // matching Qwen3.5's linear_attn.norm path.
    for (int h = 0; h < n_v_heads; ++h) {
        float* oh = y_out + h * head_v_dim;
        const float* zh = z + h * head_v_dim;

        float sum_sq = 0.0f;
        for (int i = 0; i < head_v_dim; ++i) sum_sq += oh[i] * oh[i];
        float rms_inv = 1.0f / std::sqrt(sum_sq / static_cast<float>(head_v_dim) + norm_eps);

        for (int i = 0; i < head_v_dim; ++i) {
            float scale = rms_inv;
            if (norm_weight != nullptr) scale *= norm_weight[i];
            float zv = std::max(-88.0f, std::min(88.0f, zh[i]));
            float z_act = zv / (1.0f + std::exp(-zv));
            oh[i] *= scale * z_act;
        }
    }
}


// ===================================================================
// Full transformer layer forward pass (single token)
// ===================================================================
// This fuses: rms_norm → qkv_proj → rope → kv_cache → gqa → o_proj → residual → rms_norm → ffn
// All in C++ with no Python roundtrips.

struct TransformerLayerWeights {
    const uint8_t* q_proj;     // raw GGUF weights
    const uint8_t* k_proj;
    const uint8_t* v_proj;
    const uint8_t* o_proj;
    const uint8_t* gate_proj;
    const uint8_t* up_proj;
    const uint8_t* down_proj;
    const float* attn_norm;   // RMS norm weights
    const float* ffn_norm;
    const float* k_bias;      // optional, nullptr if not present
    const float* v_bias;
    int qkv_bytes_per_row;     // Q projection bytes per row
    int k_bytes_per_row;       // K projection
    int v_bytes_per_row;       // V projection
    int o_bytes_per_row;
    int gate_bytes_per_row;
    int up_bytes_per_row;
    int down_bytes_per_row;
    int q_out_features;        // num_heads * head_dim
    int k_out_features;        // num_kv_heads * head_dim
    int v_out_features;
    int o_out_features;        // hidden_size
    int gate_out_features;     // intermediate_size
    int up_out_features;
    int down_out_features;
    const char* q_quant_type;  // "q4k" or "q6k"
    const char* k_quant_type;
    const char* v_quant_type;
    const char* o_quant_type;
    const char* gate_quant_type;
    const char* up_quant_type;
    const char* down_quant_type;
};

// Generic matmul dispatcher based on quant type
inline void quant_matmul_vec(
    const uint8_t* weights,
    const float* x,
    float* y,
    int out_features,
    int bytes_per_row,
    const char* quant_type
) {
    if (std::strcmp(quant_type, "q4k") == 0) {
        q4k_matmul_vec(weights, x, y, out_features, bytes_per_row);
    } else {
        q6k_matmul_vec(weights, x, y, out_features, bytes_per_row);
    }
}


} // namespace inference
} // namespace rftmw
