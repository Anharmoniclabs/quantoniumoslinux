/*
 * SPDX-License-Identifier: LicenseRef-SPX-Proprietary
 * Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
 *
 * rftmw_spectral_kernels.c - RFTMW Spectral-Domain Matmul v2
 * ============================================================
 *
 * Implements:
 *   1. Split-format radix-2 DIT FFT (float32, AVX2-accelerated)
 *   2. Hybrid phi-RFT forward transform (FFT + golden-ratio phase diagonal)
 *   3. Phase LUT for zero-trig dequantization in hot loop
 *   4. Quantized sparse dot product (4-bit mag + 8-bit phase)
 *   5. Full spectral GEMV with input caching
 *   6. Dense-to-quantized-spectral weight converter
 *   7. Non-uniform sparsity schedule
 */

#include "rftmw_spectral_kernels.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __AVX2__
#include <immintrin.h>
#define SPECTRAL_HAS_AVX2 1
#else
#define SPECTRAL_HAS_AVX2 0
#endif

#ifdef __FMA__
#define SPECTRAL_HAS_FMA 1
#else
#define SPECTRAL_HAS_FMA 0
#endif

/* ================================================================
 * Aligned allocation helpers
 * ================================================================ */

static void* aligned_alloc_f(size_t count) {
    void* ptr = NULL;
    size_t bytes = count * sizeof(float);
    if (bytes == 0) bytes = sizeof(float);
#if defined(_WIN32)
    ptr = _aligned_malloc(bytes, 32);
#else
    if (posix_memalign(&ptr, 32, bytes) != 0) ptr = NULL;
#endif
    if (ptr) memset(ptr, 0, bytes);
    return ptr;
}

static void aligned_free(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/* ================================================================
 * Phase LUT Initialization
 *
 * Maps uint8 phase level p -> angle = (p / 255.0) * 2*pi - pi
 * Range: [-pi, +pi]
 * ================================================================ */

void rftmw_phase_lut_init(rftmw_phase_lut_t* lut) {
    for (int p = 0; p < RFTMW_PHASE_LUT_SIZE; ++p) {
        double angle = ((double)p / 255.0) * RFTMW_TWO_PI - 3.14159265358979323846;
        lut->cos_lut[p] = (float)cos(angle);
        lut->sin_lut[p] = (float)sin(angle);
    }
}

/* ================================================================
 * Bit-reversal permutation for FFT
 * ================================================================ */

static int bit_reverse(int x, int log2_n) {
    int result = 0;
    for (int i = 0; i < log2_n; ++i) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

static void bit_reverse_permute_f32(float* re, float* im, int n, int log2_n) {
    for (int i = 0; i < n; ++i) {
        int j = bit_reverse(i, log2_n);
        if (j > i) {
            float tmp;
            tmp = re[i]; re[i] = re[j]; re[j] = tmp;
            tmp = im[i]; im[i] = im[j]; im[j] = tmp;
        }
    }
}

/* ================================================================
 * Split-format Radix-2 DIT FFT (float32)
 * ================================================================ */

void rftmw_fft_f32(float* re, float* im, int n) {
    if (n <= 1) return;

    int log2_n = 0;
    { int tmp = n; while (tmp > 1) { log2_n++; tmp >>= 1; } }

    bit_reverse_permute_f32(re, im, n, log2_n);

    for (int stage = 1; stage <= log2_n; ++stage) {
        int m = 1 << stage;
        int half = m >> 1;
        float w_re_step = cosf((float)(-RFTMW_TWO_PI / (double)m));
        float w_im_step = sinf((float)(-RFTMW_TWO_PI / (double)m));

#if SPECTRAL_HAS_AVX2 && SPECTRAL_HAS_FMA
        if (half >= 8) {
            for (int k = 0; k < n; k += m) {
                float tw_re[8] __attribute__((aligned(32)));
                float tw_im[8] __attribute__((aligned(32)));

                for (int g = 0; g < half; g += 8) {
                    int count = (g + 8 <= half) ? 8 : (half - g);

                    float wr = 1.0f, wi = 0.0f;
                    for (int i = 0; i < g; ++i) {
                        float t = wr * w_re_step - wi * w_im_step;
                        wi = wr * w_im_step + wi * w_re_step;
                        wr = t;
                    }
                    for (int i = 0; i < count; ++i) {
                        tw_re[i] = wr;
                        tw_im[i] = wi;
                        float t = wr * w_re_step - wi * w_im_step;
                        wi = wr * w_im_step + wi * w_re_step;
                        wr = t;
                    }
                    for (int i = count; i < 8; ++i) {
                        tw_re[i] = 0.0f;
                        tw_im[i] = 0.0f;
                    }

                    if (count == 8) {
                        __m256 v_tw_re = _mm256_load_ps(tw_re);
                        __m256 v_tw_im = _mm256_load_ps(tw_im);
                        __m256 v_p_re = _mm256_loadu_ps(&re[k + g]);
                        __m256 v_p_im = _mm256_loadu_ps(&im[k + g]);
                        __m256 v_q_re = _mm256_loadu_ps(&re[k + g + half]);
                        __m256 v_q_im = _mm256_loadu_ps(&im[k + g + half]);

                        __m256 t_re = _mm256_mul_ps(v_tw_re, v_q_re);
                        t_re = _mm256_fnmadd_ps(v_tw_im, v_q_im, t_re);
                        __m256 t_im = _mm256_mul_ps(v_tw_re, v_q_im);
                        t_im = _mm256_fmadd_ps(v_tw_im, v_q_re, t_im);

                        _mm256_storeu_ps(&re[k + g],        _mm256_add_ps(v_p_re, t_re));
                        _mm256_storeu_ps(&im[k + g],        _mm256_add_ps(v_p_im, t_im));
                        _mm256_storeu_ps(&re[k + g + half], _mm256_sub_ps(v_p_re, t_re));
                        _mm256_storeu_ps(&im[k + g + half], _mm256_sub_ps(v_p_im, t_im));
                    } else {
                        for (int i = 0; i < count; ++i) {
                            int p_idx = k + g + i;
                            int q_idx = p_idx + half;
                            float t_re_s = tw_re[i] * re[q_idx] - tw_im[i] * im[q_idx];
                            float t_im_s = tw_re[i] * im[q_idx] + tw_im[i] * re[q_idx];
                            re[q_idx] = re[p_idx] - t_re_s;
                            im[q_idx] = im[p_idx] - t_im_s;
                            re[p_idx] += t_re_s;
                            im[p_idx] += t_im_s;
                        }
                    }
                }
            }
        } else
#endif
        {
            for (int k = 0; k < n; k += m) {
                float wr = 1.0f, wi = 0.0f;
                for (int j = 0; j < half; ++j) {
                    int p = k + j;
                    int q = p + half;
                    float t_re_s = wr * re[q] - wi * im[q];
                    float t_im_s = wr * im[q] + wi * re[q];
                    re[q] = re[p] - t_re_s;
                    im[q] = im[p] - t_im_s;
                    re[p] += t_re_s;
                    im[p] += t_im_s;
                    float tmp = wr * w_re_step - wi * w_im_step;
                    wi = wr * w_im_step + wi * w_re_step;
                    wr = tmp;
                }
            }
        }
    }
}

void rftmw_ifft_f32(float* re, float* im, int n) {
    for (int i = 0; i < n; ++i) im[i] = -im[i];
    rftmw_fft_f32(re, im, n);
    float inv_n = 1.0f / (float)n;
    for (int i = 0; i < n; ++i) {
        re[i] *= inv_n;
        im[i] = -im[i] * inv_n;
    }
}

/* ================================================================
 * Spectral Basis Initialization
 * ================================================================ */

void rftmw_spectral_basis_init(rftmw_spectral_basis_t* basis, int block_size) {
    memset(basis, 0, sizeof(*basis));
    basis->block_size = block_size;
    basis->inv_sqrt_n = 1.0f / sqrtf((float)block_size);

    int log2_n = 0;
    { int tmp = block_size; while (tmp > 1) { log2_n++; tmp >>= 1; } }
    basis->log2_n = log2_n;

    basis->cos_table = (float*)aligned_alloc_f((size_t)block_size);
    basis->sin_table = (float*)aligned_alloc_f((size_t)block_size);

    for (int k = 0; k < block_size; ++k) {
        double frac_val = (double)(k + 1) * RFTMW_PHI;
        frac_val = frac_val - floor(frac_val);
        double theta = RFTMW_TWO_PI * frac_val;
        basis->cos_table[k] = (float)cos(theta);
        basis->sin_table[k] = (float)sin(theta);
    }

    basis->twiddle_re = NULL;
    basis->twiddle_im = NULL;
}

void rftmw_spectral_basis_free(rftmw_spectral_basis_t* basis) {
    if (basis->cos_table) { aligned_free(basis->cos_table); basis->cos_table = NULL; }
    if (basis->sin_table) { aligned_free(basis->sin_table); basis->sin_table = NULL; }
    if (basis->twiddle_re) { aligned_free(basis->twiddle_re); basis->twiddle_re = NULL; }
    if (basis->twiddle_im) { aligned_free(basis->twiddle_im); basis->twiddle_im = NULL; }
}

/* ================================================================
 * Forward Hybrid RFT Transform
 * out[k] = E[k] * FFT(x_real)[k] / sqrt(N)
 * ================================================================ */

void rftmw_spectral_forward_block(
    const rftmw_spectral_basis_t* basis,
    const float* x_real,
    float* out_re,
    float* out_im
) {
    int N = basis->block_size;

    memcpy(out_re, x_real, (size_t)N * sizeof(float));
    memset(out_im, 0, (size_t)N * sizeof(float));

    rftmw_fft_f32(out_re, out_im, N);

    float inv_sqrt = basis->inv_sqrt_n;

#if SPECTRAL_HAS_AVX2 && SPECTRAL_HAS_FMA
    __m256 v_inv = _mm256_set1_ps(inv_sqrt);
    int k = 0;
    for (; k + 7 < N; k += 8) {
        __m256 e_re = _mm256_loadu_ps(&basis->cos_table[k]);
        __m256 e_im = _mm256_loadu_ps(&basis->sin_table[k]);
        __m256 f_re = _mm256_loadu_ps(&out_re[k]);
        __m256 f_im = _mm256_loadu_ps(&out_im[k]);

        __m256 r = _mm256_mul_ps(e_re, f_re);
        r = _mm256_fnmadd_ps(e_im, f_im, r);
        r = _mm256_mul_ps(r, v_inv);

        __m256 i = _mm256_mul_ps(e_re, f_im);
        i = _mm256_fmadd_ps(e_im, f_re, i);
        i = _mm256_mul_ps(i, v_inv);

        _mm256_storeu_ps(&out_re[k], r);
        _mm256_storeu_ps(&out_im[k], i);
    }
    for (; k < N; ++k) {
#else
    for (int k = 0; k < N; ++k) {
#endif
        float e_r = basis->cos_table[k];
        float e_i = basis->sin_table[k];
        float f_r = out_re[k];
        float f_i = out_im[k];
        out_re[k] = (e_r * f_r - e_i * f_i) * inv_sqrt;
        out_im[k] = (e_r * f_i + e_i * f_r) * inv_sqrt;
#if SPECTRAL_HAS_AVX2 && SPECTRAL_HAS_FMA
    }
#else
    }
#endif
}

/* ================================================================
 * Quantized Sparse Spectral Dot Product (Hot Inner Loop)
 *
 * Dequantizes on-the-fly using phase LUT:
 *   nibble = (i%2==0) ? mag4[i/2] & 0x0F : mag4[i/2] >> 4
 *   mag   = (nibble / 15.0) * peak_mag
 *   re    = mag * cos_lut[phase8[i]]
 *   im    = mag * sin_lut[phase8[i]]
 *   dot  += re * x_spec_re[idx] + im * x_spec_im[idx]
 * ================================================================ */

float rftmw_qspectral_sparse_dot(
    const uint16_t* indices,
    const uint8_t*  mag4,
    const uint8_t*  phase8,
    float           peak_mag,
    const float*    cos_lut,
    const float*    sin_lut,
    const float*    x_spec_re,
    const float*    x_spec_im,
    int             n_nonzero
) {
    float result = 0.0f;
    float inv15 = peak_mag / 15.0f;

    for (int i = 0; i < n_nonzero; ++i) {
        /* Extract 4-bit magnitude nibble */
        uint8_t byte = mag4[i >> 1];
        uint8_t nibble = (i & 1) ? (byte >> 4) : (byte & 0x0F);

        /* Dequantize magnitude */
        float mag = (float)nibble * inv15;

        /* Dequantize phase via LUT (zero trig!) */
        uint8_t plevel = phase8[i];
        float re = mag * cos_lut[plevel];
        float im = mag * sin_lut[plevel];

        /* Accumulate sparse dot */
        int idx = (int)indices[i];
        result += re * x_spec_re[idx] + im * x_spec_im[idx];
    }
    return result;
}

/* ================================================================
 * Full Spectral GEMV: y = SpectralW @ x
 *
 * 1. Transform each block of x into spectral domain (cached)
 * 2. For each output row, compute quantized sparse dot over blocks
 * ================================================================ */

void rftmw_spectral_gemv(
    rftmw_spectral_weight_t* W,
    const float* x,
    float*       y
) {
    int block_size = W->block_size;
    int n_blocks   = W->n_blocks;
    int in_feat    = W->in_features;
    int out_feat   = W->out_features;

    /* Step 1: Transform all blocks of x into spectral domain */
    for (int b = 0; b < n_blocks; ++b) {
        int start = b * block_size;
        int end = start + block_size;

        if (end > in_feat) {
            float* padded = W->scratch_re;
            int actual = in_feat - start;
            if (actual > 0) {
                memcpy(padded, &x[start], (size_t)actual * sizeof(float));
            }
            memset(&padded[actual > 0 ? actual : 0], 0,
                   (size_t)(block_size - (actual > 0 ? actual : 0)) * sizeof(float));

            rftmw_spectral_forward_block(
                &W->basis, padded,
                &W->x_spec_re[b * block_size],
                &W->x_spec_im[b * block_size]
            );
        } else {
            rftmw_spectral_forward_block(
                &W->basis, &x[start],
                &W->x_spec_re[b * block_size],
                &W->x_spec_im[b * block_size]
            );
        }
    }

    /* Step 2: Quantized sparse dot products for each output row */
    const float* cos_lut = W->phase_lut.cos_lut;
    const float* sin_lut = W->phase_lut.sin_lut;

    for (int row = 0; row < out_feat; ++row) {
        float sum = 0.0f;
        const rftmw_qsparse_row_t* srow = &W->rows[row];

        for (int b = 0; b < srow->n_blocks; ++b) {
            const rftmw_qsparse_block_t* blk = &srow->blocks[b];
            if (blk->n_nonzero == 0) continue;

            sum += rftmw_qspectral_sparse_dot(
                blk->indices,
                blk->mag4,
                blk->phase8,
                blk->peak_mag,
                cos_lut, sin_lut,
                &W->x_spec_re[b * block_size],
                &W->x_spec_im[b * block_size],
                (int)blk->n_nonzero
            );
        }
        y[row] = sum;
    }
}

/* ================================================================
 * Weight Compression: Dense -> Quantized Sparse Spectral
 *
 * For each block:
 *   1. RFT transform
 *   2. Sort by magnitude, keep top-k
 *   3. Quantize: 4-bit mag (per-block peak scale) + 8-bit phase
 * ================================================================ */

typedef struct { int idx; float mag; } _idx_mag_t;

static int _cmp_mag_desc(const void* a, const void* b) {
    float da = ((const _idx_mag_t*)a)->mag;
    float db = ((const _idx_mag_t*)b)->mag;
    if (db > da) return 1;
    if (db < da) return -1;
    return 0;
}

/* Compare uint16 for qsort (index sort for cache locality) */
static int _cmp_u16_asc(const void* a, const void* b) {
    uint16_t va = *(const uint16_t*)a;
    uint16_t vb = *(const uint16_t*)b;
    return (va > vb) - (va < vb);
}

void rftmw_spectral_compress_row(
    const rftmw_spectral_basis_t* basis,
    const rftmw_phase_lut_t*      phase_lut,
    const float* dense_row,
    int          in_features,
    float        keep_ratio,
    rftmw_qsparse_row_t* out_row
) {
    int block_size = basis->block_size;
    int n_blocks = (in_features + block_size - 1) / block_size;
    (void)phase_lut; /* used for reference only; quantization uses direct atan2 */

    out_row->n_blocks = n_blocks;
    out_row->blocks = (rftmw_qsparse_block_t*)calloc(
        (size_t)n_blocks, sizeof(rftmw_qsparse_block_t));

    /* Temp buffers */
    float* tmp_re = (float*)aligned_alloc_f((size_t)block_size);
    float* tmp_im = (float*)aligned_alloc_f((size_t)block_size);
    _idx_mag_t* sorted = (_idx_mag_t*)malloc(sizeof(_idx_mag_t) * (size_t)block_size);

    for (int b = 0; b < n_blocks; ++b) {
        int start = b * block_size;
        int end = start + block_size;
        int actual = (end > in_features) ? (in_features - start) : block_size;

        /* Copy block with padding */
        float* input_block = (float*)aligned_alloc_f((size_t)block_size);
        if (actual > 0) {
            memcpy(input_block, &dense_row[start], (size_t)actual * sizeof(float));
        }
        if (actual < block_size) {
            memset(&input_block[actual], 0, (size_t)(block_size - actual) * sizeof(float));
        }

        /* Forward transform */
        rftmw_spectral_forward_block(basis, input_block, tmp_re, tmp_im);
        aligned_free(input_block);

        /* Compute magnitudes and sort */
        for (int k = 0; k < block_size; ++k) {
            sorted[k].idx = k;
            sorted[k].mag = sqrtf(tmp_re[k] * tmp_re[k] + tmp_im[k] * tmp_im[k]);
        }
        qsort(sorted, (size_t)block_size, sizeof(_idx_mag_t), _cmp_mag_desc);

        /* Keep top-k */
        int keep_k = (int)((float)block_size * keep_ratio);
        if (keep_k < 1) keep_k = 1;
        if (keep_k > block_size) keep_k = block_size;

        /* Find peak magnitude for quantization scale */
        float peak_mag = sorted[0].mag;
        if (peak_mag < 1e-30f) peak_mag = 1e-30f; /* prevent div-by-zero */

        /* Collect top-k indices and sort ascending for cache locality */
        uint16_t* top_idx = (uint16_t*)malloc(sizeof(uint16_t) * (size_t)keep_k);
        for (int i = 0; i < keep_k; ++i) {
            top_idx[i] = (uint16_t)sorted[i].idx;
        }
        qsort(top_idx, (size_t)keep_k, sizeof(uint16_t), _cmp_u16_asc);

        /* Allocate quantized sparse block */
        rftmw_qsparse_block_t* blk = &out_row->blocks[b];
        blk->n_nonzero = (uint16_t)keep_k;
        blk->indices = (uint16_t*)calloc((size_t)keep_k, sizeof(uint16_t));
        blk->mag4    = (uint8_t*)calloc(((size_t)keep_k + 1) / 2, 1);
        blk->phase8  = (uint8_t*)calloc((size_t)keep_k, 1);
        blk->peak_mag = peak_mag;

        /* Quantize each coefficient */
        float inv_peak = 15.0f / peak_mag;
        for (int i = 0; i < keep_k; ++i) {
            int idx = (int)top_idx[i];
            blk->indices[i] = top_idx[i];

            /* Magnitude: normalize to [0,1] and quantize to 4-bit [0,15] */
            float mag = sqrtf(tmp_re[idx] * tmp_re[idx] + tmp_im[idx] * tmp_im[idx]);
            int mag_q = (int)(mag * inv_peak + 0.5f);
            if (mag_q > 15) mag_q = 15;
            if (mag_q < 0)  mag_q = 0;

            /* Pack 4-bit: even indices in low nibble, odd in high nibble */
            if (i & 1) {
                blk->mag4[i >> 1] |= (uint8_t)(mag_q << 4);
            } else {
                blk->mag4[i >> 1] = (uint8_t)mag_q;
            }

            /* Phase: atan2(im, re) -> [-pi, +pi] -> uint8 [0, 255] */
            float phase = atan2f(tmp_im[idx], tmp_re[idx]);
            /* Map [-pi, pi] -> [0, 255] */
            float phase_norm = (phase + 3.14159265358979323846f) / (float)RFTMW_TWO_PI;
            int phase_q = (int)(phase_norm * 255.0f + 0.5f);
            if (phase_q > 255) phase_q = 255;
            if (phase_q < 0)   phase_q = 0;
            blk->phase8[i] = (uint8_t)phase_q;
        }
        free(top_idx);
    }

    aligned_free(tmp_re);
    aligned_free(tmp_im);
    free(sorted);
}

/* ================================================================
 * Full Weight Matrix Init / Free
 * ================================================================ */

void rftmw_spectral_weight_init(
    rftmw_spectral_weight_t* W,
    const float* weights,
    int out_features,
    int in_features,
    int block_size,
    float keep_ratio
) {
    memset(W, 0, sizeof(*W));
    W->out_features = out_features;
    W->in_features  = in_features;
    W->block_size   = block_size;
    W->n_blocks     = (in_features + block_size - 1) / block_size;

    /* Init basis and phase LUT */
    rftmw_spectral_basis_init(&W->basis, block_size);
    rftmw_phase_lut_init(&W->phase_lut);

    /* Allocate scratch and transform cache */
    W->scratch_re = (float*)aligned_alloc_f((size_t)block_size);
    W->scratch_im = (float*)aligned_alloc_f((size_t)block_size);
    W->x_spec_re  = (float*)aligned_alloc_f((size_t)W->n_blocks * (size_t)block_size);
    W->x_spec_im  = (float*)aligned_alloc_f((size_t)W->n_blocks * (size_t)block_size);

    /* Compress each row to quantized sparse spectral form */
    W->rows = (rftmw_qsparse_row_t*)calloc((size_t)out_features, sizeof(rftmw_qsparse_row_t));
    for (int r = 0; r < out_features; ++r) {
        rftmw_spectral_compress_row(
            &W->basis,
            &W->phase_lut,
            &weights[(size_t)r * (size_t)in_features],
            in_features,
            keep_ratio,
            &W->rows[r]
        );
    }
}

void rftmw_qsparse_row_free(rftmw_qsparse_row_t* row) {
    if (!row) return;
    for (int b = 0; b < row->n_blocks; ++b) {
        rftmw_qsparse_block_t* blk = &row->blocks[b];
        free(blk->indices);
        free(blk->mag4);
        free(blk->phase8);
    }
    free(row->blocks);
    row->blocks = NULL;
    row->n_blocks = 0;
}

void rftmw_spectral_weight_free(rftmw_spectral_weight_t* W) {
    if (!W) return;
    if (W->rows) {
        for (int r = 0; r < W->out_features; ++r) {
            rftmw_qsparse_row_free(&W->rows[r]);
        }
        free(W->rows);
        W->rows = NULL;
    }
    rftmw_spectral_basis_free(&W->basis);
    aligned_free(W->scratch_re); W->scratch_re = NULL;
    aligned_free(W->scratch_im); W->scratch_im = NULL;
    aligned_free(W->x_spec_re);  W->x_spec_re = NULL;
    aligned_free(W->x_spec_im);  W->x_spec_im = NULL;
}

/* ================================================================
 * Non-Uniform Sparsity Schedule
 *
 * Early layers (embedding-adjacent):  low info-density → aggressive
 * Core layers (middle):               high info-density → conservative
 * Late layers (output-adjacent):      moderate info → moderate
 * ================================================================ */

float rftmw_layer_keep_ratio(int layer_idx, int total_layers, float base_keep) {
    if (total_layers <= 0) return base_keep;

    /* Boundaries */
    int early_end = total_layers / 6;          /* first ~17% of layers */
    int late_start = total_layers - total_layers / 6; /* last ~17% of layers */

    float ratio;
    if (layer_idx < early_end) {
        /* Early layers: static embeddings, low spectral complexity */
        ratio = base_keep * 0.25f;
    } else if (layer_idx >= late_start) {
        /* Late layers: output projection, moderate */
        ratio = base_keep * 0.50f;
    } else {
        /* Core layers: reasoning, attention patterns → need more coefficients */
        ratio = base_keep * 1.50f;
    }

    /* Clamp to [0.02, 1.0] */
    if (ratio < 0.02f) ratio = 0.02f;
    if (ratio > 1.00f) ratio = 1.00f;
    return ratio;
}
