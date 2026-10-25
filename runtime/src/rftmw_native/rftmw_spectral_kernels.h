/*
 * SPDX-License-Identifier: LicenseRef-SPX-Proprietary
 * Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
 *
 * rftmw_spectral_kernels.h - Spectral-Domain Matmul Kernels v2
 * =============================================================
 *
 * C interface for RFTMW spectral-domain matrix-vector multiplication.
 * Weights stored as quantized sparse spectral coefficients:
 *   - 4-bit magnitude (uint4 packed) + 8-bit phase (int8)
 *   - Per-block peak magnitude scale factor
 *   - 256-entry cos/sin LUT for zero-trig dequant in hot loop
 *
 * Memory per coefficient: 1.5 bytes (vs 8 bytes float32 re/im)
 *   → 5.3x coefficient compression on top of spectral sparsity
 *
 * Non-uniform sparsity: each layer gets its own keep_ratio based on
 * a "logic vs static" schedule.
 */

#ifndef RFTMW_SPECTRAL_KERNELS_H
#define RFTMW_SPECTRAL_KERNELS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Constants
 * ================================================================ */

#define RFTMW_PHI       1.6180339887498948482
#define RFTMW_TWO_PI    6.2831853071795864769
#define RFTMW_SPECTRAL_DEFAULT_BLOCK  256
#define RFTMW_SPECTRAL_DEFAULT_KEEP   0.20f
#define RFTMW_PHASE_LUT_SIZE          256

/* ================================================================
 * Phase LUT (precomputed cos/sin for 256 phase levels)
 * ================================================================ */

typedef struct {
    float cos_lut[RFTMW_PHASE_LUT_SIZE];  /* cos(phase) for each uint8 level */
    float sin_lut[RFTMW_PHASE_LUT_SIZE];  /* sin(phase) for each uint8 level */
} rftmw_phase_lut_t;

/* Precompute phase LUT. Phase level p maps to angle = (p/255)*2pi - pi */
void rftmw_phase_lut_init(rftmw_phase_lut_t* lut);

/* ================================================================
 * Data Structures
 * ================================================================ */

/* Precomputed phi-grid diagonal for hybrid RFT */
typedef struct {
    float* cos_table;    /* E_re[k] = cos(2pi * frac((k+1)*phi)) */
    float* sin_table;    /* E_im[k] = sin(2pi * frac((k+1)*phi)) */
    float* twiddle_re;   /* FFT twiddle factors (reserved) */
    float* twiddle_im;   /* FFT twiddle factors (reserved) */
    float  inv_sqrt_n;   /* 1/sqrt(block_size) */
    int    block_size;   /* N (must be power of 2) */
    int    log2_n;       /* log2(N) */
} rftmw_spectral_basis_t;

/* Quantized sparse spectral block:
 *   4-bit magnitude (packed 2 per byte) + 8-bit phase
 *   Per-block peak_mag for dequantization:
 *     mag_f32 = (nibble / 15.0) * peak_mag
 *     phase_f32 = lut lookup from uint8 level
 *     re = mag * cos(phase), im = mag * sin(phase) */
typedef struct {
    uint16_t  n_nonzero;
    uint16_t* indices;     /* nonzero positions within block [n_nonzero] */
    uint8_t*  mag4;        /* 4-bit magnitudes packed [ceil(n_nonzero/2)] */
    uint8_t*  phase8;      /* 8-bit phase levels 0..255 [n_nonzero] */
    float     peak_mag;    /* per-block magnitude scale */
} rftmw_qsparse_block_t;

/* Quantized sparse row */
typedef struct {
    int n_blocks;
    rftmw_qsparse_block_t* blocks;
} rftmw_qsparse_row_t;

/* Full quantized spectral weight matrix */
typedef struct {
    int out_features;
    int in_features;
    int block_size;
    int n_blocks;         /* ceil(in_features / block_size) */
    rftmw_spectral_basis_t basis;
    rftmw_phase_lut_t      phase_lut;
    rftmw_qsparse_row_t*   rows;
    /* Scratch buffers for input transform (allocated once) */
    float* scratch_re;    /* [block_size] */
    float* scratch_im;    /* [block_size] */
    /* Transformed input cache */
    float* x_spec_re;    /* [n_blocks * block_size] */
    float* x_spec_im;    /* [n_blocks * block_size] */
} rftmw_spectral_weight_t;

/* ================================================================
 * FFT Kernels
 * ================================================================ */

void rftmw_fft_f32(float* re, float* im, int n);
void rftmw_ifft_f32(float* re, float* im, int n);

/* ================================================================
 * Spectral Basis
 * ================================================================ */

void rftmw_spectral_basis_init(rftmw_spectral_basis_t* basis, int block_size);
void rftmw_spectral_basis_free(rftmw_spectral_basis_t* basis);

/* ================================================================
 * Forward Transform
 * ================================================================ */

void rftmw_spectral_forward_block(
    const rftmw_spectral_basis_t* basis,
    const float* x_real,
    float* out_re,
    float* out_im
);

/* ================================================================
 * Quantized Sparse Spectral Dot Product (Hot Inner Loop)
 *
 * Dequantizes on the fly using phase LUT:
 *   mag  = (nibble / 15.0) * peak_mag
 *   re   = mag * cos_lut[phase8[i]]
 *   im   = mag * sin_lut[phase8[i]]
 *   dot += re * x_spec_re[idx] + im * x_spec_im[idx]
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
);

/* ================================================================
 * Full Spectral GEMV
 * ================================================================ */

void rftmw_spectral_gemv(
    rftmw_spectral_weight_t* W,
    const float* x,
    float*       y
);

/* ================================================================
 * Weight Construction / Conversion
 * ================================================================ */

/* Compress one dense row → quantized sparse spectral. */
void rftmw_spectral_compress_row(
    const rftmw_spectral_basis_t* basis,
    const rftmw_phase_lut_t*      phase_lut,
    const float* dense_row,
    int          in_features,
    float        keep_ratio,
    rftmw_qsparse_row_t* out_row
);

/* Compress full matrix. */
void rftmw_spectral_weight_init(
    rftmw_spectral_weight_t* W,
    const float* weights,
    int out_features,
    int in_features,
    int block_size,
    float keep_ratio
);

void rftmw_qsparse_row_free(rftmw_qsparse_row_t* row);
void rftmw_spectral_weight_free(rftmw_spectral_weight_t* W);

/* ================================================================
 * Non-Uniform Sparsity Schedule
 *
 * Returns keep_ratio for a given layer index based on:
 *   Early layers (0..n/6):     base * 0.25  (static, low-info)
 *   Core layers  (n/6..5n/6):  base * 1.5   (logic, high-info)
 *   Late layers  (5n/6..n):    base * 0.5   (output projection)
 * ================================================================ */

float rftmw_layer_keep_ratio(int layer_idx, int total_layers, float base_keep);

#ifdef __cplusplus
}
#endif

#endif /* RFTMW_SPECTRAL_KERNELS_H */
