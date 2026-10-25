// SPDX-License-Identifier: LicenseRef-SPX-Proprietary
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
//
// rftmw_spectral_gemv.wgsl — WebGPU Compute Shader for Spectral GEMV
// ====================================================================
//
// Implements the two hot paths on GPU:
//   1. FFT + φ-phase forward transform (input blocks → spectral domain)
//   2. Quantized sparse dot product (spectral weight rows × spectral input)
//
// Architecture:
//   - One dispatch per GEMV:  y[out_features] = SpectralW @ x[in_features]
//   - Workgroup per output row → parallel across rows
//   - Shared memory for input spectral coefficients per block
//
// Memory layout (matches C struct packing):
//   - x_spec_re/im: float32[n_blocks * block_size] — spectral input cache
//   - Per-row: indices (u16), mag4 (u8 packed nibble pairs), phase8 (u8), peak_mag (f32)

// ── Uniforms ──────────────────────────────────────────────────────

struct GemvUniforms {
    out_features:   u32,
    in_features:    u32,
    block_size:     u32,
    n_blocks:       u32,
    log2_block:     u32,
    phi:            f32,  // golden ratio fractional constant
    _pad0:          u32,
    _pad1:          u32,
};

@group(0) @binding(0) var<uniform> u: GemvUniforms;

// ── Phase LUT (256 cos + 256 sin = 512 floats) ──────────────────

@group(0) @binding(1) var<storage, read> phase_lut: array<f32, 512>;
// phase_lut[0..255] = cos_lut, phase_lut[256..511] = sin_lut

// ── Input vector (dense f32) ─────────────────────────────────────

@group(0) @binding(2) var<storage, read> x_in: array<f32>;

// ── Spectral input cache (written by pass 1, read by pass 2) ────

@group(0) @binding(3) var<storage, read_write> x_spec_re: array<f32>;
@group(0) @binding(4) var<storage, read_write> x_spec_im: array<f32>;

// ── Output vector ────────────────────────────────────────────────

@group(0) @binding(5) var<storage, read_write> y_out: array<f32>;

// ══════════════════════════════════════════════════════════════════
// PASS 1: FFT + φ-Phase Forward Transform
// ══════════════════════════════════════════════════════════════════
//
// One workgroup per input block.
// Each thread handles one FFT butterfly, then applies the φ-phase diagonal.
//
// Transform:  X[k] = E[k] * FFT(x_block)[k] / sqrt(N)
//   where E[k] = exp(i * 2π * frac((k+1) * φ))
//
// This IS the hybrid φ-RFT — NOT an NUFFT.

var<workgroup> wg_re: array<f32, 512>;  // max block_size = 512
var<workgroup> wg_im: array<f32, 512>;

fn bit_reverse(x_val: u32, bits: u32) -> u32 {
    var r = 0u;
    var v = x_val;
    for (var i = 0u; i < bits; i++) {
        r = (r << 1u) | (v & 1u);
        v >>= 1u;
    }
    return r;
}

@compute @workgroup_size(128)
fn fft_phi_forward(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) num_wgs: vec3<u32>
) {
    let block_idx = wg_id.x;
    let N = u.block_size;
    let log2N = u.log2_block;
    let tid = lid.x;

    // Load input with bit-reversal permutation
    // Each thread loads N/128 elements (or 1 if N <= 128)
    let elems_per_thread = max(N / 128u, 1u);

    for (var e = 0u; e < elems_per_thread; e++) {
        let i = tid * elems_per_thread + e;
        if (i < N) {
            let j = bit_reverse(i, log2N);
            let src_idx = block_idx * N + i;
            let val = select(0.0, x_in[src_idx], src_idx < u.in_features);
            wg_re[j] = val;
            wg_im[j] = 0.0;
        }
    }

    workgroupBarrier();

    // Radix-2 DIT FFT butterfly stages
    for (var stage = 0u; stage < log2N; stage++) {
        let half_size = 1u << stage;
        let full_size = half_size << 1u;

        for (var e = 0u; e < elems_per_thread; e++) {
            let i = tid * elems_per_thread + e;
            if (i < N / 2u) {
                let group = i / half_size;
                let pos = i % half_size;
                let even_idx = group * full_size + pos;
                let odd_idx = even_idx + half_size;

                let angle = -6.28318530718 * f32(pos) / f32(full_size);
                let tw_re = cos(angle);
                let tw_im = sin(angle);

                let o_re = wg_re[odd_idx];
                let o_im = wg_im[odd_idx];
                let t_re = tw_re * o_re - tw_im * o_im;
                let t_im = tw_re * o_im + tw_im * o_re;

                let e_re = wg_re[even_idx];
                let e_im = wg_im[even_idx];

                wg_re[even_idx] = e_re + t_re;
                wg_im[even_idx] = e_im + t_im;
                wg_re[odd_idx]  = e_re - t_re;
                wg_im[odd_idx]  = e_im - t_im;
            }
        }

        workgroupBarrier();
    }

    // Apply φ-phase diagonal: X[k] *= exp(i * 2π * frac((k+1) * φ)) / sqrt(N)
    let inv_sqrt_n = 1.0 / sqrt(f32(N));
    let phi = u.phi;

    for (var e = 0u; e < elems_per_thread; e++) {
        let k = tid * elems_per_thread + e;
        if (k < N) {
            let frac_val = fract(f32(k + 1u) * phi);
            let angle = 6.28318530718 * frac_val;
            let e_re = cos(angle);
            let e_im = sin(angle);

            let fft_re = wg_re[k];
            let fft_im = wg_im[k];

            // Complex multiply: E[k] * FFT[k]
            let out_re = (e_re * fft_re - e_im * fft_im) * inv_sqrt_n;
            let out_im = (e_re * fft_im + e_im * fft_re) * inv_sqrt_n;

            let dst = block_idx * N + k;
            x_spec_re[dst] = out_re;
            x_spec_im[dst] = out_im;
        }
    }
}


// ══════════════════════════════════════════════════════════════════
// PASS 2: Quantized Sparse Dot Product (per output row)
// ══════════════════════════════════════════════════════════════════
//
// Layout of per-row sparse data (packed into a flat buffer):
//
//   row_offsets[out_features + 1]:  u32 byte offsets into sparse_data
//   sparse_data[...]:  per-row packed:
//     [peak_mag: f32] [n_nonzero: u32] [n_blocks_used: u32]
//     per-block: [block_id: u32] [block_nnz: u32]
//                [indices: u16 × block_nnz] [mag4: u8 × ceil(block_nnz/2)] [phase8: u8 × block_nnz]
//
// Each workgroup handles one output row.
// Threads collaborate on the sparse dot product within each block.

@group(1) @binding(0) var<storage, read> row_offsets: array<u32>;
@group(1) @binding(1) var<storage, read> sparse_data: array<u32>;  // byte-addressed via pack

var<workgroup> wg_accum: array<f32, 128>;

@compute @workgroup_size(128)
fn sparse_dot_rows(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let row = wg_id.x;
    if (row >= u.out_features) { return; }

    let tid = lid.x;
    wg_accum[tid] = 0.0;

    // Read row header from sparse_data
    let base_byte = row_offsets[row];
    let base_u32 = base_byte / 4u;

    let peak_mag_bits = sparse_data[base_u32];
    let peak_mag = bitcast<f32>(peak_mag_bits);
    let n_nonzero_total = sparse_data[base_u32 + 1u];
    let n_blocks_used = sparse_data[base_u32 + 2u];

    let inv15 = peak_mag / 15.0;
    var ptr = base_u32 + 3u;  // start of per-block data

    var local_sum = 0.0;

    for (var b = 0u; b < n_blocks_used; b++) {
        let block_id = sparse_data[ptr];
        let block_nnz = sparse_data[ptr + 1u];
        ptr += 2u;

        // indices: u16 packed as u32 pairs
        let idx_start = ptr;
        let idx_u32s = (block_nnz + 1u) / 2u;  // ceil(nnz / 2)

        // mag4: u8 nibble pairs packed as u32
        let mag_start = idx_start + idx_u32s;
        let mag_u32s = (block_nnz + 7u) / 8u;  // ceil(nnz/2 bytes / 4)

        // phase8: u8 packed as u32
        let phase_start = mag_start + mag_u32s;
        let phase_u32s = (block_nnz + 3u) / 4u;

        let spec_base = block_id * u.block_size;

        // Each thread processes a subset of nonzero coefficients
        var i = tid;
        while (i < block_nnz) {
            // Extract index (u16 from packed u32)
            let idx_word = sparse_data[idx_start + i / 2u];
            let idx_val = select(idx_word & 0xFFFFu, idx_word >> 16u, (i & 1u) != 0u);

            // Extract 4-bit magnitude nibble
            let mag_byte_idx = i / 2u;
            let mag_word = sparse_data[mag_start + mag_byte_idx / 4u];
            let mag_byte = (mag_word >> ((mag_byte_idx % 4u) * 8u)) & 0xFFu;
            let nibble = select(mag_byte & 0x0Fu, mag_byte >> 4u, (i & 1u) != 0u);
            let mag = f32(nibble) * inv15;

            // Extract 8-bit phase level
            let phase_word = sparse_data[phase_start + i / 4u];
            let plevel = (phase_word >> ((i % 4u) * 8u)) & 0xFFu;

            // Dequantize via phase LUT (zero trig in hot loop!)
            let cos_val = phase_lut[plevel];
            let sin_val = phase_lut[256u + plevel];
            let w_re = mag * cos_val;
            let w_im = mag * sin_val;

            // Sparse dot: w · x_spec
            let spec_idx = spec_base + idx_val;
            local_sum += w_re * x_spec_re[spec_idx] + w_im * x_spec_im[spec_idx];

            i += 128u;  // stride by workgroup size
        }

        ptr = phase_start + phase_u32s;
    }

    // Workgroup reduction
    wg_accum[tid] = local_sum;
    workgroupBarrier();

    // Tree reduction
    for (var stride = 64u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            wg_accum[tid] += wg_accum[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        y_out[row] = wg_accum[0u];
    }
}


// ══════════════════════════════════════════════════════════════════
// PASS 3: Q4K/Q6K Dequant GEMV (fallback / baseline comparison)
// ══════════════════════════════════════════════════════════════════
//
// Standard quantized matrix-vector multiply (non-spectral path).
// This runs on GPU as baseline when spectral is disabled.
//
// Q4K format: 256-element super-blocks, 8 sub-blocks of 32
//   - d (f16 scale), dmin (f16 min)
//   - scales[12] (6-bit packed), qs[128] (4-bit nibble pairs)

struct Q4KUniforms {
    out_features: u32,
    in_features:  u32,
    bytes_per_row: u32,
    _pad:         u32,
};

@group(0) @binding(6) var<uniform> q4k_u: Q4KUniforms;
@group(0) @binding(7) var<storage, read> q4k_data: array<u32>;  // raw quantized weight data

var<workgroup> q4k_accum: array<f32, 256>;

@compute @workgroup_size(256)
fn q4k_gemv(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let row = wg_id.x;
    if (row >= q4k_u.out_features) { return; }

    let tid = lid.x;
    let in_feat = q4k_u.in_features;
    let bpr = q4k_u.bytes_per_row;

    // Row base in q4k_data (byte offset → u32 offset)
    let row_base_u32 = (row * bpr) / 4u;

    var local_sum = 0.0;

    // Q4K: 256-element super-blocks
    // Each super-block: 2 bytes (d, dmin as f16) + 12 bytes (scales) + 128 bytes (qs)
    // Total: 144 bytes per super-block of 256 values
    let n_superblocks = (in_feat + 255u) / 256u;

    // Each thread handles a subset of super-blocks
    var sb = tid;
    while (sb < n_superblocks) {
        // Offset into this super-block within the row
        let sb_offset = row_base_u32 + (sb * 144u) / 4u;

        // First u32 contains d (f16) and dmin (f16) packed
        let d_dmin_packed = q4k_data[sb_offset];
        let d_bits = d_dmin_packed & 0xFFFFu;
        let dmin_bits = d_dmin_packed >> 16u;

        // f16 → f32 conversion
        let d = unpack2x16float(d_dmin_packed).x;
        let dmin = unpack2x16float(d_dmin_packed).y;

        // Dequantize 8 sub-blocks of 32 values each
        // scales start at byte 4, qs at byte 16
        let scales_base = sb_offset + 1u;  // byte 4 → u32 offset +1
        let qs_base = sb_offset + 4u;      // byte 16 → u32 offset +4

        for (var sub = 0u; sub < 8u; sub++) {
            // Extract 6-bit scale for this sub-block
            let scale_byte_idx = sub * 3u / 2u;
            let scale_word = q4k_data[scales_base + scale_byte_idx / 4u];
            let scale_byte = (scale_word >> ((scale_byte_idx % 4u) * 8u)) & 0x3Fu;
            let sc = d * f32(scale_byte);

            // Dequantize 32 values from nibble pairs
            let val_offset = sb * 256u + sub * 32u;
            for (var j = 0u; j < 32u; j += 2u) {
                let byte_idx = sub * 16u + j / 2u;
                let qs_word = q4k_data[qs_base + byte_idx / 4u];
                let qs_byte = (qs_word >> ((byte_idx % 4u) * 8u)) & 0xFFu;

                let v0 = f32(qs_byte & 0x0Fu) * sc - dmin;
                let v1 = f32(qs_byte >> 4u) * sc - dmin;

                let idx0 = val_offset + j;
                let idx1 = val_offset + j + 1u;
                if (idx0 < in_feat) { local_sum += v0 * x_in[idx0]; }
                if (idx1 < in_feat) { local_sum += v1 * x_in[idx1]; }
            }
        }

        sb += 256u;  // stride by workgroup size
    }

    // Reduction
    q4k_accum[tid] = local_sum;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            q4k_accum[tid] += q4k_accum[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        y_out[row] = q4k_accum[0u];
    }
}
