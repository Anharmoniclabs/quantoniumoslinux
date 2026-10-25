#!/usr/bin/env python3
"""
Spectral Roundtrip Diagnostic — Pinpoint encode/decode correctness bugs.

Tests:
  A. Pure transform roundtrip  (forward → inverse, no quantization)
  B. Quantized roundtrip        (forward → 4-bit mag + 8-bit phase → dequant → inverse)
  C. GEMV correctness           (spectral matmul vs dense matmul)

Each test reports  ||W - W_hat||_fro / ||W||_fro  (relative Frobenius error).

Expected at keep=1.0:
  A ≈ 1e-7  (float32 FFT roundtrip)
  B ≈ 0.05–0.15  (quantization noise floor from 4-bit mag + 8-bit phase)

If A >> 1e-7 → transform itself is broken.
If B >> 0.15 at keep=1.0 → quantization/dequantization has a correctness bug.
If B jumps discontinuously between keep levels → thresholding/packing bug.
"""

import numpy as np
import struct, mmap, os, sys, time

# ── Constants (must match rftmw_spectral_kernels.h exactly) ──
PHI = 1.6180339887498948482
TWO_PI = 6.2831853071795864769
PI = 3.14159265358979323846
BLOCK_SIZE = 256
PHASE_LUT_SIZE = 256


# ═══════════════════════════════════════════════════════════════
#  GGUF Loader (from test_spectral_residual.py)
# ═══════════════════════════════════════════════════════════════

def fp16_to_f32(raw_2bytes):
    return np.frombuffer(raw_2bytes, dtype=np.float16).astype(np.float32)[0]

def dequant_q4k_block(data):
    d = fp16_to_f32(data[0:2])
    dmin = fp16_to_f32(data[2:4])
    sc_raw = data[4:16]
    qs = data[16:144]
    sc = np.zeros(8, dtype=np.float32)
    mn = np.zeros(8, dtype=np.float32)
    for j in range(4):
        sc[j] = float(sc_raw[j] & 0x3F)
        mn[j] = float(sc_raw[j+4] & 0x3F)
    for j in range(4, 8):
        sc[j] = float(sc_raw[j+4] & 0x0F) + float(sc_raw[j-4] >> 6) * 16.0
        mn[j] = float(sc_raw[j+4] >> 4) + float(sc_raw[j] >> 6) * 16.0
    out = np.zeros(256, dtype=np.float32)
    for group in range(4):
        d1 = d * sc[2*group]
        m1 = dmin * mn[2*group]
        d2 = d * sc[2*group+1]
        m2 = dmin * mn[2*group+1]
        for l in range(32):
            q = qs[32*group + l]
            out[64*group + l]      = d1 * float(q & 0xF) - m1
            out[64*group + 32 + l] = d2 * float(q >> 4)  - m2
    return out

def load_gguf_tensor(path, tensor_name):
    fd = os.open(path, os.O_RDONLY)
    sz = os.fstat(fd).st_size
    mm = mmap.mmap(fd, sz, access=mmap.ACCESS_READ)
    p = 4
    ver = struct.unpack_from('<I', mm, p)[0]; p += 4
    n_tensors = struct.unpack_from('<Q', mm, p)[0]; p += 8
    n_kv = struct.unpack_from('<Q', mm, p)[0]; p += 8

    def rs(off):
        n = struct.unpack_from('<Q', mm, off)[0]
        return mm[off+8:off+8+n].decode('utf-8', 'replace'), off+8+n
    SS = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
    def skip(off, t):
        if t == 8:
            n = struct.unpack_from('<Q', mm, off)[0]; return off + 8 + n
        if t == 9:
            at = struct.unpack_from('<I', mm, off)[0]; off += 4
            al = struct.unpack_from('<Q', mm, off)[0]; off += 8
            if at in SS: return off + al * SS[at]
            for _ in range(al): off = skip(off, at)
            return off
        return off + SS.get(t, 0)
    for _ in range(n_kv):
        _, p = rs(p)
        vt = struct.unpack_from('<I', mm, p)[0]; p += 4
        p = skip(p, vt)
    tensors = {}
    for _ in range(n_tensors):
        name, p = rs(p)
        nd = struct.unpack_from('<I', mm, p)[0]; p += 4
        dims = [struct.unpack_from('<Q', mm, p + d*8)[0] for d in range(nd)]; p += nd*8
        tt = struct.unpack_from('<I', mm, p)[0]; p += 4
        off = struct.unpack_from('<Q', mm, p)[0]; p += 8
        tensors[name] = (dims, tt, off)

    data_start = (p + 31) & ~31

    if tensor_name not in tensors:
        mm.close(); os.close(fd)
        avail = sorted(tensors.keys())[:20]
        raise KeyError(f"'{tensor_name}' not found. First 20 keys: {avail}")

    dims, tt, rel_off = tensors[tensor_name]
    abs_off = data_start + rel_off
    in_feat, out_feat = dims[0], dims[1]
    if tt != 12:
        mm.close(); os.close(fd)
        raise ValueError(f"Expected Q4_K (12), got type {tt}")

    bpr = (in_feat // 256) * 144
    weights = np.zeros((out_feat, in_feat), dtype=np.float32)
    for row in range(out_feat):
        row_start = abs_off + row * bpr
        for b in range(in_feat // 256):
            bo = row_start + b * 144
            weights[row, b*256:(b+1)*256] = dequant_q4k_block(bytes(mm[bo:bo+144]))

    mm.close(); os.close(fd)
    return weights


# ═══════════════════════════════════════════════════════════════
#  Spectral Transform (matches C exactly, but uses float64 for
#  the transform itself to isolate transform bugs from FP noise)
# ═══════════════════════════════════════════════════════════════

def build_basis(block_size=BLOCK_SIZE):
    """Build E_k = exp(j * 2π * frac((k+1)*φ)) diagonal — matches C basis_init."""
    k = np.arange(block_size, dtype=np.float64)
    frac_val = np.mod((k + 1) * PHI, 1.0)
    theta = TWO_PI * frac_val
    return np.cos(theta), np.sin(theta)  # cos_table, sin_table

def build_phase_lut():
    """Build phase LUT — matches C rftmw_phase_lut_init exactly."""
    p = np.arange(PHASE_LUT_SIZE, dtype=np.float64)
    angle = (p / 255.0) * TWO_PI - PI
    return np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)

COS_TABLE, SIN_TABLE = build_basis()
PHASE_COS_LUT, PHASE_SIN_LUT = build_phase_lut()


def spectral_forward_block(x_real, use_f32=False):
    """Forward: out = E * FFT(x) / sqrt(N).  Matches C rftmw_spectral_forward_block."""
    N = len(x_real)
    dtype = np.float32 if use_f32 else np.float64
    cdtype = np.complex64 if use_f32 else np.complex128

    x = np.array(x_real, dtype=dtype)
    fft_out = np.fft.fft(x).astype(cdtype)

    # E_k * FFT[k] / sqrt(N)
    E_re = COS_TABLE[:N].astype(dtype)
    E_im = SIN_TABLE[:N].astype(dtype)
    f_re = np.real(fft_out).astype(dtype)
    f_im = np.imag(fft_out).astype(dtype)

    inv_sqrt = dtype(1.0 / np.sqrt(N))
    out_re = (E_re * f_re - E_im * f_im) * inv_sqrt
    out_im = (E_re * f_im + E_im * f_re) * inv_sqrt
    return out_re, out_im


def spectral_inverse_block(spec_re, spec_im, use_f32=False):
    """Inverse: x = IFFT(E* * spec) * sqrt(N).  Undo forward transform."""
    N = len(spec_re)
    dtype = np.float32 if use_f32 else np.float64

    E_re = COS_TABLE[:N].astype(dtype)
    E_im = SIN_TABLE[:N].astype(dtype)

    # Multiply by E* (conjugate): (E_re - j*E_im) * (spec_re + j*spec_im)
    s_re = E_re * spec_re + E_im * spec_im
    s_im = E_re * spec_im - E_im * spec_re

    sqrt_N = dtype(np.sqrt(N))
    fft_re = s_re * sqrt_N
    fft_im = s_im * sqrt_N

    # IFFT
    fft_c = (fft_re + 1j * fft_im).astype(np.complex128 if not use_f32 else np.complex64)
    x = np.real(np.fft.ifft(fft_c))
    return x.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
#  Quantize / Dequantize (matches C compress_row + sparse_dot)
# ═══════════════════════════════════════════════════════════════

def quantize_block(spec_re, spec_im, keep_ratio=1.0):
    """
    Quantize spectral coefficients exactly as C does:
      - Sort by magnitude, keep top-k
      - 4-bit magnitude (per-block peak scale)
      - 8-bit phase (atan2 → [0,255])
    Returns: indices, mag_nibbles, phase_bytes, peak_mag, keep_k
    """
    N = len(spec_re)
    mags = np.sqrt(spec_re**2 + spec_im**2).astype(np.float32)

    # Sort descending by magnitude
    order = np.argsort(mags)[::-1]

    keep_k = max(1, int(N * keep_ratio))
    if keep_k > N:
        keep_k = N

    top_indices = np.sort(order[:keep_k])  # re-sort ascending for cache locality (matches C)

    peak_mag = float(mags[order[0]])
    if peak_mag < 1e-30:
        peak_mag = 1e-30

    # Quantize
    inv_peak = 15.0 / peak_mag
    mag_nibbles = np.zeros(keep_k, dtype=np.uint8)
    phase_bytes = np.zeros(keep_k, dtype=np.uint8)

    for i, idx in enumerate(top_indices):
        mag = float(mags[idx])
        mag_q = int(mag * inv_peak + 0.5)
        mag_q = max(0, min(15, mag_q))
        mag_nibbles[i] = mag_q

        phase = np.arctan2(float(spec_im[idx]), float(spec_re[idx]))
        phase_norm = (phase + PI) / TWO_PI
        phase_q = int(phase_norm * 255.0 + 0.5)
        phase_q = max(0, min(255, phase_q))
        phase_bytes[i] = phase_q

    return top_indices.astype(np.uint16), mag_nibbles, phase_bytes, peak_mag, keep_k


def dequantize_block(indices, mag_nibbles, phase_bytes, peak_mag, block_size=BLOCK_SIZE):
    """
    Dequantize back to full-size spectral re/im arrays.
    Matches C rftmw_qspectral_sparse_dot dequant logic.
    """
    spec_re = np.zeros(block_size, dtype=np.float32)
    spec_im = np.zeros(block_size, dtype=np.float32)

    inv15 = peak_mag / 15.0

    for i in range(len(indices)):
        nibble = int(mag_nibbles[i])
        mag = float(nibble) * inv15

        plevel = int(phase_bytes[i])
        re = mag * PHASE_COS_LUT[plevel]
        im = mag * PHASE_SIN_LUT[plevel]

        idx = int(indices[i])
        spec_re[idx] = re
        spec_im[idx] = im

    return spec_re, spec_im


# ═══════════════════════════════════════════════════════════════
#  Row-level roundtrip (process all blocks in a row)
# ═══════════════════════════════════════════════════════════════

def roundtrip_row_transform_only(row, block_size=BLOCK_SIZE, use_f32=False):
    """Forward → inverse, no quantization. Should be near machine epsilon."""
    N = len(row)
    n_blocks = (N + block_size - 1) // block_size
    recon = np.zeros(n_blocks * block_size, dtype=np.float32)

    for b in range(n_blocks):
        s = b * block_size
        e = min(s + block_size, N)
        blk = np.zeros(block_size, dtype=np.float32)
        blk[:e-s] = row[s:e]

        spec_re, spec_im = spectral_forward_block(blk, use_f32=use_f32)
        recon[b*block_size:(b+1)*block_size] = spectral_inverse_block(spec_re, spec_im, use_f32=use_f32)

    return recon[:N]


def roundtrip_row_quantized(row, keep_ratio=1.0, block_size=BLOCK_SIZE):
    """Forward → quantize → dequantize → inverse. Full pipeline."""
    N = len(row)
    n_blocks = (N + block_size - 1) // block_size
    recon = np.zeros(n_blocks * block_size, dtype=np.float32)

    for b in range(n_blocks):
        s = b * block_size
        e = min(s + block_size, N)
        blk = np.zeros(block_size, dtype=np.float32)
        blk[:e-s] = row[s:e]

        # Forward transform (use float64 for transform to isolate quant errors)
        spec_re, spec_im = spectral_forward_block(blk, use_f32=False)

        # Quantize (4-bit mag + 8-bit phase, matches C exactly)
        indices, mag_nib, phase_b, peak_mag, keep_k = quantize_block(
            spec_re.astype(np.float32), spec_im.astype(np.float32), keep_ratio)

        # Dequantize
        dq_re, dq_im = dequantize_block(indices, mag_nib, phase_b, peak_mag, block_size)

        # Inverse transform
        recon[b*block_size:(b+1)*block_size] = spectral_inverse_block(
            dq_re.astype(np.float64), dq_im.astype(np.float64), use_f32=False)

    return recon[:N]


# ═══════════════════════════════════════════════════════════════
#  GEMV test: spectral matmul vs dense matmul
# ═══════════════════════════════════════════════════════════════

def spectral_gemv_row(row, x, keep_ratio=1.0, block_size=BLOCK_SIZE):
    """
    Compute y = row @ x via spectral domain.
    Matches C rftmw_spectral_gemv: transform x blocks, then sparse dot.
    """
    N = len(row)
    n_blocks = (N + block_size - 1) // block_size

    # Step 1: Transform x blocks (matches C step 1)
    x_spec_re = np.zeros(n_blocks * block_size, dtype=np.float32)
    x_spec_im = np.zeros(n_blocks * block_size, dtype=np.float32)

    for b in range(n_blocks):
        s = b * block_size
        e = min(s + block_size, N)
        blk = np.zeros(block_size, dtype=np.float32)
        blk[:e-s] = x[s:e]
        # Use float32 here to match C behavior
        fwd_re, fwd_im = spectral_forward_block(blk, use_f32=True)
        x_spec_re[b*block_size:(b+1)*block_size] = fwd_re.astype(np.float32)
        x_spec_im[b*block_size:(b+1)*block_size] = fwd_im.astype(np.float32)

    # Step 2: Compress row (forward transform + quantize)
    row_blocks = []
    for b in range(n_blocks):
        s = b * block_size
        e = min(s + block_size, N)
        blk = np.zeros(block_size, dtype=np.float32)
        blk[:e-s] = row[s:e]
        spec_re, spec_im = spectral_forward_block(blk, use_f32=True)
        indices, mag_nib, phase_b, peak_mag, keep_k = quantize_block(
            spec_re.astype(np.float32), spec_im.astype(np.float32), keep_ratio)
        row_blocks.append((indices, mag_nib, phase_b, peak_mag))

    # Step 3: Sparse dot (matches C rftmw_qspectral_sparse_dot)
    result = 0.0
    for b, (indices, mag_nib, phase_b, peak_mag) in enumerate(row_blocks):
        inv15 = peak_mag / 15.0
        for i in range(len(indices)):
            nibble = int(mag_nib[i])
            mag = float(nibble) * inv15
            plevel = int(phase_b[i])
            re = mag * float(PHASE_COS_LUT[plevel])
            im = mag * float(PHASE_SIN_LUT[plevel])
            idx = int(indices[i])
            boff = b * block_size
            result += re * x_spec_re[boff + idx] + im * x_spec_im[boff + idx]

    return result


# ═══════════════════════════════════════════════════════════════
#  Diagnostic runner
# ═══════════════════════════════════════════════════════════════

def frob_relative(original, reconstructed):
    """||orig - recon||_F / ||orig||_F"""
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    norm_diff = np.sqrt(np.sum(diff**2))
    norm_orig = np.sqrt(np.sum(original.astype(np.float64)**2))
    if norm_orig < 1e-30:
        return float('inf')
    return norm_diff / norm_orig


def run_single_block_tests():
    """Test A: Single 256-element block, synthetic data."""
    print("=" * 72)
    print("  TEST A: Single-block roundtrip (synthetic 256-element vectors)")
    print("=" * 72)

    rng = np.random.RandomState(42)

    test_vectors = {
        "randn":      rng.randn(BLOCK_SIZE).astype(np.float32),
        "uniform":    rng.uniform(-1, 1, BLOCK_SIZE).astype(np.float32),
        "sparse":     np.zeros(BLOCK_SIZE, dtype=np.float32),
        "ones":       np.ones(BLOCK_SIZE, dtype=np.float32),
        "impulse":    np.zeros(BLOCK_SIZE, dtype=np.float32),
        "nn_like":    (rng.randn(BLOCK_SIZE) * 0.02).astype(np.float32),
    }
    test_vectors["sparse"][:10] = rng.randn(10).astype(np.float32)
    test_vectors["impulse"][0] = 1.0

    # A1: Pure transform roundtrip (float64)
    print("\n  A1. Pure transform roundtrip (float64 FFT, no quantization):")
    print(f"  {'Vector':12s} | {'||W-W_hat||/||W||':>20s} | {'Status':8s}")
    print(f"  {'-'*12} | {'-'*20} | {'-'*8}")
    for name, vec in test_vectors.items():
        recon = roundtrip_row_transform_only(vec, use_f32=False)
        err = frob_relative(vec, recon)
        status = "OK" if err < 1e-10 else ("WARN" if err < 1e-5 else "**BUG**")
        print(f"  {name:12s} | {err:20.2e} | {status}")

    # A2: Pure transform roundtrip (float32)
    print("\n  A2. Pure transform roundtrip (float32 FFT, no quantization):")
    print(f"  {'Vector':12s} | {'||W-W_hat||/||W||':>20s} | {'Status':8s}")
    print(f"  {'-'*12} | {'-'*20} | {'-'*8}")
    for name, vec in test_vectors.items():
        recon = roundtrip_row_transform_only(vec, use_f32=True)
        err = frob_relative(vec, recon)
        status = "OK" if err < 1e-5 else ("WARN" if err < 1e-3 else "**BUG**")
        print(f"  {name:12s} | {err:20.2e} | {status}")

    # A3: Quantized roundtrip, keep=1.0
    print("\n  A3. Quantized roundtrip (4-bit mag + 8-bit phase), keep=1.0:")
    print(f"  {'Vector':12s} | {'||W-W_hat||/||W||':>20s} | {'Status':8s}")
    print(f"  {'-'*12} | {'-'*20} | {'-'*8}")
    for name, vec in test_vectors.items():
        recon = roundtrip_row_quantized(vec, keep_ratio=1.0)
        err = frob_relative(vec, recon)
        # 4-bit mag + 8-bit phase: expect ~5-15% error from quantization noise
        status = "OK" if err < 0.20 else ("HIGH" if err < 0.50 else "**BUG**")
        print(f"  {name:12s} | {err:20.4f} | {status}")

    # A4: Quantized roundtrip, keep sweep
    print("\n  A4. Quantized roundtrip keep sweep (randn vector):")
    print(f"  {'keep':>6s} | {'||W-W_hat||/||W||':>20s} | {'n_kept':>6s}")
    print(f"  {'-'*6} | {'-'*20} | {'-'*6}")
    vec = test_vectors["nn_like"]
    for keep in [1.0, 0.80, 0.50, 0.30, 0.10, 0.05]:
        recon = roundtrip_row_quantized(vec, keep_ratio=keep)
        err = frob_relative(vec, recon)
        n_kept = max(1, int(BLOCK_SIZE * keep))
        print(f"  {keep:6.2f} | {err:20.6f} | {n_kept:6d}")

    return test_vectors


def run_real_weight_tests(W, tensor_name):
    """Test B: Real GGUF weight matrix roundtrip."""
    out_f, in_f = W.shape
    n_sample = min(out_f, 64)
    sample_idx = np.linspace(0, out_f - 1, n_sample, dtype=int)

    print("\n" + "=" * 72)
    print(f"  TEST B: Real weight matrix roundtrip — {tensor_name}")
    print(f"  Shape: {out_f} x {in_f}")
    print(f"  Stats: min={W.min():.6f} max={W.max():.6f} mean={W.mean():.6f} std={W.std():.6f}")
    print(f"  Sampling {n_sample} rows")
    print("=" * 72)

    # B1: Pure transform roundtrip
    print("\n  B1. Pure transform (float64), keep=1.0:")
    errors = []
    for ri in range(n_sample):
        row = W[sample_idx[ri]]
        recon = roundtrip_row_transform_only(row, use_f32=False)
        errors.append(frob_relative(row, recon))
    errors = np.array(errors)
    status = "OK" if errors.max() < 1e-10 else ("WARN" if errors.max() < 1e-5 else "**BUG**")
    print(f"    mean={errors.mean():.2e}  max={errors.max():.2e}  min={errors.min():.2e}  [{status}]")

    # B1b: Same but float32 (matches C precision)
    print("\n  B1b. Pure transform (float32), keep=1.0:")
    errors = []
    for ri in range(n_sample):
        row = W[sample_idx[ri]]
        recon = roundtrip_row_transform_only(row, use_f32=True)
        errors.append(frob_relative(row, recon))
    errors = np.array(errors)
    status = "OK" if errors.max() < 1e-5 else ("WARN" if errors.max() < 1e-3 else "**BUG**")
    print(f"    mean={errors.mean():.2e}  max={errors.max():.2e}  min={errors.min():.2e}  [{status}]")

    # B2: Quantized roundtrip at various keep levels
    print("\n  B2. Quantized roundtrip (4-bit mag + 8-bit phase):")
    print(f"  {'keep':>6s} | {'mean err':>12s} | {'max err':>12s} | {'min err':>12s} | {'status':8s}")
    print(f"  {'-'*6} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*8}")

    for keep in [1.0, 0.80, 0.50, 0.30, 0.10]:
        errors = []
        for ri in range(n_sample):
            row = W[sample_idx[ri]]
            recon = roundtrip_row_quantized(row, keep_ratio=keep)
            errors.append(frob_relative(row, recon))
        errors = np.array(errors)
        if keep == 1.0:
            status = "OK" if errors.mean() < 0.20 else ("HIGH" if errors.mean() < 0.50 else "**BUG**")
        else:
            status = f"{errors.mean():.4f}"
        print(f"  {keep:6.2f} | {errors.mean():12.6f} | {errors.max():12.6f} | {errors.min():12.6f} | {status}")


def run_gemv_test(W, tensor_name):
    """Test C: Spectral GEMV vs dense matmul."""
    out_f, in_f = W.shape
    n_sample = min(out_f, 32)
    sample_idx = np.linspace(0, out_f - 1, n_sample, dtype=int)

    rng = np.random.RandomState(42)
    x = (rng.randn(in_f) * 0.01).astype(np.float32)
    y_dense = W[sample_idx] @ x

    print("\n" + "=" * 72)
    print(f"  TEST C: GEMV correctness — {tensor_name}")
    print(f"  Computing y = W[sampled_rows] @ x for {n_sample} rows")
    print("=" * 72)

    for keep in [1.0, 0.50, 0.30, 0.10]:
        y_spectral = np.zeros(n_sample, dtype=np.float64)
        for ri in range(n_sample):
            row = W[sample_idx[ri]]
            y_spectral[ri] = spectral_gemv_row(row, x, keep_ratio=keep)

        # Cosine similarity
        cos_sim = np.dot(y_dense, y_spectral) / (
            np.linalg.norm(y_dense) * np.linalg.norm(y_spectral) + 1e-30)
        # Relative error
        rel_err = np.linalg.norm(y_dense - y_spectral) / (np.linalg.norm(y_dense) + 1e-30)
        # Max absolute deviation
        max_dev = np.max(np.abs(y_dense - y_spectral))

        status = "OK" if cos_sim > 0.95 else ("LOW" if cos_sim > 0.50 else "**BUG**")
        print(f"  keep={keep:.2f}: cosine={cos_sim:.6f}  rel_err={rel_err:.6f}  max_dev={max_dev:.2e}  [{status}]")


def run_coefficient_drill(W, tensor_name):
    """
    Test D: Drill into one block to find EXACTLY where error comes from.
    Isolates: transform error vs phase quantization error vs magnitude quantization error.
    """
    row = W[W.shape[0] // 2]  # middle row
    blk = row[:BLOCK_SIZE].copy()

    print("\n" + "=" * 72)
    print(f"  TEST D: Single-block coefficient drill — {tensor_name}, row {W.shape[0]//2}, block 0")
    print(f"  Block stats: min={blk.min():.6f} max={blk.max():.6f} std={blk.std():.6f}")
    print("=" * 72)

    # Forward transform
    spec_re, spec_im = spectral_forward_block(blk, use_f32=False)
    spec_re_f32 = spec_re.astype(np.float32)
    spec_im_f32 = spec_im.astype(np.float32)

    # D1: Transform roundtrip
    recon_exact = spectral_inverse_block(spec_re, spec_im, use_f32=False)
    err_transform = frob_relative(blk, recon_exact)
    print(f"\n  D1. Transform roundtrip error:         {err_transform:.2e}")

    # D2: Magnitude-only quantization (keep phase exact)
    mags = np.sqrt(spec_re_f32**2 + spec_im_f32**2)
    peak = mags.max()
    if peak < 1e-30:
        peak = 1e-30
    mag_q = np.clip(np.round(mags / peak * 15), 0, 15).astype(np.uint8)
    mag_dq = mag_q.astype(np.float32) * (peak / 15.0)

    # Reconstruct with exact phase but quantized magnitude
    phases_exact = np.arctan2(spec_im_f32, spec_re_f32)
    dq_re = mag_dq * np.cos(phases_exact)
    dq_im = mag_dq * np.sin(phases_exact)
    recon_mag_only = spectral_inverse_block(dq_re.astype(np.float64), dq_im.astype(np.float64))
    err_mag_only = frob_relative(blk, recon_mag_only)
    print(f"  D2. Magnitude quant only (4-bit):      {err_mag_only:.6f}")

    # D3: Phase-only quantization (keep magnitude exact)
    phase_norm = (phases_exact + PI) / TWO_PI
    phase_q = np.clip(np.round(phase_norm * 255), 0, 255).astype(np.uint8)
    phase_dq = np.zeros(BLOCK_SIZE, dtype=np.float32)
    for i in range(BLOCK_SIZE):
        phase_dq[i] = np.arctan2(PHASE_SIN_LUT[phase_q[i]], PHASE_COS_LUT[phase_q[i]])

    dq_re2 = mags * np.cos(phase_dq)
    dq_im2 = mags * np.sin(phase_dq)
    recon_phase_only = spectral_inverse_block(dq_re2.astype(np.float64), dq_im2.astype(np.float64))
    err_phase_only = frob_relative(blk, recon_phase_only)
    print(f"  D3. Phase quant only (8-bit):          {err_phase_only:.6f}")

    # D4: Both quantized (full pipeline)
    recon_both = roundtrip_row_quantized(blk, keep_ratio=1.0)
    err_both = frob_relative(blk, recon_both)
    print(f"  D4. Both mag+phase quantized:          {err_both:.6f}")

    # D5: Check if errors add up (they should roughly)
    print(f"\n  D5. Error budget sanity check:")
    print(f"      Transform noise:    {err_transform:.2e}")
    print(f"      Mag quant alone:    {err_mag_only:.6f}")
    print(f"      Phase quant alone:  {err_phase_only:.6f}")
    print(f"      Combined:           {err_both:.6f}")
    combined_est = np.sqrt(err_mag_only**2 + err_phase_only**2)
    print(f"      RSS estimate:       {combined_est:.6f}")
    ratio = err_both / (combined_est + 1e-30)
    print(f"      Actual/RSS ratio:   {ratio:.2f}  {'(OK, ~1.0)' if 0.5 < ratio < 2.0 else '(**SUSPICIOUS**)'}")

    # D6: Per-coefficient error histogram
    print(f"\n  D6. Magnitude quantization error distribution:")
    mag_err_pct = np.abs(mags - mag_dq) / (mags + 1e-30) * 100
    print(f"      Mean mag error:  {mag_err_pct.mean():.2f}%")
    print(f"      Max mag error:   {mag_err_pct.max():.2f}%")
    print(f"      >10% error:      {(mag_err_pct > 10).sum()}/{BLOCK_SIZE} coefficients")

    print(f"\n  D7. Phase quantization error distribution:")
    phase_err_rad = np.abs(phases_exact - phase_dq)
    phase_err_rad = np.minimum(phase_err_rad, TWO_PI - phase_err_rad)  # wrap
    phase_err_deg = np.degrees(phase_err_rad)
    print(f"      Mean phase error: {phase_err_deg.mean():.2f} degrees")
    print(f"      Max phase error:  {phase_err_deg.max():.2f} degrees")
    print(f"      >5 deg error:     {(phase_err_deg > 5).sum()}/{BLOCK_SIZE} coefficients")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  RFTMW Spectral Roundtrip Diagnostic                               ║")
    print("║  Goal: Find ||W - W_hat||/||W|| at keep=1.0 → locate the bug       ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    t_start = time.time()

    # ── Test A: Synthetic single-block tests ──
    test_vectors = run_single_block_tests()

    # ── Find GGUF file ──
    gguf_candidates = [
        # 7B model (the one with the perplexity bug)
        r'C:\quantoniumos\models\DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf',
        r'/workspaces/quantoniumos/models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf',
        # 3B model (backup)
        r'C:\quantoniumos\models\qwen2.5-coder-3b-instruct-q4_k_m.gguf',
        r'/workspaces/quantoniumos/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf',
    ]

    # Allow override via command line
    if len(sys.argv) > 1:
        gguf_candidates = [sys.argv[1]] + gguf_candidates

    gguf_path = None
    for c in gguf_candidates:
        if os.path.isfile(c):
            gguf_path = c
            break

    if gguf_path is None:
        print("\n  No GGUF file found. Synthetic tests complete.")
        print(f"  Searched: {gguf_candidates}")
        print(f"\n  Usage: python {sys.argv[0]} /path/to/model.gguf")
        print(f"\n  Elapsed: {time.time()-t_start:.1f}s")
        sys.exit(0)

    print(f"\n  GGUF: {gguf_path}")

    # Pick a middle-layer FFN weight (the ones the spectral path targets)
    # Try 7B (28 layers → layer 14), then 3B (36 layers → layer 18)
    tensor_candidates = [
        'blk.14.ffn_gate.weight',   # 7B middle
        'blk.14.ffn_up.weight',     # 7B middle
        'blk.14.ffn_down.weight',   # 7B middle
        'blk.18.ffn_gate.weight',   # 3B middle
        'blk.17.ffn_gate.weight',   # 3B middle
        'blk.0.ffn_gate.weight',    # fallback: first layer
    ]

    for tensor_name in tensor_candidates:
        try:
            print(f"\n  Loading {tensor_name}...", flush=True)
            t0 = time.time()
            W = load_gguf_tensor(gguf_path, tensor_name)
            print(f"  Loaded: shape={W.shape}, time={time.time()-t0:.1f}s")

            run_real_weight_tests(W, tensor_name)
            run_gemv_test(W, tensor_name)
            run_coefficient_drill(W, tensor_name)

            # One tensor is enough to diagnose
            break
        except KeyError as e:
            print(f"  {e}")
            continue
        except Exception as e:
            print(f"  Error loading {tensor_name}: {e}")
            import traceback; traceback.print_exc()
            continue

    print(f"\n{'='*72}")
    print(f"  Total elapsed: {time.time()-t_start:.1f}s")
    print(f"{'='*72}")
    print()
    print("  INTERPRETATION GUIDE:")
    print("  ─────────────────────")
    print("  A1/B1 error >> 1e-7  → FFT or E_k diagonal is wrong")
    print("  A3/B2 error >> 0.15 at keep=1.0 → quantization/dequant has a bug")
    print("  B2 error jumps discontinuously → thresholding/index packing bug")
    print("  C cosine < 0.95 at keep=1.0 → GEMV path has a correctness bug")
    print("  D4 >> sqrt(D2² + D3²) → interaction bug between mag and phase quant")
    print()
