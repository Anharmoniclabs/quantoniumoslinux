#!/usr/bin/env python3
"""
Spectral energy & residual patching diagnostic v2.
Fixed GGUF parser: tensor offsets are relative to data_start.
Uses float64 for FFT to avoid overflow.
"""
import numpy as np
import struct, mmap, os, time

PHI = 1.6180339887498948482
TWO_PI = 6.2831853071795864769

# ─── GGUF loader ───

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
    mm = mmap.mmap(fd, sz, mmap.MAP_PRIVATE, mmap.PROT_READ)
    p = 4
    ver = struct.unpack_from('<I', mm, p)[0]; p += 4
    n_tensors = struct.unpack_from('<Q', mm, p)[0]; p += 8
    n_kv = struct.unpack_from('<Q', mm, p)[0]; p += 8

    def rs(off):
        n = struct.unpack_from('<Q', mm, off)[0]
        return mm[off+8:off+8+n].decode('utf-8','replace'), off+8+n
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

    # KEY FIX: data section starts aligned after headers
    data_start = (p + 31) & ~31

    if tensor_name not in tensors:
        mm.close(); os.close(fd)
        avail = [k for k in tensors if 'blk.0' in k or 'output' in k or 'embd' in k]
        raise KeyError(f"'{tensor_name}' not found. Sample keys: {avail}")

    dims, tt, rel_off = tensors[tensor_name]
    abs_off = data_start + rel_off
    in_feat, out_feat = dims[0], dims[1]
    if tt != 12:
        mm.close(); os.close(fd)
        raise ValueError(f"Expected Q4_K (12), got {tt}")

    bpr = (in_feat // 256) * 144
    weights = np.zeros((out_feat, in_feat), dtype=np.float32)
    for row in range(out_feat):
        row_start = abs_off + row * bpr
        for b in range(in_feat // 256):
            bo = row_start + b * 144
            weights[row, b*256:(b+1)*256] = dequant_q4k_block(bytes(mm[bo:bo+144]))

    mm.close(); os.close(fd)
    return weights

# ─── Hybrid RFT (float64 for precision) ───

def hybrid_rft_forward(x, block_size=256):
    N = len(x)
    nb = (N + block_size - 1) // block_size
    result = np.zeros(nb * block_size, dtype=np.complex128)
    for b in range(nb):
        s = b * block_size
        e = min(s + block_size, N)
        blk = np.zeros(block_size, dtype=np.float64)
        blk[:e-s] = x[s:e].astype(np.float64)
        fft_out = np.fft.fft(blk)
        k = np.arange(block_size, dtype=np.float64)
        E = np.exp(1j * TWO_PI * np.mod((k+1)*PHI, 1.0))
        result[b*block_size:(b+1)*block_size] = E * fft_out / np.sqrt(block_size)
    return result

def hybrid_rft_inverse(spec, block_size=256, orig_len=None):
    nb = len(spec) // block_size
    result = np.zeros(nb * block_size, dtype=np.float64)
    for b in range(nb):
        chunk = spec[b*block_size:(b+1)*block_size]
        k = np.arange(block_size, dtype=np.float64)
        E_conj = np.exp(-1j * TWO_PI * np.mod((k+1)*PHI, 1.0))
        result[b*block_size:(b+1)*block_size] = np.real(np.fft.ifft(E_conj * chunk * np.sqrt(block_size)))
    if orig_len: result = result[:orig_len]
    return result.astype(np.float32)

# ─── Analysis ───

def analyze(weights, name, block_size=256):
    out_f, in_f = weights.shape
    print(f"\n{'='*70}")
    print(f"  {name}: {out_f}x{in_f}")
    print(f"  Weight stats: min={weights.min():.6f} max={weights.max():.6f} "
          f"mean={weights.mean():.6f} std={weights.std():.6f}")
    print(f"{'='*70}")

    n_sample = min(out_f, 64)
    idx = np.linspace(0, out_f-1, n_sample, dtype=int)

    keep_levels = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00]
    print(f"\n  Spectral Energy (block={block_size}):")
    print(f"  Keep%  | Energy%  | NRMSE%   ")
    print(f"  -------|----------|----------")

    for keep in keep_levels:
        energies, errors = [], []
        for ri in range(n_sample):
            row = weights[idx[ri]]
            spec = hybrid_rft_forward(row, block_size)
            mags = np.abs(spec)
            total_e = np.sum(mags**2)
            si = np.argsort(mags)[::-1]
            k = max(1, int(len(spec) * keep))
            mask = np.zeros_like(spec)
            mask[si[:k]] = spec[si[:k]]
            energies.append(np.sum(np.abs(mask)**2) / (total_e + 1e-30))
            recon = hybrid_rft_inverse(mask, block_size, len(row))
            errors.append(np.sqrt(np.mean((row - recon)**2)) / (np.sqrt(np.mean(row**2)) + 1e-10))
        ec = np.mean(energies)*100
        er = np.mean(errors)*100
        tag = "✓" if ec > 95 else ("~" if ec > 80 else "✗")
        print(f"  {keep*100:5.0f}%  | {ec:6.2f}%  | {er:6.2f}%  {tag}")

def test_residual(weights, name, block_size=256):
    out_f, in_f = weights.shape
    n_s = min(out_f, 32)
    idx = np.linspace(0, out_f-1, n_s, dtype=int)
    rng = np.random.RandomState(42)
    x = rng.randn(in_f).astype(np.float32) * 0.01
    y_true = weights[idx] @ x
    q4k_bpr = (in_f // 256) * 144

    print(f"\n  Residual Patching ({name}):")
    print(f"  spectral_keep | Method               | cosine   | bytes/row | vs Q4_K")
    print(f"  --------------|----------------------|----------|-----------|--------")

    for sk in [0.05, 0.10, 0.20]:
        y_s = np.zeros(n_s)
        y_rq8 = np.zeros(n_s)
        y_rq4 = np.zeros(n_s)
        sb, rb8, rb4 = 0, 0, 0

        for ri in range(n_s):
            row = weights[idx[ri]]
            spec = hybrid_rft_forward(row, block_size)
            mags = np.abs(spec)
            si = np.argsort(mags)[::-1]
            k = max(1, int(len(spec) * sk))
            mask = np.zeros_like(spec)
            mask[si[:k]] = spec[si[:k]]
            recon = hybrid_rft_inverse(mask, block_size, len(row))
            residual = row - recon

            y_s[ri] = recon @ x

            # int8 residual
            rm = np.max(np.abs(residual)) + 1e-10
            rq = np.clip(np.round(residual / rm * 127), -128, 127).astype(np.int8)
            y_rq8[ri] = recon @ x + (rq.astype(np.float32) * (rm/127.0)) @ x

            # int4 residual (per-32 block)
            rr4 = np.zeros_like(residual)
            for bi in range(0, len(residual), 32):
                blk = residual[bi:bi+32]
                bm = np.max(np.abs(blk)) + 1e-10
                bq = np.clip(np.round(blk / bm * 7), -8, 7).astype(np.int8)
                rr4[bi:bi+32] = bq.astype(np.float32) * (bm/7.0)
            y_rq4[ri] = recon @ x + rr4 @ x

            sb += k * 3.5 + k * 2  # spectral coeff bytes
            rb8 += in_f + 4
            rb4 += (in_f // 32) * 20

        for label, yp, extra in [
            ("Spectral only",    y_s,   0),
            ("Spectral+res_q8",  y_rq8, rb8),
            ("Spectral+res_q4",  y_rq4, rb4),
        ]:
            cos = np.dot(y_true, yp) / (np.linalg.norm(y_true) * np.linalg.norm(yp) + 1e-30)
            bpr = (sb + extra) / n_s
            ratio = bpr / q4k_bpr
            print(f"  {sk*100:12.0f}%  | {label:20s} | {cos:.6f} | {bpr:9.0f} | {ratio:.2f}x")

    # Also test: just plain Q8 (no spectral at all)
    y_q8 = np.zeros(n_s)
    for ri in range(n_s):
        row = weights[idx[ri]]
        rm = np.max(np.abs(row)) + 1e-10
        rq = np.clip(np.round(row / rm * 127), -128, 127).astype(np.int8)
        y_q8[ri] = (rq.astype(np.float32) * (rm/127.0)) @ x
    cos_q8 = np.dot(y_true, y_q8) / (np.linalg.norm(y_true)*np.linalg.norm(y_q8)+1e-30)
    print(f"  {'n/a':>12s}  | {'Plain Q8 (no RFT)':20s} | {cos_q8:.6f} | {in_f+4:9d} | {(in_f+4)/q4k_bpr:.2f}x")

# ─── Main ───

if __name__ == '__main__':
    gguf = '/workspaces/quantoniumos/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf'
    tests = [
        ('blk.0.attn_q.weight',    'Early attn_q'),
        ('blk.17.attn_q.weight',   'Mid attn_q'),
        ('blk.35.attn_q.weight',   'Late attn_q'),
        ('blk.17.ffn_gate.weight', 'Mid ffn_gate'),
    ]
    for tname, label in tests:
        try:
            print(f"\nLoading {tname} ({label})...")
            t0 = time.time()
            W = load_gguf_tensor(gguf, tname)
            print(f"  shape={W.shape} time={time.time()-t0:.1f}s")
            analyze(W, tname, block_size=256)
            test_residual(W, tname, block_size=256)
        except Exception as e:
            import traceback; traceback.print_exc()
