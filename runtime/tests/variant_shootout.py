"""All 27 RFTMW variants vs DFT on real + quasi-periodic signals."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import math
import warnings
warnings.filterwarnings('ignore')

from algorithms.rft.variants.registry import VARIANTS

PHI = (1 + math.sqrt(5)) / 2
N = 256

def dft_matrix(N):
    n = np.arange(N)
    return np.exp(-2j * np.pi * np.outer(n, n) / N) / math.sqrt(N)

def k99(coeffs):
    energy = np.abs(coeffs) ** 2
    total = np.sum(energy)
    if total < 1e-30:
        return len(coeffs)
    sorted_e = np.sort(energy)[::-1]
    cumsum = np.cumsum(sorted_e)
    idx = np.searchsorted(cumsum, 0.99 * total)
    return int(idx) + 1

# Build all variant matrices
print(f"Building {len(VARIANTS)} variant matrices at N={N}...")
variant_matrices = {}
for name, info in VARIANTS.items():
    try:
        mat = info.generator(N)
        if mat.shape == (N, N):
            variant_matrices[name] = mat
        else:
            print(f"  SKIP {name}: shape={mat.shape}")
    except Exception as e:
        print(f"  SKIP {name}: {e}")

F = dft_matrix(N)
print(f"Built {len(variant_matrices)} variants + DFT\n")

# ── Signal generators ──
rng = np.random.default_rng(42)

def gen_ecg_blocks():
    """Real ECG from scipy (MIT-BIH record 200)."""
    try:
        from scipy.datasets import electrocardiogram
    except ImportError:
        from scipy.misc import electrocardiogram
    ecg = electrocardiogram()
    blocks = []
    for b in range(min(100, len(ecg) // N)):
        block = ecg[b * N:(b + 1) * N].copy()
        block = block / (np.linalg.norm(block) + 1e-30)
        blocks.append(block)
    return blocks

def gen_quasi_sqrt2():
    """Quasi-periodic: sqrt(2) ratio tones."""
    blocks = []
    n = np.arange(N)
    for _ in range(100):
        f1 = rng.uniform(0.05, 0.45)
        f2 = (f1 * math.sqrt(2)) % 1.0
        amp = rng.uniform(0.3, 1.0, 2)
        sig = amp[0] * np.cos(2 * np.pi * f1 * n) + amp[1] * np.cos(2 * np.pi * f2 * n)
        sig = sig / np.linalg.norm(sig)
        blocks.append(sig)
    return blocks

def gen_quasi_pi():
    """Quasi-periodic: pi ratio tones."""
    blocks = []
    n = np.arange(N)
    for _ in range(100):
        f1 = rng.uniform(0.05, 0.3)
        f2 = (f1 * math.pi) % 1.0
        amp = rng.uniform(0.3, 1.0, 2)
        sig = amp[0] * np.cos(2 * np.pi * f1 * n) + amp[1] * np.cos(2 * np.pi * f2 * n)
        sig = sig / np.linalg.norm(sig)
        blocks.append(sig)
    return blocks

def gen_quasi_5tone():
    """5 random irrational-frequency tones."""
    blocks = []
    n = np.arange(N)
    for _ in range(100):
        freqs = rng.uniform(0.01, 0.99, 5)
        amps = rng.uniform(0.2, 1.0, 5)
        sig = sum(a * np.cos(2 * np.pi * f * n) for a, f in zip(amps, freqs))
        sig = sig / np.linalg.norm(sig)
        blocks.append(sig)
    return blocks

def gen_fibonacci_chain():
    """1D Fibonacci quasicrystal."""
    seq = [1]
    for _ in range(18):
        new = []
        for s in seq:
            if s == 1:
                new.extend([1, 0])
            else:
                new.append(1)
        seq = new
    chain = np.array(seq, dtype=float)
    blocks = []
    for b in range(min(100, len(chain) // N)):
        block = chain[b * N:(b + 1) * N].copy()
        block = block / (np.linalg.norm(block) + 1e-30)
        blocks.append(block)
    return blocks

def gen_bearing():
    """Real CWRU bearing vibration data."""
    import scipy.io, glob, os
    blocks = []
    mat_files = sorted(glob.glob('algorithms/rft/applications/vibration/*.mat'))[:5]
    for mf in mat_files:
        try:
            data = scipy.io.loadmat(mf)
        except Exception:
            continue
        de_key = [k for k in data.keys() if 'DE_time' in k]
        if not de_key:
            continue
        signal = data[de_key[0]].flatten()
        for b in range(min(20, len(signal) // N)):
            block = signal[b * N:(b + 1) * N].copy()
            block = block / (np.linalg.norm(block) + 1e-30)
            blocks.append(block)
    return blocks

# ── Run shootout ──
signals = {
    "ECG (MIT-BIH real)": gen_ecg_blocks,
    "Quasi-periodic sqrt(2)": gen_quasi_sqrt2,
    "Quasi-periodic pi": gen_quasi_pi,
    "Quasi-periodic 5-tone": gen_quasi_5tone,
    "Fibonacci chain (QC)": gen_fibonacci_chain,
    "Bearing vibration (real)": gen_bearing,
}

for sig_name, gen_fn in signals.items():
    print(f"{'='*70}")
    print(f"SIGNAL: {sig_name}")
    print(f"{'='*70}")

    blocks = gen_fn()
    if not blocks:
        print("  No data available, skipping.\n")
        continue

    # DFT baseline
    dft_k99s = [k99(F.conj().T @ b) for b in blocks]
    dft_mean = np.mean(dft_k99s)

    # All variants
    results = []
    for vname, mat in variant_matrices.items():
        try:
            vk99s = [k99(mat.conj().T @ b.astype(complex)) for b in blocks]
            vmean = np.mean(vk99s)
            ratio = vmean / dft_mean
            wins = sum(1 for v, d in zip(vk99s, dft_k99s) if v < d)
            results.append((vname, vmean, ratio, wins, len(blocks)))
        except Exception:
            pass

    # Sort by K_0.99 (lower is better)
    results.sort(key=lambda x: x[1])

    print(f"  DFT baseline K_0.99 = {dft_mean:.1f}")
    print(f"  {'Variant':<28s} {'K_0.99':>7s} {'Ratio':>7s} {'Wins':>10s}  {'vs DFT':>8s}")
    print(f"  {'-'*65}")

    for vname, vmean, ratio, wins, total in results[:10]:  # top 10
        tag = "** BEATS DFT **" if ratio < 1.0 else ""
        print(f"  {vname:<28s} {vmean:7.1f} {ratio:7.3f} {wins:4d}/{total:<4d}    {tag}")

    if results and results[0][2] < 1.0:
        print(f"\n  BEST: {results[0][0]} beats DFT by {(1-results[0][2])*100:.1f}%")
    else:
        print(f"\n  No variant beats DFT on this signal.")

    # Also show worst
    print(f"  (DFT rank: #{next(i+1 for i,(n,m,r,w,t) in enumerate(results))+1 if dft_mean < results[-1][1] else len(results)} of {len(results)+1})")
    print()
