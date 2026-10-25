"""Test RFT vs DFT on real quasicrystal and aperiodic structures."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import math

PHI = (1 + math.sqrt(5)) / 2

def canonical_U(N):
    k = np.arange(N)
    freqs = np.mod((k + 1) * PHI, 1.0)
    n = np.arange(N)
    Phi = np.exp(1j * 2 * np.pi * np.outer(n, freqs)) / math.sqrt(N)
    G = Phi.conj().T @ Phi
    eigvals, eigvecs = np.linalg.eigh(G)
    eigvals = np.maximum(eigvals, 1e-14)
    G_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.conj().T
    return Phi @ G_inv_sqrt

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

N = 256
U = canonical_U(N)
F = dft_matrix(N)
rng = np.random.default_rng(42)

print("QUASICRYSTAL & APERIODIC STRUCTURE TESTS")
print("Real constructions -- no synthetic S_phi(K) signals")
print("=" * 70)

# ── 1. FIBONACCI CHAIN (1D Quasicrystal) ──
print("\n--- 1. Fibonacci Chain (1D quasicrystal, substitution rule) ---")
print("    L->LS, S->L. The canonical 1D quasicrystal.")

def fibonacci_chain(generations):
    seq = [1]
    for _ in range(generations):
        new = []
        for s in seq:
            if s == 1:
                new.extend([1, 0])
            else:
                new.append(1)
        seq = new
    return np.array(seq, dtype=float)

rft_all, dft_all = [], []
chain = fibonacci_chain(18)
n_blocks = min(200, len(chain) // N)
for b in range(n_blocks):
    block = chain[b * N:(b + 1) * N].copy()
    block = block / (np.linalg.norm(block) + 1e-30)
    rft_all.append(k99(U.conj().T @ block))
    dft_all.append(k99(F.conj().T @ block))

rft_m, dft_m = np.mean(rft_all), np.mean(dft_all)
rft_bw = sum(1 for r, d in zip(rft_all, dft_all) if r < d)
print(f"  Blocks: {len(rft_all)}")
print(f"  RFT K_0.99 = {rft_m:.1f}  DFT K_0.99 = {dft_m:.1f}  ratio={rft_m / dft_m:.3f}")
print(f"  Block wins: RFT={rft_bw}  DFT={len(rft_all) - rft_bw}")
print(f"  --> {'RFT WINS' if rft_m < dft_m else 'DFT WINS'}")

# ── 2. FIBONACCI DIFFRACTION ──
print("\n--- 2. Fibonacci chain DIFFRACTION (Bragg peaks at m+n*phi) ---")
print("    Atom positions: cumulative L=phi, S=1 gaps.")

positions = []
pos = 0.0
for s in chain[:5000]:
    pos += PHI if s == 1 else 1.0
    positions.append(pos)
positions = np.array(positions)

rft_all, dft_all = [], []
for trial in range(100):
    q_start = rng.uniform(0, 50)
    q = np.linspace(q_start, q_start + N * 0.1, N)
    # Structure factor
    pos_sub = positions[:1000]
    sf = np.abs(np.sum(np.exp(1j * np.outer(q, pos_sub)), axis=1)) ** 2
    sf = sf / (np.linalg.norm(sf) + 1e-30)
    rft_all.append(k99(U.conj().T @ sf))
    dft_all.append(k99(F.conj().T @ sf))

rft_m, dft_m = np.mean(rft_all), np.mean(dft_all)
rft_bw = sum(1 for r, d in zip(rft_all, dft_all) if r < d)
print(f"  RFT K_0.99 = {rft_m:.1f}  DFT K_0.99 = {dft_m:.1f}  ratio={rft_m / dft_m:.3f}")
print(f"  Block wins: RFT={rft_bw}/100  DFT={100 - rft_bw}/100")
print(f"  --> {'RFT WINS' if rft_m < dft_m else 'DFT WINS'}")

# ── 3. PENROSE 1D CUT-AND-PROJECT ──
print("\n--- 3. Penrose 1D projection (cut-and-project from Z^2) ---")
print("    Slope = 1/phi, strip width = 1 + 1/phi")

strip_width = 1.0 + 1.0 / PHI
points_1d = []
for m in range(-300, 300):
    for nv in range(-300, 300):
        para = m + nv * PHI
        perp = m * PHI - nv
        if abs(perp) < strip_width:
            points_1d.append(para)
points_1d = np.sort(points_1d)
diffs = np.diff(points_1d)
diffs = diffs[(diffs > 0.01) & (diffs < 10.0)]

rft_all, dft_all = [], []
n_blocks = min(100, len(diffs) // N)
for b in range(n_blocks):
    block = diffs[b * N:(b + 1) * N].copy()
    block = block / (np.linalg.norm(block) + 1e-30)
    rft_all.append(k99(U.conj().T @ block))
    dft_all.append(k99(F.conj().T @ block))

if rft_all:
    rft_m, dft_m = np.mean(rft_all), np.mean(dft_all)
    rft_bw = sum(1 for r, d in zip(rft_all, dft_all) if r < d)
    unique_gaps = len(np.unique(np.round(diffs, 6)))
    print(f"  Unique gap sizes: {unique_gaps} (Three-Distance Theorem predicts 2-3)")
    print(f"  RFT K_0.99 = {rft_m:.1f}  DFT K_0.99 = {dft_m:.1f}  ratio={rft_m / dft_m:.3f}")
    print(f"  Block wins: RFT={rft_bw}/{len(rft_all)}  DFT={len(rft_all) - rft_bw}/{len(rft_all)}")
    print(f"  --> {'RFT WINS' if rft_m < dft_m else 'DFT WINS'}")

# ── 4. SILVER RATIO QUASICRYSTAL (NOT phi-based) ──
print("\n--- 4. Silver-ratio quasicrystal (1+sqrt(2), NOT phi) ---")
print("    Tests if RFT advantage is phi-specific or general.")

SILVER = 1 + math.sqrt(2)
silver_points = []
pos = 0.0
for k in range(10000):
    gap = SILVER if (int(k * (SILVER - 1)) % 2 == 0) else 1.0
    pos += gap
    silver_points.append(pos)
silver_diffs = np.diff(silver_points)

rft_all, dft_all = [], []
n_blocks = min(100, len(silver_diffs) // N)
for b in range(n_blocks):
    block = silver_diffs[b * N:(b + 1) * N].copy()
    block = block / (np.linalg.norm(block) + 1e-30)
    rft_all.append(k99(U.conj().T @ block))
    dft_all.append(k99(F.conj().T @ block))

rft_m, dft_m = np.mean(rft_all), np.mean(dft_all)
rft_bw = sum(1 for r, d in zip(rft_all, dft_all) if r < d)
print(f"  RFT K_0.99 = {rft_m:.1f}  DFT K_0.99 = {dft_m:.1f}  ratio={rft_m / dft_m:.3f}")
print(f"  Block wins: RFT={rft_bw}/{len(rft_all)}  DFT={len(rft_all) - rft_bw}/{len(rft_all)}")
print(f"  --> {'RFT WINS' if rft_m < dft_m else 'DFT WINS'}")

# ── 5. SUNFLOWER PHYLLOTAXIS ──
print("\n--- 5. Sunflower phyllotaxis (golden-angle seed distribution) ---")
print("    Vogel model: theta_n = n * golden_angle, r_n = sqrt(n)")

golden_angle = 2 * math.pi * (1 - 1 / PHI)

rft_all, dft_all = [], []
for trial in range(100):
    n_seeds = rng.integers(500, 2000)
    angles = np.arange(n_seeds) * golden_angle
    hist, _ = np.histogram(angles % (2 * np.pi), bins=N, range=(0, 2 * np.pi))
    sig = hist.astype(float)
    sig = sig / (np.linalg.norm(sig) + 1e-30)
    rft_all.append(k99(U.conj().T @ sig))
    dft_all.append(k99(F.conj().T @ sig))

rft_m, dft_m = np.mean(rft_all), np.mean(dft_all)
rft_bw = sum(1 for r, d in zip(rft_all, dft_all) if r < d)
print(f"  RFT K_0.99 = {rft_m:.1f}  DFT K_0.99 = {dft_m:.1f}  ratio={rft_m / dft_m:.3f}")
print(f"  Block wins: RFT={rft_bw}/100  DFT={100 - rft_bw}/100")
print(f"  --> {'RFT WINS' if rft_m < dft_m else 'DFT WINS'}")

# ── 6. THUE-MORSE SEQUENCE (aperiodic, NOT quasiperiodic) ──
print("\n--- 6. Thue-Morse sequence (aperiodic, singular continuous spectrum) ---")
print("    Neither periodic nor quasiperiodic. Continuous diffraction.")

def thue_morse(n):
    return np.array([bin(i).count('1') % 2 for i in range(n)], dtype=float)

tm = thue_morse(N * 100)
rft_all, dft_all = [], []
n_blocks = min(100, len(tm) // N)
for b in range(n_blocks):
    block = tm[b * N:(b + 1) * N].copy()
    block = block / (np.linalg.norm(block) + 1e-30)
    rft_all.append(k99(U.conj().T @ block))
    dft_all.append(k99(F.conj().T @ block))

rft_m, dft_m = np.mean(rft_all), np.mean(dft_all)
rft_bw = sum(1 for r, d in zip(rft_all, dft_all) if r < d)
print(f"  RFT K_0.99 = {rft_m:.1f}  DFT K_0.99 = {dft_m:.1f}  ratio={rft_m / dft_m:.3f}")
print(f"  Block wins: RFT={rft_bw}/{len(rft_all)}  DFT={len(rft_all) - rft_bw}/{len(rft_all)}")
print(f"  --> {'RFT WINS' if rft_m < dft_m else 'DFT WINS'}")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
