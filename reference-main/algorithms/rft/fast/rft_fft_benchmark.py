# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
"""
rft_fft_benchmark.py
=====================
Benchmark: Final Fast RFT Algorithm (Tier-1 exact) vs NumPy FFT

Tests reconstruction error, runtime, and time ratio across N = 64, 128, 256.
Precomputation (offline Lowdin step) is excluded from timing per protocol.

Status: WORK IN PROGRESS — see FAST_RFT_HYPOTHESIS.md
Patent: USPTO 19/169,399
Author: Luis M. Minier
Date:   March 2026
"""

import time
import numpy as np

PHI = (1 + np.sqrt(5)) / 2
N_TRIALS = 20
N_WARMUP = 3


# ── RFT core (Tier-1 exact) ───────────────────────────────────────────────────

def phi_frequencies(N: int) -> np.ndarray:
    """Golden-ratio irrational frequency lattice: omega_k = frac((k+1)*phi)."""
    k = np.arange(N)
    return np.modf((k + 1) * PHI)[0]  # fractional part only


def build_synthesis_matrix(N: int) -> np.ndarray:
    """Raw synthesis matrix Phi in C^{N x N} on phi-frequency lattice."""
    omega = phi_frequencies(N)          # shape (N,)
    n     = np.arange(N)                # shape (N,)
    return np.exp(2j * np.pi * np.outer(n, omega)) / np.sqrt(N)


def lowdin_orthogonalize(Phi: np.ndarray):
    """
    Offline Lowdin step: Phi_tilde = Phi (Phi^dag Phi)^{-1/2}
    Returns (Phi_tilde, S, cond_S) where S = Phi^dag Phi.
    """
    S        = Phi.conj().T @ Phi                       # Gram matrix
    eigvals, eigvecs = np.linalg.eigh(S)
    S_invsqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.conj().T
    Phi_tilde = Phi @ S_invsqrt
    cond_S    = eigvals.max() / eigvals.min()
    return Phi_tilde, S, cond_S


def rft_forward(x: np.ndarray, Phi_tilde: np.ndarray) -> np.ndarray:
    """Forward RFT: x_hat = Phi_tilde^dag x"""
    return Phi_tilde.conj().T @ x


def rft_inverse(x_hat: np.ndarray, Phi_tilde: np.ndarray) -> np.ndarray:
    """Inverse RFT: x = Phi_tilde x_hat"""
    return Phi_tilde @ x_hat


# ── Benchmark runners ─────────────────────────────────────────────────────────

def bench_rft(x: np.ndarray, Phi_tilde: np.ndarray, n_trials: int = N_TRIALS):
    """Time forward+inverse RFT (precomputation excluded)."""
    for _ in range(N_WARMUP):
        rft_inverse(rft_forward(x, Phi_tilde), Phi_tilde)
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        x_hat = rft_forward(x, Phi_tilde)
        x_rec = rft_inverse(x_hat, Phi_tilde)
        times.append(time.perf_counter() - t0)
    err = np.linalg.norm(x - x_rec)
    return err, np.mean(times) * 1e3, np.std(times) * 1e3  # ms


def bench_fft(x: np.ndarray, n_trials: int = N_TRIALS):
    """Time forward+inverse NumPy FFT."""
    for _ in range(N_WARMUP):
        np.fft.ifft(np.fft.fft(x))
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        x_hat = np.fft.fft(x)
        x_rec = np.fft.ifft(x_hat)
        times.append(time.perf_counter() - t0)
    err = np.linalg.norm(x - x_rec)
    return err, np.mean(times) * 1e3, np.std(times) * 1e3  # ms


# ── Unitarity check ───────────────────────────────────────────────────────────

def check_tight_frame(Phi_tilde: np.ndarray) -> float:
    """Return ||Phi_tilde^dag Phi_tilde - I||_F (should be ~machine epsilon)."""
    N = Phi_tilde.shape[1]
    return np.linalg.norm(Phi_tilde.conj().T @ Phi_tilde - np.eye(N))


# ── Main ──────────────────────────────────────────────────────────────────────

def run_benchmark(dimensions=(64, 128, 256), seed=42):
    rng = np.random.default_rng(seed)

    print('=' * 78)
    print('RFT vs FFT BENCHMARK — Final Fast RFT Algorithm (Tier-1 Exact)')
    print('Status: WORK IN PROGRESS HYPOTHESIS — see FAST_RFT_HYPOTHESIS.md')
    print(f'Trials: {N_TRIALS} | Warmup: {N_WARMUP} | Precision: float64')
    print('=' * 78)

    # Unitarity check
    print()
    print('UNITARITY CHECK  ||Phi_tilde^dag Phi_tilde - I||_F')
    print(f"  {'N':>5} | {'Frame error':>14} | {'Cond(S)':>12}")
    print('-' * 40)
    precomputed = {}
    for N in dimensions:
        Phi           = build_synthesis_matrix(N)
        Phi_tilde, S, cond_S = lowdin_orthogonalize(Phi)
        frame_err     = check_tight_frame(Phi_tilde)
        precomputed[N] = Phi_tilde
        print(f"  {N:>5} | {frame_err:14.3e} | {cond_S:12.2f}")

    # Reconstruction + timing
    print()
    print('RECONSTRUCTION ERROR AND RUNTIME')
    hdr = (f"  {'N':>5} | {'RFT err':>12} | {'FFT err':>12} | "
           f"{'RFT ms':>8} | {'FFT ms':>8} | "
           f"{'RFT std':>8} | {'FFT std':>8} | {'Ratio':>7}")
    print(hdr)
    print('-' * len(hdr))

    results = []
    for N in dimensions:
        x         = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        x         = x / np.linalg.norm(x)          # unit norm
        Phi_tilde = precomputed[N]

        rft_err, rft_t, rft_s = bench_rft(x, Phi_tilde)
        fft_err, fft_t, fft_s = bench_fft(x)
        ratio = rft_t / fft_t if fft_t > 0 else float('inf')

        print(f"  {N:>5} | {rft_err:12.3e} | {fft_err:12.3e} | "
              f"{rft_t:8.3f} | {fft_t:8.3f} | "
              f"{rft_s:8.3f} | {fft_s:8.3f} | {ratio:7.2f}")
        results.append(dict(N=N, rft_err=rft_err, fft_err=fft_err,
                            rft_t=rft_t, fft_t=fft_t,
                            rft_s=rft_s, fft_s=fft_s, ratio=ratio))

    print()
    print('INTERPRETATION')
    for r in results:
        tag = 'FASTER' if r['ratio'] < 1.0 else ('COMPARABLE' if r['ratio'] < 1.5 else 'SLOWER')
        print(f"  N={r['N']:>3}: RFT recon err {r['rft_err']:.2e} | "
              f"ratio {r['ratio']:.2f}x [{tag}]")
    print()
    print('NOTE: O(N^2) Tier-1 is expected to exceed FFT at N>=256.')
    print('      Tier-2 (NUFFT/Toeplitz) and Tier-3 (Fibonacci-recursive)')
    print('      accelerations are planned to close this gap.')
    print()
    print('See FAST_RFT_HYPOTHESIS.md for full theoretical context.')
    print('=' * 78)
    return results


if __name__ == '__main__':
    run_benchmark()
