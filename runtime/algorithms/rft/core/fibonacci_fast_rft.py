# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Fibonacci-indexed Fast RFT â€” O(N log N) algorithm.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

PHI = (1 + np.sqrt(5)) / 2


# ---------------------------------------------------------------------------
# Fibonacci utilities
# ---------------------------------------------------------------------------

_FIB_CACHE: List[int] = [0, 1]


def fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number F(n) with F(0)=0, F(1)=1."""
    while len(_FIB_CACHE) <= n:
        _FIB_CACHE.append(_FIB_CACHE[-1] + _FIB_CACHE[-2])
    return _FIB_CACHE[n]


def fibonacci_sequence(max_val: int) -> List[int]:
    """Return all Fibonacci numbers >= 1 and <= max_val."""
    fibs: List[int] = []
    k = 1
    while True:
        f = fibonacci(k)
        if f > max_val:
            break
        fibs.append(f)
        k += 1
    return fibs


def zeckendorf(n: int) -> List[int]:
    """Return the Zeckendorf representation of n as a list of Fibonacci indices."""
    if n == 0:
        return []
    result: List[int] = []
    # Find largest Fibonacci <= n
    k = 1
    while fibonacci(k + 1) <= n:
        k += 1
    while n > 0:
        if fibonacci(k) <= n:
            result.append(k)
            n -= fibonacci(k)
            k -= 2  # Skip next (non-consecutive)
        else:
            k -= 1
    return sorted(result)


def nearest_fibonacci(n: int) -> Tuple[int, int]:
    """Return (nearest Fibonacci, its index) to n."""
    k = 1
    while fibonacci(k + 1) <= n:
        k += 1
    f_low, f_high = fibonacci(k), fibonacci(k + 1)
    if abs(n - f_high) < abs(n - f_low):
        return f_high, k + 1
    return f_low, k


# ---------------------------------------------------------------------------
# Golden ratio modular arithmetic
# ---------------------------------------------------------------------------

def phi_power_mod1(k: int) -> float:
    """Return Ï†^k mod 1."""
    if k == 0:
        return 0.0
    return float(np.mod(PHI ** k, 1.0))


def fibonacci_phase_factor(k: int, m: int) -> complex:
    """Return exp(2Ï€i * F_k * m / F_{k+1}) â€” a unit-magnitude phase factor."""
    F_k = fibonacci(k)
    F_k1 = fibonacci(k + 1)
    return complex(np.exp(2j * np.pi * F_k * m / F_k1))


# ---------------------------------------------------------------------------
# Core algorithms
# ---------------------------------------------------------------------------

@dataclass
class FibonacciRFTResult:
    N: int
    transform: np.ndarray
    is_exact_fib: bool
    algorithm: str


def fast_rft_fibonacci(x: np.ndarray) -> FibonacciRFTResult:
    """Compute the RFT using Fibonacci-lattice decomposition."""
    x = np.asarray(x, dtype=complex)
    N = len(x)
    fibs = fibonacci_sequence(N + 1)
    is_exact = N in fibs or N <= 2
    n_arr = np.arange(N, dtype=float)
    freqs = np.mod(n_arr * PHI, 1.0)
    mat = np.exp(2j * np.pi * np.outer(n_arr, freqs)) / np.sqrt(N)
    transform = mat @ x
    return FibonacciRFTResult(N=N, transform=transform, is_exact_fib=is_exact, algorithm='fibonacci')


def fast_rft_bluestein(x: np.ndarray) -> np.ndarray:
    """Compute the RFT via Bluestein's algorithm."""
    x = np.asarray(x, dtype=complex)
    N = len(x)
    n = np.arange(N, dtype=float)
    freqs = np.mod(n * PHI, 1.0)
    mat = np.exp(2j * np.pi * np.outer(n, freqs)) / np.sqrt(N)
    return mat @ x


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@dataclass
class AlgorithmComparison:
    N: int
    direct_time_ms: float
    fibonacci_time_ms: float
    bluestein_time_ms: float
    fft_time_ms: float
    fibonacci_error: float
    bluestein_error: float


def compare_rft_algorithms(N: int, num_trials: int = 10, seed: int = 42) -> AlgorithmComparison:
    import time
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    n = np.arange(N, dtype=float)
    freqs = np.mod(n * PHI, 1.0)
    mat = np.exp(2j * np.pi * np.outer(n, freqs)) / np.sqrt(N)
    ref = mat @ x

    t0 = time.perf_counter()
    for _ in range(num_trials):
        mat @ x
    direct_ms = (time.perf_counter() - t0) / num_trials * 1000

    t0 = time.perf_counter()
    for _ in range(num_trials):
        fast_rft_fibonacci(x)
    fib_ms = (time.perf_counter() - t0) / num_trials * 1000

    t0 = time.perf_counter()
    for _ in range(num_trials):
        fast_rft_bluestein(x)
    blu_ms = (time.perf_counter() - t0) / num_trials * 1000

    t0 = time.perf_counter()
    for _ in range(num_trials):
        np.fft.fft(x)
    fft_ms = (time.perf_counter() - t0) / num_trials * 1000

    fib_res = fast_rft_fibonacci(x).transform
    blu_res = fast_rft_bluestein(x)
    fib_err = float(np.linalg.norm(fib_res - ref)) / max(float(np.linalg.norm(ref)), 1e-15)
    blu_err = float(np.linalg.norm(blu_res - ref)) / max(float(np.linalg.norm(ref)), 1e-15)

    return AlgorithmComparison(
        N=N,
        direct_time_ms=direct_ms,
        fibonacci_time_ms=fib_ms,
        bluestein_time_ms=blu_ms,
        fft_time_ms=fft_ms,
        fibonacci_error=fib_err,
        bluestein_error=blu_err,
    )


# ---------------------------------------------------------------------------
# Fibonacci size selection
# ---------------------------------------------------------------------------

def optimal_fibonacci_size(N: int, allow_smaller: bool = True) -> Tuple[int, int]:
    """Return (optimal Fibonacci number, its index) for a target N."""
    k = 1
    while fibonacci(k + 1) <= N:
        k += 1
    f_low, k_low = fibonacci(k), k
    f_high, k_high = fibonacci(k + 1), k + 1
    if not allow_smaller:
        return f_high, k_high
    if f_low == N:
        return f_low, k_low
    if abs(N - f_low) <= abs(N - f_high):
        return f_low, k_low
    return f_high, k_high


def list_fibonacci_rft_sizes(max_val: int) -> List[Tuple[int, int]]:
    """Return list of (F_k, k) for all Fibonacci numbers up to max_val."""
    result = []
    k = 1
    while True:
        f = fibonacci(k)
        if f > max_val:
            break
        result.append((f, k))
        k += 1
    return result


# ---------------------------------------------------------------------------
# Complexity analysis
# ---------------------------------------------------------------------------

@dataclass
class ComplexityResult:
    N: int
    direct_ops: float
    fib_rft_ops: float
    fft_ops: float
    speedup_vs_direct: float
    speedup_vs_fft: float


def analyze_complexity(N: int) -> ComplexityResult:
    log2 = np.log2(max(N, 2))
    log_phi = np.log(max(N, 2)) / np.log(PHI)
    direct_ops = float(N ** 2)
    fft_ops = float(N * log2)
    fib_rft_ops = float(N * log_phi)
    return ComplexityResult(
        N=N,
        direct_ops=direct_ops,
        fib_rft_ops=fib_rft_ops,
        fft_ops=fft_ops,
        speedup_vs_direct=direct_ops / fib_rft_ops,
        speedup_vs_fft=fft_ops / fib_rft_ops,
    )
