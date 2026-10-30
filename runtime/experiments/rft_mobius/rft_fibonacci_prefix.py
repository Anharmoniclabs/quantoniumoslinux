#!/usr/bin/env python3
"""Finite RFT prefix-bound experiment; no RH proof or disproof.

Each integer m uses the smallest Fibonacci frame F >= m. In that frame,
p_m is the prefix indicator, b=U_F^* mu_F, and a_m=U_F^* p_m.
M(m)=a_m^* b and the exact-arithmetic upper bound is sum_k |a_m,k b_k|.
This program estimates that bound in floating-point arithmetic.
Dense SVD costs O(F^3) time and O(F^2) memory per frame.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = os.environ.get("RFT_THREADS", "1")

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def mobius_sieve(n):
    mu = np.ones(n + 1, dtype=np.int64)
    mu[0] = 0
    prime = np.ones(n + 1, dtype=bool)
    prime[:2] = False
    for p in range(2, n + 1):
        if prime[p]:
            mu[p::p] *= -1
            mu[p*p::p*p] = 0
            prime[p*p::p] = False
    return mu


def fibonacci_frames(limit):
    frames = [1]
    a, b = 1, 2
    while frames[-1] < limit:
        frames.append(b)
        a, b = b, a + b
    return frames


def prefix_frame(F, mu, first, last):
    # Reduce the irrational phases in extended precision where available.
    phi = (np.longdouble(1) + np.sqrt(np.longdouble(5))) / 2
    frequencies = np.asarray(np.remainder(
        (np.arange(F, dtype=np.longdouble) + 1) * phi, 1
    ), dtype=np.float64)
    raw = np.exp(2j * np.pi * np.outer(np.arange(F), frequencies)) / np.sqrt(F)
    left, singular, right = np.linalg.svd(raw, full_matrices=False)
    U = left @ right  # Canonical polar factor Phi (Phi* Phi)^(-1/2).
    del raw, left, right
    unitary_error = float(np.linalg.norm(U.conj().T @ U - np.eye(F), "fro") / np.sqrt(F))
    b = U.conj().T @ mu[1:F+1]
    constant_b = U.conj().T @ np.ones(F)
    # Row m-1 becomes a_m^*: no extra conjugation is needed below.
    np.cumsum(U, axis=0, out=U)
    prefixes = U[first-1:last]
    recovered = prefixes @ b
    constant_recovered = prefixes @ constant_b
    B = np.abs(prefixes) @ np.abs(b)
    ms = np.arange(first, last + 1)
    exact = np.cumsum(mu, dtype=np.int64)[ms]
    errors = np.abs(recovered - exact)
    cs = np.sqrt(ms * int(np.sum(mu[1:F+1]**2)))
    error = float(np.max(errors))
    control_error = float(np.max(np.abs(constant_recovered - ms)))
    if (unitary_error > 1e-9 or error > 1e-9 * F or control_error > 1e-9 * F
            or np.any(B + 1e-9 * cs < np.abs(exact))
            or np.any(B > cs * (1 + 1e-9))):
        raise ArithmeticError(f"Numerical consistency check failed for frame {F}")
    condition = float((singular[0] / singular[-1])**2)
    status = "CHECK_PRECISION" if singular[-1] / singular[0] < 1e-8 else "OK"
    worst = int(np.argmax(B / np.sqrt(ms)))
    frame = dict(frame=F, first=first, last=last,
                 max_B_over_sqrt_m=float(B[worst] / np.sqrt(ms[worst])),
                 worst_cutoff=int(ms[worst]), gram_condition_estimate=condition,
                 max_identity_error=error, unitary_error=unitary_error,
                 constant_control_error=control_error, precision_status=status)
    return ms, exact, B, errors, frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1597,
                        help="Check every integer cutoff through this limit.")
    parser.add_argument("--output-root", type=Path,
                        default=Path(__file__).resolve().parent / "results",
                        help="Parent folder for unique result directories.")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    frames = fibonacci_frames(args.limit)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = (args.output_root / f"rft_fibonacci_prefix_{stamp}").resolve()
    output.mkdir(parents=True)
    started = time.perf_counter()
    mu = mobius_sieve(frames[-1])
    check_n = min(frames[-1], 5000)
    divisor = np.zeros(check_n + 1, dtype=np.int64)
    for d in range(1, check_n + 1):
        divisor[d::d] += mu[d]
    if divisor[1] != 1 or np.any(divisor[2:]):
        raise ArithmeticError("Exact Mobius divisor identity failed")
    print(f"Exact Mobius divisor check: PASS through {check_n:,}", flush=True)
    print(f"Every cutoff 1..{args.limit:,}; largest frame {frames[-1]:,}", flush=True)
    print(f"Output: {output}", flush=True)
    print("Frame   Cutoff range   max B/sqrt(m)   at m    cond(G)   max identity error", flush=True)
    bounds = np.empty(args.limit)
    assigned_frames = np.zeros(args.limit, dtype=np.int64)
    records = []
    previous = 0
    with (output / "prefixes.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["m", "frame", "M", "B_estimate", "B_over_sqrt_m", "identity_error"])
        for F in frames:
            last = min(F, args.limit)
            ms, exact, B, errors, record = prefix_frame(F, mu, previous + 1, last)
            bounds[previous:last] = B
            assigned_frames[previous:last] = F
            records.append(record)
            writer.writerows(zip(ms, [F]*len(ms), exact, B, B/np.sqrt(ms), errors))
            stream.flush()
            (output / "progress.json").write_text(json.dumps(dict(
                completed_through=last, requested_limit=args.limit, frames=records
            ), indent=2) + "\n")
            print(f"{F:<7} {previous+1:>5}..{last:<5} "
                  f"{record['max_B_over_sqrt_m']:>12.8f} "
                  f"{record['worst_cutoff']:>6} {record['gram_condition_estimate']:>10.4f} "
                  f"{record['max_identity_error']:.3e}", flush=True)
            previous = last
    if previous != args.limit or np.any(assigned_frames == 0):
        raise ArithmeticError("Coverage check failed")
    ms = np.arange(1, args.limit + 1)
    mertens = np.cumsum(mu, dtype=np.int64)[1:args.limit+1]
    constants = {}
    for eps in (0.01, 0.05, 0.10):
        values = bounds / ms**(0.5 + eps)
        index = int(np.argmax(values))
        constants[str(eps)] = dict(value=float(values[index]), cutoff=int(ms[index]))
        print(f"Finite C(epsilon={eps}): {values[index]:.8f} at m={ms[index]}")
    summary = dict(complete=True, limit=args.limit, cutoffs_checked=len(ms),
                   largest_frame=frames[-1], finite_constants=constants,
                   max_identity_error=max(r["max_identity_error"] for r in records),
                   frame_diagnostics=records,
                   extended_phase_epsilon=float(np.finfo(np.longdouble).eps),
                   elapsed_seconds=time.perf_counter() - started,
                   scope="Finite floating-point estimates; not rigorous interval bounds or an RH proof.")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axes[0].semilogx(ms, bounds / np.sqrt(ms), color="#ff8844", lw=1, label="Prefix bound / sqrt(m)")
    axes[0].semilogx(ms, np.abs(mertens) / np.sqrt(ms), color="#36dca3", lw=1, label="|M(m)| / sqrt(m)")
    axes[0].set_title(f"Fibonacci RFT frames: every integer cutoff 1 through {args.limit:,}")
    axes[0].set_ylabel("Normalized magnitude")
    for eps in (0.01, 0.05, 0.10):
        axes[1].semilogx(ms, np.maximum.accumulate(bounds / ms**(0.5 + eps)), label=f"epsilon = {eps}")
    axes[1].set_title("Running maxima over the finite tested range")
    axes[1].set_ylabel("max B(m) / m^(1/2 + epsilon)")
    for ax in axes:
        ax.set_xlabel("Integer cutoff m")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(output / "fibonacci_prefix.png", dpi=180)
    plt.close(fig)
    print(f"Coverage: PASS, every cutoff 1..{args.limit:,}")
    print(f"Output: {output}")
    print("FINITE TEST ONLY: these results do not prove or disprove RH.")


if __name__ == "__main__":
    main()
