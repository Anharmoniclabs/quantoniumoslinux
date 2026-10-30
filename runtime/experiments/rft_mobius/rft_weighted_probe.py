#!/usr/bin/env python3
"""EXPERIMENTAL: finite raw RFT coefficient and interaction measurements.

All coefficients are inspected at Fibonacci dimensions <= --limit.
Completed frames are saved and reused. No finite run proves the hypothesis.
Dense inversion uses O(q^3) time and O(q^2) memory.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = os.environ.get("RFT_THREADS", "1")

import argparse
import csv
import fcntl
import hashlib
import json
import time
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rft_fibonacci_prefix import fibonacci_frames, mobius_sieve


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def measure_frame(q, mu, random_signs):
    started = time.perf_counter()
    with localcontext() as context:
        context.prec = 70
        alpha = 2 / (1 + Decimal(5).sqrt())
        frequencies = np.array([float((k * alpha) % 1) for k in range(1, q + 1)])
    raw = np.exp(2j * np.pi * np.outer(np.arange(q), frequencies)) / np.sqrt(q)
    inverse = np.linalg.solve(raw, np.eye(q, dtype=complex))
    signed = mu[1:q+1]
    c = inverse @ signed
    magnitude = np.abs(c)
    diagonal = np.abs(inverse)**2 @ (signed**2)
    interaction = magnitude**2 - diagonal
    largest = int(np.argmax(magnitude))
    largest_interaction = int(np.argmax(interaction))
    relative_residual = float(np.linalg.norm(raw @ c - signed) / np.linalg.norm(signed))

    # Independent row and cross-term checks at extrema and boundary columns.
    checked = np.unique([0, q - 1, largest, largest_interaction])
    expected_rows = np.zeros((len(checked), q), dtype=complex)
    expected_rows[np.arange(len(checked)), checked] = 1
    inverse_error = float(np.max(np.abs(inverse[checked] @ raw - expected_rows)))
    terms = inverse[checked] * signed
    previous = np.cumsum(terms, axis=1) - terms
    direct_cross = 2 * np.real(np.sum(terms * np.conj(previous), axis=1))
    cross_error = float(np.max(np.abs(direct_cross - interaction[checked])))

    # Same squarefree support with random signs: a comparison, not a theorem.
    random_c = inverse @ (np.abs(signed) * random_signs[:q])
    # A bounded pure wave has one coefficient sqrt(q): catches normalization.
    phase_c = inverse @ (raw[:, 0] * np.sqrt(q))
    phase_target = np.zeros(q, dtype=complex)
    phase_target[0] = np.sqrt(q)
    phase_error = float(np.max(np.abs(phase_c - phase_target)))
    if (relative_residual > 1e-9 or inverse_error > 1e-8
            or cross_error > 1e-8 * max(1.0, float(np.max(magnitude**2)))
            or phase_error > 1e-8 * np.sqrt(q)
            or not np.all(np.isfinite(c))):
        raise ArithmeticError(f"Numerical checks failed at q={q}; frame not checkpointed")

    row = dict(q=q, max_coefficient=float(magnitude[largest]), largest_k=largest+1,
               coefficient_rms=float(np.linalg.norm(c) / np.sqrt(q)),
               max_interaction=float(interaction[largest_interaction]),
               interaction_k=largest_interaction+1,
               diagonal_at_largest=float(diagonal[largest]),
               interaction_at_largest=float(interaction[largest]),
               random_max=float(np.max(np.abs(random_c))),
               phase_control_max=float(np.max(np.abs(phase_c))),
               relative_residual=relative_residual, inverse_row_error=inverse_error,
               cross_identity_error=cross_error, phase_control_error=phase_error,
               seconds=time.perf_counter()-started, numerical_status="OK")
    arrays = dict(q=np.array(q), coefficients=c, diagonal=diagonal,
                  interactions=interaction, random_coefficients=random_c)
    return row, arrays


def report(output, rows, frames, epsilons):
    if not rows:
        return
    qs = np.array([r["q"] for r in rows])
    maxima = np.array([r["max_coefficient"] for r in rows])
    interactions = np.array([r["max_interaction"] for r in rows])
    constants = {}
    for eps in epsilons:
        values = maxima / qs**eps
        index = int(np.argmax(values))
        rvalues = np.maximum(interactions, 0) / qs**(2*eps)
        rindex = int(np.argmax(rvalues))
        constants[str(eps)] = dict(coefficient_constant=float(values[index]),
                                  coefficient_at_q=int(qs[index]),
                                  positive_interaction_constant=float(rvalues[rindex]),
                                  interaction_at_q=int(qs[rindex]))
    save_json(output / "summary.json", dict(
        complete=len(rows)==len(frames), requested_frames=frames,
        completed_frames=qs.tolist(), coefficients_inspected=int(np.sum(qs)),
        finite_constants=constants, measurements=rows,
        scope="Finite floating-point estimates at Fibonacci dimensions only; not an RH/GRH proof."))
    temporary = output / "frames.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / "frames.csv")
    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    axes[0].loglog(qs, maxima, "o-", label="Mobius: largest raw coefficient")
    axes[0].loglog(qs, [r["random_max"] for r in rows], "o-", label="Random signs, same support")
    axes[0].loglog(qs, [r["phase_control_max"] for r in rows], "--", label="Pure-wave control: sqrt(q)")
    axes[0].set_ylabel("Coefficient magnitude")
    axes[0].set_title("Raw Fibonacci-frame coefficients: finite measurements")
    axes[1].semilogx(qs, interactions, "o-", color="#ff8844", label="Largest signed interaction R")
    axes[1].set_ylabel("max R(q,k)")
    for eps in epsilons:
        axes[2].semilogx(qs, np.maximum.accumulate(maxima / qs**eps), label=f"epsilon={eps:g}")
    axes[2].set_ylabel("Running max of |c| / q^epsilon")
    axes[2].set_title("Finite records; a flat segment does not prove an infinite bound")
    for ax in axes:
        ax.set_xlabel("Fibonacci frame dimension q")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(output / "weighted_probe.png", dpi=160)
    plt.close(fig)
    for eps, item in constants.items():
        print(f"Finite C(epsilon={eps}): {item['coefficient_constant']:.8f} "
              f"at q={item['coefficient_at_q']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=2584, help="Largest allowed frame dimension.")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "weighted_probe")
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--eps", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    args = parser.parse_args()
    if args.limit < 1 or args.seed < 0 or any(not np.isfinite(e) or e <= 0 for e in args.eps):
        parser.error("limit must be positive, seed nonnegative, and eps values finite and positive")
    frames = [q for q in fibonacci_frames(args.limit) if q <= args.limit]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".probe.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another probe is already writing to this output directory")
        config = dict(version=1, seed=args.seed, source_sha256=digest(Path(__file__)),
                      mobius_source_sha256=digest(Path(__file__).with_name("rft_fibonacci_prefix.py")))
        manifest = output / "configuration.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != config:
                parser.error("Saved results use different code or seed; choose a new --output directory")
        else:
            save_json(manifest, config)
        mu = mobius_sieve(frames[-1])
        check_n = min(frames[-1], 5000)
        divisor = np.zeros(check_n+1, dtype=np.int64)
        for k in range(1, check_n+1):
            divisor[k::k] += mu[k]
        if divisor[1] != 1 or np.any(divisor[2:]):
            raise ArithmeticError("Mobius divisor identity failed")
        random_signs = np.random.default_rng(args.seed).choice([-1, 1], size=frames[-1])
        print(f"EXPERIMENTAL: all coefficients at {len(frames)} Fibonacci dimensions through {frames[-1]:,}")
        print(f"Output: {output}\nCompleted frames resume automatically. Dense inversion is O(q^3).", flush=True)
        print("q      max |c|      max R       RMS     random max  residual       state", flush=True)
        rows = []
        try:
            for q in frames:
                metadata = output / f"frame_{q:08d}.json"
                data_path = output / f"frame_{q:08d}.npz"
                if metadata.exists():
                    row = json.loads(metadata.read_text())
                    if row["q"] != q or not data_path.exists() or digest(data_path) != row["arrays_sha256"]:
                        raise ValueError(f"Incomplete or changed checkpoint: {metadata}")
                    state = "RESUMED"
                else:
                    print(f"Computing q={q:,} ...", flush=True)
                    row, arrays = measure_frame(q, mu, random_signs)
                    temporary = data_path.with_suffix(".npz.tmp")
                    with temporary.open("wb") as stream:
                        np.savez_compressed(stream, **arrays)
                    temporary.replace(data_path)
                    row["arrays_sha256"] = digest(data_path)
                    save_json(metadata, row)
                    state = "SAVED"
                rows.append(row)
                save_json(output / "progress.json", dict(completed_frames=[r["q"] for r in rows],
                                                         requested_frames=frames, last_frame=q))
                print(f"{q:<6} {row['max_coefficient']:>10.6f} {row['max_interaction']:>10.6f} "
                      f"{row['coefficient_rms']:>8.5f} {row['random_max']:>11.6f} "
                      f"{row['relative_residual']:.2e}  {state}", flush=True)
        except KeyboardInterrupt:
            print("Interrupted. Completed frames are saved; rerun the same command to resume.", flush=True)
            raise SystemExit(130)
        finally:
            report(output, rows, frames, args.eps)
        print(f"Chart: {output / 'weighted_probe.png'}")
        print("FINITE TEST ONLY: constants cover tested frames; this does not prove or disprove RH/GRH.")


if __name__ == "__main__":
    main()
