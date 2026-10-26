#!/usr/bin/env python3
"""Diff two bench_rftmw result dumps: max abs / relative error per vector."""
import struct
import sys


def load(path):
    out = {}
    with open(path, "rb") as f:
        while True:
            name = f.read(32)
            if len(name) < 32:
                return out
            (n,) = struct.unpack("<Q", f.read(8))
            out[name.rstrip(b"\0").decode()] = struct.unpack(f"<{n}d", f.read(8 * n))


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
    worst = 0.0
    for name in a:
        if name not in b:
            print(f"MISSING {name}")
            worst = float("inf")
            continue
        x, y = a[name], b[name]
        scale = max(max(abs(v) for v in x), 1e-30)
        err = max(abs(p - q) for p, q in zip(x, y)) / scale
        worst = max(worst, err)
        print(f"{'ok ' if err <= tol else 'BAD'} {name:<22} rel_err {err:.2e}")
    print(f"worst rel_err {worst:.2e} (tol {tol:g})")
    sys.exit(0 if worst <= tol else 1)


if __name__ == "__main__":
    main()
