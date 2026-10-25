#!/usr/bin/env python3
"""
Canonical RFT matrix diagnostics (Phase 1 + Phase 2)

Phase 1: build the canonical (Gram-normalized) φ-grid basis U and plot |U|.
Phase 2: take off-diagonal blocks and inspect singular value decay.

Notes:
- Building U uses Gram normalization (eigendecomposition) and is O(N^3).
- N=512 is feasible for iteration on CPU; N=2048 is typically too slow/heavy.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from algorithms.rft.core.resonant_fourier_transform import rft_basis_matrix


@dataclass(frozen=True)
class BlockSVD:
    r0: int
    r1: int
    c0: int
    c1: int
    s: np.ndarray


def _parse_block(spec: str) -> Tuple[int, int, int, int]:
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Bad block spec: {spec}. Expected r0,r1,c0,c1")
    r0, r1, c0, c1 = (int(x) for x in parts)
    return r0, r1, c0, c1


def _default_blocks(n: int, b: int) -> Iterable[Tuple[int, int, int, int]]:
    if b <= 0 or b > n:
        return []
    # A few representative off-diagonal blocks.
    # (Top-left vs far-right) and (mid vs far).
    blocks = []
    blocks.append((0, b, n - b, n))
    if n >= 2 * b:
        blocks.append((0, b, b, 2 * b))
        blocks.append((b, 2 * b, 0, b))
    if n >= 4 * b:
        blocks.append((0, b, 3 * b, 4 * b))
        blocks.append((2 * b, 3 * b, 0, b))
    return blocks


def _svd_block(U: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> BlockSVD:
    B = U[r0:r1, c0:c1]
    s = np.linalg.svd(B, compute_uv=False)
    return BlockSVD(r0=r0, r1=r1, c0=c0, c1=c1, s=s)


def _effective_rank(s: np.ndarray, energy: float = 0.99) -> int:
    # Frobenius energy: sum(s^2)
    e = np.cumsum(s * s)
    total = e[-1]
    if total <= 0:
        return 0
    k = int(np.searchsorted(e, energy * total) + 1)
    return k


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512, help="Matrix size N (default: 512)")
    parser.add_argument("--outdir", default=os.path.join("results", "matrix_diagnostics"))
    parser.add_argument("--block", type=int, default=128, help="Block size for default SVD blocks (default: 128)")
    parser.add_argument(
        "--blocks",
        action="append",
        default=[],
        help="Explicit block spec(s) 'r0,r1,c0,c1' (may be repeated).",
    )
    parser.add_argument("--log-heatmap", action="store_true", help="Plot log10(|U|+eps) instead of |U|")
    parser.add_argument(
        "--band-profile",
        action="store_true",
        help="Compute mean(|U[i,j]|) vs |i-j| and save a bandedness plot + summary.",
    )
    parser.add_argument(
        "--band-width",
        type=int,
        default=32,
        help="Half-bandwidth for summary energy fraction (default: 32).",
    )
    args = parser.parse_args()

    n = int(args.n)
    outdir = os.path.join(args.outdir, f"n{n}")
    os.makedirs(outdir, exist_ok=True)

    print(f"Building canonical basis U (Gram-normalized) for N={n} ...")
    U = rft_basis_matrix(n, n, use_gram_normalization=True)
    # Sanity: unitary check (Fro norm)
    I = np.eye(n, dtype=np.complex128)
    err = float(np.linalg.norm(U.conj().T @ U - I))
    print(f"Unitarity error ||U^H U - I||_F = {err:.3e}")

    # Phase 1: heatmap.
    try:
        import matplotlib

        if os.environ.get("MPLBACKEND") is None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"matplotlib is required for plotting: {exc}") from exc

    absU = np.abs(U)
    if args.log_heatmap:
        eps = 1e-12
        im_data = np.log10(absU + eps)
        title = f"log10(|U|+eps) - Canonical RFT basis (N={n})"
        fname = "canonical_U_abs_log10.png"
    else:
        im_data = absU
        title = f"|U| - Canonical RFT basis (N={n})"
        fname = "canonical_U_abs.png"

    plt.figure(figsize=(8, 7))
    plt.imshow(im_data, cmap="magma", aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("k (basis index)")
    plt.ylabel("n (time index)")
    out_path = os.path.join(outdir, fname)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Wrote: {out_path}")

    if args.band_profile:
        offsets = np.arange(n, dtype=np.int32)
        mean_by_off = np.zeros(n, dtype=np.float64)
        for d in offsets:
            if d == 0:
                mean_by_off[d] = float(absU.diagonal(0).mean())
            else:
                diag1 = absU.diagonal(d)
                diag2 = absU.diagonal(-d)
                mean_by_off[d] = float(np.concatenate([diag1, diag2]).mean())

        w = max(0, min(int(args.band_width), n - 1))
        total_energy = float(np.sum(absU * absU))
        band_energy = 0.0
        for d in range(-w, w + 1):
            band_energy += float(np.sum(np.square(np.abs(np.diagonal(U, d)))))
        band_frac = (band_energy / total_energy) if total_energy > 0 else 0.0

        plt.figure(figsize=(8, 4))
        plt.semilogy(offsets[1:], mean_by_off[1:] + 1e-18, label="mean |U| (off-diagonals)")
        plt.axvline(w, color="C3", linestyle="--", linewidth=1, label=f"band_width={w}")
        plt.grid(alpha=0.3)
        plt.xlabel("|i-j| (diagonal offset)")
        plt.ylabel("mean magnitude (log scale)")
        plt.title(f"Mean magnitude vs diagonal offset - Canonical RFT (N={n})")
        plt.legend(fontsize=8)
        out_path = os.path.join(outdir, "band_profile_mean_abs_vs_offset.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Wrote: {out_path}")
        print(f"Bandedness: energy within |i-j|<= {w} is {band_frac:.4f} of total |U|^2 energy")

    # Phase 2: SVD on off-diagonal blocks.
    blocks = list(_default_blocks(n, int(args.block)))
    blocks.extend(_parse_block(s) for s in args.blocks)
    # De-duplicate while preserving order.
    seen = set()
    blocks = [b for b in blocks if not (b in seen or seen.add(b))]
    if not blocks:
        print("No blocks requested; done.")
        return

    print(f"Computing SVD for {len(blocks)} block(s) ...")
    svds = [_svd_block(U, *b) for b in blocks]

    # Plot singular value decay.
    plt.figure(figsize=(8, 5))
    for sv in svds:
        x = np.arange(1, len(sv.s) + 1)
        plt.semilogy(x, sv.s, label=f"[{sv.r0}:{sv.r1}, {sv.c0}:{sv.c1}]")
    plt.grid(alpha=0.3)
    plt.xlabel("singular value index")
    plt.ylabel("sigma (log scale)")
    plt.title(f"Off-diagonal block singular values - Canonical RFT (N={n})")
    plt.legend(fontsize=8)
    out_path = os.path.join(outdir, "offdiag_block_singular_values.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Wrote: {out_path}")

    # Markdown report.
    report = os.path.join(outdir, "report.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write("# Canonical RFT matrix diagnostics\n\n")
        f.write(f"- N: `{n}`\n")
        f.write(f"- Unitarity error (Fro): `{err:.3e}`\n")
        f.write(f"- Heatmap: `{fname}`\n\n")
        f.write("## Off-diagonal block SVD\n\n")
        f.write("| Block (rows, cols) | size | k@99% Fro | k@99.9% Fro | sigma1 | sigma_last |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for sv in svds:
            k99 = _effective_rank(sv.s, 0.99)
            k999 = _effective_rank(sv.s, 0.999)
            f.write(
                f"| `{sv.r0}:{sv.r1}, {sv.c0}:{sv.c1}` | {sv.s.size} | {k99} | {k999}"
                f" | {sv.s[0]:.3e} | {sv.s[-1]:.3e} |\n"
            )
        f.write("\n")

    print(f"Wrote: {report}")


if __name__ == "__main__":
    main()
