#!/usr/bin/env python3
"""
Phase 3: Overlapping-window fix experiment (STFT-like)

Goal
----
Quantify whether overlap-add (Hann windows, hop=win/2) reduces the boundary
artifacts introduced by hard blockwise canonical RFT along the seq axis.

We compare:
  1) Non-overlap blocks: win=256, hop=256
  2) Overlap-add:        win=256, hop=128

Crucially, we compare at *equal total kept coefficient budget*.

Outputs
-------
Writes a small report and an error-vs-position plot under:
  results/overlap_experiment/
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np

from algorithms.rft.core.resonant_fourier_transform import rft_basis_matrix


@dataclass(frozen=True)
class MethodResult:
    method: str
    win: int
    hop: int
    frames: int
    k_per_frame: int
    total_kept: int
    rel_err: float
    seconds: float


def _rel_err(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.linalg.norm(x.ravel()) + 1e-12)
    return float(np.linalg.norm((x - y).ravel()) / denom)


def _frame_count(S: int, win: int, hop: int) -> int:
    if S <= win:
        return 1
    return int(np.ceil((S - win) / hop)) + 1


def _pad_to_frames(x: np.ndarray, win: int, hop: int) -> tuple[np.ndarray, int]:
    S0 = int(x.shape[0])
    frames = _frame_count(S0, win, hop)
    S_pad = (frames - 1) * hop + win
    if S_pad <= S0:
        return x, S0
    x2 = np.pad(x, (0, S_pad - S0))
    return x2, S0


def _canonical_frame_transform(frames: np.ndarray, PhiH: np.ndarray) -> np.ndarray:
    # frames: (F, win) real/complex
    return frames.astype(np.complex64, copy=False) @ PhiH.T  # (F, win)


def _canonical_frame_inverse(coeffs: np.ndarray, Phi: np.ndarray) -> np.ndarray:
    return coeffs.astype(np.complex64, copy=False) @ Phi.T  # (F, win)


def _topk_indices_by_energy(coeffs: np.ndarray, k: int) -> np.ndarray:
    # coeffs: (F, win)
    energy = np.mean(np.abs(coeffs) ** 2, axis=0)
    k = int(max(1, min(k, coeffs.shape[1])))
    idx = np.argpartition(energy, -k)[-k:]
    return np.sort(idx)


def _apply_blockwise(
    x: np.ndarray,
    *,
    win: int,
    hop: int,
    k_per_frame: int,
    Phi: np.ndarray,
    PhiH: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (y_recon, err_profile).
    err_profile is per-position |x-y| / (||x||/sqrt(S)).
    """
    x0 = np.asarray(x, dtype=np.float32)
    S0 = int(x0.shape[0])

    # If we use a tapering window (Hann), pad both ends so original samples
    # are not forced to live at window endpoints where wsum≈0.
    crop_start = 0
    x = x0
    if hop < win:
        pad = win - hop
        x = np.pad(x, (pad, pad))
        crop_start = pad

    x_pad, _S_prepad = _pad_to_frames(x, win, hop)
    S = int(x_pad.shape[0])
    frames = _frame_count(S, win, hop)

    window = np.ones(win, dtype=np.float32) if hop == win else np.hanning(win).astype(np.float32)

    # Analysis frames
    X = np.zeros((frames, win), dtype=np.float32)
    for fi in range(frames):
        start = fi * hop
        X[fi, :] = x_pad[start : start + win] * window

    C = _canonical_frame_transform(X, PhiH)  # (F, win)
    kept_idx = _topk_indices_by_energy(C, k_per_frame)
    C2 = np.zeros_like(C)
    C2[:, kept_idx] = C[:, kept_idx]

    # Synthesis
    Yf = _canonical_frame_inverse(C2, Phi).real.astype(np.float32, copy=False)
    y = np.zeros(S, dtype=np.float32)
    wsum = np.zeros(S, dtype=np.float32)
    for fi in range(frames):
        start = fi * hop
        y[start : start + win] += Yf[fi, :] * window
        wsum[start : start + win] += window * window
    y = y / np.maximum(wsum, 1e-8)
    y = y[crop_start : crop_start + S0]

    scale = float(np.linalg.norm(x0) / np.sqrt(x0.size) + 1e-12)
    err_profile = np.abs(x0 - y) / scale
    return y, err_profile


def _golden_signal(S: int, *, tones: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    n = np.arange(S, dtype=np.float32)
    s = np.zeros(S, dtype=np.complex64)
    k_idx = rng.choice(S, size=int(tones), replace=False)
    for k in k_idx:
        f = float(((k + 1) * phi) % 1.0)
        amp = np.complex64(rng.standard_normal() + 1j * rng.standard_normal())
        s += amp * np.exp(1j * 2 * np.pi * f * n).astype(np.complex64)
    s = s / np.linalg.norm(s)
    return s.real.astype(np.float32, copy=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=2048, help="Sequence length (default: 2048)")
    ap.add_argument("--win", type=int, default=256, help="Window size (default: 256)")
    ap.add_argument("--tones", type=int, default=3, help="Number of golden tones (default: 3)")
    ap.add_argument("--keep_ratio", type=float, default=0.25, help="Keep ratio for non-overlap baseline (default: 0.25)")
    ap.add_argument("--outdir", default=os.path.join("results", "overlap_experiment"))
    args = ap.parse_args()

    S = int(args.S)
    win = int(args.win)
    hop0 = win
    hop1 = win // 2
    keep_ratio = float(args.keep_ratio)

    os.makedirs(args.outdir, exist_ok=True)

    x = _golden_signal(S, tones=int(args.tones), seed=42)

    # Canonical basis for win
    Phi = rft_basis_matrix(win, win, use_gram_normalization=True).astype(np.complex64, copy=False)
    PhiH = Phi.conj().T.astype(np.complex64, copy=False)

    # Equal total coefficient budget:
    # total_kept = frames * k_per_frame
    frames0 = _frame_count(S, win, hop0)
    k0 = max(1, int(round(win * keep_ratio)))
    total_budget = frames0 * k0

    frames1 = _frame_count(S, win, hop1)
    k1 = max(1, int(round(total_budget / frames1)))

    results: list[MethodResult] = []

    t0 = time.perf_counter()
    y0, e0 = _apply_blockwise(x, win=win, hop=hop0, k_per_frame=k0, Phi=Phi, PhiH=PhiH)
    dt0 = time.perf_counter() - t0
    results.append(
        MethodResult(
            method="block_nonoverlap",
            win=win,
            hop=hop0,
            frames=frames0,
            k_per_frame=k0,
            total_kept=frames0 * k0,
            rel_err=_rel_err(x, y0),
            seconds=float(dt0),
        )
    )

    t1 = time.perf_counter()
    y1, e1 = _apply_blockwise(x, win=win, hop=hop1, k_per_frame=k1, Phi=Phi, PhiH=PhiH)
    dt1 = time.perf_counter() - t1
    results.append(
        MethodResult(
            method="overlap_hann_hop128",
            win=win,
            hop=hop1,
            frames=frames1,
            k_per_frame=k1,
            total_kept=frames1 * k1,
            rel_err=_rel_err(x, y1),
            seconds=float(dt1),
        )
    )

    # Plot error profiles to expose boundary artifacts.
    try:
        import matplotlib

        if os.environ.get("MPLBACKEND") is None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        plt = None

    if plt is not None:
        plt.figure(figsize=(10, 4))
        plt.plot(e0, label=f"non-overlap (K/frame={k0})", linewidth=1)
        plt.plot(e1, label=f"overlap (K/frame={k1})", linewidth=1)
        for b in range(win, S, win):
            plt.axvline(b, color="k", alpha=0.08, linewidth=1)
        plt.yscale("log")
        plt.grid(alpha=0.2)
        plt.xlabel("position n")
        plt.ylabel("normalized abs error (log)")
        plt.title(f"Canonical RFT windowing: boundary artifacts (S={S}, win={win})")
        plt.legend(fontsize=8)
        out_png = os.path.join(args.outdir, "error_profile.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

    # Markdown report
    out_md = os.path.join(args.outdir, "report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Canonical overlap vs block experiment\n\n")
        f.write(f"- S: `{S}`\n")
        f.write(f"- win: `{win}`\n")
        f.write(f"- tones: `{int(args.tones)}`\n")
        f.write(f"- keep_ratio (baseline non-overlap): `{keep_ratio}`\n")
        f.write(f"- equal total-kept budget: `{total_budget}` coefficients\n\n")
        f.write("## Results\n\n")
        f.write("| Method | hop | frames | K/frame | total kept | rel err | seconds |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in results:
            f.write(
                f"| `{r.method}` | {r.hop} | {r.frames} | {r.k_per_frame} | {r.total_kept} "
                f"| {r.rel_err:.4e} | {r.seconds:.3f} |\n"
            )
        if plt is not None:
            f.write("\n## Plots\n\n")
            f.write("- `error_profile.png`\n")

    print(f"Wrote: {out_md}")
    if plt is not None:
        print(f"Wrote: {os.path.join(args.outdir, 'error_profile.png')}")
    for r in results:
        print(
            f"{r.method}: rel_err={r.rel_err:.4e}, total_kept={r.total_kept}, "
            f"frames={r.frames}, K/frame={r.k_per_frame}, time={r.seconds:.3f}s"
        )


if __name__ == "__main__":
    main()
