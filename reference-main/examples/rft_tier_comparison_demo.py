#!/usr/bin/env python3
"""
RFT tier comparison demo (FFT vs FastRFT vs OversampledSurrogate vs CanonicalTrueRFT).

Writes: rft_tier_comparison.png (in the current working directory)
"""

from __future__ import annotations

import os
import numpy as np

from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT
from algorithms.rft.core.fast_rft import FastRFT
from algorithms.rft.core.fast_rft_surrogate import OversampledSurrogate


def _psnr(a: np.ndarray, b: np.ndarray, *, eps: float = 1e-300) -> float:
    mse = float(np.mean(np.abs(a - b) ** 2))
    peak = float(np.max(np.abs(a)))
    return float(20.0 * np.log10(peak / np.sqrt(mse + eps)))


def run_demo(*, N: int = 256, k: int = 7, oversample: int = 8, snap: str = "lanczos2") -> None:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    n = np.arange(N, dtype=np.float64)

    # Single tone on the golden grid (frequency in [0,1)).
    f_target = (k * phi) % 1.0
    s = np.exp(1j * 2.0 * np.pi * f_target * n).astype(np.complex128) / np.sqrt(float(N))

    # Tier 0: unitary FFT.
    hat_x_fft = np.fft.fft(s, norm="ortho")

    # Tier 1: FastRFT (D3) unitary operator Ψ = Dφ·Cσ·F.
    tier1 = FastRFT(N)
    hat_x_t1 = tier1.forward(s)

    # Tier 2: Oversampled snapping surrogate (not unitary; closer in coefficient space).
    tier2 = OversampledSurrogate(N, oversample=oversample, snap=snap)
    hat_x_t2 = tier2.forward(s)

    # Tier 3: CanonicalTrueRFT (Gram-normalized φ-grid basis).
    tier3 = CanonicalTrueRFT(N)
    hat_x_t3 = tier3.forward_transform(s)

    print(f"N={N} k={k} oversample={oversample} snap={snap}")
    print(f"PSNR(FastRFT vs Canonical): { _psnr(hat_x_t3, hat_x_t1):.2f} dB")
    print(f"PSNR(Oversampled vs Canonical): { _psnr(hat_x_t3, hat_x_t2):.2f} dB")
    print(f"PSNR(FFT vs Canonical): { _psnr(hat_x_t3, hat_x_fft):.2f} dB")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"matplotlib is required for plotting: {exc}") from exc

    titles = [
        "Tier 0: Standard FFT",
        "Tier 1: Fast RFT (D3)",
        f"Tier 2: Oversampled Surrogate ({oversample}×, {snap})",
        "Tier 3: Canonical RFT",
    ]
    data = [hat_x_fft, hat_x_t1, hat_x_t2, hat_x_t3]

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    plt.subplots_adjust(hspace=0.4)

    for i, ax in enumerate(axes):
        magnitude = np.abs(data[i])
        ax.stem(magnitude, markerfmt=" ", basefmt="C7-", linefmt="C0-")
        ax.set_title(titles[i], fontweight="bold")
        ax.set_ylabel("Magnitude")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.1)

    axes[-1].set_xlabel("Frequency Bin (Index)")
    plt.suptitle(f"RFT Hierarchy: Eliminating Spectral Blur (N={N})", fontsize=16)

    out = "rft_tier_comparison.png"
    print(f"Saving: {out}")
    plt.savefig(out, dpi=300)

    if os.getenv("SHOW_PLOT", "0") == "1":
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    run_demo()

