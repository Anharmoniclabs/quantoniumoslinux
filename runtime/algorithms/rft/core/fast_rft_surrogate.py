# SPDX-License-Identifier: LicenseRef-SPX-Proprietary
# Copyright (c) 2024-2026 Luis M. Minier. All rights reserved.
# Patent Pending: USPTO Application 19/169,399
"""
FastRFTSurrogate: oversampled-DFT approximation to the canonical RFT.

Instead of building the O(N^3) Gram-normalised basis, we approximate
the canonical RFT coefficient at each φ-grid frequency by interpolating
from an oversampled DFT. Two snapping modes are supported:
  - 'nearest'   : nearest-bin value (default)
  - 'lanczos2'  : Lanczos-2 kernel interpolation (higher PSNR)
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Literal

PHI = (1.0 + np.sqrt(5.0)) / 2.0


class FastRFTSurrogate:
    """
    Approximates the canonical RFT forward transform via an oversampled FFT.

    Parameters
    ----------
    N        : Signal length.
    oversample : Oversampling factor for the intermediate DFT (default 8).
    snap     : Interpolation mode - 'nearest' or 'lanczos2'.
    """

    def __init__(
        self,
        N: int,
        oversample: int = 8,
        snap: Literal['nearest', 'lanczos2'] = 'nearest',
    ) -> None:
        self.N = N
        self.oversample = oversample
        self.snap = snap
        self._M = N * oversample          # padded DFT length
        # Precompute target bin positions in the oversampled DFT
        k = np.arange(N, dtype=np.float64)
        freqs = np.mod((k + 1) * PHI, 1.0)   # f_k = frac((k+1)*phi)
        # Continuous bin index in [0, M)
        self._bin_pos = freqs * self._M       # float bin positions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _oversample_dft(self, x: NDArray) -> NDArray[np.complex128]:
        """Compute zero-padded DFT of length M."""
        return np.fft.fft(x, n=self._M)

    def _nearest(self, X_over: NDArray) -> NDArray[np.complex128]:
        bins = np.round(self._bin_pos).astype(int) % self._M
        return X_over[bins] / np.sqrt(self.N)

    def _lanczos2(self, X_over: NDArray) -> NDArray[np.complex128]:
        """Lanczos-2 kernel interpolation (a=2)."""
        out = np.zeros(self.N, dtype=np.complex128)
        M = self._M
        for k in range(self.N):
            x0 = self._bin_pos[k]
            # Tap positions: x0-1, x0, x0+1, x0+2 (4-tap kernel)
            s = 0.0 + 0.0j
            for di in range(-1, 3):
                xi = int(np.floor(x0)) + di
                t = x0 - xi
                w = self._lanczos_kernel(t - di + (di > 0) * 0, a=2)
                s += X_over[xi % M] * w
            out[k] = s / np.sqrt(self.N)
        return out

    @staticmethod
    def _lanczos_kernel(t: float, a: int = 2) -> float:
        if t == 0:
            return 1.0
        if abs(t) >= a:
            return 0.0
        pt = np.pi * t
        return a * np.sin(pt) * np.sin(pt / a) / (pt * pt)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x: NDArray) -> NDArray[np.complex128]:
        """Approximate canonical RFT forward transform."""
        x = np.asarray(x, dtype=np.complex128)
        X_over = self._oversample_dft(x)
        if self.snap == 'lanczos2':
            return self._lanczos2(X_over)
        return self._nearest(X_over)
