# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Wave-space soft logic operators.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


class WaveComputer:
    """Spectral support and soft-logic operations on complex coefficient vectors."""

    def __init__(self, N: int) -> None:
        self.N = N

    # ------------------------------------------------------------------
    # Support mask
    # ------------------------------------------------------------------

    def spectral_support_mask(
        self, c: np.ndarray, frac_of_max: float = 0.25
    ) -> np.ndarray:
        """Return boolean mask of indices where |c[k]| >= frac_of_max * max|c|."""
        c = np.asarray(c, dtype=complex)
        threshold = frac_of_max * float(np.max(np.abs(c)))
        return np.abs(c) >= threshold

    # ------------------------------------------------------------------
    # Soft AND
    # ------------------------------------------------------------------

    def wave_and_soft(
        self,
        c1: np.ndarray,
        c2: np.ndarray,
        frac_of_max: float = 0.25,
    ) -> np.ndarray:
        """
        Soft AND: zero-out components outside the intersection of supports.
        The result is non-zero only where both c1 and c2 exceed the threshold.
        """
        c1 = np.asarray(c1, dtype=complex)
        c2 = np.asarray(c2, dtype=complex)
        m1 = self.spectral_support_mask(c1, frac_of_max)
        m2 = self.spectral_support_mask(c2, frac_of_max)
        intersection = m1 & m2
        # Blend: geometric mean amplitude, intersection support
        amp1 = np.abs(c1)
        amp2 = np.abs(c2)
        phase1 = np.angle(c1)
        phase2 = np.angle(c2)
        blended_amp = np.sqrt(amp1 * amp2)
        blended_phase = (phase1 + phase2) / 2.0
        result = blended_amp * np.exp(1j * blended_phase)
        result[~intersection] = 0.0
        return result

    # ------------------------------------------------------------------
    # Soft conditional select
    # ------------------------------------------------------------------

    def conditional_select_soft(
        self,
        cond: np.ndarray,
        then_c: np.ndarray,
        else_c: np.ndarray,
    ) -> np.ndarray:
        """
        Soft conditional: interpolate between then_c and else_c based on
        the normalised amplitude of cond.

        alpha = ||cond||_inf / (||cond||_inf + epsilon)
        output = alpha * then_c + (1-alpha) * else_c
        """
        cond = np.asarray(cond, dtype=complex)
        then_c = np.asarray(then_c, dtype=complex)
        else_c = np.asarray(else_c, dtype=complex)
        cond_max = float(np.max(np.abs(cond)))
        alpha = cond_max / (cond_max + 1e-30)
        return alpha * then_c + (1.0 - alpha) * else_c
