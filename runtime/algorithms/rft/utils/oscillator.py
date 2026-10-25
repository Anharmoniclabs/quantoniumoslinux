#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# MIGRATED FROM: algorithms/rft/core/oscillator.py
"""
RFT Oscillator — quantoniumos RFT utils

Oscillator based on Golden Ratio frequencies. Used by GeometricContainer.
"""

import math
import numpy as np
from typing import Optional

PHI = (1 + math.sqrt(5)) / 2


class Oscillator:
    """
    RFT Oscillator based on Golden Ratio frequencies.
    Frequencies: f_k = (k+1) * PHI
    """

    def __init__(self, mode: int = 0, phase: float = 0.0):
        self.mode = mode
        self.phase = phase
        self.amplitude = 1.0
        self._update_frequency()

    def _update_frequency(self):
        self.frequency = (self.mode + 1) * PHI

    def set_mode(self, mode: int):
        self.mode = mode
        self._update_frequency()

    def get_value(self, t: float) -> float:
        omega = 2 * math.pi * self.frequency
        return self.amplitude * math.sin(omega * t + self.phase)

    def encode_value(self, value: float):
        self.amplitude = value

    def decode_value(self, signal: np.ndarray, t: np.ndarray) -> float:
        omega = 2 * math.pi * self.frequency
        reference = np.sin(omega * t + self.phase)
        numerator = np.dot(signal, reference)
        denominator = np.dot(reference, reference)
        if denominator == 0:
            return 0.0
        return numerator / denominator
