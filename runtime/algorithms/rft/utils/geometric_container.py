#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# MIGRATED FROM: algorithms/rft/core/geometric_container.py
# IMPORT UPDATE: .symbolic_wave_computer and .oscillator now resolve
#                within algorithms/rft/utils/ (same directory)
"""
Geometric Container — quantoniumos RFT utils

Geometric containers for resonant frequency encoding using RFT.
"""

import numpy as np
from typing import List, Optional
from .oscillator import Oscillator


class LinearRegion:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end

    def contains(self, value: float) -> bool:
        return self.start <= value <= self.end

    def __repr__(self):
        return f"LinearRegion({self.start}, {self.end})"


class GeometricContainer:
    """
    Container holding data encoded as geometric waveforms using RFT principles.
    NOTE: symbolic_wave_computer dependency removed — was circular.
    Uses BinaryRFT from core directly instead.
    """

    def __init__(self, id: str, capacity_bits: int = 256):
        self.id = id
        self.capacity_bits = capacity_bits
        self.wave_form: Optional[np.ndarray] = None
        self.encoded_data_len: int = 0
        self.oscillators: List[Oscillator] = []
        self.resonant_frequencies: List[float] = []

    def encode_data(self, data: str) -> None:
        from algorithms.rft.core import BinaryRFT
        brft = BinaryRFT(num_bits=max(self.capacity_bits, len(data.encode()) * 8))
        byte_data = data.encode('utf-8')
        self.wave_form = brft.encode(int.from_bytes(byte_data[:self.capacity_bits // 8], 'big'))
        self.encoded_data_len = len(byte_data)
        phi = (1 + np.sqrt(5)) / 2
        self.resonant_frequencies = [(k + 1) * phi for k in range(self.capacity_bits)]

    def check_resonance(self, frequency: float, tolerance: float = 1e-3) -> bool:
        phi = (1 + np.sqrt(5)) / 2
        k_approx = (frequency / phi) - 1
        k = int(round(k_approx))
        if 0 <= k < self.capacity_bits:
            expected_freq = (k + 1) * phi
            if abs(frequency - expected_freq) < tolerance:
                return True
        return False

    def get_data(self) -> Optional[str]:
        return None  # Decoding requires BinaryRFT instance; implement as needed.
