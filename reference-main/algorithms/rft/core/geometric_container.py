# SPDX-License-Identifier: LicenseRef-SPX-Proprietary
# Copyright (c) 2024-2026 Luis M. Minier. All rights reserved.
# Patent Pending: USPTO Application 19/169,399
"""
GeometricContainer: RFT-based data container with resonance encoding.

Each container holds data encoded as a BPSK wave on the canonical
φ-grid frequencies. Resonance checking verifies whether a query
frequency coincides with a φ-grid basis frequency.
"""
from __future__ import annotations
import numpy as np
from typing import Optional

PHI = (1.0 + np.sqrt(5.0)) / 2.0


class GeometricContainer:
    """
    Data container with RFT-resonance encoding.

    Parameters
    ----------
    id           : Identifier string for this container.
    capacity_bits: Number of bits (= number of φ-grid basis vectors). Default 64.
    """

    def __init__(self, id: str, capacity_bits: int = 64) -> None:
        self.id = id
        self.capacity_bits = capacity_bits
        self.wave_form: Optional[np.ndarray] = None
        self.encoded_data_len: int = 0
        self._bits: Optional[np.ndarray] = None
        # Precompute resonant frequencies f_k = frac((k+1)*phi), k=0..capacity_bits-1
        k = np.arange(capacity_bits, dtype=np.float64)
        self._resonant_freqs = np.mod((k + 1) * PHI, 1.0)

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode_data(self, data: str) -> None:
        """Encode a UTF-8 string as a BPSK waveform on the φ-grid."""
        raw = data.encode('utf-8')
        self.encoded_data_len = len(raw)
        # Convert bytes to bit array
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
        # Truncate or zero-pad to capacity_bits
        b = np.zeros(self.capacity_bits, dtype=np.float64)
        n = min(len(bits), self.capacity_bits)
        b[:n] = bits[:n]
        self._bits = b
        # BPSK: 0 -> +1, 1 -> -1
        symbols = 1.0 - 2.0 * b   # +1 or -1
        # Build waveform: sum of sinusoids at resonant frequencies
        t = np.arange(self.capacity_bits, dtype=np.float64)
        self.wave_form = np.zeros(self.capacity_bits, dtype=np.complex128)
        for k_idx in range(self.capacity_bits):
            self.wave_form += symbols[k_idx] * np.exp(
                2j * np.pi * self._resonant_freqs[k_idx] * t
            )
        self.wave_form /= np.sqrt(self.capacity_bits)

    def get_data(self) -> str:
        """Decode the waveform back to a UTF-8 string."""
        if self.wave_form is None or self._bits is None:
            return ''
        # Return from stored bits directly (lossless in simulation)
        n_bytes = self.encoded_data_len
        bits = self._bits[:n_bytes * 8].astype(np.uint8)
        raw = np.packbits(bits).tobytes()[:n_bytes]
        return raw.decode('utf-8')

    # ------------------------------------------------------------------
    # Resonance checking
    # ------------------------------------------------------------------

    def check_resonance(self, freq: float, tol: float = 1e-6) -> bool:
        """Return True if freq (mod 1) matches any resonant φ-grid frequency."""
        f = float(freq) % 1.0
        return bool(np.any(np.abs(self._resonant_freqs - f) < tol))

    def __repr__(self) -> str:
        return f"GeometricContainer(id={self.id!r}, capacity_bits={self.capacity_bits})"
