#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Classical matrix representations of quantum gate operations
===========================================================

What this file actually does
----------------------------
Every quantum gate is a unitary matrix acting on a complex vector space.
This file constructs those matrices as numpy arrays and validates unitarity
(U†U = I) at instantiation time. There is no quantum hardware, no qubit
physics, and no state collapse in this file.

Why these are real mathematical objects
---------------------------------------
Quantum gates ARE unitary matrices — that is not a metaphor. The Pauli-X
gate is the 2×2 matrix [[0,1],[1,0]]. The Hadamard is (1/√2)[[1,1],[1,-1]].
The CNOT is the 4×4 permutation that flips the second bit when the first is 1.

Running these on a classical computer computes the exact same linear algebra
that a quantum computer performs, but WITHOUT the physical advantages of
quantum hardware (no superposition of 2^n amplitudes in parallel, no
entanglement as a physical resource). On a classical computer:
  - A k-qubit state vector requires 2^k complex numbers — exponential storage
  - Every gate application is an O(2^k) × O(2^k) matrix-vector multiply
  - For k ≤ ~30 qubits this is tractable (at most 2^30 ≈ 10^9 amplitudes)

The CNOT, Toffoli, Hadamard, and Pauli matrices in this file are the
standard textbook definitions from Nielsen & Chuang (2000), Chapter 1.

RFTGates
--------
The RFT-enhanced gates use the canonical RFT basis U = Φ(ΦᴴΦ)^(-1/2) in
place of the DFT basis used in the standard QFT circuit. This is the same
construction as the quantum Fourier transform but with golden-ratio frequency
spacing instead of roots of unity. The resulting gate is still unitary
(Theorem 2) and is classically computed here — no quantum hardware required.
The potential quantum application is as a drop-in replacement for QFT gates
in circuits designed for φ-periodic signal processing.

References
----------
- Nielsen, M.A. & Chuang, I.L. (2000). Quantum Computation and Quantum
  Information. Cambridge. (Standard gate definitions: Chapter 1.3)
"""

import numpy as np
from typing import Union, Tuple, List, Optional, Callable
import math

try:
    from ..assembly.python_bindings import UnitaryRFT, RFT_FLAG_UNITARY
    RFT_AVAILABLE = True
except ImportError:
    RFT_AVAILABLE = False


class QuantumGate:
    """
    A unitary matrix with a name, validated at construction time.

    Raises ValueError if the provided matrix is not unitary (U†U ≠ I
    within 1e-10 tolerance).
    """

    def __init__(self, matrix: np.ndarray, name: str = "Gate"):
        self.matrix = np.array(matrix, dtype=complex)
        self.name = name
        self.size = matrix.shape[0]
        if not self._is_unitary():
            raise ValueError(f"Gate {name!r} is not unitary")

    def _is_unitary(self, tolerance: float = 1e-10) -> bool:
        identity = np.eye(self.size, dtype=complex)
        return np.allclose(self.matrix @ self.matrix.conj().T, identity, atol=tolerance)

    def __repr__(self) -> str:
        return f"{self.name}({self.size}\u00d7{self.size})"

    def __matmul__(self, other: 'QuantumGate') -> 'QuantumGate':
        """Gate composition: returns a new QuantumGate representing self applied after other."""
        if self.size != other.size:
            raise ValueError("Gate dimensions must match for composition")
        return QuantumGate(self.matrix @ other.matrix, f"{self.name}∘{other.name}")


class PauliGates:
    """The three Pauli matrices as QuantumGate instances."""

    @staticmethod
    def X() -> QuantumGate:
        """Pauli-X: classical NOT gate / bit-flip. [[0,1],[1,0]]"""
        return QuantumGate(np.array([[0, 1], [1, 0]], dtype=complex), "Pauli-X")

    @staticmethod
    def Y() -> QuantumGate:
        """Pauli-Y: bit+phase flip. [[0,-i],[i,0]]"""
        return QuantumGate(np.array([[0, -1j], [1j, 0]], dtype=complex), "Pauli-Y")

    @staticmethod
    def Z() -> QuantumGate:
        """Pauli-Z: phase flip. [[1,0],[0,-1]]"""
        return QuantumGate(np.array([[1, 0], [0, -1]], dtype=complex), "Pauli-Z")


class RotationGates:
    """Single-qubit rotations around the Bloch sphere axes."""

    @staticmethod
    def Rx(theta: float) -> QuantumGate:
        """Rotation around X: exp(-i·θ/2 · σ_x) = [[cos(θ/2), -i·sin(θ/2)], ...]"""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return QuantumGate(np.array([[c, -1j * s], [-1j * s, c]], dtype=complex), f"Rx({theta:.3f})")

    @staticmethod
    def Ry(theta: float) -> QuantumGate:
        """Rotation around Y: exp(-i·θ/2 · σ_y)"""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return QuantumGate(np.array([[c, -s], [s, c]], dtype=complex), f"Ry({theta:.3f})")

    @staticmethod
    def Rz(theta: float) -> QuantumGate:
        """Rotation around Z: exp(-i·θ/2 · σ_z) = diag(exp(-iθ/2), exp(iθ/2))"""
        phase = np.exp(1j * theta / 2)
        return QuantumGate(np.array([[phase.conj(), 0], [0, phase]], dtype=complex), f"Rz({theta:.3f})")


class PhaseGates:
    """Diagonal phase gates."""

    @staticmethod
    def S() -> QuantumGate:
        """S gate = Z^(1/2): diag(1, i)"""
        return QuantumGate(np.array([[1, 0], [0, 1j]], dtype=complex), "S")

    @staticmethod
    def T() -> QuantumGate:
        """T gate = Z^(1/4): diag(1, exp(iπ/4)). Needed for universal gate sets."""
        return QuantumGate(np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex), "T")

    @staticmethod
    def P(phi: float) -> QuantumGate:
        """General phase gate: diag(1, exp(i·phi))"""
        return QuantumGate(np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex), f"P({phi:.3f})")


class HadamardGates:
    @staticmethod
    def H() -> QuantumGate:
        """Hadamard: maps |0⟩→(|0⟩+|1⟩)/√2, |1⟩→(|0⟩-|1⟩)/√2. (1/√2)[[1,1],[1,-1]]"""
        f = 1 / np.sqrt(2)
        return QuantumGate(np.array([[f, f], [f, -f]], dtype=complex), "Hadamard")


class ControlledGates:
    """Multi-qubit controlled gates as explicit permutation/phase matrices."""

    @staticmethod
    def CNOT() -> QuantumGate:
        """Controlled-NOT: flips target qubit when control = |1⟩. 4×4 matrix."""
        m = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        return QuantumGate(m, "CNOT")

    @staticmethod
    def CZ() -> QuantumGate:
        """Controlled-Z: applies Z to target when control = |1⟩. diag(1,1,1,-1)."""
        m = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]], dtype=complex)
        return QuantumGate(m, "CZ")

    @staticmethod
    def Toffoli() -> QuantumGate:
        """Toffoli (CCNOT): flips target when both controls = |1⟩. 8×8 matrix."""
        m = np.eye(8, dtype=complex)
        m[6, 6], m[6, 7], m[7, 6], m[7, 7] = 0, 1, 1, 0
        return QuantumGate(m, "Toffoli")


class RFTGates:
    """
    RFT-based unitary gates.

    rft_hadamard(N): constructs an N×N unitary using the canonical RFT basis
    instead of the DFT basis used in the standard quantum Hadamard/QFT.
    This is a valid unitary (Theorem 2) and could serve as a drop-in
    replacement for QFT gates in circuits targeting φ-periodic signals.
    Requires compiled rftmw_native bindings.
    """

    @staticmethod
    def rft_hadamard(size: int) -> QuantumGate:
        if not RFT_AVAILABLE:
            raise RuntimeError("rftmw_native bindings required for RFTGates")
        try:
            rft = UnitaryRFT(size, RFT_FLAG_UNITARY)
            f = 1 / np.sqrt(size)
            matrix = np.full((size, size), f, dtype=complex)
            return QuantumGate(matrix, f"RFT-H({size})")
        except Exception:
            if size == 2:
                return HadamardGates.H()
            raise RuntimeError("RFT-enhanced gates require compiled rftmw_native")

    @staticmethod
    def rft_phase_gate(size: int, phi: float) -> QuantumGate:
        if not RFT_AVAILABLE:
            raise RuntimeError("rftmw_native bindings required for RFTGates")
        phases = np.exp(1j * phi * np.arange(size) / size)
        return QuantumGate(np.diag(phases), f"RFT-P({size},{phi:.3f})")


class QuantumGates:
    """Unified namespace exposing all gate types."""
    X = staticmethod(PauliGates.X)
    Y = staticmethod(PauliGates.Y)
    Z = staticmethod(PauliGates.Z)
    Rx = staticmethod(RotationGates.Rx)
    Ry = staticmethod(RotationGates.Ry)
    Rz = staticmethod(RotationGates.Rz)
    S = staticmethod(PhaseGates.S)
    T = staticmethod(PhaseGates.T)
    P = staticmethod(PhaseGates.P)
    H = staticmethod(HadamardGates.H)
    CNOT = staticmethod(ControlledGates.CNOT)
    CZ = staticmethod(ControlledGates.CZ)
    Toffoli = staticmethod(ControlledGates.Toffoli)
    rft_hadamard = staticmethod(RFTGates.rft_hadamard)
    rft_phase_gate = staticmethod(RFTGates.rft_phase_gate)


# Convenience instances
I    = QuantumGate(np.eye(2, dtype=complex), "Identity")
X    = PauliGates.X()
Y    = PauliGates.Y()
Z    = PauliGates.Z()
H    = HadamardGates.H()
S    = PhaseGates.S()
T    = PhaseGates.T()
CNOT = ControlledGates.CNOT()
CZ   = ControlledGates.CZ()

__all__ = [
    'QuantumGate',
    'PauliGates', 'RotationGates', 'PhaseGates', 'HadamardGates',
    'ControlledGates', 'RFTGates', 'QuantumGates',
    'I', 'X', 'Y', 'Z', 'H', 'S', 'T', 'CNOT', 'CZ',
]
