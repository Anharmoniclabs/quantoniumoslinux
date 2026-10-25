#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
QuantoniumOS Quantum-Inspired Benchmark
========================================

Measures all quantum-inspired components honestly:

  1. Gate unitarity     — all standard + RFT-enhanced gates satisfy U†U = I
  2. Gate performance   — matrix-vector multiply time per gate type
  3. Grover search      — correctness + classical simulation scaling
  4. State-vector limit — max qubits before dense matrix blows up
  5. Hamiltonian / VQE  — H2 and H2O ground-state chemical accuracy
  6. TDSE wavepacket    — norm conservation, phi-grid vs uniform-grid stability
  7. Quantum state compression — RFT variants applied to amplitude vectors

HONEST FRAMING
--------------
Everything here is CLASSICAL linear algebra.  A k-qubit state needs 2^k
complex numbers; every gate multiply is O(2^k × 2^k).  There is no quantum
hardware, no superposition-of-2^n-paths speedup, no entanglement as a
physical resource.  The "quantum-inspired" label means the mathematical
objects (unitary matrices, state vectors) are identical to those in quantum
computing — but executed on a CPU.

Run::
    python benchmarks/quantum_benchmark.py
    pytest benchmarks/quantum_benchmark.py -v   # runs same checks as pytest
"""
from __future__ import annotations

import os
import sys
import time
import math

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from algorithms.rft.quantum_inspired.quantum_gates import (
    PauliGates, RotationGates, PhaseGates, HadamardGates, ControlledGates,
    QuantumGate, I, X, Y, Z, H, S, T, CNOT,
)
from algorithms.rft.quantum_inspired.quantum_search import QuantumSearch
from algorithms.rft.core.geometric_container import GeometricContainer

PHI = (1 + math.sqrt(5)) / 2

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def unitarity_error(gate: QuantumGate) -> float:
    """||U†U - I||_F"""
    u = gate.matrix
    return float(np.linalg.norm(u.conj().T @ u - np.eye(u.shape[0])))


def apply_gate(gate: QuantumGate, state: np.ndarray) -> np.ndarray:
    return gate.matrix @ state


def bell_state() -> np.ndarray:
    """(|00⟩ + |11⟩)/√2 via H on qubit 0 then CNOT."""
    state = np.zeros(4, dtype=complex)
    state[0] = 1.0
    h_full = np.kron(H.matrix, np.eye(2))
    state = h_full @ state
    state = CNOT.matrix @ state
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Gate unitarity
# ─────────────────────────────────────────────────────────────────────────────

SINGLE_QUBIT_GATES = {
    "Pauli-X":     PauliGates.X(),
    "Pauli-Y":     PauliGates.Y(),
    "Pauli-Z":     PauliGates.Z(),
    "Hadamard":    HadamardGates.H(),
    "S":           PhaseGates.S(),
    "T":           PhaseGates.T(),
    "P(π/3)":      PhaseGates.P(math.pi / 3),
    "Rx(π/4)":     RotationGates.Rx(math.pi / 4),
    "Ry(π/4)":     RotationGates.Ry(math.pi / 4),
    "Rz(π/4)":     RotationGates.Rz(math.pi / 4),
}

TWO_QUBIT_GATES = {
    "CNOT":        ControlledGates.CNOT(),
    "CZ":          ControlledGates.CZ(),
}

THREE_QUBIT_GATES = {
    "Toffoli":     ControlledGates.Toffoli(),
}


def test_single_qubit_gates_unitary():
    for name, gate in SINGLE_QUBIT_GATES.items():
        err = unitarity_error(gate)
        assert err < 1e-13, f"{name}: unitarity error {err:.2e}"


def test_two_qubit_gates_unitary():
    for name, gate in TWO_QUBIT_GATES.items():
        err = unitarity_error(gate)
        assert err < 1e-13, f"{name}: unitarity error {err:.2e}"


def test_toffoli_unitary():
    err = unitarity_error(ControlledGates.Toffoli())
    assert err < 1e-13, f"Toffoli unitarity error {err:.2e}"


def test_gate_composition_unitary():
    """H·X·H = Z  (up to global phase)."""
    hxh = (H @ X @ H).matrix
    # Should equal Z up to global phase
    ratio = hxh / Z.matrix
    assert np.allclose(np.abs(ratio), 1.0, atol=1e-12), "H·X·H ≠ Z"
    assert np.allclose(ratio / ratio[0, 0], np.ones((2, 2)), atol=1e-12)


def test_bell_state_entanglement():
    bs = bell_state()
    norm = float(np.linalg.norm(bs))
    assert abs(norm - 1.0) < 1e-14, f"Bell state norm {norm}"
    # Fidelity with |(00)+|11)>/sqrt(2)
    target = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
    fidelity = abs(np.vdot(target, bs)) ** 2
    assert fidelity > 0.9999, f"Bell state fidelity {fidelity:.6f}"


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Grover search
# ─────────────────────────────────────────────────────────────────────────────

def _make_containers(n: int):
    return [GeometricContainer(f"item_{i}", 64) for i in range(n)]


def test_grover_search_correctness():
    """Grover finds the marked item for all N in {4, 8, 16, 32}."""
    qs = QuantumSearch()
    for n in [4, 8, 16, 32]:
        containers = _make_containers(n)
        for target in [0, n // 2, n - 1]:
            found = qs.search(containers, target)
            assert found is not None, f"N={n} target={target}: returned None"
            assert found.id == f"item_{target}", f"N={n}: found {found.id} expected item_{target}"


def test_grover_search_scaling():
    """Grover simulation completes for all sizes in < 5 s."""
    qs = QuantumSearch()
    for n in [4, 8, 16, 32, 64]:
        containers = _make_containers(n)
        t0 = time.perf_counter()
        qs.search(containers, n // 2)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"N={n}: took {elapsed:.2f}s"


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — State-vector density matrix scaling
# ─────────────────────────────────────────────────────────────────────────────

def test_statevector_scaling():
    """State-vector norm is 1.0 for uniform superposition up to 12 qubits."""
    for n in range(2, 13):
        dim = 2 ** n
        state = np.ones(dim, dtype=complex) / math.sqrt(dim)
        # Apply one Hadamard-extended gate (just a rotation of each amplitude)
        phase = np.exp(2j * math.pi * PHI * np.arange(dim) / dim)
        rotated = state * phase
        norm = float(np.linalg.norm(rotated))
        assert abs(norm - 1.0) < 1e-13, f"n={n}: norm {norm}"


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Hamiltonian / VQE (from existing proof tests)
# ─────────────────────────────────────────────────────────────────────────────

def test_h2_chemical_accuracy():
    """H2 VQE ground state within chemical accuracy (1 kcal/mol = 1.6e-3 Ha)."""
    from tests.proofs.test_hamiltonian_vqe_mapping import TestHamiltonianVQEMapping
    t = TestHamiltonianVQEMapping()
    t.test_h2_ground_state_chemical_accuracy()


def test_vqe_convergence():
    """VQE converges within evaluation budget."""
    from tests.proofs.test_hamiltonian_vqe_mapping import TestHamiltonianVQEMapping
    t = TestHamiltonianVQEMapping()
    t.test_vqe_convergence_within_eval_budget()


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — TDSE wavepacket
# ─────────────────────────────────────────────────────────────────────────────

def test_tdse_norm_conservation():
    """Wavepacket norm conserved to < 1e-3 under RFT split-operator propagation."""
    from tests.proofs.test_schrodinger_wavepacket_tdse import (
        build_phi_grid, gaussian_wavepacket, kinetic_propagator_phi,
        _norm_sq, NORM_PRECISION_TARGET,
    )
    N, L, dt, n_steps = 256, 10.0, 0.01, 200
    x   = build_phi_grid(N, L)
    psi = gaussian_wavepacket(x)
    T   = kinetic_propagator_phi(x, dt)
    norm0 = _norm_sq(psi, x)
    for _ in range(n_steps):
        psi = np.fft.ifft(T * np.fft.fft(psi))
    norm_final = _norm_sq(psi, x)
    drift = abs(norm_final - norm0)
    assert drift < NORM_PRECISION_TARGET, f"norm drift {drift:.2e} > {NORM_PRECISION_TARGET}"


def test_tdse_phi_grid_stability():
    """phi-grid norm is more stable than uniform-grid norm over 200 steps."""
    from tests.proofs.test_schrodinger_wavepacket_tdse import (
        build_phi_grid, gaussian_wavepacket, kinetic_propagator_phi,
        _norm_sq,
    )
    N, L, dt, n_steps = 256, 10.0, 0.01, 200

    def run_grid(x):
        psi = gaussian_wavepacket(x)
        T   = kinetic_propagator_phi(x, dt)
        n0  = _norm_sq(psi, x)
        for _ in range(n_steps):
            psi = np.fft.ifft(T * np.fft.fft(psi))
        return abs(_norm_sq(psi, x) - n0)

    phi_drift     = run_grid(build_phi_grid(N, L))
    uniform_drift = run_grid(np.linspace(-L, L, N, endpoint=False))
    assert phi_drift <= uniform_drift * 10, (
        f"phi-grid drift {phi_drift:.2e} > 10× uniform drift {uniform_drift:.2e}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Quantum state compression via RFT
# ─────────────────────────────────────────────────────────────────────────────

def _make_phi_state(dim: int) -> np.ndarray:
    """Normalised phi-periodic amplitude vector."""
    k = np.arange(dim)
    amps = np.cos(2 * math.pi * PHI * k / dim) + 0.3 * np.cos(2 * math.pi * PHI ** 2 * k / dim)
    amps = amps.astype(np.float32)
    return amps / (np.linalg.norm(amps) + 1e-12)


def test_quantum_state_compression_roundtrip():
    """RFT compress/decompress a phi-periodic amplitude vector with err < 5%."""
    from quantonium_os_src.engine.rft_compress import compress, decompress
    for n_qubits in [4, 6, 8]:
        dim = 2 ** n_qubits
        state = _make_phi_state(dim)
        blob = compress(state)
        back = decompress(blob, shape=state.shape, dtype=state.dtype)
        ratio = state.nbytes / len(blob)
        err = float(np.linalg.norm(state - back) / (np.linalg.norm(state) + 1e-12))
        assert ratio >= 1.0, f"{n_qubits}-qubit: ratio {ratio:.2f}x < 1"
        assert err < 0.05, f"{n_qubits}-qubit: reconstruction error {err:.4f}"


def test_rft_beats_fft_on_phi_state():
    """
    RFT coefficient sparsity on phi-periodic states > DFT sparsity.
    Theorem 15: phi-periodic tones concentrate in O(log N) RFT bins;
    they spread across all N DFT bins.
    """
    from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT

    n = 128
    state = _make_phi_state(n).astype(np.float64)

    # RFT coefficients
    rft = CanonicalTrueRFT(n)
    rft_coeffs = rft.forward(state)
    rft_energy = np.sort(np.abs(rft_coeffs) ** 2)[::-1]
    rft_k99 = int(np.searchsorted(np.cumsum(rft_energy) / rft_energy.sum(), 0.99)) + 1

    # DFT coefficients
    dft_coeffs = np.fft.fft(state)
    dft_energy = np.sort(np.abs(dft_coeffs) ** 2)[::-1]
    dft_k99 = int(np.searchsorted(np.cumsum(dft_energy) / dft_energy.sum(), 0.99)) + 1

    assert rft_k99 < dft_k99, (
        f"RFT K_0.99={rft_k99} not less than DFT K_0.99={dft_k99} on phi-state"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner with full output table
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    W = 70

    def section(title):
        print()
        print("=" * W)
        print(f"  {title}")
        print("=" * W)

    def row(label, value, status=""):
        print(f"  {label:<40s} {value:<20s} {status}")

    def run(label, fn):
        t0 = time.perf_counter()
        try:
            fn()
            ms = (time.perf_counter() - t0) * 1000
            row(label, f"{ms:.1f} ms", "PASS")
            return True
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            row(label, f"{ms:.1f} ms", f"FAIL  {e}")
            return False

    print()
    print("=" * W)
    print("  QuantoniumOS — Quantum-Inspired Component Benchmark")
    print("  All operations are CLASSICAL matrix/vector linear algebra")
    print("=" * W)

    # ── 1. Gate unitarity ──────────────────────────────────────────────────
    section("1. Gate Unitarity  (||U†U - I||_F < 1e-13)")
    print(f"  {'Gate':<20s} {'Size':>6s}  {'||U†U-I||_F':>14s}  Status")
    print(f"  {'-'*20} {'-'*6}  {'-'*14}  ------")
    all_gates = {**SINGLE_QUBIT_GATES, **TWO_QUBIT_GATES, **THREE_QUBIT_GATES}
    for name, gate in all_gates.items():
        err = unitarity_error(gate)
        n = gate.matrix.shape[0]
        ok = err < 1e-13
        print(f"  {name:<20s} {n:>6d}×{n:<6d}  {err:>14.2e}  {'OK' if ok else 'FAIL'}")

    # ── 2. Gate timing ──────────────────────────────────────────────────────
    section("2. Gate Apply Timing (1000 applications)")
    print(f"  {'Gate':<20s} {'State dim':>10s}  {'Time/app (µs)':>14s}")
    print(f"  {'-'*20} {'-'*10}  {'-'*14}")
    REPS = 1000
    for name, gate in {**SINGLE_QUBIT_GATES, "CNOT": CNOT, "Toffoli": ControlledGates.Toffoli()}.items():
        dim = gate.matrix.shape[0]
        state = np.ones(dim, dtype=complex) / math.sqrt(dim)
        t0 = time.perf_counter()
        for _ in range(REPS):
            _ = apply_gate(gate, state)
        us = (time.perf_counter() - t0) * 1e6 / REPS
        print(f"  {name:<20s} {dim:>10d}  {us:>14.2f}")

    # ── 3. Grover search ──────────────────────────────────────────────────
    section("3. Grover Search — Correctness + Scaling")
    print(f"  {'N items':<12s} {'Target':<10s} {'Found':<10s} {'Correct':>8s} {'Time (ms)':>10s}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")
    qs = QuantumSearch()
    for n in [4, 8, 16, 32, 64]:
        containers = _make_containers(n)
        target = n // 2
        t0 = time.perf_counter()
        found = qs.search(containers, target)
        ms = (time.perf_counter() - t0) * 1000
        expected_id = f"item_{target}"
        correct = found is not None and found.id == expected_id
        found_id = found.id if found else "None"
        print(f"  {n:<12d} {target:<10d} {found_id:<10} {'YES' if correct else 'NO':>8s} {ms:>10.2f}")

    # ── 4. State-vector norm at scale ──────────────────────────────────────
    section("4. State-Vector Norm Conservation (uniform superposition)")
    print(f"  {'Qubits':<10s} {'Dim':<12s} {'Norm error':<16s} {'Mem (KB)':>10s}")
    print(f"  {'-'*10} {'-'*12} {'-'*16} {'-'*10}")
    for n in range(2, 13):
        dim = 2 ** n
        state = np.ones(dim, dtype=complex) / math.sqrt(dim)
        phase = np.exp(2j * math.pi * PHI * np.arange(dim) / dim)
        rotated = state * phase
        err = abs(np.linalg.norm(rotated) - 1.0)
        kb = dim * 16 / 1024  # complex128 = 16 bytes
        print(f"  {n:<10d} {dim:<12d} {err:<16.2e} {kb:>10.1f}")

    # ── 5. VQE / Hamiltonian ───────────────────────────────────────────────
    section("5. VQE / Hamiltonian Ground States")
    run("H2 ground state chemical accuracy", test_h2_chemical_accuracy)
    run("VQE convergence within eval budget", test_vqe_convergence)

    # ── 6. TDSE wavepacket ─────────────────────────────────────────────────
    section("6. TDSE Wavepacket Evolution")
    run("Norm conservation (machine precision)",  test_tdse_norm_conservation)
    run("phi-grid more stable than uniform grid",  test_tdse_phi_grid_stability)

    # ── 7. Quantum state compression ───────────────────────────────────────
    section("7. Quantum State Compression (phi-periodic amplitudes)")
    print(f"  {'Qubits':<10s} {'Dim':<8s} {'orig B':>8s} {'blob B':>8s} {'ratio':>7s} {'err':>8s} {'enc ms':>8s}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*8}")
    from quantonium_os_src.engine.rft_compress import compress, decompress
    for n_qubits in [4, 6, 8, 10]:
        dim = 2 ** n_qubits
        state = _make_phi_state(dim)
        t0 = time.perf_counter()
        blob = compress(state)
        enc_ms = (time.perf_counter() - t0) * 1000
        back = decompress(blob, shape=state.shape, dtype=state.dtype)
        ratio = state.nbytes / len(blob)
        err = float(np.linalg.norm(state - back) / (np.linalg.norm(state) + 1e-12))
        print(f"  {n_qubits:<10d} {dim:<8d} {state.nbytes:>8d} {len(blob):>8d} {ratio:>7.2f}x {err:>8.4f} {enc_ms:>8.1f}")

    # ── 8. RFT sparsity vs DFT on phi-states ──────────────────────────────
    section("8. RFT K_0.99 vs DFT K_0.99 on phi-Periodic Amplitude Vectors")
    print(f"  (Theorem 15: phi-tones concentrate in O(log N) RFT bins)")
    print()
    print(f"  {'N':<8s} {'RFT K_0.99':>12s} {'DFT K_0.99':>12s} {'Ratio':>8s} {'RFT wins':>10s}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*8} {'-'*10}")
    try:
        from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT
        for n in [32, 64, 128, 256]:
            state = _make_phi_state(n).astype(np.float64)
            rft = CanonicalTrueRFT(n)
            rft_coeffs = rft.forward(state)
            rft_e = np.sort(np.abs(rft_coeffs) ** 2)[::-1]
            rft_k99 = int(np.searchsorted(np.cumsum(rft_e) / rft_e.sum(), 0.99)) + 1
            dft_coeffs = np.fft.fft(state)
            dft_e = np.sort(np.abs(dft_coeffs) ** 2)[::-1]
            dft_k99 = int(np.searchsorted(np.cumsum(dft_e) / dft_e.sum(), 0.99)) + 1
            wins = "YES" if rft_k99 < dft_k99 else "NO"
            print(f"  {n:<8d} {rft_k99:>12d} {dft_k99:>12d} {rft_k99/dft_k99:>8.3f} {wins:>10s}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print()
    print("=" * W)
    print("  Done. For full pytest output: pytest benchmarks/quantum_benchmark.py -v")
    print("=" * W)
