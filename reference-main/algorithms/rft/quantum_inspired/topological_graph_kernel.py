#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Callable
import math
import cmath
import numpy as np
from numpy import linalg as LA

from algorithms.rft.quantum_inspired.surface_topology_core import (
    SurfaceMesh,
    SurfaceTopology,
    Edge,
    VertexID,
    canonical_edge,
    compute_surface_topology,
    triangulate_torus,
    triangulate_klein_bottle,
    build_edge_phase_field,
    fundamental_torus_cycles,
    wilson_loop,
    path_holonomy,
)


# ── SU(2) PAULI BASIS ─────────────────────────────────────────────────────────

_I2 = np.eye(2, dtype=complex)
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)

PAULI = {"I": _I2, "X": _SX, "Y": _SY, "Z": _SZ}


def su2_rotation(axis: np.ndarray, theta: float) -> np.ndarray:
    """
    SU(2) rotation matrix: exp(-i * theta/2 * n.sigma)
    axis: unit 3-vector [nx, ny, nz]
    """
    nx, ny, nz = axis / LA.norm(axis)
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return c * _I2 - 1j * s * (nx * _SX + ny * _SY + nz * _SZ)


def u1_phase_matrix(phi: float) -> np.ndarray:
    """Diagonal U(1) matrix: diag(e^{i phi}, e^{-i phi})"""
    return np.diag([cmath.exp(1j * phi), cmath.exp(-1j * phi)])


# ── PHI-GRID STATE MAPPING ────────────────────────────────────────────────────

PHI = (1.0 + math.sqrt(5.0)) / 2.0


def phi_grid_state(vertex_id: int, dim: int = 2) -> np.ndarray:
    """
    Map vertex index onto a normalized complex state vector using
    quasi-periodic golden-ratio sampling.

    dim=2: qubit (SU(2) target space)
    dim=4: two-qubit / ququart
    """
    angles = [2.0 * math.pi * ((PHI ** k * vertex_id) % 1.0) for k in range(1, dim + 1)]
    raw = np.array([cmath.exp(1j * a) for a in angles], dtype=complex)
    return raw / LA.norm(raw)


def mesh_state_map(mesh: SurfaceMesh, dim: int = 2) -> Dict[int, np.ndarray]:
    """Assign a phi-grid quantum state to every vertex."""
    return {v: phi_grid_state(v, dim=dim) for v in mesh.vertices}


# ── LOCAL SU(2) EDGE OPERATORS ────────────────────────────────────────────────

@dataclass
class EdgeOperator:
    """
    SU(2) operator assigned to a directed edge (a -> b).
    Represents a discrete gauge connection / parallel transporter.
    """
    edge: Edge
    matrix: np.ndarray          # 2x2 complex unitary
    u1_phase: complex            # scalar U(1) component from phase field

    def adjoint(self) -> "EdgeOperator":
        return EdgeOperator(
            edge=(self.edge[1], self.edge[0]),
            matrix=self.matrix.conj().T,
            u1_phase=np.conj(self.u1_phase),
        )


def build_edge_operators(
    mesh: SurfaceMesh,
    edge_phase: Dict[Edge, complex],
    axis_fn: Optional[Callable[[int, int], np.ndarray]] = None,
) -> Dict[Edge, EdgeOperator]:
    """
    Attach SU(2) x U(1) operators to every canonical edge.

    axis_fn(a, b) -> 3-vector: rotation axis for edge (a,b).
    Default: axis derived from phi-grid vertex states (Bloch sphere projection).
    """
    state_map = mesh_state_map(mesh, dim=2)

    def default_axis(a: int, b: int) -> np.ndarray:
        sa, sb = state_map[a], state_map[b]
        def bloch(s):
            rho = np.outer(s, s.conj())
            return np.array([
                2 * rho[0, 1].real,
                2 * rho[0, 1].imag,
                (rho[0, 0] - rho[1, 1]).real,
            ])
        ax = bloch(sb) - bloch(sa)
        n = LA.norm(ax)
        return ax / n if n > 1e-12 else np.array([0., 0., 1.])

    ops: Dict[Edge, EdgeOperator] = {}
    for a, b in mesh.edges():
        u1 = edge_phase[(a, b)]
        theta = 2.0 * abs(cmath.phase(u1))
        axis = axis_fn(a, b) if axis_fn else default_axis(a, b)
        mat = su2_rotation(axis, theta)
        ops[(a, b)] = EdgeOperator(edge=(a, b), matrix=mat, u1_phase=u1)
    return ops


# ── DISCRETE TRANSPORT ────────────────────────────────────────────────────────

def su2_path_transport(
    path: List[int],
    edge_ops: Dict[Edge, EdgeOperator],
) -> np.ndarray:
    """
    Compute the ordered product of SU(2) matrices along a vertex path.
    Handles both forward and reverse edge traversal.
    Returns a 2x2 unitary matrix (the holonomy matrix).
    """
    U = np.eye(2, dtype=complex)
    for a, b in zip(path[:-1], path[1:]):
        e = canonical_edge(a, b)
        op = edge_ops[e]
        if (a, b) == e:
            U = op.matrix @ U
        else:
            U = op.adjoint().matrix @ U
    return U


def su2_wilson_loop(
    path: List[int],
    edge_ops: Dict[Edge, EdgeOperator],
) -> Tuple[np.ndarray, complex]:
    """
    Compute SU(2) holonomy matrix and its U(1) trace for a closed loop.
    Returns: (holonomy_matrix 2x2, wilson_trace complex)
    wilson_trace = (1/2) * Tr[U] — the standard Wilson loop observable
    """
    if path[0] != path[-1]:
        raise ValueError("Wilson loop path must be closed.")
    U = su2_path_transport(path, edge_ops)
    W = 0.5 * np.trace(U)
    return U, W


# ── MEASURABLE OUTPUTS ────────────────────────────────────────────────────────

@dataclass
class CycleHolonomy:
    cycle_name: str
    path: List[int]
    u1_holonomy: complex
    su2_matrix: np.ndarray
    wilson_trace: complex
    phase_arg: float
    deviation_from_identity: float


@dataclass
class QuasiPeriodicResponse:
    vertex: int
    state: np.ndarray
    bloch: np.ndarray
    phase_mod: float
    resonance_index: float


@dataclass
class LoopInterferenceMetric:
    cycle_a: str
    cycle_b: str
    overlap: complex
    interference: float
    phase_difference: float


@dataclass
class GraphSpectralData:
    eigenvalues: np.ndarray
    spectral_gap: float
    algebraic_connectivity: float
    spectral_spread: float


def compute_cycle_holonomies(
    mesh: SurfaceMesh,
    cycles: Dict[str, List[int]],
    edge_phase: Dict[Edge, complex],
    edge_ops: Dict[Edge, EdgeOperator],
) -> List[CycleHolonomy]:
    out = []
    for name, path in cycles.items():
        u1 = wilson_loop(path, edge_phase)
        U, W = su2_wilson_loop(path, edge_ops)
        dev = LA.norm(U - np.eye(2, dtype=complex), "fro")
        out.append(CycleHolonomy(
            cycle_name=name,
            path=path,
            u1_holonomy=u1,
            su2_matrix=U,
            wilson_trace=W,
            phase_arg=cmath.phase(u1),
            deviation_from_identity=dev,
        ))
    return out


def compute_quasi_periodic_response(
    mesh: SurfaceMesh,
    vertex_ids: Optional[List[int]] = None,
) -> List[QuasiPeriodicResponse]:
    vids = vertex_ids if vertex_ids is not None else list(mesh.vertices.keys())
    out = []
    for v in vids:
        psi = phi_grid_state(v, dim=2)
        rho = np.outer(psi, psi.conj())
        bloch = np.array([
            2 * rho[0, 1].real,
            2 * rho[0, 1].imag,
            (rho[0, 0] - rho[1, 1]).real,
        ])
        phase_mod = (PHI * v) % 1.0
        resonance = abs(bloch[2])
        out.append(QuasiPeriodicResponse(v, psi, bloch, phase_mod, resonance))
    return out


def compute_loop_interference(
    holonomies: List[CycleHolonomy],
) -> List[LoopInterferenceMetric]:
    metrics = []
    n = len(holonomies)
    for i in range(n):
        for j in range(i + 1, n):
            ha, hb = holonomies[i], holonomies[j]
            overlap = ha.wilson_trace * np.conj(hb.wilson_trace)
            metrics.append(LoopInterferenceMetric(
                cycle_a=ha.cycle_name,
                cycle_b=hb.cycle_name,
                overlap=overlap,
                interference=abs(overlap),
                phase_difference=cmath.phase(overlap),
            ))
    return metrics


def compute_graph_spectrum(mesh: SurfaceMesh) -> GraphSpectralData:
    """
    Compute graph Laplacian spectrum of the 1-skeleton.
    L = D - A
    """
    vlist = sorted(mesh.vertices.keys())
    idx = {v: i for i, v in enumerate(vlist)}
    n = len(vlist)
    A = np.zeros((n, n), dtype=float)
    for a, b in mesh.edges():
        i, j = idx[a], idx[b]
        A[i, j] = A[j, i] = 1.0
    D = np.diag(A.sum(axis=1))
    L = D - A
    eigvals = np.sort(LA.eigvalsh(L).real)
    nz = eigvals[eigvals > 1e-10]
    gap = float(nz[0]) if len(nz) > 0 else 0.0
    return GraphSpectralData(
        eigenvalues=eigvals,
        spectral_gap=gap,
        algebraic_connectivity=gap,
        spectral_spread=float(eigvals[-1] - eigvals[0]),
    )


# ── TOP-LEVEL KERNEL ──────────────────────────────────────────────────────────

@dataclass
class TopologicalKernelResult:
    surface_type: str
    topology: SurfaceTopology
    cycle_holonomies: List[CycleHolonomy]
    quasi_periodic: List[QuasiPeriodicResponse]
    loop_interference: List[LoopInterferenceMetric]
    spectral: GraphSpectralData


def run_topological_kernel(
    mesh: SurfaceMesh,
    cycles: Dict[str, List[int]],
) -> TopologicalKernelResult:
    topo = compute_surface_topology(mesh)
    edge_phase = build_edge_phase_field(mesh)
    edge_ops = build_edge_operators(mesh, edge_phase)

    holonomies = compute_cycle_holonomies(mesh, cycles, edge_phase, edge_ops)
    qpr = compute_quasi_periodic_response(mesh, list(mesh.vertices.keys())[:16])
    interference = compute_loop_interference(holonomies)
    spectral = compute_graph_spectrum(mesh)

    return TopologicalKernelResult(
        surface_type=mesh.metadata.get("surface_type", "unknown"),
        topology=topo,
        cycle_holonomies=holonomies,
        quasi_periodic=qpr,
        loop_interference=interference,
        spectral=spectral,
    )


# ── DEMO ──────────────────────────────────────────────────────────────────────

def _demo() -> None:
    print("=" * 72)
    print("TOPOLOGICAL GRAPH KERNEL — DEMO")
    print("=" * 72)

    torus = triangulate_torus(8, 10)
    cycles = fundamental_torus_cycles(8, 10)
    result = run_topological_kernel(torus, cycles)

    t = result.topology
    print(f"Surface : {result.surface_type}")
    print(f"Topology: V={t.vertex_count} E={t.edge_count} F={t.face_count} "
          f"chi={t.euler_characteristic} genus={t.genus} orientable={t.orientable}")
    print()

    print("Cycle Holonomies:")
    for h in result.cycle_holonomies:
        print(f"  [{h.cycle_name}]")
        print(f"    U(1) holonomy  : {h.u1_holonomy:.6f}  arg={h.phase_arg:.6f} rad")
        print(f"    SU(2) W-trace  : {h.wilson_trace:.6f}")
        print(f"    ||U - I||_F    : {h.deviation_from_identity:.6f}")
    print()

    print("Quasi-Periodic Response (first 4 vertices):")
    for q in result.quasi_periodic[:4]:
        print(f"  v={q.vertex:3d}  phase_mod={q.phase_mod:.6f}  "
              f"resonance={q.resonance_index:.6f}  bloch={np.round(q.bloch,4)}")
    print()

    if result.loop_interference:
        li = result.loop_interference[0]
        print(f"Loop Interference ({li.cycle_a} x {li.cycle_b}):")
        print(f"  overlap={li.overlap:.6f}  |interference|={li.interference:.6f}  "
              f"phase_diff={li.phase_difference:.6f} rad")
    print()

    s = result.spectral
    print(f"Graph Spectrum:")
    print(f"  spectral_gap (Fiedler) = {s.spectral_gap:.6f}")
    print(f"  spectral_spread        = {s.spectral_spread:.6f}")
    print(f"  eigenvalues (first 6)  = {np.round(s.eigenvalues[:6], 4)}")
    print("=" * 72)


if __name__ == "__main__":
    _demo()
