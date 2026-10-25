#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
# operator_algebra_v2.py — adds all symbols required by benchmark_suite.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math
import cmath
import numpy as np
from numpy import linalg as LA

from algorithms.rft.quantum_inspired.surface_topology_core import (
    SurfaceMesh, Edge, canonical_edge,
    triangulate_torus, build_edge_phase_field,
    fundamental_torus_cycles, wilson_loop,
)
from algorithms.rft.quantum_inspired.topological_graph_kernel import (
    EdgeOperator, phi_grid_state, mesh_state_map,
    PHI, _I2, _SX, _SY, _SZ, PAULI, su2_rotation,
)


# ── CORE MATH UTILITIES ───────────────────────────────────────────────────────

def fro_norm(A: np.ndarray) -> float:
    """Frobenius norm."""
    return float(LA.norm(A, "fro"))

def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A

def anticommutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B + B @ A

def commutator_norm(A: np.ndarray, B: np.ndarray) -> float:
    return fro_norm(commutator(A, B))


# ── STATE OBSERVABLES ─────────────────────────────────────────────────────────

def expectation_value(psi: np.ndarray, O: np.ndarray) -> complex:
    """<psi|O|psi>"""
    return complex(psi.conj() @ O @ psi)

def variance_value(psi: np.ndarray, O: np.ndarray) -> float:
    """Var(O) = <O^2> - <O>^2"""
    ev = expectation_value(psi, O)
    return float((expectation_value(psi, O @ O) - ev**2).real)

def bloch_vector(psi: np.ndarray) -> np.ndarray:
    """Bloch vector [<X>,<Y>,<Z>] for a 2-component pure state."""
    rho = np.outer(psi, psi.conj())
    return np.array([
        2 * rho[0, 1].real,
        2 * rho[0, 1].imag,
        (rho[0, 0] - rho[1, 1]).real,
    ])

def vertex_expectation(mesh: SurfaceMesh, pauli: str = "Z") -> Dict[int, float]:
    O = PAULI[pauli]
    return {v: expectation_value(psi, O).real for v, psi in mesh_state_map(mesh).items()}


# ── EDGE OPERATOR FACTORIES ───────────────────────────────────────────────────

def build_curved_edge_operators(
    mesh: SurfaceMesh,
    base_edge_phase: Dict[Edge, complex],
    field_strength: float = 0.0,
    field_mode: str = "geometric_flux",
) -> Dict[Edge, EdgeOperator]:
    """
    Build SU(2) edge operators with injected field strength.

    The rotation axis for each edge is derived from the difference of
    phi-grid Bloch vectors at its endpoints, ensuring the commutator
    with sigma_Z is non-trivial when field_strength > 0.

    base_edge_phase provides the U(1) component carried on each edge.
    field_strength is added to the base rotation angle.
    """
    sm = mesh_state_map(mesh)
    ops: Dict[Edge, EdgeOperator] = {}
    for a, b in mesh.edges():
        u1 = base_edge_phase[(a, b)]
        base_theta = 2.0 * abs(cmath.phase(u1))
        ax = bloch_vector(sm[b]) - bloch_vector(sm[a])
        n = LA.norm(ax)
        axis = ax / n if n > 1e-12 else np.array([1., 0., 0.])
        mat = su2_rotation(axis, base_theta + field_strength)
        ops[(a, b)] = EdgeOperator(edge=(a, b), matrix=mat, u1_phase=u1)
    return ops


# ── TRANSPORT HELPERS ─────────────────────────────────────────────────────────

def su2_transport(path: List[int], edge_ops: Dict[Edge, EdgeOperator]) -> np.ndarray:
    """Ordered product of SU(2) matrices along a vertex path."""
    U = np.eye(2, dtype=complex)
    for a, b in zip(path[:-1], path[1:]):
        e = canonical_edge(a, b)
        op = edge_ops[e]
        U = (op.matrix @ U) if (a, b) == e else (op.matrix.conj().T @ U)
    return U

def su2_wilson(path: List[int], edge_ops: Dict[Edge, EdgeOperator]) -> Tuple[np.ndarray, complex]:
    """SU(2) holonomy matrix and (1/2)Tr[U] for a closed loop."""
    if path[0] != path[-1]:
        raise ValueError("Wilson loop path must be closed.")
    U = su2_transport(path, edge_ops)
    return U, complex(0.5 * np.trace(U))

def u1_wilson_from_edge_ops(path: List[int], edge_ops: Dict[Edge, EdgeOperator]) -> complex:
    """Scalar U(1) holonomy from edge operator u1_phase field."""
    if path[0] != path[-1]:
        raise ValueError("Wilson loop path must be closed.")
    h = 1. + 0j
    for a, b in zip(path[:-1], path[1:]):
        e = canonical_edge(a, b)
        u1 = edge_ops[e].u1_phase
        h *= u1 if (a, b) == e else np.conj(u1)
    return h


# ── CURVATURE OBSERVABLES ─────────────────────────────────────────────────────

def curvature_observable_su2(
    path: List[int],
    edge_ops: Dict[Edge, EdgeOperator],
) -> float:
    """||U_loop - I||_F  (0 = flat, max = 2*sqrt(2))."""
    if path[0] != path[-1]:
        raise ValueError("Must be a closed path.")
    U = su2_transport(path, edge_ops)
    return fro_norm(U - np.eye(2, dtype=complex))

def curvature_observable_u1(
    path: List[int],
    edge_ops: Dict[Edge, EdgeOperator],
) -> float:
    """Phase deviation of the U(1) Wilson loop from 1."""
    w = u1_wilson_from_edge_ops(path, edge_ops)
    return float(abs(cmath.phase(w)))


@dataclass
class CurvatureField:
    face_curvatures: Dict[int, float]
    wilson_traces: Dict[int, complex]
    mean_curvature: float
    max_curvature: float
    total_curvature: float

def compute_curvature_field(
    mesh: SurfaceMesh,
    edge_ops: Dict[Edge, EdgeOperator],
) -> CurvatureField:
    fc: Dict[int, float] = {}
    wt: Dict[int, complex] = {}
    for fi, face in enumerate(mesh.faces):
        i, j, k = face
        path = [i, j, k, i]
        U = su2_transport(path, edge_ops)
        fc[fi] = fro_norm(U - np.eye(2, dtype=complex))
        wt[fi] = complex(0.5 * np.trace(U))
    vals = list(fc.values())
    return CurvatureField(
        face_curvatures=fc,
        wilson_traces=wt,
        mean_curvature=float(np.mean(vals)),
        max_curvature=float(np.max(vals)),
        total_curvature=float(np.sum(vals)),
    )


# ── LOCAL COMMUTATOR SCAN ─────────────────────────────────────────────────────

def local_commutator_scan(
    mesh: SurfaceMesh,
    edge_ops: Dict[Edge, EdgeOperator],
    limit: int = 256,
) -> List[Tuple[Edge, Edge, float]]:
    """
    For each pair of edges sharing a vertex, compute ||[U_e1, U_e2]||_F.
    Returns list of (e1, e2, defect).
    """
    adj: Dict[int, set] = {}
    for a, b in mesh.edges():
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    results: List[Tuple[Edge, Edge, float]] = []
    seen: set = set()
    for v, neighbors in adj.items():
        nlist = sorted(neighbors)
        for i, n1 in enumerate(nlist):
            for n2 in nlist[i + 1:]:
                e1 = canonical_edge(v, n1)
                e2 = canonical_edge(v, n2)
                key = (min(e1, e2), max(e1, e2))
                if key in seen:
                    continue
                seen.add(key)
                if e1 in edge_ops and e2 in edge_ops:
                    A = edge_ops[e1].matrix
                    B = edge_ops[e2].matrix
                    results.append((e1, e2, fro_norm(A @ B - B @ A)))
                if len(results) >= limit:
                    return results
    return results


# ── DEMO ──────────────────────────────────────────────────────────────────────

def _demo() -> None:
    print("=" * 72)
    print("OPERATOR ALGEBRA v2 — DEMO")
    print("=" * 72)
    torus = triangulate_torus(8, 10)
    base_ep = build_edge_phase_field(torus)
    for fs in [0.0, math.pi/4, math.pi/2, math.pi]:
        ops = build_curved_edge_operators(torus, base_ep, field_strength=fs)
        cf = compute_curvature_field(torus, ops)
        scan = local_commutator_scan(torus, ops, limit=64)
        mean_def = float(np.mean([d for _,_,d in scan])) if scan else 0.
        print(f"  fs={fs:.4f}  mean_curv={cf.mean_curvature:.4f}  "
              f"max_curv={cf.max_curvature:.4f}  mean_comm_def={mean_def:.4f}")
    print("=" * 72)

if __name__ == "__main__":
    _demo()
