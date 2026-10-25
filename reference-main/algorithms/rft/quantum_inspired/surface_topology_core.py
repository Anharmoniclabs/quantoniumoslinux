#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Iterable, Optional
import math
import cmath
import numpy as np


VertexID = int
Edge = Tuple[int, int]
Face = Tuple[int, int, int]


def canonical_edge(a: int, b: int) -> Edge:
    if a == b:
        raise ValueError("Self-loop edge is not allowed in a simplicial surface.")
    return (a, b) if a < b else (b, a)


@dataclass
class SurfaceMesh:
    """
    Pure topological/combinatorial surface mesh with optional vertex coordinates.

    vertices:
        Dict[vertex_id] -> np.ndarray([x, y, z]) or other coordinate vector.
        Coordinates are optional geometry; topology does not depend on them.

    faces:
        List of oriented triangles (i, j, k). Their ordering matters for
        orientability checks and boundary extraction.
    """
    vertices: Dict[VertexID, np.ndarray]
    faces: List[Face]
    metadata: Dict[str, object] = field(default_factory=dict)

    def edges(self) -> Set[Edge]:
        e: Set[Edge] = set()
        for i, j, k in self.faces:
            e.add(canonical_edge(i, j))
            e.add(canonical_edge(j, k))
            e.add(canonical_edge(k, i))
        return e

    def face_edges_oriented(self, face: Face) -> List[Tuple[int, int]]:
        i, j, k = face
        return [(i, j), (j, k), (k, i)]

    def edge_face_incidence(self) -> Dict[Edge, List[int]]:
        incidence: Dict[Edge, List[int]] = {e: [] for e in self.edges()}
        for f_idx, face in enumerate(self.faces):
            for a, b in self.face_edges_oriented(face):
                incidence[canonical_edge(a, b)].append(f_idx)
        return incidence

    def oriented_edge_face_incidence(self) -> Dict[Edge, List[Tuple[int, int]]]:
        """
        Returns:
            canonical_edge -> [(face_index, sign), ...]
        sign = +1 if face uses edge in canonical direction, -1 otherwise
        """
        incidence: Dict[Edge, List[Tuple[int, int]]] = {e: [] for e in self.edges()}
        for f_idx, face in enumerate(self.faces):
            for a, b in self.face_edges_oriented(face):
                e = canonical_edge(a, b)
                sign = +1 if (a, b) == e else -1
                incidence[e].append((f_idx, sign))
        return incidence

    def vertex_neighbors(self) -> Dict[int, Set[int]]:
        nbrs: Dict[int, Set[int]] = {v: set() for v in self.vertices}
        for a, b in self.edges():
            nbrs[a].add(b)
            nbrs[b].add(a)
        return nbrs

    def boundary_edges(self) -> Set[Edge]:
        incidence = self.edge_face_incidence()
        return {e for e, faces in incidence.items() if len(faces) == 1}

    def boundary_vertices(self) -> Set[int]:
        out: Set[int] = set()
        for a, b in self.boundary_edges():
            out.add(a)
            out.add(b)
        return out

    def is_closed(self) -> bool:
        return len(self.boundary_edges()) == 0

    def validate(self) -> None:
        if not self.faces:
            raise ValueError("SurfaceMesh must contain at least one face.")
        vertex_ids = set(self.vertices.keys())
        for f_idx, (i, j, k) in enumerate(self.faces):
            if len({i, j, k}) != 3:
                raise ValueError(f"Degenerate face at index {f_idx}: {(i, j, k)}")
            if i not in vertex_ids or j not in vertex_ids or k not in vertex_ids:
                raise ValueError(f"Face {f_idx} references missing vertices.")
        incidence = self.edge_face_incidence()
        for e, uses in incidence.items():
            if len(uses) > 2:
                raise ValueError(
                    f"Non-manifold edge {e}: used by {len(uses)} faces (expected <= 2)."
                )


@dataclass
class SurfaceTopology:
    vertex_count: int
    edge_count: int
    face_count: int
    euler_characteristic: int
    orientable: bool
    genus: Optional[int]
    crosscap_number: Optional[int]
    boundary_component_count: int
    closed: bool


def _build_face_adjacency(mesh: SurfaceMesh) -> Dict[int, List[Tuple[int, Edge, int, int]]]:
    oriented_incidence = mesh.oriented_edge_face_incidence()
    adj: Dict[int, List[Tuple[int, Edge, int, int]]] = {i: [] for i in range(len(mesh.faces))}
    for edge, entries in oriented_incidence.items():
        if len(entries) == 2:
            (f0, s0), (f1, s1) = entries
            adj[f0].append((f1, edge, s0, s1))
            adj[f1].append((f0, edge, s1, s0))
    return adj


def _is_orientable(mesh: SurfaceMesh) -> bool:
    adj = _build_face_adjacency(mesh)
    if not adj:
        return True

    face_flip: Dict[int, int] = {}

    for start in range(len(mesh.faces)):
        if start in face_flip:
            continue
        face_flip[start] = 1
        stack = [start]

        while stack:
            f = stack.pop()
            for g, _edge, s_f, s_g in adj[f]:
                required = -face_flip[f] * s_f * s_g
                if g in face_flip:
                    if face_flip[g] != required:
                        return False
                else:
                    face_flip[g] = required
                    stack.append(g)
    return True


def _count_boundary_components(mesh: SurfaceMesh) -> int:
    b_edges = mesh.boundary_edges()
    if not b_edges:
        return 0

    b_adj: Dict[int, Set[int]] = {}
    for a, b in b_edges:
        b_adj.setdefault(a, set()).add(b)
        b_adj.setdefault(b, set()).add(a)

    seen: Set[int] = set()
    components = 0
    for v in b_adj:
        if v in seen:
            continue
        components += 1
        stack = [v]
        seen.add(v)
        while stack:
            x = stack.pop()
            for y in b_adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
    return components


def compute_surface_topology(mesh: SurfaceMesh) -> SurfaceTopology:
    mesh.validate()

    V = len(mesh.vertices)
    E = len(mesh.edges())
    F = len(mesh.faces)
    chi = V - E + F
    closed = mesh.is_closed()
    orientable = _is_orientable(mesh)
    b = _count_boundary_components(mesh)

    genus: Optional[int] = None
    crosscap_number: Optional[int] = None

    if closed:
        if orientable:
            if (2 - chi) % 2 == 0:
                genus = (2 - chi) // 2
            else:
                genus = None
        else:
            crosscap_number = 2 - chi

    return SurfaceTopology(
        vertex_count=V,
        edge_count=E,
        face_count=F,
        euler_characteristic=chi,
        orientable=orientable,
        genus=genus,
        crosscap_number=crosscap_number,
        boundary_component_count=b,
        closed=closed,
    )


def _torus_xyz(u: float, v: float, R: float = 3.0, r: float = 1.0) -> np.ndarray:
    cu, su = math.cos(u), math.sin(u)
    cv, sv = math.cos(v), math.sin(v)
    x = (R + r * cv) * cu
    y = (R + r * cv) * su
    z = r * sv
    return np.array([x, y, z], dtype=float)


def _klein_xyz(u: float, v: float, scale: float = 2.0) -> np.ndarray:
    cu, su = math.cos(u), math.sin(u)
    cv, sv = math.cos(v), math.sin(v)

    if 0 <= u < math.pi:
        x = (6 * cu * (1 + su) + 4 * (1 - cu / 2) * cv) / 8.0
        z = (16 * su + 4 * (1 - cu / 2) * sv) / 8.0
    else:
        x = (6 * cu * (1 + su) + 4 * (1 - cu / 2) * math.cos(v + math.pi)) / 8.0
        z = (16 * su) / 8.0

    y = (4 * (1 - cu / 2) * sv) / 8.0
    return scale * np.array([x, y, z], dtype=float)


def triangulate_torus(nu: int, nv: int, R: float = 3.0, r: float = 1.0) -> SurfaceMesh:
    if nu < 3 or nv < 3:
        raise ValueError("torus resolution must satisfy nu,nv >= 3")

    vertices: Dict[int, np.ndarray] = {}

    def vid(i: int, j: int) -> int:
        return i * nv + j

    for i in range(nu):
        u = 2.0 * math.pi * i / nu
        for j in range(nv):
            v = 2.0 * math.pi * j / nv
            vertices[vid(i, j)] = _torus_xyz(u, v, R=R, r=r)

    faces: List[Face] = []
    for i in range(nu):
        i1 = (i + 1) % nu
        for j in range(nv):
            j1 = (j + 1) % nv
            a = vid(i, j)
            b = vid(i1, j)
            c = vid(i1, j1)
            d = vid(i, j1)
            faces.append((a, b, c))
            faces.append((a, c, d))

    return SurfaceMesh(
        vertices=vertices,
        faces=faces,
        metadata={
            "surface_type": "torus",
            "nu": nu,
            "nv": nv,
            "closed": True,
            "identification": "periodic_u_periodic_v",
        },
    )


def triangulate_klein_bottle(nu: int, nv: int, scale: float = 2.0) -> SurfaceMesh:
    if nu < 3 or nv < 3:
        raise ValueError("klein bottle resolution must satisfy nu,nv >= 3")

    vertices: Dict[int, np.ndarray] = {}

    def raw_vid(i: int, j: int) -> int:
        return i * nv + j

    for i in range(nu):
        u = 2.0 * math.pi * i / nu
        for j in range(nv):
            v = 2.0 * math.pi * j / nv
            vertices[raw_vid(i, j)] = _klein_xyz(u, v, scale=scale)

    def seam_v(j: int) -> int:
        return (nv - j) % nv

    faces: List[Face] = []
    for i in range(nu):
        i1 = i + 1
        wrap_twist = (i1 == nu)

        for j in range(nv):
            j1 = (j + 1) % nv

            a = raw_vid(i, j)
            d = raw_vid(i, j1)

            if not wrap_twist:
                b = raw_vid(i1, j)
                c = raw_vid(i1, j1)
            else:
                b = raw_vid(0, seam_v(j))
                c = raw_vid(0, seam_v(j1))

            faces.append((a, b, c))
            faces.append((a, c, d))

    return SurfaceMesh(
        vertices=vertices,
        faces=faces,
        metadata={
            "surface_type": "klein_bottle",
            "nu": nu,
            "nv": nv,
            "closed": True,
            "identification": "periodic_u_twisted_v",
        },
    )


def extract_face_boundary(face: Face) -> List[Tuple[int, int]]:
    i, j, k = face
    return [(i, j), (j, k), (k, i)]


def build_edge_phase_field(
    mesh: SurfaceMesh,
    phi: float = (1.0 + math.sqrt(5.0)) / 2.0,
    mode: str = "golden_vertex_phase",
) -> Dict[Edge, complex]:
    if mode != "golden_vertex_phase":
        raise ValueError(f"Unsupported phase mode: {mode}")

    vertex_phase: Dict[int, complex] = {}
    for v in mesh.vertices:
        theta = 2.0 * math.pi * ((phi * v) % 1.0)
        vertex_phase[v] = cmath.exp(1j * theta)

    edge_phase: Dict[Edge, complex] = {}
    for a, b in mesh.edges():
        edge_phase[(a, b)] = np.conj(vertex_phase[a]) * vertex_phase[b]
    return edge_phase


def path_holonomy(path: List[int], edge_phase: Dict[Edge, complex]) -> complex:
    if len(path) < 2:
        return 1.0 + 0.0j

    h = 1.0 + 0.0j
    for a, b in zip(path[:-1], path[1:]):
        e = canonical_edge(a, b)
        if e not in edge_phase:
            raise ValueError(f"Path uses non-edge {(a, b)}")
        if (a, b) == e:
            h *= edge_phase[e]
        else:
            h *= np.conj(edge_phase[e])
    return h


def wilson_loop(path: List[int], edge_phase: Dict[Edge, complex]) -> complex:
    if len(path) < 3 or path[0] != path[-1]:
        raise ValueError("Wilson loop path must be closed.")
    return path_holonomy(path, edge_phase)


def fundamental_torus_cycles(nu: int, nv: int) -> Dict[str, List[int]]:
    if nu < 3 or nv < 3:
        raise ValueError("Need nu,nv >= 3")

    def vid(i: int, j: int) -> int:
        return i * nv + j

    meridian = [vid(i, 0) for i in range(nu)] + [vid(0, 0)]
    longitude = [vid(0, j) for j in range(nv)] + [vid(0, 0)]

    return {
        "u_cycle": meridian,
        "v_cycle": longitude,
    }


def summarize_surface(mesh: SurfaceMesh) -> Dict[str, object]:
    topo = compute_surface_topology(mesh)
    return {
        "metadata": dict(mesh.metadata),
        "vertex_count": topo.vertex_count,
        "edge_count": topo.edge_count,
        "face_count": topo.face_count,
        "euler_characteristic": topo.euler_characteristic,
        "orientable": topo.orientable,
        "genus": topo.genus,
        "crosscap_number": topo.crosscap_number,
        "boundary_component_count": topo.boundary_component_count,
        "closed": topo.closed,
    }


def _demo() -> None:
    print("=" * 72)
    print("TOPOLOGY ENGINE DEMO")
    print("=" * 72)

    torus = triangulate_torus(8, 10)
    torus_topo = compute_surface_topology(torus)
    print("Torus:")
    print(f"  V={torus_topo.vertex_count}, E={torus_topo.edge_count}, F={torus_topo.face_count}")
    print(f"  chi={torus_topo.euler_characteristic}, orientable={torus_topo.orientable}, genus={torus_topo.genus}")

    klein = triangulate_klein_bottle(8, 10)
    klein_topo = compute_surface_topology(klein)
    print("Klein bottle:")
    print(f"  V={klein_topo.vertex_count}, E={klein_topo.edge_count}, F={klein_topo.face_count}")
    print(
        f"  chi={klein_topo.euler_characteristic}, orientable={klein_topo.orientable}, "
        f"crosscap_number={klein_topo.crosscap_number}"
    )

    edge_phase = build_edge_phase_field(torus)
    cycles = fundamental_torus_cycles(8, 10)

    Wu = wilson_loop(cycles["u_cycle"], edge_phase)
    Wv = wilson_loop(cycles["v_cycle"], edge_phase)

    print("Torus Wilson loops:")
    print(f"  W_u = {Wu}  |arg|={abs(cmath.phase(Wu)):.6f}")
    print(f"  W_v = {Wv}  |arg|={abs(cmath.phase(Wv)):.6f}")

    print("=" * 72)


if __name__ == "__main__":
    _demo()
