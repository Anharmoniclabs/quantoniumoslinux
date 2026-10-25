#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Iterable, Any
import math
import cmath
import json

import numpy as np
from numpy import linalg as LA

from algorithms.rft.quantum_inspired.surface_topology_core import (
    SurfaceMesh, Edge, canonical_edge,
    triangulate_torus, triangulate_klein_bottle,
    build_edge_phase_field, fundamental_torus_cycles,
    wilson_loop as surface_u1_wilson_loop,
)
from algorithms.rft.quantum_inspired.topological_graph_kernel import (
    PHI, PAULI, EdgeOperator, phi_grid_state, mesh_state_map,
    build_edge_operators,
    _I2, _SX, _SY, _SZ,
)
from algorithms.rft.quantum_inspired.operator_algebra_v2 import (
    expectation_value, variance_value, bloch_vector,
    build_curved_edge_operators,
    su2_transport, su2_wilson,
    u1_wilson_from_edge_ops,
    curvature_observable_su2, curvature_observable_u1,
    local_commutator_scan, fro_norm,
)


# ── DATA MODELS ───────────────────────────────────────────────────────────────

@dataclass
class LoopMetrics:
    name: str
    length: int
    surface_u1: complex
    kernel_u1: complex
    curved_u1: complex
    kernel_su2_trace: complex
    curved_su2_trace: complex
    kernel_u1_curvature: float
    curved_u1_curvature: float
    kernel_su2_curvature: float
    curved_su2_curvature: float
    kernel_su2_deviation_from_I: float
    curved_su2_deviation_from_I: float

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        for k, v in list(out.items()):
            if isinstance(v, complex):
                out[k] = {"real": float(v.real), "imag": float(v.imag),
                           "abs": float(abs(v)), "phase": float(cmath.phase(v))}
        return out


@dataclass
class ScheduleMetrics:
    schedule_name: str
    path_length: int
    final_state_distance_vs_kernel: float
    final_state_distance_vs_curved: float
    kernel_transport_deviation_from_I: float
    curved_transport_deviation_from_I: float
    kernel_trace_real: float
    curved_trace_real: float
    accumulated_local_commutator_defect: float
    max_step_commutator_defect: float
    endpoint_expectation_X_real: float
    endpoint_expectation_Y_real: float
    endpoint_expectation_Z_real: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkReport:
    surface_type: str
    nu: int
    nv: int
    field_strength: float
    field_mode: str
    vertex_count: int
    edge_count: int
    mean_local_commutator_defect: float
    max_local_commutator_defect: float
    loop_metrics: List[LoopMetrics]
    schedule_metrics: List[ScheduleMetrics]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "surface_type": self.surface_type, "nu": self.nu, "nv": self.nv,
            "field_strength": self.field_strength, "field_mode": self.field_mode,
            "vertex_count": self.vertex_count, "edge_count": self.edge_count,
            "mean_local_commutator_defect": self.mean_local_commutator_defect,
            "max_local_commutator_defect": self.max_local_commutator_defect,
            "loop_metrics": [m.to_dict() for m in self.loop_metrics],
            "schedule_metrics": [m.to_dict() for m in self.schedule_metrics],
            "summary": self.summary,
        }


# ── UTILITIES ─────────────────────────────────────────────────────────────────

def complex_close_to_unit(z: complex, tol: float = 1e-10) -> bool:
    return abs(abs(z) - 1.0) < tol

def trace_real(U: np.ndarray) -> float:
    return float((0.5 * np.trace(U)).real)

def mean_or_zero(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0

def cycle_to_open_path(cycle: List[int]) -> List[int]:
    if len(cycle) >= 2 and cycle[0] == cycle[-1]:
        return cycle[:-1]
    return cycle[:]

def reverse_path(path: List[int]) -> List[int]:
    return list(reversed(path))

def concat_paths(*paths: List[int]) -> List[int]:
    out: List[int] = []
    for p in paths:
        if not p: continue
        if not out: out.extend(p)
        elif out[-1] == p[0]: out.extend(p[1:])
        else: out.extend(p)
    return out

def make_commutator_loop(u_cycle: List[int], v_cycle: List[int]) -> List[int]:
    u = cycle_to_open_path(u_cycle)
    v = cycle_to_open_path(v_cycle)
    path = concat_paths(u, v, reverse_path(u), reverse_path(v))
    if path[0] != path[-1]: path.append(path[0])
    return path

def adjacent_path_edges(path: List[int]) -> List[Tuple[Tuple[int,int], Tuple[int,int]]]:
    if len(path) < 3: return []
    steps = list(zip(path[:-1], path[1:]))
    return [(steps[i], steps[i+1]) for i in range(len(steps)-1)]

def local_path_commutator_profile(
    path: List[int], edge_ops: Dict[Edge, EdgeOperator]
) -> Tuple[float, float]:
    pairs = adjacent_path_edges(path)
    if not pairs: return 0.0, 0.0
    defects = []
    for (a1,b1),(a2,b2) in pairs:
        e1=canonical_edge(a1,b1); e2=canonical_edge(a2,b2)
        if e1 not in edge_ops or e2 not in edge_ops: continue
        A = edge_ops[e1].matrix if (a1,b1)==e1 else edge_ops[e1].matrix.conj().T
        B = edge_ops[e2].matrix if (a2,b2)==e2 else edge_ops[e2].matrix.conj().T
        defects.append(fro_norm(A@B - B@A))
    if not defects: return 0.0, 0.0
    return float(sum(defects)), float(max(defects))

def endpoint_expectations(state: np.ndarray) -> Tuple[float, float, float]:
    return (
        float(expectation_value(state, _SX).real),
        float(expectation_value(state, _SY).real),
        float(expectation_value(state, _SZ).real),
    )


# ── LOOP BENCHMARKS ───────────────────────────────────────────────────────────

def benchmark_loops(
    mesh: SurfaceMesh,
    cycles: Dict[str, List[int]],
    base_edge_phase: Dict[Edge, complex],
    kernel_edge_ops: Dict[Edge, EdgeOperator],
    curved_edge_ops: Dict[Edge, EdgeOperator],
) -> List[LoopMetrics]:
    out = []
    for name, path in cycles.items():
        su1 = surface_u1_wilson_loop(path, base_edge_phase)
        ku1 = u1_wilson_from_edge_ops(path, kernel_edge_ops)
        cu1 = u1_wilson_from_edge_ops(path, curved_edge_ops)
        Uk, Wk = su2_wilson(path, kernel_edge_ops)
        Uc, Wc = su2_wilson(path, curved_edge_ops)
        out.append(LoopMetrics(
            name=name, length=len(path)-1,
            surface_u1=su1, kernel_u1=ku1, curved_u1=cu1,
            kernel_su2_trace=Wk, curved_su2_trace=Wc,
            kernel_u1_curvature=curvature_observable_u1(path, kernel_edge_ops),
            curved_u1_curvature=curvature_observable_u1(path, curved_edge_ops),
            kernel_su2_curvature=curvature_observable_su2(path, kernel_edge_ops),
            curved_su2_curvature=curvature_observable_su2(path, curved_edge_ops),
            kernel_su2_deviation_from_I=float(fro_norm(Uk-_I2)),
            curved_su2_deviation_from_I=float(fro_norm(Uc-_I2)),
        ))
    return out


# ── SCHEDULE BENCHMARKS ───────────────────────────────────────────────────────

def build_schedule_family(mesh: SurfaceMesh, cycles: Dict[str, List[int]]) -> Dict[str, List[int]]:
    u_cycle = cycles["u_cycle"]; v_cycle = cycles["v_cycle"]
    u = cycle_to_open_path(u_cycle); v = cycle_to_open_path(v_cycle)
    schedules = {"u_only": u_cycle, "v_only": v_cycle}
    uv = concat_paths(u, v)
    if uv[0]!=uv[-1]: uv.append(uv[0])
    schedules["u_then_v"] = uv
    vu = concat_paths(v, u)
    if vu[0]!=vu[-1]: vu.append(vu[0])
    schedules["v_then_u"] = vu
    schedules["commutator_loop"] = make_commutator_loop(u_cycle, v_cycle)
    # Use face triangle from actual mesh for mixed_reverse
    face_tri = mesh.faces[0]
    upr = concat_paths(u[:max(2,len(u)//2)], reverse_path(v), u[max(2,len(u)//2):])
    if upr and upr[0]!=upr[-1]: upr.append(upr[0])
    schedules["mixed_reverse"] = upr

    # Validate all schedule paths against mesh edge set
    mesh_edges = mesh.edges()
    validated = {}
    for name, path in schedules.items():
        valid = True
        for a, b in zip(path[:-1], path[1:]):
            if canonical_edge(a, b) not in mesh_edges:
                valid = False
                break
        if valid:
            validated[name] = path
    return validated


def benchmark_schedules(
    schedules: Dict[str, List[int]],
    kernel_edge_ops: Dict[Edge, EdgeOperator],
    curved_edge_ops: Dict[Edge, EdgeOperator],
    initial_state: np.ndarray,
) -> List[ScheduleMetrics]:
    out = []
    for name, path in schedules.items():
        Uk = su2_transport(path, kernel_edge_ops)
        Uc = su2_transport(path, curved_edge_ops)
        psi_k = Uk @ initial_state
        psi_c = Uc @ initial_state
        acc_def, max_def = local_path_commutator_profile(path, curved_edge_ops)
        ex, ey, ez = endpoint_expectations(psi_c)
        out.append(ScheduleMetrics(
            schedule_name=name, path_length=len(path)-1,
            final_state_distance_vs_kernel=float(LA.norm(psi_k-initial_state)),
            final_state_distance_vs_curved=float(LA.norm(psi_c-initial_state)),
            kernel_transport_deviation_from_I=float(fro_norm(Uk-_I2)),
            curved_transport_deviation_from_I=float(fro_norm(Uc-_I2)),
            kernel_trace_real=trace_real(Uk),
            curved_trace_real=trace_real(Uc),
            accumulated_local_commutator_defect=acc_def,
            max_step_commutator_defect=max_def,
            endpoint_expectation_X_real=ex,
            endpoint_expectation_Y_real=ey,
            endpoint_expectation_Z_real=ez,
        ))
    return out


# ── FULL DRIVER ───────────────────────────────────────────────────────────────

def build_mesh(surface_type: str, nu: int, nv: int) -> SurfaceMesh:
    st = surface_type.lower()
    if st == "torus": return triangulate_torus(nu, nv)
    if st in {"klein", "klein_bottle"}: return triangulate_klein_bottle(nu, nv)
    raise ValueError(f"Unsupported surface_type: {surface_type}")


def run_benchmark(
    surface_type: str = "torus",
    nu: int = 8, nv: int = 10,
    field_strength: float = 0.0,
    field_mode: str = "geometric_flux",
    initial_vertex: int = 0,
) -> BenchmarkReport:
    mesh = build_mesh(surface_type, nu, nv)
    base_ep = build_edge_phase_field(mesh)
    kernel_ops = build_edge_operators(mesh, base_ep)
    curved_ops = build_curved_edge_operators(mesh, base_ep,
                                              field_strength=field_strength,
                                              field_mode=field_mode)
    if surface_type.lower() == "torus":
        cycles = fundamental_torus_cycles(nu, nv)
    else:
        def vid(i,j): return i*nv+j
        cycles = {
            "u_cycle": [vid(i,0) for i in range(nu)] + [vid(0,0)],
            "v_cycle": [vid(0,j) for j in range(nv)] + [vid(0,0)],
        }

    loop_m = benchmark_loops(mesh, cycles, base_ep, kernel_ops, curved_ops)
    schedules = build_schedule_family(mesh, cycles)
    initial_state = phi_grid_state(initial_vertex, dim=2)
    sched_m = benchmark_schedules(schedules, kernel_ops, curved_ops, initial_state)

    comm_scan = local_commutator_scan(mesh, curved_ops, limit=256)
    comm_vals = [x[2] for x in comm_scan]

    summary = {
        "surface_layer_flat_u1": all(complex_close_to_unit(m.surface_u1) for m in loop_m),
        "kernel_loop_mean_su2_curvature": mean_or_zero([m.kernel_su2_curvature for m in loop_m]),
        "curved_loop_mean_su2_curvature": mean_or_zero([m.curved_su2_curvature for m in loop_m]),
        "curved_loop_max_su2_curvature": max((m.curved_su2_curvature for m in loop_m), default=0.),
        "schedule_mean_commutator_accumulation": mean_or_zero([m.accumulated_local_commutator_defect for m in sched_m]),
        "schedule_max_final_state_distance": max((m.final_state_distance_vs_curved for m in sched_m), default=0.),
        "field_strength": field_strength,
        "field_mode": field_mode,
    }

    return BenchmarkReport(
        surface_type=surface_type, nu=nu, nv=nv,
        field_strength=field_strength, field_mode=field_mode,
        vertex_count=len(mesh.vertices), edge_count=len(mesh.edges()),
        mean_local_commutator_defect=mean_or_zero(comm_vals),
        max_local_commutator_defect=max(comm_vals) if comm_vals else 0.,
        loop_metrics=loop_m, schedule_metrics=sched_m, summary=summary,
    )


def run_field_sweep(
    field_strengths: Iterable[float],
    surface_type: str = "torus",
    nu: int = 8, nv: int = 10,
    field_mode: str = "geometric_flux",
) -> List[BenchmarkReport]:
    return [run_benchmark(surface_type=surface_type, nu=nu, nv=nv,
                          field_strength=s, field_mode=field_mode)
            for s in field_strengths]


def summarize_sweep(reports: List[BenchmarkReport]) -> Dict[str, Any]:
    if not reports: return {}
    mc = [r.summary["curved_loop_mean_su2_curvature"] for r in reports]
    return {
        "field_strengths": [r.field_strength for r in reports],
        "mean_loop_curvature": mc,
        "max_loop_curvature": [r.summary["curved_loop_max_su2_curvature"] for r in reports],
        "mean_schedule_commutator_accumulation": [r.summary["schedule_mean_commutator_accumulation"] for r in reports],
        "max_schedule_final_state_distance": [r.summary["schedule_max_final_state_distance"] for r in reports],
        "mean_loop_curvature_monotone_non_decreasing": all(
            mc[i] <= mc[i+1]+1e-12 for i in range(len(mc)-1)),
    }


def save_report_json(report: BenchmarkReport, filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, sort_keys=True)

def save_sweep_json(reports: List[BenchmarkReport], filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({"reports": [r.to_dict() for r in reports],
                   "summary": summarize_sweep(reports)}, f, indent=2, sort_keys=True)


# ── SELF-TEST ─────────────────────────────────────────────────────────────────

def _self_test() -> None:
    PASS="PASS"; FAIL="FAIL"
    results = []
    def chk(name, cond, detail=""):
        results.append((name, PASS if cond else FAIL, detail))

    r0 = run_benchmark(surface_type="torus", nu=8, nv=10, field_strength=0.0)
    chk("V>0", r0.vertex_count>0, f"V={r0.vertex_count}")
    chk("E>0", r0.edge_count>0,   f"E={r0.edge_count}")
    chk("Surface U1 unit", all(abs(abs(m.surface_u1)-1.)<1e-10 for m in r0.loop_metrics))
    chk("Kernel SU2 trace bounded",  all(abs(m.kernel_su2_trace.real)<=1.+1e-10 for m in r0.loop_metrics))
    chk("Curved SU2 trace bounded",  all(abs(m.curved_su2_trace.real)<=1.+1e-10 for m in r0.loop_metrics))

    sweep = run_field_sweep([0.0, math.pi/4, math.pi/2, math.pi], nu=8, nv=10)
    ss = summarize_sweep(sweep)
    chk("Sweep monotone", bool(ss["mean_loop_curvature_monotone_non_decreasing"]),
        str(ss["mean_loop_curvature"]))

    r1 = run_benchmark(nu=8, nv=10, field_strength=math.pi/2)
    moved = any(abs(a.curved_su2_curvature-b.curved_su2_curvature)>1e-8
                for a,b in zip(r0.loop_metrics, r1.loop_metrics))
    chk("Field moves curvature", moved)

    finite = all(
        np.isfinite(m.final_state_distance_vs_curved) and
        np.isfinite(m.accumulated_local_commutator_defect) and
        np.isfinite(m.endpoint_expectation_X_real)
        for m in r1.schedule_metrics
    )
    chk("Schedule metrics finite", finite, f"n={len(r1.schedule_metrics)}")

    cycles = fundamental_torus_cycles(8, 10)
    mesh = triangulate_torus(8, 10)
    scheds = build_schedule_family(mesh, cycles)
    if "commutator_loop" in scheds:
        cl = scheds["commutator_loop"]
        chk("Commutator loop closed", cl[0]==cl[-1], f"len={len(cl)-1}")
    chk("Ordered schedules differ", scheds.get("u_then_v")!=scheds.get("v_then_u"))

    exp_ok = all(
        abs(m.endpoint_expectation_X_real)<=1.+1e-10 and
        abs(m.endpoint_expectation_Y_real)<=1.+1e-10 and
        abs(m.endpoint_expectation_Z_real)<=1.+1e-10
        for m in r1.schedule_metrics
    )
    chk("Endpoint expectations bounded", exp_ok)

    passed=sum(1 for _,s,_ in results if s==PASS)
    print("="*80)
    print("BENCHMARK SUITE SELF-TEST")
    print("="*80)
    print(f"{'TEST':<46} {'STATUS':<8} DETAIL")
    print("-"*80)
    for name,status,detail in results:
        symbol = "\u2705" if status==PASS else "\u274c"
        print(f"{name:<46} {symbol} {status:<6} {detail}")
    print("-"*80)
    print(f"RESULT: {passed}/{len(results)} {'ALL CLEAR \u2705' if passed==len(results) else 'FAILURES \u274c'}")
    print()
    print("Field sweep:")
    print(json.dumps(ss, indent=2))


if __name__ == "__main__":
    _self_test()
