# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""SmartWeights manifests for safely growing Eldrae without mutating base weights.

SmartWeights are small, auditable growth packs layered above a frozen base model:
LoRA adapters, RFT/RFTMW delta metadata, tool-use heads, visual-token heads, and
knowledge snapshots. This module intentionally uses only the Python standard
library so validation can run before any AI/training stack is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "smartweights/v1"
DEFAULT_RUNTIME_MODEL = "qwen2.5-coder:3b"
DEFAULT_TRAINING_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
DEFAULT_ALLOWED_WRITE_ROOTS = [
    "ai/training/models/",
    "ai/training/reports/",
    "smartweights/packs/",
]
SAFE_KIND_VALUES = {
    "lora_adapter",
    "rft_delta",
    "tool_head",
    "visual_token_head",
    "knowledge_snapshot",
    "composite",
}
SAFE_CAPABILITIES = {
    "chat",
    "memory",
    "retrieval",
    "tool_planning",
    "file_read",
    "file_write_scoped",
    "shell_scoped",
    "voice",
    "visual_programs",
    "visual_tokens",
}
MODEL_PROFILES: Dict[str, Dict[str, Any]] = {
    # Qwen2.5 Coder 3B uses GQA, so k/v projections are smaller than q/o.
    # These numbers are for budgeting SmartWeights overlays, not for replacing
    # exact framework inspection when a model is loaded locally.
    "qwen2.5-coder:3b": {
        "base_parameters": 3_000_000_000,
        "layers": 36,
        "modules": {
            "q_proj": [2048, 2048],
            "k_proj": [2048, 256],
            "v_proj": [2048, 256],
            "o_proj": [2048, 2048],
            "gate_proj": [2048, 11008],
            "up_proj": [2048, 11008],
            "down_proj": [11008, 2048],
        },
    },
    "Qwen/Qwen2.5-Coder-3B-Instruct": {
        "alias": "qwen2.5-coder:3b",
    },
    "eldrae:smartweights": {
        "alias": "qwen2.5-coder:3b",
    },
}
DEFAULT_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
DEFAULT_LORA_PHASES = [
    {
        "id": "phase01_attention_route",
        "name": "Attention Route",
        "target_modules": ["q_proj", "v_proj"],
        "capabilities": ["chat", "tool_planning", "retrieval"],
        "artifact_path": "ai/training/models/eldrae_qwen3b_lora_phase01_attention_route",
        "manifest_path": "smartweights/packs/eldrae-core/phases/phase01_attention_route.json",
    },
    {
        "id": "phase02_key_output",
        "name": "Key Output",
        "target_modules": ["k_proj", "o_proj"],
        "capabilities": ["memory", "retrieval", "tool_planning"],
        "artifact_path": "ai/training/models/eldrae_qwen3b_lora_phase02_key_output",
        "manifest_path": "smartweights/packs/eldrae-core/phases/phase02_key_output.json",
    },
    {
        "id": "phase03_gate_habits",
        "name": "Gate Habits",
        "target_modules": ["gate_proj"],
        "capabilities": ["chat", "memory", "tool_planning"],
        "artifact_path": "ai/training/models/eldrae_qwen3b_lora_phase03_gate_habits",
        "manifest_path": "smartweights/packs/eldrae-core/phases/phase03_gate_habits.json",
    },
    {
        "id": "phase04_up_visual_tools",
        "name": "Up Visual Tools",
        "target_modules": ["up_proj"],
        "capabilities": ["tool_planning", "visual_programs", "visual_tokens"],
        "artifact_path": "ai/training/models/eldrae_qwen3b_lora_phase04_up_visual_tools",
        "manifest_path": "smartweights/packs/eldrae-core/phases/phase04_up_visual_tools.json",
    },
    {
        "id": "phase05_down_synthesis",
        "name": "Down Synthesis",
        "target_modules": ["down_proj"],
        "capabilities": ["chat", "memory", "visual_programs"],
        "artifact_path": "ai/training/models/eldrae_qwen3b_lora_phase05_down_synthesis",
        "manifest_path": "smartweights/packs/eldrae-core/phases/phase05_down_synthesis.json",
    },
]


class SmartWeightsError(ValueError):
    """Raised when a SmartWeights manifest fails validation."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _clean_rel_path(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        raise SmartWeightsError("path must be a non-empty string")
    raw = raw.replace("\\", "/").strip()
    check_raw = raw.rstrip("/")
    path = Path(check_raw)
    if path.is_absolute() or re.match(r"^[a-zA-Z]:", raw):
        raise SmartWeightsError(f"path must be repo-relative: {raw}")
    if any(part in {"..", ""} for part in check_raw.split("/")):
        raise SmartWeightsError(f"path cannot contain '..' or empty segments: {raw}")
    return raw


def _resolve_repo_path(raw: str) -> Path:
    return REPO_ROOT / _clean_rel_path(raw)


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for child in sorted(path.rglob("*")):
        if child.is_file():
            yield child


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_path(path: Path) -> Tuple[str, int, int]:
    """Return deterministic digest, total bytes, and file count for file or dir."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        return sha256_file(path), path.stat().st_size, 1

    h = hashlib.sha256()
    total = 0
    count = 0
    root = path.resolve()
    for file_path in _iter_files(path):
        rel = file_path.resolve().relative_to(root).as_posix()
        digest = sha256_file(file_path)
        size = file_path.stat().st_size
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(str(size).encode("ascii"))
        h.update(b"\0")
        h.update(digest.encode("ascii"))
        h.update(b"\n")
        total += size
        count += 1
    return h.hexdigest(), total, count


def collect_artifact(raw_path: str, *, optional: bool = False) -> Dict[str, Any]:
    rel = _clean_rel_path(raw_path)
    path = _resolve_repo_path(rel)
    artifact: Dict[str, Any] = {
        "path": rel,
        "optional": bool(optional),
    }
    if not path.exists():
        artifact.update({"present": False, "kind": "missing"})
        return artifact

    digest, total_bytes, file_count = digest_path(path)
    artifact.update(
        {
            "present": True,
            "kind": "directory" if path.is_dir() else "file",
            "sha256": digest,
            "bytes": total_bytes,
            "file_count": file_count,
        }
    )
    return artifact


def default_safety_block() -> Dict[str, Any]:
    return {
        "base_weight_mutation": "forbidden",
        "loads_as_overlay": True,
        "requires_manifest_validation": True,
        "destructive_actions": "requires_user_approval",
        "network_access": "off_by_default",
        "pii_redaction": "enabled_for_training_corpora",
        "allowed_write_roots": list(DEFAULT_ALLOWED_WRITE_ROOTS),
    }


def _resolve_model_profile(model: str) -> Dict[str, Any]:
    profile = MODEL_PROFILES.get(model)
    if profile and "alias" in profile:
        profile = MODEL_PROFILES.get(str(profile["alias"]))
    if not profile:
        raise SmartWeightsError(f"no SmartWeights budget profile for model: {model}")
    return profile


def _normalize_targets(target_modules: Optional[List[str] | str]) -> List[str]:
    if target_modules is None or target_modules == "auto":
        return list(DEFAULT_LORA_TARGETS)
    if isinstance(target_modules, str):
        target_modules = [part.strip() for part in target_modules.split(",")]
    targets = [target for target in target_modules if target]
    if not targets:
        raise SmartWeightsError("at least one LoRA target module is required")
    return targets


def estimate_lora_budget(
    *,
    model: str = DEFAULT_RUNTIME_MODEL,
    rank: int = 8,
    target_modules: Optional[List[str] | str] = None,
    dtype_bytes: int = 2,
) -> Dict[str, Any]:
    """Estimate compact LoRA SmartWeights size for a known local model profile."""
    if rank <= 0:
        raise SmartWeightsError("LoRA rank must be positive")
    if dtype_bytes <= 0:
        raise SmartWeightsError("dtype byte width must be positive")

    profile = _resolve_model_profile(model)
    targets = _normalize_targets(target_modules)
    modules = profile["modules"]
    per_layer_params = 0
    module_params: Dict[str, int] = {}
    for target in targets:
        if target not in modules:
            raise SmartWeightsError(f"unknown target module for {model}: {target}")
        in_features, out_features = modules[target]
        params = rank * (int(in_features) + int(out_features))
        module_params[target] = params
        per_layer_params += params

    layers = int(profile["layers"])
    trainable_params = per_layer_params * layers
    fp_bytes = trainable_params * dtype_bytes
    return {
        "model": model,
        "base_parameters": int(profile["base_parameters"]),
        "active_dense_parameters": int(profile["base_parameters"]),
        "lora_rank": rank,
        "target_modules": targets,
        "layers": layers,
        "trainable_overlay_parameters": trainable_params,
        "parameters_per_layer": per_layer_params,
        "parameters_per_module_per_layer": module_params,
        "storage_estimate": {
            "dtype_bytes": dtype_bytes,
            "fp_bytes": fp_bytes,
            "fp_mb": round(fp_bytes / (1024 * 1024), 2),
            "q8_mb": round(trainable_params / (1024 * 1024), 2),
            "q4_mb": round(trainable_params * 0.5 / (1024 * 1024), 2),
        },
    }


def estimate_lora_phases(
    *,
    model: str = DEFAULT_RUNTIME_MODEL,
    rank: int = 8,
    dtype_bytes: int = 2,
) -> Dict[str, Any]:
    """Estimate a phased LoRA plan whose phases add up to the full target pack."""
    phases: List[Dict[str, Any]] = []
    total_params = 0
    total_fp_bytes = 0
    for index, phase in enumerate(DEFAULT_LORA_PHASES, start=1):
        budget = estimate_lora_budget(
            model=model,
            rank=rank,
            target_modules=phase["target_modules"],
            dtype_bytes=dtype_bytes,
        )
        params = int(budget["trainable_overlay_parameters"])
        fp_bytes = int(budget["storage_estimate"]["fp_bytes"])
        total_params += params
        total_fp_bytes += fp_bytes
        phases.append(
            {
                "index": index,
                "id": phase["id"],
                "name": phase["name"],
                "target_modules": list(phase["target_modules"]),
                "capabilities": list(phase["capabilities"]),
                "artifact_path": phase["artifact_path"],
                "manifest_path": phase["manifest_path"],
                "trainable_overlay_parameters": params,
                "storage_estimate": budget["storage_estimate"],
            }
        )

    full_budget = estimate_lora_budget(
        model=model,
        rank=rank,
        target_modules=DEFAULT_LORA_TARGETS,
        dtype_bytes=dtype_bytes,
    )
    return {
        "model": model,
        "lora_rank": rank,
        "strategy": "phase_train_disjoint_target_groups_then_combine_or_stack_for_inference",
        "total_trainable_overlay_parameters": total_params,
        "total_storage_estimate": {
            "dtype_bytes": dtype_bytes,
            "fp_bytes": total_fp_bytes,
            "fp_mb": round(total_fp_bytes / (1024 * 1024), 2),
            "q8_mb": round(total_params / (1024 * 1024), 2),
            "q4_mb": round(total_params * 0.5 / (1024 * 1024), 2),
        },
        "matches_full_rank8_budget": total_params == int(full_budget["trainable_overlay_parameters"]),
        "final_combined_artifact": "ai/training/models/eldrae_qwen3b_lora",
        "combine_policy": "phase adapters target disjoint modules, so their adapter tensors can be stacked at runtime or merged into one adapter directory after validation",
        "phases": phases,
    }


def build_efficiency_block(
    *,
    model: str = DEFAULT_RUNTIME_MODEL,
    rank: int = 8,
    target_modules: Optional[List[str] | str] = None,
    actual_trainable_parameters: Optional[int] = None,
) -> Dict[str, Any]:
    estimate = estimate_lora_budget(model=model, rank=rank, target_modules=target_modules)
    if actual_trainable_parameters is not None:
        estimate["actual_trainable_overlay_parameters"] = int(actual_trainable_parameters)
    return {
        "goal": "small_physical_overlay_high_effective_inference_capacity",
        "budget": estimate,
        "phase_plan": estimate_lora_phases(
            model=model,
            rank=rank,
            dtype_bytes=2,
        ),
        "middleware_strategy": [
            {
                "stage": "manifest_gate",
                "role": "validate hashes, capabilities, write scopes, and overlay-only loading before inference",
            },
            {
                "stage": "rftmw_delta",
                "role": "compress stable adapter/head tensors into compact RFTMW delta packs",
            },
            {
                "stage": "retrieval_memory",
                "role": "add fresh knowledge and recall without baking volatile facts into adapter weights",
            },
            {
                "stage": "tool_router",
                "role": "activate local tools and visual pipelines only when the prompt needs them",
            },
        ],
        "activation_policy": "load the frozen Qwen 3B base plus only the SmartWeights overlays relevant to the current task",
        "scaling_rule": "increase effective capability through routed overlays, memory, tools, and RFTMW packs before increasing dense base size",
    }


def build_lora_manifest(
    *,
    out_path: str,
    adapter_path: str,
    name: str = "Eldrae Core SmartWeights",
    version: str = "0.1.0",
    runtime_model: str = DEFAULT_RUNTIME_MODEL,
    training_model: str = DEFAULT_TRAINING_MODEL,
    datasets: Optional[List[str]] = None,
    reports: Optional[List[str]] = None,
    capabilities: Optional[List[str]] = None,
    efficiency: Optional[Dict[str, Any]] = None,
    optional_artifact: bool = False,
    notes: str = "",
) -> Dict[str, Any]:
    datasets = datasets or []
    reports = reports or []
    capabilities = capabilities or ["chat", "memory", "retrieval", "tool_planning"]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "version": version,
        "kind": "lora_adapter",
        "created_at": _now_iso(),
        "identity": {
            "assistant": "Eldrae",
            "role": "seed local OS intelligence for QuantoniumOS",
        },
        "base": {
            "runtime_model": runtime_model,
            "training_model": training_model,
            "frozen": True,
        },
        "rft_space": {
            "family": "RFTMW",
            "encoding": "adapter_delta_overlay",
            "base_tensor_mutation": "none",
            "future_compression": "RFT delta pack once adapter tensors are finalized",
        },
        "artifacts": [collect_artifact(adapter_path, optional=optional_artifact)],
        "training": {
            "datasets": [_clean_rel_path(p) for p in datasets],
            "reports": [_clean_rel_path(p) for p in reports],
            "guardrails": "required unless explicitly disabled by operator",
        },
        "capabilities": sorted(set(capabilities)),
        "efficiency": efficiency
        or build_efficiency_block(
            model=runtime_model,
            rank=8,
            target_modules=DEFAULT_LORA_TARGETS,
        ),
        "safety": default_safety_block(),
        "notes": notes,
    }
    write_manifest(out_path, manifest)
    return manifest


def write_manifest(out_path: str, manifest: Dict[str, Any]) -> None:
    rel = _clean_rel_path(out_path)
    path = _resolve_repo_path(rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_manifest(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def _require(condition: bool, errors: List[str], message: str) -> None:
    if not condition:
        errors.append(message)


def validate_manifest(path: str | Path, *, strict: bool = False) -> Tuple[List[str], List[str]]:
    manifest_path = Path(path)
    manifest = load_manifest(manifest_path)
    errors: List[str] = []
    warnings: List[str] = []

    _require(manifest.get("schema_version") == SCHEMA_VERSION, errors, "schema_version must be smartweights/v1")
    _require(bool(manifest.get("name")), errors, "name is required")
    _require(bool(manifest.get("version")), errors, "version is required")
    _require(manifest.get("kind") in SAFE_KIND_VALUES, errors, f"kind must be one of {sorted(SAFE_KIND_VALUES)}")

    base = manifest.get("base") or {}
    _require(base.get("frozen") is True, errors, "base.frozen must be true")
    _require(bool(base.get("runtime_model")), errors, "base.runtime_model is required")
    _require(bool(base.get("training_model")), errors, "base.training_model is required")

    safety = manifest.get("safety") or {}
    _require(safety.get("base_weight_mutation") == "forbidden", errors, "base weight mutation must be forbidden")
    _require(safety.get("loads_as_overlay") is True, errors, "SmartWeights must load as an overlay")
    allowed_roots = safety.get("allowed_write_roots") or []
    _require(isinstance(allowed_roots, list) and allowed_roots, errors, "safety.allowed_write_roots must be non-empty")
    for root in allowed_roots:
        try:
            _clean_rel_path(root)
        except SmartWeightsError as exc:
            errors.append(f"invalid allowed write root {root!r}: {exc}")

    capabilities = manifest.get("capabilities") or []
    _require(isinstance(capabilities, list), errors, "capabilities must be a list")
    for capability in capabilities:
        if capability not in SAFE_CAPABILITIES:
            warnings.append(f"unknown capability {capability!r}; add it deliberately if this is intended")

    artifacts = manifest.get("artifacts") or []
    _require(isinstance(artifacts, list), errors, "artifacts must be a list")
    for artifact in artifacts:
        rel = artifact.get("path")
        try:
            rel = _clean_rel_path(rel)
        except SmartWeightsError as exc:
            errors.append(f"invalid artifact path: {exc}")
            continue
        artifact_path = _resolve_repo_path(rel)
        optional = bool(artifact.get("optional"))
        if not artifact_path.exists():
            message = f"artifact missing: {rel}"
            if strict and not optional:
                errors.append(message)
            else:
                warnings.append(message)
            continue
        digest, total_bytes, file_count = digest_path(artifact_path)
        expected = artifact.get("sha256")
        if expected and expected != digest:
            errors.append(f"artifact digest mismatch for {rel}")
        if artifact.get("bytes") not in (None, total_bytes):
            errors.append(f"artifact byte count mismatch for {rel}")
        if artifact.get("file_count") not in (None, file_count):
            errors.append(f"artifact file count mismatch for {rel}")

    for section in ("datasets", "reports"):
        for raw in (manifest.get("training") or {}).get(section, []) or []:
            try:
                _clean_rel_path(raw)
            except SmartWeightsError as exc:
                errors.append(f"invalid training {section} path {raw!r}: {exc}")

    efficiency = manifest.get("efficiency")
    if efficiency is not None:
        _require(isinstance(efficiency, dict), errors, "efficiency must be an object")
        budget = efficiency.get("budget") if isinstance(efficiency, dict) else None
        if budget is not None:
            _require(isinstance(budget, dict), errors, "efficiency.budget must be an object")
            if isinstance(budget, dict):
                base_params = budget.get("base_parameters")
                active_params = budget.get("active_dense_parameters")
                overlay_params = budget.get("trainable_overlay_parameters")
                _require(isinstance(base_params, int) and base_params > 0, errors, "efficiency.budget.base_parameters must be positive")
                _require(isinstance(active_params, int) and active_params > 0, errors, "efficiency.budget.active_dense_parameters must be positive")
                _require(isinstance(overlay_params, int) and overlay_params >= 0, errors, "efficiency.budget.trainable_overlay_parameters must be non-negative")
                if isinstance(base_params, int) and isinstance(active_params, int) and active_params < base_params:
                    warnings.append("active dense parameters are lower than base parameters; verify this is intentional")
                storage = budget.get("storage_estimate") or {}
                if storage:
                    _require(isinstance(storage, dict), errors, "efficiency.budget.storage_estimate must be an object")
                    fp_mb = storage.get("fp_mb") if isinstance(storage, dict) else None
                    if isinstance(fp_mb, (int, float)) and fp_mb > 256:
                        warnings.append("SmartWeights fp16 overlay estimate exceeds 256 MB; consider lower rank or fewer targets")
        phase_plan = efficiency.get("phase_plan") if isinstance(efficiency, dict) else None
        if phase_plan is not None:
            _require(isinstance(phase_plan, dict), errors, "efficiency.phase_plan must be an object")
            phases = phase_plan.get("phases") if isinstance(phase_plan, dict) else None
            _require(isinstance(phases, list) and bool(phases), errors, "efficiency.phase_plan.phases must be a non-empty list")
            if isinstance(phases, list):
                seen_targets: set[str] = set()
                total = 0
                for phase in phases:
                    _require(isinstance(phase, dict), errors, "each SmartWeights phase must be an object")
                    if not isinstance(phase, dict):
                        continue
                    phase_id = phase.get("id")
                    _require(isinstance(phase_id, str) and bool(phase_id), errors, "each SmartWeights phase needs an id")
                    for path_key in ("artifact_path", "manifest_path"):
                        raw = phase.get(path_key)
                        if raw:
                            try:
                                _clean_rel_path(raw)
                            except SmartWeightsError as exc:
                                errors.append(f"invalid phase {path_key} {raw!r}: {exc}")
                    targets = phase.get("target_modules") or []
                    _require(isinstance(targets, list) and bool(targets), errors, f"phase {phase_id!r} must list target modules")
                    iter_targets = targets if isinstance(targets, list) else []
                    for target in iter_targets:
                        if target in seen_targets:
                            warnings.append(f"target module {target!r} appears in more than one phase")
                        seen_targets.add(str(target))
                    params = phase.get("trainable_overlay_parameters")
                    _require(isinstance(params, int) and params > 0, errors, f"phase {phase_id!r} must have positive trainable params")
                    if isinstance(params, int):
                        total += params
                declared_total = phase_plan.get("total_trainable_overlay_parameters") if isinstance(phase_plan, dict) else None
                if isinstance(declared_total, int) and declared_total != total:
                    errors.append("efficiency.phase_plan total_trainable_overlay_parameters does not match phase sum")

    return errors, warnings


def _cmd_hash(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    digest, total_bytes, file_count = digest_path(path)
    print(json.dumps({"path": _repo_rel(path), "sha256": digest, "bytes": total_bytes, "file_count": file_count}, indent=2))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    errors, warnings = validate_manifest(args.manifest, strict=args.strict)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"SmartWeights manifest OK: {args.manifest}")
    return 0


def _cmd_build_lora(args: argparse.Namespace) -> int:
    datasets = args.dataset or []
    reports = args.report or []
    targets = _normalize_targets(args.target_modules)
    efficiency = build_efficiency_block(
        model=args.runtime_model,
        rank=args.lora_rank,
        target_modules=targets,
    )
    manifest = build_lora_manifest(
        out_path=args.out,
        adapter_path=args.adapter,
        name=args.name,
        version=args.version,
        runtime_model=args.runtime_model,
        training_model=args.training_model,
        datasets=datasets,
        reports=reports,
        capabilities=args.capability or None,
        efficiency=efficiency,
        optional_artifact=args.optional_artifact,
        notes=args.notes or "",
    )
    print(f"Wrote SmartWeights manifest: {args.out}")
    print(json.dumps({"name": manifest["name"], "version": manifest["version"], "artifact": manifest["artifacts"][0]}, indent=2))
    return 0


def _cmd_estimate_lora(args: argparse.Namespace) -> int:
    estimate = estimate_lora_budget(
        model=args.model,
        rank=args.rank,
        target_modules=args.target_modules,
        dtype_bytes=args.dtype_bytes,
    )
    print(json.dumps(estimate, indent=2))
    return 0


def _cmd_estimate_phases(args: argparse.Namespace) -> int:
    plan = estimate_lora_phases(
        model=args.model,
        rank=args.rank,
        dtype_bytes=args.dtype_bytes,
    )
    print(json.dumps(plan, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and build QuantoniumOS SmartWeights manifests.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_hash = sub.add_parser("hash", help="Hash a file or directory as a SmartWeights artifact.")
    p_hash.add_argument("path")
    p_hash.set_defaults(func=_cmd_hash)

    p_validate = sub.add_parser("validate", help="Validate a SmartWeights manifest.")
    p_validate.add_argument("manifest")
    p_validate.add_argument("--strict", action="store_true", help="Fail if non-optional artifacts are missing.")
    p_validate.set_defaults(func=_cmd_validate)

    p_build = sub.add_parser("build-lora", help="Build a SmartWeights manifest for a LoRA adapter directory.")
    p_build.add_argument("--adapter", default="ai/training/models/eldrae_qwen3b_lora")
    p_build.add_argument("--out", default="smartweights/packs/eldrae-core/smartweight.json")
    p_build.add_argument("--name", default="Eldrae Core SmartWeights")
    p_build.add_argument("--version", default=os.getenv("SMARTWEIGHTS_VERSION", "0.1.0"))
    p_build.add_argument("--runtime-model", default=DEFAULT_RUNTIME_MODEL)
    p_build.add_argument("--training-model", default=DEFAULT_TRAINING_MODEL)
    p_build.add_argument("--dataset", action="append", default=[])
    p_build.add_argument("--report", action="append", default=[])
    p_build.add_argument("--capability", action="append", default=[])
    p_build.add_argument("--lora-rank", type=int, default=8)
    p_build.add_argument("--target-modules", default="auto")
    p_build.add_argument("--optional-artifact", action="store_true")
    p_build.add_argument("--notes", default="")
    p_build.set_defaults(func=_cmd_build_lora)

    p_estimate = sub.add_parser("estimate-lora", help="Estimate compact LoRA SmartWeights parameter and storage budget.")
    p_estimate.add_argument("--model", default=DEFAULT_RUNTIME_MODEL)
    p_estimate.add_argument("--rank", type=int, default=8)
    p_estimate.add_argument("--target-modules", default="auto")
    p_estimate.add_argument("--dtype-bytes", type=int, default=2)
    p_estimate.set_defaults(func=_cmd_estimate_lora)

    p_phases = sub.add_parser("estimate-phases", help="Estimate phased SmartWeights chunks that add up to the full LoRA budget.")
    p_phases.add_argument("--model", default=DEFAULT_RUNTIME_MODEL)
    p_phases.add_argument("--rank", type=int, default=8)
    p_phases.add_argument("--dtype-bytes", type=int, default=2)
    p_phases.set_defaults(func=_cmd_estimate_phases)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except SmartWeightsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: path not found: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
