# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Agent router config loader — reads AGENT_ROUTER_CONFIG.yaml and SAGE_EXPERIMENTS.yaml.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Cached config
_router_config: Optional[Dict[str, Any]] = None
_experiments_config: Optional[Dict[str, Any]] = None


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file, falling back to basic parsing if PyYAML not available."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if yaml:
        return yaml.safe_load(text) or {}
    # Minimal fallback for simple structures
    return {}


def get_router_config() -> Dict[str, Any]:
    """Load and cache AGENT_ROUTER_CONFIG.yaml."""
    global _router_config
    if _router_config is None:
        _router_config = _load_yaml(_PROJECT_ROOT / "AGENT_ROUTER_CONFIG.yaml")
    return _router_config


def get_experiments_config() -> Dict[str, Any]:
    """Load and cache SAGE_EXPERIMENTS.yaml."""
    global _experiments_config
    if _experiments_config is None:
        _experiments_config = _load_yaml(_PROJECT_ROOT / "SAGE_EXPERIMENTS.yaml")
    return _experiments_config


def get_agent_model(agent_name: str) -> str:
    """Get the model assignment for an agent from config."""
    config = get_router_config()
    models = config.get("agent_models", {})
    return models.get(agent_name, "gag0/qwen35-opus-distil")


def get_agent_skill_tree(agent_name: str) -> Dict[str, Any]:
    """Get skill tree config for an agent."""
    config = get_router_config()
    trees = config.get("skill_trees", {})
    return trees.get(agent_name, {})


def route_task(task: str, file_path: Optional[str] = None) -> tuple[str, int, str]:
    """
    Route a task to the best agent using AGENT_ROUTER_CONFIG.yaml rules.
    
    Returns: (agent_name, priority_boost, reason)
    """
    config = get_router_config()
    task_lower = task.lower()
    
    # Check file path routing first (more specific)
    if file_path:
        path_routing = config.get("path_routing", {})
        for path_pattern, agent in path_routing.items():
            if path_pattern.endswith("/"):
                if file_path.startswith(path_pattern) or f"/{path_pattern}" in file_path:
                    return (agent, 5, f"Path match: {path_pattern}")
            elif path_pattern.startswith("*"):
                ext = path_pattern[1:]
                if file_path.endswith(ext):
                    return (agent, 3, f"Extension match: {ext}")
    
    # Check pattern-based routing rules
    routing_rules = config.get("routing_rules", [])
    for rule in routing_rules:
        patterns = rule.get("patterns", [])
        for pattern in patterns:
            if pattern.lower() in task_lower:
                return (
                    rule.get("agent", "SAGE"),
                    rule.get("priority_boost", 0),
                    rule.get("reason", f"Pattern match: {pattern}")
                )
    
    # Default to SAGE for analysis
    return ("SAGE", 0, "Default routing")


def get_experiment_templates() -> Dict[str, Dict[str, Any]]:
    """Get all available experiment templates for SAGE."""
    config = get_experiments_config()
    return config.get("experiments", {})


def get_experiment(name: str) -> Optional[Dict[str, Any]]:
    """Get a specific experiment template by name."""
    experiments = get_experiment_templates()
    return experiments.get(name)


# Agent model size mapping for token budget decisions
AGENT_TOKEN_BUDGETS = {
    "ORACLE": 2000,
    "SAGE": 2000,
    "FORGE": 1000,
    "VIAL": 1000,
    "WARD": 1000,
    "LORE": 500,
    "SPARK": 500,
    "TIDE": 500,
}


def get_token_budget(agent_name: str) -> int:
    """Get token budget for an agent based on its model size."""
    return AGENT_TOKEN_BUDGETS.get(agent_name, 500)
