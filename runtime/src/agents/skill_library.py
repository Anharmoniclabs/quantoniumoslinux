# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""Voyager-style skill library — reusable task functions indexed by name."""

from __future__ import annotations

import datetime
import json
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

_STATE_DIR = Path(__file__).resolve().parents[2] / "data" / "agent_state"


@dataclass
class Skill:
    name: str
    description: str
    agent: str                    # which agent owns this skill
    category: str = "general"     # "build", "test", "scan", "report"
    times_used: int = 0
    last_used: str = ""
    avg_duration_ms: float = 0.0
    success_rate: float = 1.0

    def to_dict(self):
        return asdict(self)


class SkillLibrary:
    """Registry of callable task functions with usage tracking."""

    _STATE_PATH = _STATE_DIR / "skill_library.json"

    def __init__(self):
        self._skills: Dict[str, Skill] = {}
        self._executors: Dict[str, Callable] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if self._STATE_PATH.exists():
            try:
                data = json.loads(self._STATE_PATH.read_text(encoding="utf-8"))
                for d in data:
                    s = Skill(**d)
                    self._skills[s.name] = s
            except Exception:
                pass

    def _save(self):
        self._STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = [s.to_dict() for s in self._skills.values()]
        self._STATE_PATH.write_text(json.dumps(data, indent=1), encoding="utf-8")

    def register(self, name: str, description: str, agent: str,
                 executor: Callable, category: str = "general"):
        with self._lock:
            if name not in self._skills:
                self._skills[name] = Skill(
                    name=name, description=description,
                    agent=agent, category=category,
                )
            self._executors[name] = executor
            self._save()

    def execute(self, name: str, **kwargs) -> str:
        executor = self._executors.get(name)
        if not executor:
            raise ValueError(f"Skill '{name}' not found in library")

        import time
        t0 = time.monotonic()
        try:
            result = executor(**kwargs)
            duration = (time.monotonic() - t0) * 1000
            with self._lock:
                skill = self._skills[name]
                skill.times_used += 1
                skill.last_used = datetime.datetime.now(datetime.timezone.utc).isoformat()
                n = skill.times_used
                skill.avg_duration_ms = (skill.avg_duration_ms * (n - 1) + duration) / n
                total_success = skill.success_rate * (n - 1) + 1.0
                skill.success_rate = total_success / n
                self._save()
            return result
        except Exception as exc:
            with self._lock:
                skill = self._skills.get(name)
                if skill:
                    skill.times_used += 1
                    n = skill.times_used
                    total_success = skill.success_rate * (n - 1)
                    skill.success_rate = total_success / n
                    self._save()
            raise

    def list_skills(self, agent: Optional[str] = None) -> List[Skill]:
        with self._lock:
            if agent:
                return [s for s in self._skills.values() if s.agent == agent]
            return list(self._skills.values())
