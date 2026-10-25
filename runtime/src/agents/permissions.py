# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""Per-agent permission definitions — principle of least privilege."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

@dataclass(frozen=True)
class AgentPermissions:
    allowed_paths: List[str]
    allowed_commands: List[str] = field(default_factory=list)
    can_write: bool = False
    write_paths: List[str] = field(default_factory=list)
    can_execute: bool = False
    max_subprocess_timeout: int = 60
    max_output_bytes: int = 500_000
    max_tool_calls_per_tick: int = 10
    max_subprocess_per_tick: int = 3
    max_file_reads_per_tick: int = 20
    max_file_writes_per_tick: int = 5

AGENT_PERMISSIONS = {
    "FORGE": AgentPermissions(
        allowed_paths=["src/rftmw_native/", "benchmarks/", "data/agent_reports/"],
        allowed_commands=["cmake", "make"],
        can_execute=True,
        can_write=True,
        write_paths=["data/agent_reports/"],
        max_subprocess_timeout=120,
        max_output_bytes=1_000_000,
    ),
    "VIAL": AgentPermissions(
        allowed_paths=["tests/", "test_*.py", "benchmarks/", "src/"],
        allowed_commands=["pytest"],
        can_execute=True,
        can_write=True,
        write_paths=["data/agent_reports/"],
        max_subprocess_timeout=900,
        max_output_bytes=2_000_000,
    ),
    "WARD": AgentPermissions(
        allowed_paths=["*"],
        allowed_commands=["pip-audit", "grep"],
        can_execute=True,
        can_write=True,
        write_paths=["data/agent_reports/", "data/security/"],
        max_subprocess_timeout=60,
    ),
    "LORE": AgentPermissions(
        allowed_paths=["docs/", "*.md"],
        allowed_commands=[],
        can_execute=False,
        can_write=True,
        write_paths=["data/agent_reports/"],
    ),
    "SPARK": AgentPermissions(
        allowed_paths=["src/apps/", "ui/", "quantonium_os_src/apps/"],
        allowed_commands=["python"],
        can_execute=True,
        can_write=True,
        write_paths=["data/agent_reports/"],
        max_subprocess_timeout=30,
    ),
    "TIDE": AgentPermissions(
        allowed_paths=["mobile/", "quantonium_os_src/frontend/"],
        allowed_commands=["npm", "npx"],
        can_execute=True,
        can_write=True,
        write_paths=["data/agent_reports/"],
        max_subprocess_timeout=120,
        max_output_bytes=1_000_000,
    ),
    "ORACLE": AgentPermissions(
        allowed_paths=["*"],
        allowed_commands=["git"],
        can_write=True,
        write_paths=["data/agent_reports/"],
        can_execute=True,
        max_subprocess_timeout=30,
    ),
    "SAGE": AgentPermissions(
        allowed_paths=["*"],
        allowed_commands=["python"],
        can_execute=True,
        can_write=True,
        write_paths=["data/agent_reports/"],
        max_subprocess_timeout=120,
        max_output_bytes=500_000,
        max_subprocess_per_tick=2,
        max_file_reads_per_tick=30,
        max_file_writes_per_tick=3,
    ),
}
