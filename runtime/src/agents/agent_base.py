# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
BaseAgent — abstract foundation for all QUE Guild agents.

Each agent has:
  - RPG stats (level, xp, hp, mp)
  - Domain permissions (least privilege)
  - Memory stream (append-only event log)
  - Skill library (reusable task functions)
  - Quest processing loop (plan → execute → report)
  - Audit trail (provenance chain)
  - Circuit breaker (cooldown on repeated failure)

Safety:
  - Agents NEVER write source code
  - Agents NEVER run shell=True
  - All subprocess commands validated against allowlist
  - Output sanitized for injection markers
  - Full audit logging per GOVERNANCE.md §III
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .memory_stream import MemoryStream
from .permissions import AGENT_PERMISSIONS, AgentPermissions
from .quest_board import Quest, QuestBoard, QuestStatus
from .skill_library import SkillLibrary

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_STATE_DIR = _PROJECT_ROOT / "data" / "agent_state"

# Shell metacharacter blocklist — never allow these in commands
_SHELL_METACHAR = re.compile(r'[;&|`$(){}<>]')

# Prompt injection markers — stripped from outputs
_INJECTION_RE = re.compile(
    r'<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>'
    r'|<tool_call>|</tool_call>'
    r'|<\|system\|>|<\|user\|>|<\|assistant\|>'
    r'|\[INST\]|\[/INST\]'
    r'|<<SYS>>|<</SYS>>'
    r'|Human:|Assistant:',
    re.IGNORECASE,
)

# ── RFTMW compressed-engine hook ────────────────────────────────
# Set by rpg_headless.py once the Qwen25Engine packs are loaded.
# _rftmw_engine_heavy  → 14b pack (ORACLE, SAGE)
# _rftmw_engine        → 7b/3b pack (all other agents; fallback for heavy)
_rftmw_engine = None        # type: Any | None
_rftmw_engine_heavy = None  # type: Any | None

# Agents that prefer the heavy (14b) engine
_HEAVY_AGENTS = frozenset({"ORACLE", "SAGE"})

# ── Ollama Model Assignments ─────────────────────────────────────
# Map each agent to the best-fit Ollama model.
# Heavier agents (ORACLE, FORGE) get larger models; lighter agents
# use the fast 3b for quick reasoning.
AGENT_MODELS = {
    "ORACLE": "gag0/qwen35-opus-distil",
    "FORGE":  "gag0/qwen35-opus-distil",
    "VIAL":   "gag0/qwen35-opus-distil",
    "WARD":   "gag0/qwen35-opus-distil",
    "LORE":   "gag0/qwen35-opus-distil",
    "SPARK":  "gag0/qwen35-opus-distil",
    "TIDE":   "gag0/qwen35-opus-distil",
    "SAGE":   "gag0/qwen35-opus-distil",
}

# ── RPG Stats ─────────────────────────────────────────────────────

AGENT_PROFILES = {
    "FORGE": {"rpg_class": "Blacksmith", "icon": "⚔️", "color": "#d4a056",
              "home": "The Foundry", "desc": "Kernel Smith — builds & benchmarks native code"},
    "VIAL":  {"rpg_class": "Alchemist",  "icon": "🧪", "color": "#8b5cf6",
              "home": "The Laboratory", "desc": "Alchemist — runs tests & tracks coverage"},
    "WARD":  {"rpg_class": "Paladin",    "icon": "🛡️", "color": "#3b82f6",
              "home": "The Watchtower", "desc": "Guardian — security scans & audits"},
    "LORE":  {"rpg_class": "Bard",       "icon": "📜", "color": "#10b981",
              "home": "The Archive", "desc": "Chronicler — validates docs & links"},
    "SPARK": {"rpg_class": "Artificer",  "icon": "⚡", "color": "#f59e0b",
              "home": "The Workshop", "desc": "Artificer — smoke-tests UI & apps"},
    "TIDE":  {"rpg_class": "Ranger",     "icon": "🌊", "color": "#06b6d4",
              "home": "The Docks", "desc": "Navigator — validates mobile & frontend"},
    "ORACLE":{"rpg_class": "Archmage",   "icon": "🔮", "color": "#ec4899",
              "home": "Crystal Tower", "desc": "Coordinator — dispatches quests & aggregates"},
    "SAGE":  {"rpg_class": "Scholar",    "icon": "🔬", "color": "#9333ea",
              "home": "The Laboratory", "desc": "Research Scientist — studies codebase & runs experiments"},
}


@dataclass
class AgentStats:
    name: str
    rpg_class: str = ""
    level: int = 1
    xp: int = 0
    hp: int = 100
    mp: int = 50
    strength: int = 10
    wisdom: int = 10
    quests_completed: int = 0
    quests_failed: int = 0
    consecutive_errors: int = 0
    cooldown_until: str = ""
    total_tool_calls: int = 0

    def to_dict(self):
        return asdict(self)

    @property
    def xp_to_next(self) -> int:
        return self.level * 100

    def add_xp(self, amount: int):
        self.xp += amount
        while self.xp >= self.xp_to_next:
            self.xp -= self.xp_to_next
            self.level += 1
            self.hp = min(100, self.hp + 10)
            self.mp = min(100, self.mp + 5)
            self.strength += 2
            self.wisdom += 1
            logger.info("🎉 %s leveled up to %d!", self.name, self.level)


# ── Base Agent ────────────────────────────────────────────────────

class BaseAgent(ABC):
    """Abstract base agent — all 8 guild members inherit from this."""

    MAX_CONSECUTIVE_ERRORS = 3
    IDLE_INTERVAL = 120  # seconds between idle Ollama calls (saves CPU)
    ERROR_COOLDOWN_SECONDS = 300

    def __init__(self, name: str, quest_board: QuestBoard, skill_library: SkillLibrary):
        self.name = name
        profile = AGENT_PROFILES.get(name, {})
        self.icon = profile.get("icon", "?")
        self.color = profile.get("color", "#808080")
        self.home = profile.get("home", "Village")
        self.description = profile.get("desc", "")

        self.permissions = AGENT_PERMISSIONS.get(name, AgentPermissions(allowed_paths=["*"]))
        self.memory = MemoryStream(name)
        self.quest_board = quest_board
        self.skill_library = skill_library
        self.stats = AgentStats(name=name, rpg_class=profile.get("rpg_class", "Adventurer"))

        # Tick tracking
        self._tool_calls_this_tick = 0
        self._subprocess_calls_this_tick = 0
        self._file_reads_this_tick = 0
        self._file_writes_this_tick = 0
        self._lock = threading.Lock()

        # Position in RPG world (tile coords)
        self.x: int = 0
        self.y: int = 0
        self.target_x: int = 0
        self.target_y: int = 0
        self.state: str = "idle"   # idle, walking, working, resting, cooldown

        # Current quest being worked on
        self.current_quest: Optional[Quest] = None

        # Event callbacks for UI
        self._event_callbacks: list = []

        self._load_stats()
        self._register_skills()

    # ── Persistence ───────────────────────────────────────────────

    def _stats_path(self) -> Path:
        return _STATE_DIR / f"{self.name.lower()}_stats.json"

    def _load_stats(self):
        p = self._stats_path()
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                for k, v in d.items():
                    if hasattr(self.stats, k):
                        setattr(self.stats, k, v)
            except Exception:
                pass

    # Throttled persistence: stats files change on almost every tick (xp),
    # but we only need durability on level-up, cooldown entry, and shutdown.
    # Between those, defer writes to every SAVE_STATS_INTERVAL seconds.
    SAVE_STATS_INTERVAL = 30.0
    _last_save_at: float = 0.0

    def _save_stats(self, *, force: bool = False):
        now = time.time()
        if not force and (now - self._last_save_at) < self.SAVE_STATS_INTERVAL:
            return
        self._last_save_at = now
        p = self._stats_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.stats.to_dict(), indent=1), encoding="utf-8")

    # ── Event System ──────────────────────────────────────────────

    def on_event(self, callback):
        self._event_callbacks.append(callback)

    def _emit(self, event_type: str, data: dict):
        for cb in self._event_callbacks:
            try:
                cb(self.name, event_type, data)
            except Exception:
                pass

    # ── LLM Reasoning (Ollama) ───────────────────────────────────

    def ollama_think(self, prompt: str, *, max_tokens: int = 256,
                     temperature: float = 0.7) -> str:
        """Ask the agent's assigned model to reason about a task.

        Uses the RFTMW compressed engine when available (faster, no Ollama
        required). Falls back to Ollama HTTP if the engine is not loaded.
        Returns the model's response text, or an error string starting
        with 'Error' if both paths fail.
        """
        # ── Fast path: RFTMW in-process engine ──────────────────
        import src.agents.agent_base as _self_mod
        # Heavy agents (ORACLE, SAGE) prefer the 14b engine; fall back to worker engine.
        engine = None
        if self.name in _self_mod._HEAVY_AGENTS and _self_mod._rftmw_engine_heavy is not None:
            engine = _self_mod._rftmw_engine_heavy
        elif _self_mod._rftmw_engine is not None:
            engine = _self_mod._rftmw_engine
        if engine is not None:
            try:
                # Build recent-memory context the same way as Ollama path
                recent = self.memory.recent(5)
                memory_ctx = ""
                if recent:
                    memory_lines = [f"- [{m.event_type}] {m.summary[:100]}"
                                    for m in recent]
                    memory_ctx = "\nRecent memory:\n" + "\n".join(memory_lines)
                system_text = (
                    f"You are {self.name}, a {self.stats.rpg_class} agent in the QUE Guild "
                    f"(Quantonium OS). {self.description}\n"
                    f"Level {self.stats.level}, {self.stats.quests_completed} quests completed.\n"
                    f"You work autonomously on tasks in a software project.\n"
                    f"Be concise — respond in 1-3 sentences max.{memory_ctx}"
                )
                chatml = (
                    f"<|im_start|>system\n{system_text}<|im_end|>\n"
                    f"<|im_start|>user\n{prompt}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                response = engine.generate(
                    chatml, max_tokens=max_tokens,
                    temperature=max(temperature, 0.01),
                )
                response = _INJECTION_RE.sub("[FILTERED]", response)
                self.memory.add("llm_thought", f"[rftmw] {response[:200]}",
                                importance=4, source_agent=self.name)
                return response
            except Exception as exc:
                logger.warning("%s rftmw_think failed, falling back to Ollama: %s",
                               self.name, exc)

        # ── Fallback: Ollama HTTP ────────────────────────────────
        try:
            from src.apps.ollama_client import ollama_chat
        except ImportError:
            try:
                import sys
                sys.path.insert(0, str(_PROJECT_ROOT))
                from src.apps.ollama_client import ollama_chat
            except ImportError:
                return "Error: ollama_client not available"

        model = AGENT_MODELS.get(self.name, "gag0/qwen35-opus-distil")

        # Build context from recent memory
        recent = self.memory.recent(5)
        memory_ctx = ""
        if recent:
            memory_lines = [f"- [{m.event_type}] {m.summary[:100]}"
                            for m in recent]
            memory_ctx = "\nRecent memory:\n" + "\n".join(memory_lines)

        system_prompt = (
            f"You are {self.name}, a {self.stats.rpg_class} agent in the QUE Guild "
            f"(Quantonium OS). {self.description}\n"
            f"Level {self.stats.level}, {self.stats.quests_completed} quests completed.\n"
            f"You work autonomously on tasks in a software project.\n"
            f"Be concise — respond in 1-3 sentences max.{memory_ctx}"
        )

        try:
            response = ollama_chat(
                user_text=prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # Sanitize
            response = _INJECTION_RE.sub("[FILTERED]", response)
            self.memory.add("llm_thought", f"[{model}] {response[:200]}",
                            importance=4, source_agent=self.name)
            return response
        except Exception as exc:
            logger.warning("%s ollama_think failed: %s", self.name, exc)
            return f"Error: Ollama unavailable ({type(exc).__name__})"

    # ── Safety: Subprocess ────────────────────────────────────────

    def safe_subprocess(self, cmd: str) -> Tuple[int, str]:
        """Run a command with full safety checks. Returns (returncode, output)."""
        if not self.permissions.can_execute:
            raise PermissionError(f"{self.name} has no execute permission")

        # Check subprocess budget for this tick
        with self._lock:
            if self._subprocess_calls_this_tick >= self.permissions.max_subprocess_per_tick:
                raise PermissionError(f"{self.name} subprocess budget exhausted this tick")
            self._subprocess_calls_this_tick += 1

        # Block shell metacharacters
        if _SHELL_METACHAR.search(cmd):
            raise PermissionError(f"Shell metacharacters blocked in: {cmd[:80]}")

        # Validate against command allowlist
        cmd_lower = cmd.strip().lower()
        allowed = False
        for prefix in self.permissions.allowed_commands:
            if cmd_lower.startswith(prefix.lower()):
                allowed = True
                break
        if not allowed:
            raise PermissionError(
                f"{self.name} not authorized for command: {cmd[:60]}. "
                f"Allowed: {self.permissions.allowed_commands}"
            )

        # Execute with timeout and output cap
        try:
            args = shlex.split(cmd)
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=self.permissions.max_subprocess_timeout,
                cwd=str(_PROJECT_ROOT),
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            output = (result.stdout or "") + (result.stderr or "")
            # Cap output size
            if len(output) > self.permissions.max_output_bytes:
                output = output[:self.permissions.max_output_bytes] + "\n...[truncated]"
            # Sanitize
            output = _INJECTION_RE.sub("[FILTERED]", output)
            return result.returncode, output
        except subprocess.TimeoutExpired:
            return -1, f"Command timed out after {self.permissions.max_subprocess_timeout}s"
        except FileNotFoundError:
            return -1, f"Command not found: {shlex.split(cmd)[0]}"
        except Exception as exc:
            return -1, f"Subprocess error: {type(exc).__name__}: {exc}"

    # ── Safety: File Write ────────────────────────────────────────

    def safe_write(self, path: str, content: str, *,
                   backup: bool = True) -> str:
        """Write a file within allowed write-paths. Creates .bak backup first.

        Returns "OK: ..." on success or "Error: ..." on failure.

        Safeguards:
          - Agent must have can_write=True
          - Path must be inside project root (no traversal)
          - Path must match one of write_paths prefixes
          - Content size capped at 100 KB
          - Automatic .bak backup of existing files
          - File write count tracked per tick
        """
        if not self.permissions.can_write:
            return f"Error: {self.name} has no write permission"

        # Budget check
        with self._lock:
            if self._file_writes_this_tick >= self.permissions.max_file_writes_per_tick:
                return "Error: file write budget exhausted this tick"
            self._file_writes_this_tick += 1

        # Content size cap (100 KB)
        if len(content) > 100_000:
            return "Error: content exceeds 100 KB limit"

        # Resolve and sandbox
        resolved = (_PROJECT_ROOT / path).resolve()
        try:
            resolved.relative_to(_PROJECT_ROOT)
        except ValueError:
            return "Error: path outside project sandbox"

        # Check write_paths whitelist
        rel = str(resolved.relative_to(_PROJECT_ROOT)).replace("\\", "/")
        allowed = False
        for wp in self.permissions.write_paths:
            if wp.endswith("/") and rel.startswith(wp):
                allowed = True
                break
            elif rel == wp or rel.startswith(wp + "/"):
                allowed = True
                break
        if not allowed:
            return f"Error: {self.name} not authorized to write {rel}"

        # Disallow writing Python source files (agents can't modify code)
        if resolved.suffix in (".py", ".pyx", ".pyi"):
            return "Error: agents cannot write Python source files"

        # Disallow overwriting critical files
        critical = {"requirements.txt", "pyproject.toml", "setup.py",
                    ".gitignore", "Dockerfile", "LICENSE"}
        if resolved.name in critical:
            return f"Error: cannot overwrite critical file {resolved.name}"

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)

            # Backup existing file
            if backup and resolved.exists():
                bak = resolved.with_suffix(resolved.suffix + ".bak")
                bak.write_bytes(resolved.read_bytes())

            # Sanitize content for injection markers
            clean = _INJECTION_RE.sub("[FILTERED]", content)
            resolved.write_text(clean, encoding="utf-8")

            self.memory.add("file_write", f"Wrote {rel} ({len(clean)} bytes)",
                            importance=6, source_agent=self.name)
            return f"OK: wrote {rel} ({len(clean)} bytes)"

        except Exception as exc:
            return f"Error writing {rel}: {type(exc).__name__}: {exc}"

    # ── Safety: File Read ─────────────────────────────────────────

    def safe_read(self, path: str, max_lines: int = 200) -> str:
        """Read a file within allowed paths. Returns content or error string."""
        with self._lock:
            if self._file_reads_this_tick >= self.permissions.max_file_reads_per_tick:
                return "Error: file read budget exhausted this tick"
            self._file_reads_this_tick += 1

        resolved = (_PROJECT_ROOT / path).resolve()
        try:
            resolved.relative_to(_PROJECT_ROOT)
        except ValueError:
            return "Error: path outside project sandbox"

        if not resolved.exists():
            return f"Error: not found: {path}"
        if not resolved.is_file():
            return f"Error: not a file: {path}"

        # Check allowed paths
        if "*" not in self.permissions.allowed_paths:
            rel = str(resolved.relative_to(_PROJECT_ROOT)).replace("\\", "/")
            allowed = False
            for pattern in self.permissions.allowed_paths:
                if pattern.endswith("/") and rel.startswith(pattern):
                    allowed = True
                    break
                elif "*" in pattern:
                    import fnmatch
                    if fnmatch.fnmatch(rel, pattern):
                        allowed = True
                        break
            if not allowed:
                return f"Error: {self.name} not authorized to read {path}"

        try:
            text = resolved.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            if len(lines) > max_lines:
                lines = lines[:max_lines]
                lines.append(f"...[{len(text.splitlines()) - max_lines} more lines]")
            output = "\n".join(lines)
            return _INJECTION_RE.sub("[FILTERED]", output)
        except Exception as exc:
            return f"Error reading: {type(exc).__name__}"

    # ── Tick Lifecycle ────────────────────────────────────────────

    def reset_tick_budget(self):
        with self._lock:
            self._tool_calls_this_tick = 0
            self._subprocess_calls_this_tick = 0
            self._file_reads_this_tick = 0
            self._file_writes_this_tick = 0

    def is_on_cooldown(self) -> bool:
        if not self.stats.cooldown_until:
            return False
        try:
            until = datetime.datetime.fromisoformat(self.stats.cooldown_until)
            if datetime.datetime.now(datetime.timezone.utc) < until:
                return True
            self.stats.cooldown_until = ""
            self._save_stats()
        except Exception:
            self.stats.cooldown_until = ""
        return False

    def enter_cooldown(self, reason: str = "circuit_breaker"):
        until = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
            seconds=self.ERROR_COOLDOWN_SECONDS
        )
        self.stats.cooldown_until = until.isoformat()
        self.state = "cooldown"
        self.memory.add("cooldown", f"Entering cooldown: {reason}", importance=8)
        self._emit("cooldown", {"reason": reason, "until": self.stats.cooldown_until})
        self._save_stats(force=True)
        logger.warning("%s %s entering cooldown: %s", self.icon, self.name, reason)

    def tick(self) -> Optional[dict]:
        """One execution tick. Returns event dict for RPG renderer or None."""
        if self.is_on_cooldown():
            return {"agent": self.name, "event": "cooldown", "state": "resting"}

        self.reset_tick_budget()
        self.state = "idle"

        try:
            # 1. Check for pending quests
            quests = self.quest_board.pending_for(self.name)
            quest = quests[0] if quests else None

            if quest:
                # 2. Claim and execute
                claimed = self.quest_board.claim_quest(quest.quest_id)
                if claimed:
                    self.current_quest = claimed
                    self.state = "working"
                    self._emit("quest_start", {"quest": claimed.title})
                    self.memory.add("quest_start", f"Starting: {claimed.title}",
                                    quest_id=claimed.quest_id, importance=6)

                    # 3. Execute the quest
                    result = self.execute_quest(claimed)

                    # 4. Complete
                    success = not result.startswith("Error")
                    self.quest_board.complete_quest(claimed.quest_id, result, success=success)

                    if success:
                        self.stats.quests_completed += 1
                        self.stats.add_xp(claimed.xp_reward)
                        self.stats.consecutive_errors = 0
                        self.memory.add("quest_done", f"Completed: {claimed.title}",
                                        details=result[:500], quest_id=claimed.quest_id,
                                        importance=7)
                        self._emit("quest_done", {
                            "quest": claimed.title, "xp": claimed.xp_reward,
                            "result": result[:200], "level": self.stats.level,
                        })
                    else:
                        self.stats.quests_failed += 1
                        self.stats.consecutive_errors += 1
                        self.memory.add("error", f"Failed: {claimed.title}",
                                        details=result[:500], quest_id=claimed.quest_id,
                                        importance=8)
                        self._emit("quest_fail", {"quest": claimed.title, "error": result[:200]})

                        if self.stats.consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                            self.enter_cooldown("circuit_breaker")

                    self.current_quest = None
                    self.state = "idle"
                    self._save_stats()

                    return {
                        "agent": self.name,
                        "event": "quest_done" if success else "quest_fail",
                        "quest": claimed.title,
                        "result": result[:200],
                        "xp": claimed.xp_reward if success else 0,
                        "level": self.stats.level,
                        "state": self.state,
                    }
            else:
                # No quest — do idle patrol / self-proposed task
                # Throttle idle Ollama calls: only fire every IDLE_INTERVAL seconds
                now = time.time()
                if now - getattr(self, "_last_idle_at", 0) < self.IDLE_INTERVAL:
                    return None
                self._last_idle_at = now
                idle_result = self.idle_action()
                if idle_result:
                    return {"agent": self.name, "event": "idle_action",
                            "result": idle_result[:200], "state": "idle"}

        except PermissionError as exc:
            self.memory.add("blocked", str(exc), importance=7)
            self._emit("blocked", {"error": str(exc)[:200]})
            logger.warning("%s %s blocked: %s", self.icon, self.name, exc)
        except Exception as exc:
            self.stats.consecutive_errors += 1
            self.memory.add("error", f"Tick error: {type(exc).__name__}: {exc}", importance=8)
            self._emit("error", {"error": str(exc)[:200]})
            logger.error("%s %s tick error: %s", self.icon, self.name, exc, exc_info=True)
            if self.stats.consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                self.enter_cooldown("circuit_breaker")
            self._save_stats()

        return None

    # ── Abstract Methods ──────────────────────────────────────────

    @abstractmethod
    def _register_skills(self):
        """Register this agent's skills into the skill library."""
        pass

    @abstractmethod
    def execute_quest(self, quest: Quest) -> str:
        """Execute a quest. Return result string. Start with 'Error' on failure."""
        pass

    def idle_action(self) -> Optional[str]:
        """Optional self-directed action when no quests pending. Override in subclass."""
        return None
