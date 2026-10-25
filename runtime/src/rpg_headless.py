# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Headless QUE Guild tick loop — Godot frontend bridge.

Spawned by Godot as a child process. Emits one JSON event per line on
stdout. Reads JSON commands (one per line) on stdin.

Outbound events (stdout):
    {"type": "ready",     "agents": [...], "homes": {...}}
    {"type": "stats",     "name": "FORGE", "level": 1, "xp": 0, ...}
    {"type": "quest_start", "name": "FORGE", "quest": "..."}
    {"type": "quest_done",  "name": "FORGE", "quest": "...", "xp": 10, "level": 1}
    {"type": "quest_fail",  "name": "FORGE", "quest": "...", "error": "..."}
    {"type": "cooldown",    "name": "FORGE", "reason": "..."}
    {"type": "idle_action", "name": "FORGE", "result": "..."}
    {"type": "walk_to",     "name": "FORGE", "x": 4, "y": 2}
    {"type": "log",         "name": "FORGE", "msg": "...", "category": "..."}
    {"type": "system",      "msg": "..."}

Inbound commands (stdin):
    {"cmd": "patrol"}         dispatch patrol quests via ORACLE
    {"cmd": "auto", "on": true}  toggle auto-tick
    {"cmd": "report"}         emit full status snapshot
    {"cmd": "chat", "text": "..."} ask Oracle (gathers agent discourse)
    {"cmd": "quit"}           clean shutdown
"""
from __future__ import annotations

import json
import queue
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

# Self-bootstrap: ensure repo root is on sys.path so `src.*` imports resolve
# regardless of the caller's cwd (Godot spawns us from godot/bronx-rpg/).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.agents.agent_base import AGENT_PROFILES, _STATE_DIR
from src.agents.quest_board import QuestBoard
from src.agents.skill_library import SkillLibrary
from src.agents.agent_forge import ForgeAgent
from src.agents.agent_vial import VialAgent
from src.agents.agent_ward import WardAgent
from src.agents.agent_lore import LoreAgent
from src.agents.agent_spark import SparkAgent
from src.agents.agent_tide import TideAgent
from src.agents.agent_oracle import OracleAgent
from src.agents.agent_sage import SageAgent
from src.rpg.world import ROOMS, MEETING_POINT, agent_home_pos

# ── Tuning ──
# Staggered tick: one agent per AGENT_TICK_DELAY seconds.
# Full sweep of 8 agents takes 8 * 5 = 40s, then SWEEP_COOLDOWN.
AGENT_TICK_DELAY = 10    # seconds between individual agent ticks
SWEEP_COOLDOWN = 30      # seconds rest after a full sweep

_out_lock = threading.Lock()
_discourse_lock = threading.Lock()  # held during Oracle chat — pauses tick loop

# ── RFTMW compressed-engine bootstrap ────────────────────────────
# The .rftmwpk files are lightweight index stubs — actual weights are
# read directly from the GGUF blobs in Ollama's cache at runtime via
# a .meta.json sidecar.  Pack file size is NOT an indicator of model
# size; use the parameter_count in the sidecar instead.
#
# Two engines are loaded:
#   _rftmw_engine_heavy  — 14b (ORACLE + SAGE)
#   _rftmw_engine        — 7b  (all other agents, fallback for heavy)


def _pack_sidecar(pack_path: Path) -> Path:
    """Return metadata sidecar, falling back from *_finetuned to base pack."""
    sidecar = _pack_sidecar(pack_path)
    if sidecar.exists():
        return sidecar
    if pack_path.stem.endswith("_finetuned"):
        base_stem = pack_path.stem[:-len("_finetuned")]
        base_sidecar = pack_path.parent / f"{base_stem}.rftmwpk.meta.json"
        if base_sidecar.exists():
            return base_sidecar
    return sidecar


def _pack_param_count(pack_path: Path) -> int:
    """Return parameter count from the .meta.json sidecar (0 if missing)."""
    sidecar = _pack_sidecar(pack_path)
    try:
        import json as _json
        meta = _json.loads(sidecar.read_text(encoding="utf-8"))
        return int(meta.get("provenance", {}).get("parameter_count", 0))
    except Exception:
        return 0


def _pack_is_usable(pack_path: Path) -> bool:
    """Check if pack has embedded weights or a valid external GGUF reference.
    
    Index-only stubs (< 1MB) require the referenced GGUF to exist on disk.
    Large packs (> 100MB) have embedded weights and are always usable.
    """
    import json as _json
    size_mb = pack_path.stat().st_size / 1e6
    if size_mb > 100:  # Embedded weights
        return True
    # Check if external GGUF exists
    sidecar = pack_path.with_suffix(".rftmwpk.meta.json")
    if not sidecar.exists():
        return False
    try:
        meta = _json.loads(sidecar.read_text(encoding="utf-8"))
        gguf_path = meta.get("provenance", {}).get("gguf_path", "")
        if gguf_path and Path(gguf_path).exists():
            return True
    except Exception:
        pass
    return False


def _find_pack_by_params(target_b: float, tolerance: float = 0.25) -> str | None:
    """Find a usable pack whose parameter count is closest to target_b billion."""
    target = int(target_b * 1e9)
    best: tuple[int, int, Path] | None = None
    for p in _REPO_ROOT.glob("*.rftmwpk"):
        count = _pack_param_count(p)
        if count == 0:
            continue
        if not _pack_is_usable(p):
            continue  # Skip index stubs with missing GGUF
        diff = abs(count - target)
        if diff / target <= tolerance:
            preference = 0 if p.stem.endswith("_finetuned") else 1
            if best is None or (diff, preference) < (best[0], best[1]):
                best = (diff, preference, p)
    return str(best[2]) if best else None


def _chatml(system: str, user: str) -> str:
    """Format a Qwen2.5 ChatML prompt string."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _rftmw_stream(system: str, user: str, on_chunk, max_tokens: int = 256,
                  temperature: float = 0.7) -> str:
    """Stream tokens from the RFTMW engine; returns full assembled text."""
    import src.agents.agent_base as _ab
    engine = _ab._rftmw_engine
    if engine is None:
        return ""
    prompt = _chatml(system, user)
    parts: list[str] = []
    for token in engine.generate_stream(
        prompt, max_tokens=max_tokens,
        temperature=max(temperature, 0.01),
    ):
        parts.append(token)
        on_chunk(token)
    return "".join(parts)


def _load_one_engine(pack: str) -> "Any":
    """Load a single Qwen25Engine from a pack path. Raises on failure."""
    from quantonium_os_src.engine.rftmw_numpy_engine import (
        Qwen25Engine, _get_native_infer,
    )
    _get_native_infer()
    engine = Qwen25Engine(pack)
    engine.load()
    return engine


def _resolve_gguf_from_pack(pack_path: str) -> str | None:
    """Resolve the GGUF file from a .rftmwpk sidecar meta.json."""
    pack = Path(pack_path)
    sidecar = pack.with_suffix(pack.suffix + ".meta.json")
    if not sidecar.exists():
        return None
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return None
    gp = (meta.get("provenance", {}).get("gguf_path", "")
          or meta.get("gguf_path", ""))
    sha = (meta.get("provenance", {}).get("gguf_file_sha256", "")
           or meta.get("gguf_file_sha256", ""))
    if gp:
        candidate = Path(gp)
        if candidate.exists():
            return str(candidate)
        for base in (pack.parent, _REPO_ROOT):
            resolved = base / candidate
            if resolved.exists():
                return str(resolved)
    if sha:
        blob = Path.home() / '.ollama' / 'models' / 'blobs' / f'sha256-{sha}'
        if blob.exists():
            return str(blob)
    return None


def _load_native_engine(pack_path: str) -> "Any":
    """Load SpectralNativeEngine from a pack's GGUF. Raises on failure."""
    # Lazy import — chatbox_network has PyQt5 at module level
    try:
        from src.apps.chatbox_network import SpectralNativeEngine
    except ImportError:
        raise ImportError(
            "SpectralNativeEngine unavailable (PyQt5 not installed?). "
            "Install PyQt5 or use Ollama fallback."
        )
    gguf = _resolve_gguf_from_pack(pack_path)
    if gguf is None:
        raise FileNotFoundError(f"Cannot resolve GGUF from pack: {pack_path}")
    # Resolve relative GGUF paths against repo root
    gguf_abs = str((_REPO_ROOT / gguf).resolve()) if not Path(gguf).is_absolute() else gguf
    engine = SpectralNativeEngine(
        gguf_abs, max_seq=2048, spectral=False, kv_compress=False,
    )
    engine.load()
    return engine


def _boot_rftmw_engine() -> None:
    """Load RFTMW engines on a background thread then inject into agent_base.

    Loads two engines when both packs are available:
      35b/14b → _rftmw_engine_heavy  (ORACLE, SAGE)
       7b/3b  → _rftmw_engine        (all other agents)
    Falls back to whichever single pack is available.
    """
    import os
    import src.agents.agent_base as _ab

    os.environ.setdefault("RFTMW_REQUIRE_NATIVE", "1")

    # ── Fast path: --pack CLI arg → native C++ engine for ALL agents ──
    if _CLI_PACK is not None:
        pack_stem = Path(_CLI_PACK).stem
        try:
            emit({"type": "system", "msg": f"Loading native C++ engine [{pack_stem}]..."})
            engine = _load_native_engine(str((_REPO_ROOT / _CLI_PACK).resolve()
                                             if not Path(_CLI_PACK).is_absolute()
                                             else _CLI_PACK))
            _ab._rftmw_engine = engine
            _ab._rftmw_engine_heavy = engine  # all agents share native engine
            emit({"type": "system", "msg": f"✅ Native engine ready [{pack_stem}] — all agents wired."})
            return
        except Exception as exc:
            emit({"type": "system",
                  "msg": f"Native engine load failed ({type(exc).__name__}: {exc}) — falling back to pack scan"})

    # ── Fallback: auto-discover packs by parameter count ──
    # 35B tier for qwen35-opus-distil (34.6B params)
    pack_35b = os.environ.get("RFTMWPK_35B") or _find_pack_by_params(35.0)
    pack_14b = os.environ.get("RFTMWPK_14B") or _find_pack_by_params(14.0)
    pack_7b  = os.environ.get("RFTMWPK_7B")  or _find_pack_by_params(7.0)
    pack_3b  = os.environ.get("RFTMWPK_3B")  or _find_pack_by_params(3.0)

    # --- load heavy engine (35b preferred, 14b fallback / ORACLE + SAGE) ---
    heavy_pack = pack_35b or pack_14b
    if heavy_pack:
        label = "35b" if pack_35b else "14b"
        try:
            emit({"type": "system", "msg": f"Loading RFTMW {label} engine [{Path(heavy_pack).stem}]..."})
            _ab._rftmw_engine_heavy = _load_one_engine(heavy_pack)
            emit({"type": "system", "msg": f"RFTMW {label} ready — ORACLE + SAGE use compressed {label}."})
        except Exception as exc:
            emit({"type": "system", "msg": f"RFTMW {label} load failed ({type(exc).__name__}) — ORACLE/SAGE will use Ollama"})

    # --- load worker engine (7b preferred, 3b fallback, heavy fallback) ---
    worker_pack = pack_7b or pack_3b
    if worker_pack:
        try:
            emit({"type": "system", "msg": f"Loading RFTMW worker engine [{Path(worker_pack).stem}]..."})
            _ab._rftmw_engine = _load_one_engine(worker_pack)
            emit({"type": "system", "msg": f"RFTMW worker engine ready [{Path(worker_pack).stem}] — all agents now use compressed model."})
        except Exception as exc:
            emit({"type": "system", "msg": f"RFTMW worker engine load failed ({type(exc).__name__}) — using Ollama"})
    elif _ab._rftmw_engine_heavy is not None:
        # No small model; share heavy engine for all agents
        _ab._rftmw_engine = _ab._rftmw_engine_heavy
        emit({"type": "system", "msg": "No 7b/3b pack — all agents share heavy engine."})

    if not heavy_pack and not worker_pack:
        emit({"type": "system", "msg": "No RFTMW packs found — all inference via Ollama"})

# Only mirror human-readable events to stderr when attached to a terminal.
# Under the Godot/WS bridges, stderr is piped and every line becomes a
# push_warning / [py.err] spam line — pure overhead with no reader.
_STDERR_IS_TTY = False
try:
    _STDERR_IS_TTY = bool(sys.stderr.isatty())
except Exception:
    _STDERR_IS_TTY = False

# ── ANSI colors for terminal ──
_COLORS = {
    "FORGE": "\033[33m",   "VIAL": "\033[35m",   "WARD": "\033[34m",
    "LORE": "\033[32m",    "SPARK": "\033[93m",  "TIDE": "\033[36m",
    "ORACLE": "\033[95m",  "SAGE": "\033[94m",   "SYSTEM": "\033[90m",
}
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

_EVENT_ICONS = {
    "ready": "🟢", "stats": "📊", "quest_start": "⚔️ ", "quest_done": "✅",
    "quest_fail": "❌", "cooldown": "💤", "idle_action": "💭",
    "walk_to": "🚶", "chat_response": "💬", "log": "📝", "system": "⚙️ ",
    "error": "🔥", "blocked": "🚫",
}


def _pretty(obj: Dict[str, Any]) -> str:
    """Format an event for human-readable terminal output."""
    etype = obj.get("type", "?")
    name = obj.get("name", obj.get("from", ""))
    icon = _EVENT_ICONS.get(etype, "•")
    color = _COLORS.get(name, _COLORS.get("SYSTEM", ""))

    if etype == "stats":
        return (f"{color}{icon} {name:<7}{_RESET} "
                f"Lv{obj.get('level',0)} {obj.get('state','?')} "
                f"Q:{obj.get('quests_completed',0)}")
    if etype == "quest_start":
        return f"{color}{icon}{_BOLD}{name:<7}{_RESET} starts [{obj.get('quest','?')}]"
    if etype == "quest_done":
        return (f"{color}{icon}{_BOLD}{name:<7}{_RESET} done [{obj.get('quest','?')}] "
                f"+{obj.get('xp',0)}XP")
    if etype == "quest_fail":
        return f"{color}{icon}{_BOLD}{name:<7}{_RESET} FAILED [{obj.get('quest','?')}]"
    if etype == "idle_action":
        result = obj.get("result", "")[:120]
        return f"{color}{icon} {name:<7}{_RESET} {_DIM}{result}{_RESET}"
    if etype == "walk_to":
        return f"{color}{icon} {name:<7}{_RESET} → ({obj.get('x',0)},{obj.get('y',0)})"
    if etype == "chat_response":
        return f"{color}{icon}{_BOLD} {name:<7}{_RESET} {obj.get('msg','')[:200]}"
    if etype == "system":
        return f"{_DIM}{icon} {obj.get('msg','')}{_RESET}"
    if etype == "log":
        cat = obj.get("category", "")
        return f"{color}{icon} {name:<7}{_RESET} [{cat}] {obj.get('msg','')[:150]}"
    if etype == "cooldown":
        return f"{color}{icon} {name:<7}{_RESET} cooldown: {obj.get('reason','?')}"
    if etype == "ready":
        agents = obj.get("agents", [])
        return f"{icon} Ready — {len(agents)} agents: {', '.join(str(a) for a in agents)}"
    return f"{icon} {etype}: {json.dumps(obj, default=str)[:150]}"


def emit(obj: Dict[str, Any]) -> None:
    line = json.dumps(obj, separators=(",", ":"), default=str)
    with _out_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
        if _STDERR_IS_TTY:
            try:
                sys.stderr.write(_pretty(obj) + "\n")
                sys.stderr.flush()
            except Exception:
                pass


def _agent_stats_payload(name: str, agent) -> Dict[str, Any]:
    s = agent.stats
    p = AGENT_PROFILES.get(name, {})
    return {
        "type": "stats",
        "name": name,
        "rpg_class": p.get("rpg_class", ""),
        "color": p.get("color", "#888888"),
        "icon": p.get("icon", "?"),
        "level": s.level,
        "xp": s.xp,
        "xp_to_next": s.xp_to_next,
        "hp": s.hp,
        "mp": s.mp,
        "state": getattr(agent, "state", "idle"),
        "quests_completed": s.quests_completed,
        "quests_failed": s.quests_failed,
    }


# ── Movement destinations ──
# Maps quest keywords to rooms/locations agents walk to
_ROOM_POSITIONS: Dict[str, tuple[int, int]] = {}
for rkey, (rx, ry, rw, rh) in ROOMS.items():
    _ROOM_POSITIONS[rkey] = (rx + rw // 2, ry + rh // 2)

_CONFERENCE = MEETING_POINT


# Give each agent a stable index for spread offsets
_AGENT_ORDER = ["FORGE", "VIAL", "WARD", "LORE", "SPARK", "TIDE", "ORACLE", "SAGE"]
_AGENT_IDX = {n: i for i, n in enumerate(_AGENT_ORDER)}

# Spread offsets around a center point (ring of 8 positions)
_SPREAD = [(-1, -1), (0, -1), (1, -1), (-1, 0),
           (1, 0), (-1, 1), (0, 1), (1, 1)]


def _spread(center: tuple[int, int], agent_name: str) -> tuple[int, int]:
    """Offset a destination so agents don't stack on the same tile."""
    idx = _AGENT_IDX.get(agent_name, 0)
    dx, dy = _SPREAD[idx % len(_SPREAD)]
    x = max(1, min(29, center[0] + dx))
    y = max(1, min(15, center[1] + dy))
    return (x, y)


def _pick_walk_dest(agent_name: str, event: str) -> tuple[int, int] | None:
    """Pick a destination for an agent to walk to based on the event."""
    home = agent_home_pos(agent_name)

    if event == "quest_start":
        return home
    elif event == "quest_done":
        if random.random() < 0.4:
            return _spread(_CONFERENCE, agent_name)
        return _random_hallway_pos()
    elif event == "idle":
        r = random.random()
        if r < 0.3:
            return _spread(_CONFERENCE, agent_name)
        elif r < 0.5:
            return _spread((9, 2), agent_name)    # coffee area
        elif r < 0.65:
            return _spread((21, 2), agent_name)   # water cooler area
        elif r < 0.8:
            return _random_hallway_pos()
        else:
            return home
    elif event == "cooperate":
        return _spread(_CONFERENCE, agent_name)
    return None


def _random_hallway_pos() -> tuple[int, int]:
    """Pick a random walkable hallway position."""
    x = random.randint(8, 22)
    y = random.choice([1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15])
    return (x, y)


# CLI pack override — set by main() from --pack argument
_CLI_PACK: str | None = None


def main() -> int:
    global _CLI_PACK
    import argparse
    parser = argparse.ArgumentParser(description="QUE Guild headless agent loop")
    parser.add_argument("--pack", default=None,
                        help="Path to .rftmwpk pack (uses native C++ engine)")
    args, _unknown = parser.parse_known_args()
    if args.pack:
        _CLI_PACK = args.pack

    _STATE_DIR.mkdir(parents=True, exist_ok=True)

    quest_board = QuestBoard()
    skill_library = SkillLibrary()

    agent_classes = {
        "FORGE": ForgeAgent, "VIAL": VialAgent, "WARD": WardAgent,
        "LORE": LoreAgent, "SPARK": SparkAgent, "TIDE": TideAgent,
        "ORACLE": OracleAgent, "SAGE": SageAgent,
    }

    agents: Dict[str, Any] = {}
    cmd_q: queue.Queue = queue.Queue()
    tick_on = [False]  # Start paused until RFTMW engine loads
    running = [True]

    def _make_handler(agent_name: str):
        def _handler(_src, event_type: str, data: dict):
            payload = {"type": event_type, "name": agent_name}
            payload.update(data or {})
            emit(payload)
            ag = agents.get(agent_name)

            # Emit walk_to events based on agent activity
            if event_type == "quest_start":
                dest = _pick_walk_dest(agent_name, "quest_start")
                if dest:
                    emit({"type": "walk_to", "name": agent_name,
                          "x": dest[0], "y": dest[1]})
            elif event_type in ("quest_done", "quest_fail"):
                if ag is not None:
                    emit(_agent_stats_payload(agent_name, ag))
                dest = _pick_walk_dest(agent_name, "quest_done")
                if dest:
                    emit({"type": "walk_to", "name": agent_name,
                          "x": dest[0], "y": dest[1]})
            elif event_type == "idle_action":
                dest = _pick_walk_dest(agent_name, "idle")
                if dest:
                    emit({"type": "walk_to", "name": agent_name,
                          "x": dest[0], "y": dest[1]})
        return _handler

    for name, cls in agent_classes.items():
        agent = cls(quest_board=quest_board, skill_library=skill_library)
        agents[name] = agent
        agent.on_event(_make_handler(name))

    # Announce readiness
    emit({
        "type": "ready",
        "agents": list(agents.keys()),
        "profiles": {
            n: {
                "rpg_class": AGENT_PROFILES[n]["rpg_class"],
                "color": AGENT_PROFILES[n]["color"],
                "icon": AGENT_PROFILES[n]["icon"],
                "home": AGENT_PROFILES[n]["home"],
            }
            for n in agents
        },
    })
    for name, agent in agents.items():
        emit(_agent_stats_payload(name, agent))

    emit({"type": "system", "msg": "Penthouse office online. 8 guild agents standing by."})
    emit({"type": "system", "msg": "Loading RFTMW 3b engine... (agents idle until ready)"})

    # Spin up RFTMW compressed engine in foreground blocking, so agents
    # don't try to use Ollama while waiting.
    def _boot_and_dispatch():
        _boot_rftmw_engine()
        import src.agents.agent_base as _ab
        if _ab._rftmw_engine is not None:
            emit({"type": "system", "msg": "✅ RFTMW engine ready — enabling auto-tick..."})
            tick_on[0] = True  # Enable agent ticking now that engine is ready
            # Now dispatch patrol since engine is ready
            oracle = agents.get("ORACLE")
            if oracle and hasattr(oracle, "_skill_dispatch_patrol"):
                try:
                    oracle._skill_dispatch_patrol()
                    emit({"type": "system", "msg": "ORACLE dispatched patrol quests."})
                except Exception as exc:
                    emit({"type": "system", "msg": f"Patrol dispatch failed: {exc}"})
            emit({"type": "system", "msg": "Auto-tick ON — agents running."})
        else:
            emit({"type": "system", "msg": "⚠️ No RFTMW engine — using Ollama (needs ollama server)"})
            tick_on[0] = True  # Still allow ticking, will use Ollama fallback

    threading.Thread(target=_boot_and_dispatch, daemon=True).start()

    emit({"type": "system", "msg": "Waiting for engine... (press PATROL when ready)"})

    # ── stdin command reader thread ──
    def _stdin_reader():
        for raw in sys.stdin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                cmd = json.loads(raw)
            except Exception as exc:
                emit({"type": "system", "msg": f"bad cmd: {exc}"})
                continue
            cmd_q.put(cmd)

    threading.Thread(target=_stdin_reader, daemon=True).start()

    # ── Staggered tick loop ──
    # Ticks ONE agent at a time with delay between each,
    # preventing CPU spike from all 8 running simultaneously.
    def _tick_loop():
        agent_names = list(agents.keys())
        while running[0]:
            if tick_on[0]:
                for name in agent_names:
                    if not running[0] or not tick_on[0]:
                        break
                    # Skip ticking while Oracle chat discourse is active
                    if _discourse_lock.locked():
                        break
                    try:
                        agents[name].tick()
                    except Exception as exc:
                        emit({
                            "type": "log", "name": name,
                            "msg": f"{type(exc).__name__}: {exc}",
                            "category": "error",
                        })
                    # Sleep between agents to spread CPU load
                    for _ in range(AGENT_TICK_DELAY * 10):
                        if not running[0]:
                            return
                        time.sleep(0.1)

                # Auto-dispatch patrol if the board drains low after a sweep
                if tick_on[0] and len(quest_board.active_quests()) < 8:
                    orc = agents.get("ORACLE")
                    if orc and hasattr(orc, "_skill_dispatch_patrol"):
                        try:
                            orc._skill_dispatch_patrol()
                            emit({"type": "system",
                                  "msg": "Board low — ORACLE auto-dispatched patrol quests."})
                        except Exception:
                            pass

                # Rest after full sweep
                for _ in range(SWEEP_COOLDOWN * 10):
                    if not running[0] or not tick_on[0]:
                        break
                    time.sleep(0.1)
            else:
                time.sleep(0.5)

    threading.Thread(target=_tick_loop, daemon=True).start()

    def _dispatch_patrol():
        orc = agents.get("ORACLE")
        if orc and hasattr(orc, "_skill_dispatch_patrol"):
            threading.Thread(
                target=orc._skill_dispatch_patrol, daemon=True,
            ).start()
            emit({"type": "system", "msg": "ORACLE dispatched patrol quests"})

    def _sage_deepscan():
        sage = agents.get("SAGE")
        if not sage or not hasattr(sage, "_skill_deep_optimization_scan"):
            emit({"type": "system", "msg": "SAGE unavailable for deep scan"})
            return
        emit({"type": "system",
              "msg": "SAGE deep scan starting — repo learn + 14b experiments..."})
        sage_home = agent_home_pos("SAGE")
        emit({"type": "walk_to", "name": "SAGE",
              "x": sage_home[0], "y": sage_home[1]})

        def _progress(msg: str) -> None:
            emit({"type": "log", "name": "SAGE",
                  "msg": msg, "category": "deep_scan"})

        try:
            result = sage._skill_deep_optimization_scan(emit_fn=_progress)
            emit({"type": "system", "msg": f"SAGE deep scan: {result[:280]}"})
            emit({"type": "chat_response", "from": "SAGE", "msg": result[:600]})
        except Exception as exc:
            emit({"type": "system",
                  "msg": f"SAGE deep scan failed: {type(exc).__name__}: {exc}"})

    def _send_report():
        # Re-emit ready event so late-joining browsers get profiles
        # (fixes race condition where WS connects before Phaser scenes subscribe)
        emit({
            "type": "ready",
            "agents": list(agents.keys()),
            "profiles": {
                n: {
                    "rpg_class": AGENT_PROFILES[n]["rpg_class"],
                    "color": AGENT_PROFILES[n]["color"],
                    "icon": AGENT_PROFILES[n]["icon"],
                    "home": AGENT_PROFILES[n]["home"],
                }
                for n in agents
            },
        })
        for name, agent in agents.items():
            emit(_agent_stats_payload(name, agent))
        active = quest_board.active_quests()
        emit({
            "type": "system",
            "msg": (f"Report: {len([q for q in active if q.status == 'in_progress'])} in-progress, "
                    f"{len([q for q in active if q.status == 'pending'])} pending"),
        })

    def _send_reports_snapshot() -> None:
        """Read all agent-report JSON files and push a single reports_snapshot event."""
        reports_dir = _REPO_ROOT / "data" / "agent_reports"
        reports: Dict[str, Any] = {}
        if reports_dir.exists():
            for fp in sorted(reports_dir.glob("*.json")):
                try:
                    reports[fp.stem] = json.loads(fp.read_text(encoding="utf-8"))
                except Exception:
                    reports[fp.stem] = {"error": "parse_failed"}
        # Also include a live quest-board summary
        active = quest_board.active_quests()
        reports["_board"] = {
            "pending": len([q for q in active if q.status == "pending"]),
            "in_progress": len([q for q in active if q.status == "in_progress"]),
        }
        emit({"type": "reports_snapshot", "reports": reports})

    def _oracle_chat(text: str):
        """Oracle gathers discourse from relevant agents, then responds."""
        with _discourse_lock:  # pause tick loop while agents are in conference
            orc = agents.get("ORACLE")
            if not orc or not hasattr(orc, "ollama_think"):
                emit({"type": "chat_response", "from": "SYSTEM",
                      "msg": "Oracle unavailable"})
                return

            # Walk all agents to conference room for discourse (spread out)
            for aname in agents:
                sx, sy = _spread(_CONFERENCE, aname)
                emit({"type": "walk_to", "name": aname, "x": sx, "y": sy})

            emit({"type": "system", "msg": "Agents gathering in conference room..."})
            emit({"type": "chat_response", "from": "ORACLE",
                  "msg": "[thinking — gathering specialist input...]"})
            time.sleep(1)  # Brief pause for visual effect

            # Fan out specialist briefs in parallel — each Ollama call is
            # independent and typically 3-15s. Running them serially would
            # stack into 30-120s; parallel runs in the slowest-model time.
            agent_inputs: list[str] = []
            inputs_lock = threading.Lock()

            def _gather(aname: str, ag) -> None:
                try:
                    brief = ag.ollama_think(
                        f"The user asks: '{text}'\n"
                        f"As {aname}, give a 1-sentence response from your "
                        f"area of expertise. Be specific and concise."
                    )
                    if brief.startswith("Error"):
                        return
                    with inputs_lock:
                        agent_inputs.append(f"{aname}: {brief.strip()}")
                    emit({"type": "log", "name": aname,
                          "msg": f"Contributing: {brief[:80]}",
                          "category": "action"})
                except Exception:
                    pass

            threads: list[threading.Thread] = []
            for aname, ag in agents.items():
                if aname == "ORACLE":
                    continue
                t = threading.Thread(target=_gather, args=(aname, ag), daemon=True)
                t.start()
                threads.append(t)
            for t in threads:
                t.join(timeout=90)

            # Oracle synthesizes all input — stream tokens live so the UI
            # can render partial answers instead of staring at a frozen bubble.
            discourse = "\n".join(agent_inputs) if agent_inputs else "No agent input available."
            synthesis_prompt = (
                f"The user asked: '{text}'\n\n"
                f"Your agents contributed the following:\n{discourse}\n\n"
                f"As ORACLE the coordinator, synthesize their findings into a "
                f"clear, helpful response. Address the user directly."
            )

            oracle_response = ""
            oracle_sys = (
                f"You are ORACLE, guild coordinator. Level {orc.stats.level}. "
                f"Be concise — 1-3 sentences."
            )

            import src.agents.agent_base as _ab
            if _ab._rftmw_engine is not None:
                # ── Fast path: RFTMW compressed engine ──────────
                try:
                    emit({"type": "chat_stream", "from": "ORACLE", "event": "start"})

                    def _on_chunk(delta: str) -> None:
                        emit({"type": "chat_chunk", "from": "ORACLE", "delta": delta})

                    oracle_response = _rftmw_stream(
                        oracle_sys, synthesis_prompt,
                        on_chunk=_on_chunk,
                        max_tokens=256, temperature=0.7,
                    )
                    emit({"type": "chat_stream", "from": "ORACLE", "event": "end"})
                except Exception as exc:
                    emit({"type": "system",
                          "msg": f"rftmw stream error: {type(exc).__name__}"})
                    oracle_response = orc.ollama_think(synthesis_prompt)
            else:
                # ── Fallback: Ollama HTTP ────────────────────────
                try:
                    from src.apps.ollama_client import ollama_chat_stream
                    from src.agents.agent_base import AGENT_MODELS

                    emit({"type": "chat_stream", "from": "ORACLE", "event": "start"})

                    def _on_chunk(delta: str) -> None:  # type: ignore[no-redef]
                        emit({"type": "chat_chunk", "from": "ORACLE", "delta": delta})

                    oracle_response = ollama_chat_stream(
                        user_text=synthesis_prompt,
                        on_chunk=_on_chunk,
                        system_prompt=oracle_sys,
                        model=AGENT_MODELS.get("ORACLE", "gag0/qwen35-opus-distil"),
                        temperature=0.7,
                        max_tokens=256,
                    )
                    emit({"type": "chat_stream", "from": "ORACLE", "event": "end"})
                except Exception as exc:
                    emit({"type": "system",
                          "msg": f"stream fallback: {type(exc).__name__}"})
                    oracle_response = orc.ollama_think(synthesis_prompt)

            emit({"type": "chat_response", "from": "ORACLE",
                  "msg": oracle_response or "(no response)"})

            # Send agents back to their desks
            time.sleep(2)
            for aname in agents:
                home = agent_home_pos(aname)
                emit({"type": "walk_to", "name": aname,
                      "x": home[0], "y": home[1]})

    # ── command pump (main thread) ──
    while running[0]:
        try:
            cmd = cmd_q.get(timeout=0.5)
        except queue.Empty:
            continue
        c = (cmd.get("cmd") or "").lower()
        if c == "patrol":
            _dispatch_patrol()
        elif c == "auto":
            tick_on[0] = bool(cmd.get("on", not tick_on[0]))
            emit({
                "type": "system",
                "msg": f"Auto-tick {'ON' if tick_on[0] else 'OFF'}",
            })
        elif c == "report":
            _send_report()
        elif c == "reports":
            threading.Thread(target=_send_reports_snapshot, daemon=True).start()
        elif c == "sage_scan":
            threading.Thread(target=_sage_deepscan, daemon=True).start()
        elif c == "sage_experiments":
            # Run all template experiments from SAGE_EXPERIMENTS.yaml
            def _run_experiments():
                sage = agents.get("SAGE")
                if not sage or not hasattr(sage, "run_all_template_experiments"):
                    emit({"type": "system", "msg": "SAGE template experiments unavailable"})
                    return
                emit({"type": "system", "msg": "SAGE running template experiments from SAGE_EXPERIMENTS.yaml..."})
                
                def _progress(msg: str) -> None:
                    emit({"type": "log", "name": "SAGE", "msg": msg, "category": "experiment"})
                
                try:
                    result = sage.run_all_template_experiments(emit_fn=_progress)
                    emit({"type": "chat_response", "from": "SAGE", "msg": result[:1500]})
                except Exception as exc:
                    emit({"type": "system", "msg": f"SAGE experiments failed: {type(exc).__name__}: {exc}"})
            
            threading.Thread(target=_run_experiments, daemon=True).start()
        elif c == "list_experiments":
            # List available experiment templates
            sage = agents.get("SAGE")
            if sage and hasattr(sage, "_skill_list_experiments"):
                result = sage._skill_list_experiments()
                emit({"type": "chat_response", "from": "SAGE", "msg": result})
        elif c == "chat":
            text = cmd.get("text", "")
            if text:
                # Run Oracle discourse in background to not block command pump
                threading.Thread(target=_oracle_chat, args=(text,),
                                 daemon=True).start()
            else:
                emit({"type": "system", "msg": "Empty message"})
        elif c == "quit":
            running[0] = False
            break
        else:
            emit({"type": "system", "msg": f"unknown cmd: {c}"})

    # Shutdown
    tick_on[0] = False
    for agent in agents.values():
        try:
            agent._save_stats(force=True)
        except Exception:
            pass
    emit({"type": "system", "msg": "shutdown"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
