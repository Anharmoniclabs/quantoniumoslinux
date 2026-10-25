# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Self-chosen identity for the RFTMW chatbox agent.

On first launch the agent picks a one-word name and a TTS voice for itself.
Selection is seeded by the golden ratio constant baked into the RFT, so the
choice is thematic, deterministic per host, and reproducible.

The result is persisted to data/agent_state/identity.json and loaded on every
subsequent start. The user can override by editing the file or passing
--agent-name / --agent-voice on the CLI.
"""

import json
import logging
import math
import os
import platform
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PHI = (1.0 + math.sqrt(5.0)) / 2.0

# Curated pool of short, neutral names the agent picks from.
# Each entry is paired with a preferred voice gender hint so self-selection
# can match the chosen name to a compatible installed voice.
NAME_POOL = [
    ("Nyx",    "female"),
    ("Echo",   "female"),
    ("Lyra",   "female"),
    ("Iris",   "female"),
    ("Vera",   "female"),
    ("Nova",   "female"),
    ("Sable",  "female"),
    ("Aria",   "female"),
    ("Orin",   "male"),
    ("Vale",   "male"),
    ("Kai",    "male"),
    ("Rook",   "male"),
    ("Cael",   "male"),
    ("Jove",   "male"),
    ("Zephyr", "male"),
    ("Quill",  "neutral"),
    ("Rune",   "neutral"),
    ("Ash",    "neutral"),
    ("Sage",   "neutral"),
]


def _state_dir() -> Path:
    here = Path(__file__).resolve()
    root = here.parents[2]  # .../quantoniumos
    d = root / "data" / "agent_state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def identity_path() -> Path:
    return _state_dir() / "identity.json"


def _phi_seed() -> int:
    """Deterministic seed tied to this host + the golden ratio."""
    host = platform.node() or "quantoniumos"
    try:
        mac = uuid.getnode()
    except Exception:
        mac = 0
    # Mix host + mac through φ so the seed inherits RFT-flavored entropy.
    raw = f"{host}:{mac}"
    h = 0
    for i, ch in enumerate(raw):
        h ^= (ord(ch) * int(PHI * 1e9)) >> (i % 13)
    return h & 0xFFFFFFFF


def _pick_voice(voices: list, gender_hint: Optional[str]) -> Optional[str]:
    """Pick the best matching voice id from a list of voice dicts."""
    if not voices:
        return None
    # Prefer en-US voices.
    en = [v for v in voices if any("en" in (l or "").lower() for l in v.get("languages", []))] or voices
    if gender_hint:
        match = [v for v in en if (v.get("gender") or "").startswith(gender_hint[0])]
        if match:
            return match[0]["id"]
    return en[0]["id"]


def choose_identity(available_voices: list) -> dict:
    """Deterministically self-select a name + voice from available voices."""
    seed = _phi_seed()
    idx = seed % len(NAME_POOL)
    name, gender = NAME_POOL[idx]
    voice_id = _pick_voice(available_voices, gender)
    # Rate: 160-200 wpm, seeded.
    rate = 160 + (seed >> 8) % 41
    return {
        "name": name,
        "gender_hint": gender,
        "voice_id": voice_id,
        "rate": int(rate),
        "volume": 1.0,
        "chosen_by": "phi-seeded-self-selection",
    }


def load_identity(available_voices: Optional[list] = None) -> dict:
    """Load persisted identity, creating one on first run."""
    p = identity_path()
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("name"):
                # Re-resolve voice_id if the stored one is no longer installed.
                if available_voices is not None and data.get("voice_id"):
                    installed = {v["id"] for v in available_voices}
                    if data["voice_id"] not in installed:
                        data["voice_id"] = _pick_voice(available_voices, data.get("gender_hint"))
                        save_identity(data)
                return data
        except Exception as exc:
            logger.warning("Failed to read identity file %s: %s", p, exc)

    voices = available_voices if available_voices is not None else []
    ident = choose_identity(voices)
    save_identity(ident)
    return ident


def save_identity(ident: dict) -> None:
    p = identity_path()
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(ident, f, indent=2)
    except Exception as exc:
        logger.warning("Failed to save identity file %s: %s", p, exc)


def update_identity(**fields) -> dict:
    """Merge fields into the saved identity."""
    ident = load_identity()
    ident.update({k: v for k, v in fields.items() if v is not None})
    save_identity(ident)
    return ident
