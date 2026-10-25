# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""Per-agent memory stream — append-only event log with recency retrieval."""

from __future__ import annotations

import datetime
import json
import os
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

_STATE_DIR = Path(__file__).resolve().parents[2] / "data" / "agent_state"


@dataclass
class MemoryRecord:
    timestamp: str
    event_type: str          # "quest_start", "quest_done", "error", "message", "observation"
    summary: str
    details: str = ""
    importance: int = 5      # 1-10
    source_agent: str = ""
    quest_id: str = ""

    def to_dict(self):
        return asdict(self)


class MemoryStream:
    """Thread-safe, append-only memory for one agent."""

    MAX_MEMORIES = 200

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self._records: List[MemoryRecord] = []
        self._lock = threading.Lock()
        self._path = _STATE_DIR / f"{agent_name.lower()}_memory.json"
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                for r in data[-self.MAX_MEMORIES:]:
                    self._records.append(MemoryRecord(**r))
            except Exception:
                pass

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in self._records[-self.MAX_MEMORIES:]]
        self._path.write_text(json.dumps(data, indent=1), encoding="utf-8")

    def add(self, event_type: str, summary: str, details: str = "",
            importance: int = 5, source_agent: str = "", quest_id: str = ""):
        record = MemoryRecord(
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            event_type=event_type,
            summary=summary,
            details=details[:2000],
            importance=importance,
            source_agent=source_agent or self.agent_name,
            quest_id=quest_id,
        )
        with self._lock:
            self._records.append(record)
            if len(self._records) > self.MAX_MEMORIES:
                self._records = self._records[-self.MAX_MEMORIES:]
            self._save()
        return record

    def recent(self, n: int = 10) -> List[MemoryRecord]:
        with self._lock:
            return list(self._records[-n:])

    def by_quest(self, quest_id: str) -> List[MemoryRecord]:
        with self._lock:
            return [r for r in self._records if r.quest_id == quest_id]

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._records)
