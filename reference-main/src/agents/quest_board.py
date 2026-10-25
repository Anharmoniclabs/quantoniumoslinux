# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""Quest board — task queue for agent assignments."""

from __future__ import annotations

import datetime
import json
import threading
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class QuestStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class QuestPriority(int, Enum):
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class Quest:
    quest_id: str
    title: str
    description: str
    assigned_to: str              # agent name
    priority: int = QuestPriority.NORMAL
    status: str = QuestStatus.PENDING
    xp_reward: int = 25
    created_at: str = ""
    completed_at: str = ""
    result_summary: str = ""
    parent_quest_id: str = ""     # provenance chain
    triggered_by: str = ""        # which agent or "scheduler"

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    def to_dict(self):
        return asdict(self)


class QuestBoard:
    """Thread-safe quest board shared across all agents."""

    _STATE_PATH = Path(__file__).resolve().parents[2] / "data" / "agent_state" / "quest_board.json"
    MAX_HISTORY = 500

    def __init__(self):
        self._quests: Dict[str, Quest] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if self._STATE_PATH.exists():
            try:
                data = json.loads(self._STATE_PATH.read_text(encoding="utf-8"))
                for d in data:
                    q = Quest(**d)
                    self._quests[q.quest_id] = q
            except Exception:
                pass

    def _save(self):
        self._STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Keep only recent history
        all_q = sorted(self._quests.values(), key=lambda q: q.created_at)
        if len(all_q) > self.MAX_HISTORY:
            for old in all_q[:-self.MAX_HISTORY]:
                del self._quests[old.quest_id]
        data = [q.to_dict() for q in self._quests.values()]
        self._STATE_PATH.write_text(json.dumps(data, indent=1), encoding="utf-8")

    def post_quest(self, title: str, description: str, assigned_to: str,
                   priority: int = QuestPriority.NORMAL, xp_reward: int = 25,
                   triggered_by: str = "scheduler", parent_quest_id: str = "") -> Quest:
        quest = Quest(
            quest_id=f"q-{uuid.uuid4().hex[:8]}",
            title=title,
            description=description,
            assigned_to=assigned_to,
            priority=priority,
            xp_reward=xp_reward,
            triggered_by=triggered_by,
            parent_quest_id=parent_quest_id,
        )
        with self._lock:
            self._quests[quest.quest_id] = quest
            self._save()
        return quest

    def claim_quest(self, quest_id: str) -> Optional[Quest]:
        with self._lock:
            q = self._quests.get(quest_id)
            if q and q.status == QuestStatus.PENDING:
                q.status = QuestStatus.IN_PROGRESS
                self._save()
                return q
        return None

    def complete_quest(self, quest_id: str, result_summary: str, success: bool = True):
        with self._lock:
            q = self._quests.get(quest_id)
            if q:
                q.status = QuestStatus.COMPLETED if success else QuestStatus.FAILED
                q.completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
                q.result_summary = result_summary[:2000]
                self._save()

    def pending_for(self, agent_name: str) -> List[Quest]:
        with self._lock:
            return sorted(
                [q for q in self._quests.values()
                 if q.assigned_to == agent_name and q.status == QuestStatus.PENDING],
                key=lambda q: -q.priority,
            )

    def active_quests(self) -> List[Quest]:
        with self._lock:
            return [q for q in self._quests.values()
                    if q.status in (QuestStatus.PENDING, QuestStatus.IN_PROGRESS)]

    def recent_completed(self, n: int = 20) -> List[Quest]:
        with self._lock:
            done = [q for q in self._quests.values()
                    if q.status in (QuestStatus.COMPLETED, QuestStatus.FAILED)]
            return sorted(done, key=lambda q: q.completed_at, reverse=True)[:n]
