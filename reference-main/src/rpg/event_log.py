# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
RPG event log — scrolling battle-style event feed, rendered via pygame.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pygame


@dataclass
class EventEntry:
    timestamp: float
    agent: str
    icon: str
    color: str
    message: str
    category: str = "action"


class EventLog:
    """Scrolling retro RPG event feed."""

    MAX_VISIBLE = 12

    def __init__(self):
        self.entries: List[EventEntry] = []
        self._scroll = 0
        self._font: Optional[pygame.font.Font] = None

    def _init_font(self):
        if self._font is None:
            self._font = pygame.font.SysFont("consolas", 11)

    def post(self, agent: str, icon: str, color: str, message: str,
             category: str = "action"):
        self.entries.append(EventEntry(
            timestamp=time.monotonic(), agent=agent, icon=icon,
            color=color, message=message, category=category,
        ))
        if len(self.entries) > 200:
            self.entries = self.entries[-200:]
        self._scroll = max(0, len(self.entries) - self.MAX_VISIBLE)

    def post_system(self, message: str):
        self.post("SYSTEM", ">>", "#888888", message, "system")

    def scroll(self, delta: int):
        self._scroll = max(0, min(
            len(self.entries) - self.MAX_VISIBLE,
            self._scroll - delta * 3,
        ))

    def draw(self, target: pygame.Surface,
             viewport: Tuple[int, int, int, int]):
        self._init_font()
        vx, vy, vw, vh = viewport

        pygame.draw.rect(target, (14, 14, 14), (vx, vy, vw, vh))
        pygame.draw.line(target, (34, 34, 34),
                         (vx, vy), (vx + vw, vy))

        line_h = 14

        # Header
        hdr = self._font.render("-- EVENT LOG --", True, (85, 85, 85))
        target.blit(hdr, (vx + 6, vy + 4))

        y = vy + 4 + line_h
        visible = self.entries[self._scroll:
                               self._scroll + self.MAX_VISIBLE]

        cat_colors = {
            "action":   (192, 192, 192),
            "level_up": (251, 191, 36),
            "error":    (239, 68, 68),
            "system":   (102, 102, 102),
        }

        for entry in visible:
            if y + line_h > vy + vh:
                break

            # Timestamp
            t = f"{entry.timestamp % 10000:07.1f}"
            ts = self._font.render(t, True, (68, 68, 68))
            target.blit(ts, (vx + 4, y))

            # Agent tag
            tag = f"{entry.agent}"
            tg = self._font.render(tag, True, pygame.Color(entry.color))
            target.blit(tg, (vx + 70, y))

            # Message
            msg_x = vx + 70 + tg.get_width() + 8
            msg_color = cat_colors.get(entry.category, (192, 192, 192))
            avail = vx + vw - msg_x - 8
            text = entry.message
            rendered = self._font.render(text, True, msg_color)
            if rendered.get_width() > avail > 0:
                while (self._font.size(text + "...")[0] > avail
                       and len(text) > 5):
                    text = text[:-1]
                text += "..."
                rendered = self._font.render(text, True, msg_color)
            target.blit(rendered, (msg_x, y))

            y += line_h
