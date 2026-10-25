# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
RPG HUD — agent stat cards and quest log, rendered via pygame.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pygame


class HUDPanel:
    """Right-side HUD combining agent stats and quest log."""

    def __init__(self):
        self.agents: Dict[str, dict] = {}
        self.quest_entries: List[Tuple[str, str]] = []   # (text, color_hex)
        self._font_title: Optional[pygame.font.Font] = None
        self._font_sm: Optional[pygame.font.Font] = None
        self._font_xs: Optional[pygame.font.Font] = None

    def _init_fonts(self):
        if self._font_title is None:
            self._font_title = pygame.font.SysFont("consolas", 14, bold=True)
            self._font_sm = pygame.font.SysFont("consolas", 11, bold=True)
            self._font_xs = pygame.font.SysFont("consolas", 9)

    def add_agent(self, name: str, rpg_class: str, color: str, icon: str):
        self.agents[name] = {
            "rpg_class": rpg_class, "color": color, "icon": icon,
            "level": 1, "xp": 0, "hp": 100, "mp": 50,
            "state": "idle", "quests": 0,
        }

    def update_agent(self, name: str, level: int, xp: int,
                     hp: int, mp: int, state: str, quests_done: int):
        if name in self.agents:
            a = self.agents[name]
            a["level"] = level
            a["xp"] = xp
            a["hp"] = hp
            a["mp"] = mp
            a["state"] = state
            a["quests"] = quests_done

    def add_quest_entry(self, text: str, color: str = "#888888"):
        self.quest_entries.append((text, color))
        if len(self.quest_entries) > 100:
            self.quest_entries = self.quest_entries[-100:]

    def finalize(self):
        """Compatibility stub — not needed for pygame version."""

    def draw(self, target: pygame.Surface,
             viewport: Tuple[int, int, int, int]):
        self._init_fonts()
        vx, vy, vw, vh = viewport

        # Background
        pygame.draw.rect(target, (18, 18, 18), (vx, vy, vw, vh))

        # Title
        title = self._font_title.render("QUE GUILD", True, (224, 224, 224))
        target.blit(title, (vx + vw // 2 - title.get_width() // 2, vy + 6))

        y = vy + 26
        card_h = 70

        for name, a in self.agents.items():
            if y + card_h > vy + vh - 120:
                break

            # Card background
            pygame.draw.rect(target, (26, 26, 26),
                             (vx + 4, y, vw - 8, card_h - 2))
            pygame.draw.rect(target, (51, 51, 51),
                             (vx + 4, y, vw - 8, card_h - 2), 1)

            color = pygame.Color(a["color"])

            # Colored dot + name
            pygame.draw.rect(target, color, (vx + 10, y + 6, 8, 8))
            nm = self._font_sm.render(name, True, color)
            target.blit(nm, (vx + 22, y + 4))

            # Class + state
            info = (f'Lv.{a["level"]} {a["rpg_class"]}  |  '
                    f'{a["state"].upper()}  |  Q:{a["quests"]}')
            inf = self._font_xs.render(info, True, (136, 136, 136))
            target.blit(inf, (vx + 10, y + 20))

            # Stat bars
            bar_x = vx + 10
            bar_w = vw - 24
            xp_max = a["level"] * 100
            hp_max = 100 + a["level"] * 10
            mp_max = 50 + a["level"] * 5
            self._draw_bar(target, bar_x, y + 32, bar_w, 8,
                           a["hp"], hp_max, (239, 68, 68), "HP")
            self._draw_bar(target, bar_x, y + 42, bar_w, 8,
                           a["mp"], mp_max, (59, 130, 246), "MP")
            self._draw_bar(target, bar_x, y + 52, bar_w, 8,
                           a["xp"], xp_max, (245, 158, 11), "XP")

            y += card_h

        # ── Quest log section ──
        if y + 20 < vy + vh:
            ql = self._font_sm.render("QUEST LOG", True, (245, 158, 11))
            target.blit(ql, (vx + 10, y + 4))
            y += 20
            for text, qcolor in self.quest_entries[-15:]:
                if y + 12 > vy + vh:
                    break
                entry = self._font_xs.render(text[:35], True,
                                             pygame.Color(qcolor))
                target.blit(entry, (vx + 8, y))
                y += 12

    def _draw_bar(self, target: pygame.Surface, x: int, y: int,
                  w: int, h: int, val: int, max_val: int,
                  color: Tuple[int, int, int], label: str):
        pygame.draw.rect(target, (42, 42, 42), (x, y, w, h))
        fill_w = int((val / max(max_val, 1)) * w)
        if fill_w > 0:
            pygame.draw.rect(target, color, (x, y, fill_w, h))
        lbl = self._font_xs.render(f"{label} {val}/{max_val}", True,
                                   (255, 255, 255))
        target.blit(lbl, (x + 3, y))
