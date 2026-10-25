# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
RPG Dashboard — standalone pygame-ce window for the QUE Guild RPG world.

Launch:
    python -m src.rpg

Combines the tile-map renderer, HUD stat cards, and event log into
a single window.  Instantiates all 7 guild agents, wires their events
via a thread-safe queue, and runs the agent tick loop on a background
thread.

Controls:
    P         — Dispatch patrol quests
    A         — Toggle auto-patrol
    R         — Generate status report
    Arrows    — Scroll map camera
    Mouse-wheel — Scroll event log
    ESC / close — Quit
"""

from __future__ import annotations

import logging
import queue
import random
import sys
import threading
import time
from typing import Dict

import pygame

from .renderer import MapRenderer
from .hud import HUDPanel
from .event_log import EventLog
from .world import AGENT_HOMES, VILLAGE_SQUARE, agent_home_pos

# Agent imports
from ..agents.agent_base import AGENT_PROFILES, _STATE_DIR
from ..agents.quest_board import QuestBoard
from ..agents.skill_library import SkillLibrary
from ..agents.agent_forge import ForgeAgent
from ..agents.agent_vial import VialAgent
from ..agents.agent_ward import WardAgent
from ..agents.agent_lore import LoreAgent
from ..agents.agent_spark import SparkAgent
from ..agents.agent_tide import TideAgent
from ..agents.agent_oracle import OracleAgent
from ..agents.agent_sage import SageAgent

logger = logging.getLogger(__name__)

TICK_INTERVAL = 30   # seconds between full tick sweeps
FPS = 20

# ── Layout constants ──
WIN_W, WIN_H = 1280, 720
MAP_W_PX = 1000
MAP_H_PX = 540
HUD_W = WIN_W - MAP_W_PX        # 280
HUD_H = MAP_H_PX
LOG_H = WIN_H - MAP_H_PX        # 180
CTRL_W = HUD_W
CTRL_H = LOG_H


class _Event:
    """Lightweight event envelope for the agent→main-thread queue."""
    __slots__ = ("kind", "data")
    def __init__(self, kind: str, **data):
        self.kind = kind
        self.data = data


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    _STATE_DIR.mkdir(parents=True, exist_ok=True)

    # ── pygame init ──
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("QUE Guild \u2014 Penthouse")
    clock = pygame.time.Clock()

    # ── Components ──
    renderer = MapRenderer()
    hud = HUDPanel()
    event_log = EventLog()

    # ── Agent infrastructure ──
    quest_board = QuestBoard()
    skill_library = SkillLibrary()
    agents: Dict[str, object] = {}
    eq: queue.Queue = queue.Queue()

    agent_classes = {
        "FORGE": ForgeAgent, "VIAL": VialAgent, "WARD": WardAgent,
        "LORE": LoreAgent, "SPARK": SparkAgent, "TIDE": TideAgent,
        "ORACLE": OracleAgent, "SAGE": SageAgent,
    }

    for name, cls in agent_classes.items():
        p = AGENT_PROFILES[name]
        agent = cls(quest_board=quest_board, skill_library=skill_library)
        agents[name] = agent
        renderer.add_agent(name, p["color"], p["icon"])
        hud.add_agent(name, p["rpg_class"], p["color"], p["icon"])

        # Wire agent event callbacks → queue
        def _make_handler(agent_name):
            def _handler(src, event_type, data):
                eq.put(_Event("agent", name=agent_name,
                              etype=event_type, edata=data))
            return _handler
        agent.on_event(_make_handler(name))

    # ── Tick-loop state ──
    tick_running = [False]

    def dispatch_patrol():
        oracle = agents.get("ORACLE")
        if oracle and hasattr(oracle, "_skill_dispatch_patrol"):
            threading.Thread(target=oracle._skill_dispatch_patrol,
                             daemon=True).start()
            event_log.post_system("ORACLE dispatched patrol quests")

    def toggle_auto():
        if tick_running[0]:
            tick_running[0] = False
            return
        tick_running[0] = True

        def _loop():
            while tick_running[0]:
                for name, agent in agents.items():
                    if not tick_running[0]:
                        break
                    try:
                        agent.tick()
                    except Exception as e:
                        logger.error("Agent %s tick: %s", name, e)
                        p = AGENT_PROFILES.get(name, {})
                        eq.put(_Event("log", name=name,
                                      icon=p.get("icon", "?"),
                                      color=p.get("color", "#888"),
                                      msg=str(e), category="error"))
                if tick_running[0]:
                    time.sleep(TICK_INTERVAL)

        threading.Thread(target=_loop, daemon=True).start()

    def gen_report():
        lines = ["-- QUE Guild Status Report --"]
        for name, agent in agents.items():
            s = agent.stats
            p = AGENT_PROFILES[name]
            lines.append(
                f'{name} Lv.{s.level} '
                f'XP:{s.xp}/{s.xp_to_next} '
                f'Q:{s.quests_completed}ok {s.quests_failed}fail'
            )
        lines.append(
            f"\nActive quests: {len(quest_board.active_quests())}")
        event_log.post_system("\n".join(lines))

    def update_hud(name):
        agent = agents.get(name)
        if not agent:
            return
        s = agent.stats
        hud.update_agent(name, s.level, s.xp, s.hp, s.mp,
                         getattr(agent, "_state", "idle"),
                         s.quests_completed)

    def process_events():
        """Drain the event queue (called every frame on main thread)."""
        for _ in range(50):       # max 50 events per frame
            try:
                ev = eq.get_nowait()
            except queue.Empty:
                break
            if ev.kind == "agent":
                _handle_agent_event(ev.data)
            elif ev.kind == "log":
                d = ev.data
                event_log.post(d["name"], d.get("icon", "?"),
                               d.get("color", "#888"),
                               d.get("msg", ""), d.get("category", "error"))

    def _handle_agent_event(d):
        name = d["name"]
        etype = d["etype"]
        data = d["edata"]
        p = AGENT_PROFILES.get(name, {})
        color = p.get("color", "#888")

        if etype == "quest_start":
            renderer.set_agent_state(name, "working",
                                     agent_home_pos(name))
            event_log.post(name, "", color,
                           f'starts [{data.get("quest", "?")}]', "action")

        elif etype == "quest_done":
            title = data.get("quest", "?")
            xp = data.get("xp", 0)
            home = agent_home_pos(name)
            renderer.set_agent_state(
                name, "idle",
                (home[0] + random.randint(-2, 2),
                 home[1] + random.randint(-2, 2)))
            renderer.add_bubble(name, f"Done: {title}")
            event_log.post(name, "", color,
                           f'completed [{title}] +{xp}XP', "action")
            hud.add_quest_entry(f'{name}: {title} +{xp}XP', color)
            update_hud(name)

        elif etype == "quest_fail":
            renderer.set_agent_state(name, "idle")
            event_log.post(name, "", color,
                           f'FAILED [{data.get("quest", "?")}]', "error")
            update_hud(name)

        elif etype == "cooldown":
            renderer.set_agent_state(name, "cooldown")
            event_log.post(name, "", color,
                           f'cooldown: {data.get("reason", "?")}', "error")

        elif etype == "idle_action":
            event_log.post(name, "", color,
                           data.get("result", "idle patrol"), "action")

    # ── Welcome messages ──
    event_log.post_system("Penthouse office online...")
    event_log.post_system("7 guild agents standing by")
    event_log.post_system("[P] Patrol  [A] Auto  [R] Report  [ESC] Quit")

    # ── Button rectangles ──
    btn_w, btn_h = 140, 34
    btn_x = MAP_W_PX + (HUD_W - btn_w) // 2
    btn_patrol = pygame.Rect(btn_x, MAP_H_PX + 20, btn_w, btn_h)
    btn_auto   = pygame.Rect(btn_x, MAP_H_PX + 62, btn_w, btn_h)
    btn_report = pygame.Rect(btn_x, MAP_H_PX + 104, btn_w, btn_h)
    btn_font = pygame.font.SysFont("consolas", 13, bold=True)
    status_font = pygame.font.SysFont("consolas", 10)
    status_msg = "Ready -- press PATROL or P"

    cam_offset = [0, 0]

    # ═══════════════════ MAIN LOOP ═══════════════════
    running = True
    while running:
        clock.tick(FPS)

        # ── Events ──
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    dispatch_patrol()
                    status_msg = "Patrol quests dispatched"
                elif event.key == pygame.K_a:
                    toggle_auto()
                    status_msg = ("Auto ON" if tick_running[0]
                                  else "Auto OFF")
                elif event.key == pygame.K_r:
                    gen_report()
                    status_msg = "Report generated"
            elif event.type == pygame.MOUSEWHEEL:
                event_log.scroll(event.y)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if btn_patrol.collidepoint(event.pos):
                    dispatch_patrol()
                    status_msg = "Patrol quests dispatched"
                elif btn_auto.collidepoint(event.pos):
                    toggle_auto()
                    status_msg = ("Auto ON" if tick_running[0]
                                  else "Auto OFF")
                elif btn_report.collidepoint(event.pos):
                    gen_report()
                    status_msg = "Report generated"

        # Arrow keys scroll camera
        keys = pygame.key.get_pressed()
        cam_speed = 6
        if keys[pygame.K_LEFT]:
            cam_offset[0] -= cam_speed
        if keys[pygame.K_RIGHT]:
            cam_offset[0] += cam_speed
        if keys[pygame.K_UP]:
            cam_offset[1] -= cam_speed
        if keys[pygame.K_DOWN]:
            cam_offset[1] += cam_speed

        # ── Process agent events ──
        process_events()

        # ── Update ──
        renderer.update()

        # ── Draw ──
        screen.fill((14, 14, 14))

        # Map area
        renderer.draw(screen, (0, 0, MAP_W_PX, MAP_H_PX),
                       tuple(cam_offset))

        # HUD area
        hud.draw(screen, (MAP_W_PX, 0, HUD_W, HUD_H))

        # Event log (bottom-left)
        event_log.draw(screen, (0, MAP_H_PX, MAP_W_PX, LOG_H))

        # Controls area (bottom-right)
        pygame.draw.rect(screen, (18, 18, 18),
                         (MAP_W_PX, MAP_H_PX, CTRL_W, CTRL_H))

        mouse = pygame.mouse.get_pos()
        for rect, label, color in [
            (btn_patrol, "PATROL [P]",
             (212, 160, 86)),
            (btn_auto,
             "STOP [A]" if tick_running[0] else "AUTO [A]",
             (239, 68, 68) if tick_running[0] else (74, 222, 128)),
            (btn_report, "REPORT [R]",
             (59, 130, 246)),
        ]:
            hover = rect.collidepoint(mouse)
            bg = (37, 37, 37) if hover else (26, 26, 26)
            pygame.draw.rect(screen, bg, rect, border_radius=4)
            border = color if hover else (51, 51, 51)
            pygame.draw.rect(screen, border, rect, 1, border_radius=4)
            txt = btn_font.render(label, True, color)
            screen.blit(txt, (rect.x + rect.w // 2 - txt.get_width() // 2,
                              rect.y + rect.h // 2 - txt.get_height() // 2))

        # Status bar
        st = status_font.render(status_msg, True, (85, 85, 85))
        screen.blit(st, (MAP_W_PX + 20, WIN_H - 18))

        pygame.display.flip()

    # ── Cleanup ──
    tick_running[0] = False
    for agent in agents.values():
        try:
            agent._save_stats(force=True)
        except Exception:
            pass
    pygame.quit()


if __name__ == "__main__":
    main()
