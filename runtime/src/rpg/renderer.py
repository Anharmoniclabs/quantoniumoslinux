# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Map renderer — draws the QUE Guild office tile map, agents, and overlays.

Uses 32×32 native tiles (SCALE=1) from sprite_factory + world.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pygame

from .world import (
    MAP_W, MAP_H, TILE_SIZE,
    HARDWOOD, CARPET, GLASS_WALL, MARBLE,
    OFFICE_WALL, OFFICE_DOOR, CONF_CARPET, ELEVATOR, ELEV_DOOR,
    DESK, OFFICE_CHAIR, POTTED_PLANT, WHITEBOARD,
    COFFEE_MACHINE, WATER_COOLER, SERVER_RACK, BOOKSHELF,
    ROOMS, AGENT_HOMES, MEETING_POINT,
    TileMap, agent_home_pos,
)
from .sprite_factory import (
    tile_hardwood, tile_carpet, tile_glass_wall, tile_marble,
    tile_office_wall, tile_office_door, tile_conf_carpet,
    tile_elevator, tile_elevator_door,
    sprite_desk, sprite_office_chair, sprite_potted_plant,
    sprite_whiteboard, sprite_coffee_machine, sprite_water_cooler,
    sprite_server_rack, sprite_bookshelf,
    character_sprite, status_icon,
    T, CH_W, CH_H,
)

SCALE = 1  # tiles are already 32px — no scaling needed
CHAR_H = CH_H  # 40


# ─────────── helpers ───────────

_LABEL_FONT: Optional[pygame.font.Font] = None


def _font():
    global _LABEL_FONT
    if _LABEL_FONT is None:
        _LABEL_FONT = pygame.font.SysFont("consolas", 11, bold=True)
    return _LABEL_FONT


# ─────────── speech bubble ───────────

@dataclass
class SpeechBubble:
    text: str
    born: float = field(default_factory=time.time)
    duration: float = 4.0

    def alive(self):
        return time.time() - self.born < self.duration


# ─────────── agent sprite ───────────

@dataclass
class AgentSprite:
    name: str
    color: str
    icon: str
    x: float = 0.0
    y: float = 0.0
    target_x: float = 0.0
    target_y: float = 0.0
    state: str = "idle"  # idle | working | cooldown
    direction: int = 0   # 0=down 1=left 2=right 3=up
    frame: int = 0
    _frame_time: float = field(default=0.0, init=False)
    bubbles: List[SpeechBubble] = field(default_factory=list)

    def update(self, dt: float):
        # Walk toward target
        speed = 2.0
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > 0.15:
            step = min(speed * dt, dist)
            self.x += dx / dist * step
            self.y += dy / dist * step
            # Direction
            if abs(dx) > abs(dy):
                self.direction = 2 if dx > 0 else 1
            else:
                self.direction = 0 if dy > 0 else 3
            # Walk cycle
            self._frame_time += dt
            if self._frame_time > 0.3:
                self.frame = 1 - self.frame
                self._frame_time = 0.0
        else:
            self.x = self.target_x
            self.y = self.target_y
            self.frame = 0
        # Remove dead bubbles
        self.bubbles = [b for b in self.bubbles if b.alive()]


# ─────────── tile function map ───────────

def _tile_fn(tile_id: int, variant: int = 0):
    """Return the surface for a ground tile."""
    if tile_id == HARDWOOD:
        return tile_hardwood(variant)
    if tile_id == CARPET:
        return tile_carpet(variant)
    if tile_id == GLASS_WALL:
        return tile_glass_wall()
    if tile_id == MARBLE:
        return tile_marble()
    if tile_id == OFFICE_WALL:
        return tile_office_wall()
    if tile_id == OFFICE_DOOR:
        return tile_office_door()
    if tile_id == CONF_CARPET:
        return tile_conf_carpet()
    if tile_id == ELEVATOR:
        return tile_elevator()
    if tile_id == ELEV_DOOR:
        return tile_elevator_door()
    return tile_hardwood(0)


def _deco_fn(deco_id: int, variant: int = 0):
    if deco_id == DESK:
        return sprite_desk(variant)
    if deco_id == OFFICE_CHAIR:
        return sprite_office_chair()
    if deco_id == POTTED_PLANT:
        return sprite_potted_plant(variant)
    if deco_id == WHITEBOARD:
        return sprite_whiteboard()
    if deco_id == COFFEE_MACHINE:
        return sprite_coffee_machine()
    if deco_id == WATER_COOLER:
        return sprite_water_cooler()
    if deco_id == SERVER_RACK:
        return sprite_server_rack()
    if deco_id == BOOKSHELF:
        return sprite_bookshelf()
    return None


# ════════════════════ MapRenderer ════════════════════


class MapRenderer:
    def __init__(self):
        self.tilemap = TileMap()
        self.agents: Dict[str, AgentSprite] = {}
        self._ground: Optional[pygame.Surface] = None
        self._last_tick = time.time()
        self._water_frame = 0
        self._water_timer = 0.0

    # ── public API ──

    def add_agent(self, name: str, color: str, icon: str):
        hx, hy = agent_home_pos(name)
        sp = AgentSprite(name=name, color=color, icon=icon,
                         x=float(hx), y=float(hy),
                         target_x=float(hx), target_y=float(hy))
        self.agents[name] = sp

    def set_agent_state(self, name: str, state: str,
                        target: Optional[Tuple[int, int]] = None):
        sp = self.agents.get(name)
        if not sp:
            return
        sp.state = state
        if target:
            sp.target_x = float(target[0])
            sp.target_y = float(target[1])

    def add_bubble(self, name: str, text: str, duration: float = 4.0):
        sp = self.agents.get(name)
        if sp:
            sp.bubbles.append(SpeechBubble(text, duration=duration))

    def update(self):
        now = time.time()
        dt = now - self._last_tick
        self._last_tick = now
        for sp in self.agents.values():
            sp.update(dt)
        # Animate water
        self._water_timer += dt
        if self._water_timer > 0.5:
            self._water_timer = 0.0
            self._water_frame = (self._water_frame + 1) % 4
            self._ground = None  # force rebake for water anim

    def draw(self, screen: pygame.Surface,
             area: Tuple[int, int, int, int],
             cam_offset: Tuple[int, int] = (0, 0)):
        ax, ay, aw, ah = area
        clip = pygame.Rect(ax, ay, aw, ah)
        screen.set_clip(clip)

        # Bake static ground
        if self._ground is None:
            self._bake_ground()

        # Camera
        world_w = MAP_W * T * SCALE
        world_h = MAP_H * T * SCALE
        cx = max(0, min(cam_offset[0], world_w - aw))
        cy = max(0, min(cam_offset[1], world_h - ah))

        # Draw ground
        screen.blit(self._ground, (ax - cx, ay - cy))

        # Decorations (Y-sorted with agents for depth)
        drawables: List[Tuple[float, str, object]] = []

        # Collect decos
        for (dx, dy), did in self.tilemap.decorations.items():
            variant = self.tilemap.deco_variants.get((dx, dy), 0)
            surf = _deco_fn(did, variant)
            if surf:
                px = dx * T * SCALE - cx + ax
                py = dy * T * SCALE - cy + ay
                drawables.append((dy + 0.9, "deco", (surf, px, py)))

        # Collect agents
        for sp in self.agents.values():
            px = sp.x * T * SCALE - cx + ax
            py = sp.y * T * SCALE - cy + ay
            drawables.append((sp.y + 0.5, "agent", (sp, px, py)))

        # Sort by Y
        drawables.sort(key=lambda d: d[0])

        for _, kind, payload in drawables:
            if kind == "deco":
                surf, px, py = payload
                screen.blit(surf, (px, py))
            elif kind == "agent":
                sp, px, py = payload
                self._draw_agent(screen, sp, px, py)

        # Building labels
        self._draw_labels(screen, ax, ay, cx, cy)

        screen.set_clip(None)

    # ── internals ──

    def _bake_ground(self):
        """Pre-render ground tiles to a big surface."""
        w = MAP_W * T * SCALE
        h = MAP_H * T * SCALE
        self._ground = pygame.Surface((w, h))
        self._ground.fill((30, 30, 30))

        for gy in range(MAP_H):
            for gx in range(MAP_W):
                tid = self.tilemap.tiles[gy][gx]
                variant = (gx * 7 + gy * 13) % 8
                surf = _tile_fn(tid, variant)
                px = gx * T * SCALE
                py = gy * T * SCALE
                if SCALE == 1:
                    self._ground.blit(surf, (px, py))
                else:
                    self._ground.blit(
                        pygame.transform.scale(
                            surf, (T * SCALE, T * SCALE)),
                        (px, py))

    def _draw_agent(self, screen: pygame.Surface,
                    sp: AgentSprite, px: float, py: float):
        """Draw one agent character + status icon + bubble."""
        cs = character_sprite(sp.name, sp.direction, sp.frame)
        # Center horizontally on tile, feet at tile bottom
        draw_x = px + (T * SCALE - CH_W * SCALE) // 2
        draw_y = py + T * SCALE - CHAR_H * SCALE
        if SCALE == 1:
            screen.blit(cs, (draw_x, draw_y))
        else:
            screen.blit(
                pygame.transform.scale(
                    cs, (CH_W * SCALE, CHAR_H * SCALE)),
                (draw_x, draw_y))

        # Status icon
        icon_map = {"idle": "zzz", "working": "working",
                    "cooldown": "alert"}
        icon_key = icon_map.get(sp.state, "zzz")
        si = status_icon(icon_key)
        screen.blit(si, (draw_x + CH_W * SCALE - 6, draw_y - 10))

        # Name label
        f = _font()
        ntxt = f.render(sp.name, True, pygame.Color(sp.color))
        nx = px + (T * SCALE - ntxt.get_width()) // 2
        screen.blit(ntxt, (nx, draw_y - 14))

        # Speech bubbles
        for i, bub in enumerate(sp.bubbles):
            self._draw_bubble(screen, bub.text,
                              px + T * SCALE // 2,
                              draw_y - 22 - i * 24)

    def _draw_bubble(self, screen, text, cx, top_y):
        f = _font()
        txt = f.render(text, True, (20, 20, 20))
        tw, th = txt.get_size()
        pad = 4
        bw = tw + pad * 2
        bh = th + pad * 2
        bx = int(cx - bw // 2)
        by = int(top_y - bh)
        bg = pygame.Surface((bw, bh), pygame.SRCALPHA)
        bg.fill((255, 255, 255, 210))
        screen.blit(bg, (bx, by))
        screen.blit(txt, (bx + pad, by + pad))

    def _draw_labels(self, screen, ax, ay, cx, cy):
        """Room name labels above their area."""
        f = _font()
        for rname, (rx, ry, rw, _rh) in ROOMS.items():
            label = rname.replace("_", " ").upper()
            txt = f.render(label, True, (220, 220, 220))
            px = ax + (rx + rw / 2) * T * SCALE - cx - txt.get_width() // 2
            py = ay + ry * T * SCALE - cy - 14
            screen.blit(txt, (int(px), int(py)))
