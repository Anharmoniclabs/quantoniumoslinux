# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
World map — QUE Guild Penthouse Office.

32×32-pixel tile grid.  Top-floor luxury office with floor-to-ceiling
glass walls, 8 private offices (4 per side), a central hallway, a
conference area, and an elevator lobby.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

TILE_SIZE = 32

# ── tile material (office) ──
HARDWOOD    = 0   # hallway / common floor
CARPET      = 1   # private office carpet
GLASS_WALL  = 2   # perimeter windows
MARBLE      = 3   # elevator lobby
OFFICE_WALL = 4   # interior partitions
OFFICE_DOOR = 5   # frosted glass doors
CONF_CARPET = 6   # conference room carpet
ELEVATOR    = 7   # elevator shaft
ELEV_DOOR   = 8   # open elevator entrance

# ── decoration IDs (office) ──
DESK           = 20
OFFICE_CHAIR   = 21
POTTED_PLANT   = 22
WHITEBOARD     = 23
COFFEE_MACHINE = 24
WATER_COOLER   = 25
SERVER_RACK    = 26
BOOKSHELF      = 27

# ═══════════════ map dimensions ═══════════════
MAP_W = 31
MAP_H = 17

# ═══════════════ rooms ═══════════════
# Each room: (x, y, w, h)
# West offices (cols 1-6), East offices (cols 24-29)
ROOMS = {
    "forge_office":   (1,  1,  6, 3),
    "vial_office":    (1,  5,  6, 3),
    "ward_office":    (1,  9,  6, 3),
    "lore_office":    (1,  13, 6, 3),
    "spark_office":   (24, 1,  6, 3),
    "tide_office":    (24, 5,  6, 3),
    "oracle_office":  (24, 9,  6, 3),
    "sage_office":    (24, 13, 6, 3),
    "conference":     (12, 6,  7, 4),
    "elevator_lobby": (13, 14, 6, 2),
}

AGENT_HOMES = {
    "FORGE":  "forge_office",
    "VIAL":   "vial_office",
    "WARD":   "ward_office",
    "LORE":   "lore_office",
    "SPARK":  "spark_office",
    "TIDE":   "tide_office",
    "ORACLE": "oracle_office",
    "SAGE":   "sage_office",
}

# Conference center for gatherings
MEETING_POINT = (15, 8)
# Legacy alias
VILLAGE_SQUARE = MEETING_POINT

# ── room labels: key → (color, label) ──
_ROOM_LABELS = {
    "forge_office":  ("#d4a056", "FORGE"),
    "vial_office":   ("#8b5cf6", "VIAL"),
    "ward_office":   ("#3b82f6", "WARD"),
    "lore_office":   ("#10b981", "LORE"),
    "spark_office":  ("#f59e0b", "SPARK"),
    "tide_office":   ("#06b6d4", "TIDE"),
    "oracle_office": ("#ec4899", "ORACLE"),
    "sage_office":   ("#9333ea", "SAGE"),
    "conference":    ("#ffffff", "CONFERENCE"),
}

# Legacy aliases for renderer compatibility
BUILDINGS = ROOMS
_STOREFRONTS = _ROOM_LABELS

# Legacy tile aliases (for renderer imports that haven't been updated)
SIDEWALK    = HARDWOOD
ASPHALT     = CARPET
WATER       = GLASS_WALL
GRASS       = MARBLE
BROWNSTONE  = OFFICE_WALL
DOOR        = OFFICE_DOOR
WINDOW_TILE = CONF_CARPET
ROOF        = ELEVATOR
STOREFRONT  = ELEV_DOOR
TREE        = DESK
HYDRANT     = OFFICE_CHAIR
SIGNPOST    = POTTED_PLANT
BBALL_HOOP  = WHITEBOARD
CAR         = COFFEE_MACHINE


def agent_home_pos(name: str) -> tuple[int, int]:
    """Return (tile_x, tile_y) for an agent's desk (center of office)."""
    bkey = AGENT_HOMES.get(name)
    if bkey and bkey in ROOMS:
        rx, ry, rw, rh = ROOMS[bkey]
        return (rx + rw // 2, ry + rh // 2)
    return MEETING_POINT


@dataclass
class TileMap:
    tiles: list = field(default_factory=list)
    decorations: dict = field(default_factory=dict)
    deco_variants: dict = field(default_factory=dict)
    storefront_data: dict = field(default_factory=dict)

    def __post_init__(self):
        self.tiles = [[HARDWOOD] * MAP_W for _ in range(MAP_H)]
        self._build_office()

    # ── query ──

    def get_tile(self, x: int, y: int) -> int:
        if not self._in_bounds(x, y):
            return GLASS_WALL
        return self.tiles[y][x]

    def is_walkable(self, x: int, y: int) -> bool:
        t = self.get_tile(x, y)
        return t in (HARDWOOD, CARPET, MARBLE, OFFICE_DOOR, CONF_CARPET,
                     ELEV_DOOR)

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < MAP_W and 0 <= y < MAP_H

    # ── office builder ──

    def _build_office(self):
        rng = random.Random(42)

        # ── Fill entire floor with hardwood ──
        for y in range(MAP_H):
            for x in range(MAP_W):
                self.tiles[y][x] = HARDWOOD

        # ── Perimeter glass walls ──
        for x in range(MAP_W):
            self.tiles[0][x] = GLASS_WALL
            self.tiles[MAP_H - 1][x] = GLASS_WALL
        for y in range(MAP_H):
            self.tiles[y][0] = GLASS_WALL
            self.tiles[y][MAP_W - 1] = GLASS_WALL

        # ── Interior walls: vertical partitions at col 7 and col 23 ──
        for y in range(1, MAP_H - 1):
            self.tiles[y][7] = OFFICE_WALL
            self.tiles[y][23] = OFFICE_WALL

        # ── Horizontal dividers at rows 4, 8, 12 (office separators) ──
        for row in (4, 8, 12):
            for x in range(1, 8):
                self.tiles[row][x] = OFFICE_WALL
            for x in range(23, MAP_W - 1):
                self.tiles[row][x] = OFFICE_WALL

        # ── Fill office interiors with carpet ──
        west_offices = [
            (1, 1, 6, 3),   # forge
            (1, 5, 6, 3),   # vial
            (1, 9, 6, 3),   # ward
            (1, 13, 6, 3),  # lore
        ]
        east_offices = [
            (24, 1, 6, 3),   # spark
            (24, 5, 6, 3),   # tide
            (24, 9, 6, 3),   # oracle
            (24, 13, 6, 3),  # sage
        ]
        for ox, oy, ow, oh in west_offices + east_offices:
            for ry in range(oy, oy + oh):
                for rx in range(ox, ox + ow):
                    if self._in_bounds(rx, ry):
                        self.tiles[ry][rx] = CARPET

        # ── Office doors ──
        # West doors at col 7, east doors at col 23
        for door_y in (2, 6, 10, 14):
            self.tiles[door_y][7] = OFFICE_DOOR
            self.tiles[door_y][23] = OFFICE_DOOR

        # ── Conference room (center) ──
        for ry in range(6, 10):
            for rx in range(12, 19):
                self.tiles[ry][rx] = CONF_CARPET

        # ── Elevator lobby ──
        for ry in range(14, 16):
            for rx in range(13, 19):
                self.tiles[ry][rx] = MARBLE
        # Elevator shaft
        self.tiles[15][17] = ELEVATOR
        self.tiles[15][18] = ELEVATOR
        # Elevator entrance
        self.tiles[14][17] = ELEV_DOOR

        # ══════ Decorations ══════

        # ── Desks and chairs in each office ──
        # West offices: desk near glass wall (col 2), chair at col 3
        west_agents = ["FORGE", "VIAL", "WARD", "LORE"]
        for i, (ox, oy, ow, oh) in enumerate(west_offices):
            desk_x, desk_y = ox + 1, oy + 1
            self.decorations[(desk_x, desk_y)] = DESK
            self.deco_variants[(desk_x, desk_y)] = i % 4
            self.decorations[(desk_x + 2, desk_y)] = OFFICE_CHAIR

        # East offices: desk near glass wall (col 28), chair at col 27
        east_agents = ["SPARK", "TIDE", "ORACLE", "SAGE"]
        for i, (ox, oy, ow, oh) in enumerate(east_offices):
            desk_x, desk_y = ox + 4, oy + 1
            self.decorations[(desk_x, desk_y)] = DESK
            self.deco_variants[(desk_x, desk_y)] = (i + 2) % 4
            self.decorations[(desk_x - 2, desk_y)] = OFFICE_CHAIR

        # ── Bookshelves in some offices ──
        for bx, by in [(5, 1), (5, 9), (5, 13), (25, 9), (25, 13)]:
            self.decorations[(bx, by)] = BOOKSHELF

        # ── Server racks ──
        for sx, sy in [(5, 5), (25, 1)]:
            self.decorations[(sx, sy)] = SERVER_RACK

        # ── Common area amenities ──
        # Coffee machine - north hallway
        self.decorations[(9, 1)] = COFFEE_MACHINE
        # Water cooler - north hallway east side
        self.decorations[(21, 1)] = WATER_COOLER

        # ── Potted plants at hallway corners ──
        for px, py in [(8, 4), (22, 4), (8, 12), (22, 12),
                        (11, 8), (19, 8)]:
            self.decorations[(px, py)] = POTTED_PLANT
            self.deco_variants[(px, py)] = rng.randint(0, 2)

        # ── Whiteboard in conference room ──
        self.decorations[(15, 6)] = WHITEBOARD
