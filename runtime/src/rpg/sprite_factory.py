# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Hi-res procedural pixel-art sprite factory — Jerome Ave, Bronx.

32×32 tile textures, 24×40 chibi character sprites.
Inspired by the real Jerome Ave / Cromwell Ave streetscape:
  wide asphalt boulevard, yellow lane markings, brownstone apartments
  with fire escapes, storefronts with awnings, street trees, hydrants,
  traffic lights, parked cars.
"""

from __future__ import annotations

from functools import lru_cache

import pygame

T = 32  # tile size in pixels
CH_W, CH_H = 24, 40  # character sprite dimensions


# ───────────────────── helpers ──────────────────────

def _c(h: str):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)


def _surf(w: int = T, h: int = T, alpha: bool = True) -> pygame.Surface:
    s = pygame.Surface((w, h), pygame.SRCALPHA if alpha else 0)
    if alpha:
        s.fill((0, 0, 0, 0))
    return s


def _px(s: pygame.Surface, x: int, y: int, color):
    if isinstance(color, str):
        color = _c(color)
    if 0 <= x < s.get_width() and 0 <= y < s.get_height():
        s.set_at((x, y), color)


def _fr(s: pygame.Surface, x: int, y: int, w: int, h: int, color):
    if isinstance(color, str):
        color = _c(color)
    s.fill(color, pygame.Rect(x, y, w, h))


def _hline(s, x1, x2, y, color):
    _fr(s, x1, y, x2 - x1 + 1, 1, color)


def _vline(s, x, y1, y2, color):
    _fr(s, x, y1, 1, y2 - y1 + 1, color)


# ═══════════════════ TILES (32×32) ═══════════════════


@lru_cache(maxsize=64)
def tile_sidewalk(variant: int = 0) -> pygame.Surface:
    """Concrete sidewalk with expansion joints and subtle cracks."""
    s = _surf(alpha=False)
    s.fill(_c("#a0a09a"))
    # Expansion joint grid
    for x in range(T):
        _px(s, x, 15, _c("#8a8a84"))
        _px(s, x, 16, _c("#8a8a84"))
    _vline(s, 15, 0, T - 1, _c("#8a8a84"))
    _vline(s, 16, 0, T - 1, _c("#8a8a84"))
    # Specks
    seeds = [(4, 3), (10, 7), (22, 5), (7, 22), (25, 18),
             (14, 27), (3, 12), (28, 10), (18, 25), (8, 30)]
    for i, (sx, sy) in enumerate(seeds):
        if (i + variant) % 3 == 0:
            _px(s, sx, sy, "#929290")
        elif (i + variant) % 4 == 0:
            _px(s, sx, sy, "#ababab")
    # Crack on some variants
    if variant % 4 == 1:
        for d in range(8):
            _px(s, 10 + d, 20 + (d % 3) - 1, "#8a8a80")
    return s


@lru_cache(maxsize=8)
def tile_asphalt(variant: int = 0) -> pygame.Surface:
    """Dark asphalt. variant: 0=plain, 1=horiz yellow dashes, 2=vert dashes,
    3=crosswalk horiz, 4=crosswalk vert."""
    s = _surf(alpha=False)
    s.fill(_c("#3a3a3a"))
    # Rough texture
    for i, (px, py) in enumerate([(3, 5), (12, 9), (22, 3), (8, 20),
                                   (26, 15), (17, 28), (5, 25), (29, 7)]):
        _px(s, px, py, "#333333" if i % 2 == 0 else "#444444")
    if variant == 1:  # horiz yellow center dashes
        _fr(s, 0, 14, 12, 4, _c("#c8a830"))
        _fr(s, 20, 14, 12, 4, _c("#c8a830"))
    elif variant == 2:  # vert yellow center dashes
        _fr(s, 14, 0, 4, 12, _c("#c8a830"))
        _fr(s, 14, 20, 4, 12, _c("#c8a830"))
    elif variant == 3:  # crosswalk horiz white stripes
        for y in range(2, T - 2, 6):
            _fr(s, 0, y, T, 3, _c("#d0d0d0"))
    elif variant == 4:  # crosswalk vert white stripes
        for x in range(2, T - 2, 6):
            _fr(s, x, 0, 3, T, _c("#d0d0d0"))
    return s


@lru_cache(maxsize=64)
def tile_grass(variant: int = 0) -> pygame.Surface:
    s = _surf(alpha=False)
    s.fill(_c("#3a7a2e"))
    dark, light = _c("#2d6623"), _c("#4e9a3a")
    for i, (sx, sy) in enumerate([(5, 4), (14, 2), (24, 8), (8, 18),
                                   (18, 24), (28, 14), (3, 28), (20, 10),
                                   (12, 20), (26, 26)]):
        if (i + variant) % 3 == 0:
            _px(s, sx, sy, dark)
            _px(s, sx, sy - 1, dark)
        elif (i + variant) % 3 == 1:
            _px(s, sx, sy, light)
    return s


@lru_cache(maxsize=1)
def tile_brownstone() -> pygame.Surface:
    """Red-brown brick wall with mortar lines."""
    s = _surf(alpha=False)
    s.fill(_c("#8b4513"))
    mortar = _c("#6b3410")
    light = _c("#a0522d")
    # Brick rows every 4px, offset pattern
    for y in range(0, T, 4):
        _hline(s, 0, T - 1, y, mortar)
        offset = 6 if (y // 4) % 2 else 0
        for vy in range(1, 4):
            if y + vy < T:
                for vx in range(offset, T, 12):
                    _px(s, vx, y + vy, mortar)
    # Highlight bricks
    for bx, by in [(5, 2), (18, 6), (10, 14), (25, 10), (3, 22), (20, 26)]:
        if by < T:
            _px(s, bx, by, light)
            if bx + 1 < T:
                _px(s, bx + 1, by, light)
    return s


@lru_cache(maxsize=1)
def tile_brownstone_window() -> pygame.Surface:
    """Brownstone wall with a detailed window + fire escape bar."""
    s = tile_brownstone().copy()
    # Window hole
    _fr(s, 8, 4, 16, 18, _c("#1a2a4a"))
    # Glass panes — 2x2 grid
    _fr(s, 9, 5, 7, 8, _c("#2a4a6a"))
    _fr(s, 17, 5, 7, 8, _c("#2a4a6a"))
    _fr(s, 9, 14, 7, 7, _c("#2a4a6a"))
    _fr(s, 17, 14, 7, 7, _c("#2a4a6a"))
    # Window frame (brown wood)
    frame = _c("#5a4535")
    _hline(s, 7, 24, 3, frame)
    _hline(s, 7, 24, 22, frame)
    _hline(s, 7, 24, 13, frame)
    _vline(s, 7, 3, 22, frame)
    _vline(s, 16, 3, 22, frame)
    _vline(s, 24, 3, 22, frame)
    # Glass highlights
    _px(s, 10, 6, _c("#5a8aba"))
    _px(s, 18, 6, _c("#5a8aba"))
    # Fire escape railing below window
    _fr(s, 5, 23, 22, 2, _c("#555555"))
    _fr(s, 5, 25, 2, 5, _c("#555555"))
    _fr(s, 25, 25, 2, 5, _c("#555555"))
    # Railing bars
    for x in range(8, 25, 4):
        _vline(s, x, 23, 27, _c("#666666"))
    return s


@lru_cache(maxsize=1)
def tile_door() -> pygame.Surface:
    """Building entrance door with stoop steps."""
    s = _surf(alpha=False)
    s.fill(_c("#8b4513"))
    # Door recess
    _fr(s, 6, 0, 20, T, _c("#5a4030"))
    # Double door panels
    _fr(s, 7, 2, 9, 24, _c("#6a4a2a"))
    _fr(s, 17, 2, 9, 24, _c("#6a4a2a"))
    # Frame
    frame = _c("#3a2a18")
    _vline(s, 6, 0, 27, frame)
    _vline(s, 16, 0, 27, frame)
    _vline(s, 26, 0, 27, frame)
    _hline(s, 6, 26, 1, frame)
    _hline(s, 6, 26, 26, frame)
    # Handles
    _fr(s, 14, 12, 2, 3, _c("#d4aa44"))
    _fr(s, 18, 12, 2, 3, _c("#d4aa44"))
    # Transom window above
    _fr(s, 8, 0, 16, 2, _c("#3a5a7a"))
    # Steps
    _fr(s, 4, 27, 24, 2, _c("#888888"))
    _fr(s, 2, 29, 28, 3, _c("#777777"))
    return s


@lru_cache(maxsize=1)
def tile_roof() -> pygame.Surface:
    """Flat roof with AC unit, pipes, and parapet edge."""
    s = _surf(alpha=False)
    s.fill(_c("#555555"))
    # Tar/gravel texture
    for y in range(0, T, 3):
        for x in range(0, T, 5):
            _px(s, x, y, _c("#4a4a4a"))
    # AC unit
    _fr(s, 18, 6, 10, 8, _c("#6a6a6a"))
    _fr(s, 19, 7, 8, 6, _c("#777777"))
    _fr(s, 21, 9, 4, 2, _c("#888888"))  # fan grill
    # Pipe
    _fr(s, 4, 3, 3, 20, _c("#4a4a4a"))
    _fr(s, 5, 2, 1, 1, _c("#555555"))
    # Parapet edge (top)
    _fr(s, 0, 0, T, 2, _c("#666666"))
    return s


@lru_cache(maxsize=4)
def tile_water(frame: int = 0) -> pygame.Surface:
    s = _surf(alpha=False)
    s.fill(_c("#1a5fb4"))
    off = frame % 4
    for y in range(0, T, 4):
        for x in range(T):
            wx = (x + off * 3 + y) % 10
            if wx < 3:
                _px(s, x, y, "#5b9fd4")
            elif wx > 7:
                _px(s, x, y, "#134c8f")
    return s


@lru_cache(maxsize=8)
def tile_awning(color: str = "#cc4444") -> pygame.Surface:
    """Storefront awning with scalloped edge."""
    s = _surf()
    c = _c(color)
    dark = (max(0, c[0] - 50), max(0, c[1] - 50), max(0, c[2] - 50), 255)
    # Striped awning body
    for y in range(0, 14):
        for x in range(T):
            _px(s, x, y, c if (x // 5) % 2 == 0 else dark)
    # Scalloped bottom edge
    for x in range(0, T, 4):
        _fr(s, x, 14, 3, 2, c)
        _px(s, x + 1, 16, c)
    # Shadow below
    _fr(s, 0, 17, T, 3, (0, 0, 0, 50))
    return s


@lru_cache(maxsize=8)
def tile_storefront(sign_text: str = "SHOP",
                    awning_color: str = "#cc4444") -> pygame.Surface:
    """Storefront ground floor — glass window, awning, signage."""
    s = _surf(alpha=False)
    s.fill(_c("#8b4513"))  # brick surround
    # Large display window
    _fr(s, 3, 8, 26, 18, _c("#1a3a5a"))
    # Glass reflections
    _fr(s, 5, 10, 10, 14, _c("#2a4a6a"))
    _fr(s, 17, 10, 10, 14, _c("#2a4a6a"))
    _px(s, 6, 11, _c("#5a8aba"))
    _px(s, 18, 11, _c("#5a8aba"))
    # Window frame
    _vline(s, 3, 8, 26, _c("#5a4535"))
    _vline(s, 15, 8, 26, _c("#5a4535"))
    _vline(s, 28, 8, 26, _c("#5a4535"))
    _hline(s, 3, 28, 8, _c("#5a4535"))
    _hline(s, 3, 28, 26, _c("#5a4535"))
    # Awning at top
    c = _c(awning_color)
    dk = (max(0, c[0] - 50), max(0, c[1] - 50), max(0, c[2] - 50), 255)
    for y in range(0, 7):
        for x in range(T):
            s.set_at((x, y), c if (x // 4) % 2 == 0 else dk)
    # Curb/step at bottom
    _fr(s, 0, 28, T, 4, _c("#888888"))
    return s


# ═══════════════════ DECORATIONS (32×32) ═══════════════════


@lru_cache(maxsize=4)
def sprite_tree(variant: int = 0) -> pygame.Surface:
    """Street tree with full canopy and metal grate at base."""
    s = _surf()
    # Trunk
    _fr(s, 13, 20, 6, 10, _c("#5a3a1a"))
    _fr(s, 14, 22, 4, 8, _c("#4a2a10"))
    # Canopy — large round
    for y in range(0, 20):
        hw = max(1, min(14, 15 - abs(y - 8)))
        for dx in range(-hw, hw + 1):
            c = _c("#2a7a3a")
            if abs(dx) >= hw - 1 or y <= 1:
                c = _c("#1a5a2a")
            elif abs(dx) < hw - 3 and y > 3:
                c = _c("#3a9a4a")
            _px(s, 16 + dx, y, c)
    # Light dapple
    for bx, by in [(12, 5), (18, 3), (14, 10), (20, 8)]:
        _px(s, bx, by, _c("#50b858"))
    # Metal tree grate at base
    _fr(s, 8, 30, 16, 2, _c("#666666"))
    for x in range(10, 24, 3):
        _px(s, x, 30, _c("#555555"))
    return s


@lru_cache(maxsize=1)
def sprite_hydrant() -> pygame.Surface:
    s = _surf()
    body = _c("#cc2222")
    # Base
    _fr(s, 10, 24, 12, 4, _c("#aa1111"))
    # Body
    _fr(s, 11, 12, 10, 14, body)
    # Bonnet (top)
    _fr(s, 10, 8, 12, 5, _c("#dd3333"))
    _fr(s, 13, 5, 6, 4, _c("#dd3333"))
    # Side nozzles
    _fr(s, 7, 14, 4, 4, body)
    _fr(s, 21, 14, 4, 4, body)
    # Operating nut
    _fr(s, 14, 6, 4, 2, _c("#ee5555"))
    # Chain
    _px(s, 8, 13, _c("#888888"))
    _px(s, 23, 13, _c("#888888"))
    # Highlight
    _px(s, 14, 14, _c("#ee4444"))
    return s


@lru_cache(maxsize=1)
def sprite_signpost() -> pygame.Surface:
    """NYC-style green street sign on pole."""
    s = _surf()
    # Pole
    _fr(s, 15, 10, 2, 22, _c("#666666"))
    # Green sign plate
    _fr(s, 2, 2, 28, 8, _c("#1a5a1a"))
    _fr(s, 2, 2, 28, 1, _c("#2a7a2a"))
    _fr(s, 2, 9, 28, 1, _c("#0a4a0a"))
    # White text lines
    _fr(s, 5, 4, 22, 1, _c("#e0e0e0"))
    _fr(s, 7, 7, 18, 1, _c("#e0e0e0"))
    # Traffic light below sign
    _fr(s, 13, 11, 6, 10, _c("#333333"))
    _fr(s, 14, 12, 4, 2, _c("#cc2222"))   # red
    _fr(s, 14, 15, 4, 2, _c("#c8a830"))   # yellow
    _fr(s, 14, 18, 4, 2, _c("#22aa22"))   # green
    return s


@lru_cache(maxsize=1)
def sprite_bball_hoop() -> pygame.Surface:
    s = _surf()
    # Pole
    _fr(s, 14, 14, 4, 18, _c("#888888"))
    # Backboard
    _fr(s, 6, 2, 20, 12, _c("#e0e0e0"))
    _fr(s, 6, 2, 20, 1, _c("#cccccc"))
    _fr(s, 8, 4, 16, 8, _c("#dddddd"))
    # Red target square
    _fr(s, 12, 6, 8, 4, _c("#cc4400"))
    _fr(s, 13, 7, 6, 2, _c("#e0e0e0"))
    # Rim
    _fr(s, 10, 14, 12, 2, _c("#cc4400"))
    # Net strings
    for dy in range(2, 8):
        _px(s, 11, 14 + dy, _c("#dddddd"))
        _px(s, 14, 14 + dy, _c("#dddddd"))
        _px(s, 17, 14 + dy, _c("#dddddd"))
        _px(s, 20, 14 + dy, _c("#dddddd"))
    return s


@lru_cache(maxsize=4)
def sprite_car(variant: int = 0) -> pygame.Surface:
    """Parked car — facing right. variant changes color."""
    colors = [_c("#2a2a8a"), _c("#8a2a2a"), _c("#e0e0e0"), _c("#2a2a2a")]
    body = colors[variant % len(colors)]
    s = _surf()
    # Body
    _fr(s, 2, 14, 28, 10, body)
    # Roof
    _fr(s, 8, 8, 16, 7, body)
    # Dark body line
    dark = (max(0, body[0] - 40), max(0, body[1] - 40),
            max(0, body[2] - 40), 255)
    _hline(s, 2, 29, 24, dark)
    # Windshield
    _fr(s, 8, 9, 6, 5, _c("#5a8aba"))
    # Rear window
    _fr(s, 18, 9, 6, 5, _c("#5a8aba"))
    # Headlights
    _fr(s, 28, 16, 2, 3, _c("#ffffaa"))
    # Taillights
    _fr(s, 2, 16, 2, 3, _c("#cc2222"))
    # Wheels
    _fr(s, 6, 24, 6, 6, _c("#222222"))
    _fr(s, 8, 26, 2, 2, _c("#888888"))  # hubcap
    _fr(s, 20, 24, 6, 6, _c("#222222"))
    _fr(s, 22, 26, 2, 2, _c("#888888"))
    # Door line
    _vline(s, 15, 14, 23, dark)
    return s


# ═══════════════════ CHARACTERS (24×40) ═══════════════════


def _draw_character(s, skin, hair, outfit, accent, direction, frame,
                    hat_type="none"):
    """Draw a chibi character on a 24×40 surface.
    direction: 0=down 1=left 2=right 3=up.  frame: 0 or 1."""
    sk, hr, ot, ac = _c(skin), _c(hair), _c(outfit), _c(accent)
    ot_dk = (max(0, ot[0] - 30), max(0, ot[1] - 30),
             max(0, ot[2] - 30), 255)
    W = 24

    # ── Head (y 2..14) ──
    if direction == 0:  # facing down
        _fr(s, 5, 1, 14, 5, hr)          # hair top
        _px(s, 4, 3, hr); _px(s, 19, 3, hr)
        _fr(s, 5, 6, 14, 8, sk)          # face
        _px(s, 4, 7, sk); _px(s, 4, 8, sk)
        _px(s, 19, 7, sk); _px(s, 19, 8, sk)
        # Eyes
        _fr(s, 7, 8, 3, 3, _c("#1a1a2e"))
        _fr(s, 14, 8, 3, 3, _c("#1a1a2e"))
        _px(s, 7, 8, _c("#ffffff")); _px(s, 14, 8, _c("#ffffff"))
        # Mouth
        _fr(s, 10, 12, 4, 1, _c("#c47070"))
        # Bangs
        _fr(s, 5, 5, 4, 3, hr); _fr(s, 15, 5, 4, 3, hr)
    elif direction == 3:  # up
        _fr(s, 5, 1, 14, 13, hr)
        _px(s, 4, 3, hr); _px(s, 19, 3, hr)
        _px(s, 4, 5, hr); _px(s, 19, 5, hr)
        _px(s, 4, 8, sk); _px(s, 19, 8, sk)
    elif direction == 1:  # left
        _fr(s, 7, 1, 12, 5, hr)
        _fr(s, 7, 6, 12, 8, sk)
        _px(s, 5, 3, hr); _px(s, 5, 4, hr); _px(s, 5, 5, hr)
        _px(s, 5, 7, hr); _px(s, 5, 8, hr); _px(s, 5, 9, hr)
        _px(s, 19, 7, sk); _px(s, 19, 8, sk)
        _fr(s, 9, 8, 3, 3, _c("#1a1a2e"))
        _px(s, 9, 8, _c("#ffffff"))
        _px(s, 14, 10, _c("#c47070"))
    else:  # right
        _fr(s, 5, 1, 12, 5, hr)
        _fr(s, 5, 6, 12, 8, sk)
        _px(s, 18, 3, hr); _px(s, 18, 4, hr); _px(s, 18, 5, hr)
        _px(s, 18, 7, hr); _px(s, 18, 8, hr); _px(s, 18, 9, hr)
        _px(s, 4, 7, sk); _px(s, 4, 8, sk)
        _fr(s, 12, 8, 3, 3, _c("#1a1a2e"))
        _px(s, 14, 8, _c("#ffffff"))
        _px(s, 9, 10, _c("#c47070"))

    # ── Hat / headgear ──
    if hat_type == "cap":
        _fr(s, 4, 0, 16, 3, ac)
        if direction == 0:
            _fr(s, 3, 3, 18, 2, ac)
        elif direction == 1:
            _fr(s, 2, 3, 10, 2, ac)
        elif direction == 2:
            _fr(s, 12, 3, 10, 2, ac)
    elif hat_type == "goggles":
        _fr(s, 5, 1, 14, 3, hr)
        _fr(s, 4, 5, 16, 2, ac)
        _fr(s, 7, 5, 4, 2, _c("#e0e0e0"))
        _fr(s, 13, 5, 4, 2, _c("#e0e0e0"))
    elif hat_type == "beanie":
        _fr(s, 5, 0, 14, 3, ac)
        _fr(s, 4, 2, 16, 2, ac)
        ac_dk = (max(0, ac[0] - 40), max(0, ac[1] - 40),
                 max(0, ac[2] - 40), 255)
        _fr(s, 4, 4, 16, 1, ac_dk)
    elif hat_type == "hardhat":
        _fr(s, 4, 0, 16, 4, ac)
        _px(s, 3, 3, ac); _px(s, 20, 3, ac)
        _hline(s, 5, 18, 1, (min(255, ac[0]+30), min(255, ac[1]+30),
                              min(255, ac[2]+30), 255))
    elif hat_type == "headphones":
        _fr(s, 6, 0, 12, 2, _c("#333333"))
        if direction != 3:
            _fr(s, 2, 5, 4, 5, _c("#333333"))
            _fr(s, 3, 6, 2, 3, ac)
            _fr(s, 18, 5, 4, 5, _c("#333333"))
            _fr(s, 19, 6, 2, 3, ac)
    elif hat_type == "glasses":
        if direction in (0, 2):
            _hline(s, 6, 17, 8, _c("#c0c0c0"))
            _fr(s, 7, 7, 4, 3, _c("#aaaacc"))
            _fr(s, 13, 7, 4, 3, _c("#aaaacc"))

    # ── Body (y 14..26) ──
    _fr(s, 6, 14, 12, 12, ot)
    _fr(s, 9, 14, 6, 2, sk)       # neck
    _fr(s, 6, 24, 12, 1, ot_dk)   # belt
    _fr(s, 10, 24, 4, 1, ac)      # buckle
    # Arms
    if frame == 0:
        _fr(s, 2, 16, 4, 8, ot)
        _fr(s, 18, 16, 4, 8, ot)
        _fr(s, 2, 23, 4, 2, sk)
        _fr(s, 18, 23, 4, 2, sk)
    else:
        _fr(s, 2, 15, 4, 8, ot)
        _fr(s, 18, 17, 4, 8, ot)
        _fr(s, 2, 22, 4, 2, sk)
        _fr(s, 18, 24, 4, 2, sk)
    # Shoulder details
    _px(s, 6, 16, ac); _px(s, 17, 16, ac)

    # Outfit details
    if hat_type == "cap":
        _fr(s, 8, 18, 3, 3, ot_dk)
        _fr(s, 13, 18, 3, 3, ot_dk)
    elif hat_type == "glasses":
        for y in range(16, 22):
            _px(s, 11, y, ac)

    # ── Legs (y 26..39) ──
    pants = ot_dk
    boot = _c("#2a2a2a")
    wo = 2 if frame == 1 else 0
    _fr(s, 7, 26, 5, 7 + wo, pants)
    _fr(s, 7, 33 + wo, 5, 4, boot)
    _fr(s, 12, 26, 5, 7 - wo, pants)
    _fr(s, 12, 33 - wo, 5, 4 + wo, boot)

    # Shadow under feet
    for x in range(5, 19):
        _px(s, x, 39, (0, 0, 0, 60))


# ═══════════════════ OFFICE TILES (32×32) ═══════════════════


@lru_cache(maxsize=64)
def tile_hardwood(variant: int = 0) -> pygame.Surface:
    """Warm hardwood plank floor for hallways / common areas."""
    s = _surf(alpha=False)
    s.fill(_c("#8b6914"))
    light, dark = _c("#a07828"), _c("#70560e")
    # Horizontal plank lines
    for y in range(0, T, 8):
        _hline(s, 0, T - 1, y, dark)
        # Offset join every other plank row
        off = 14 if ((y // 8) + variant) % 2 else 0
        if off > 0 and y > 0:
            _vline(s, off, y - 7, y, dark)
        _vline(s, (off + 22) % T, y + 1, min(y + 7, T - 1), dark)
    # Wood grain dots
    seeds = [(4, 3), (10, 11), (22, 5), (7, 19), (25, 15),
             (14, 27), (3, 23), (28, 9), (18, 25), (8, 30)]
    for i, (sx, sy) in enumerate(seeds):
        if (i + variant) % 3 == 0:
            _px(s, sx, sy, light)
        elif (i + variant) % 4 == 0:
            _px(s, sx, sy, dark)
    return s


@lru_cache(maxsize=64)
def tile_carpet(variant: int = 0) -> pygame.Surface:
    """Blue-gray office carpet for private offices."""
    s = _surf(alpha=False)
    s.fill(_c("#3a4a5a"))
    dark, light = _c("#2e3e4e"), _c("#4a5a6a")
    # Cross-hatch texture
    for y in range(0, T, 4):
        for x in range(0, T, 4):
            _px(s, x, y, dark)
    # Speckling
    seeds = [(3, 5), (11, 7), (23, 3), (7, 21), (26, 17),
             (15, 29), (2, 13), (29, 11), (19, 23), (9, 27)]
    for i, (sx, sy) in enumerate(seeds):
        if (i + variant) % 3 == 0:
            _px(s, sx, sy, light)
        elif (i + variant) % 4 == 0:
            _px(s, sx, sy, dark)
    return s


@lru_cache(maxsize=1)
def tile_glass_wall() -> pygame.Surface:
    """Floor-to-ceiling window — dark reflective glass with city skyline."""
    s = _surf(alpha=False)
    s.fill(_c("#0a1a2a"))
    # Metal frame
    _hline(s, 0, T - 1, 0, _c("#444444"))
    _hline(s, 0, T - 1, T - 1, _c("#444444"))
    _vline(s, 0, 0, T - 1, _c("#444444"))
    _vline(s, T - 1, 0, T - 1, _c("#444444"))
    _vline(s, 15, 0, T - 1, _c("#333333"))
    _vline(s, 16, 0, T - 1, _c("#333333"))
    # Skyline silhouette
    buildings = [(3, 12, 4, 18), (8, 8, 3, 18), (12, 14, 3, 18),
                 (17, 6, 4, 18), (22, 10, 3, 18), (26, 14, 4, 18)]
    for bx, by, bw, bh in buildings:
        _fr(s, bx, by, bw, T - 1 - by, _c("#141e28"))
        # Little window lights
        for wy in range(by + 1, T - 2, 3):
            for wx in range(bx + 1, bx + bw - 1):
                if (wx + wy) % 5 < 2:
                    _px(s, wx, wy, _c("#3a5a70"))
    # Reflection streaks
    for dy in range(4, 28, 6):
        _px(s, 5, dy, _c("#1a3a5a"))
        _px(s, 6, dy + 1, _c("#1a3a5a"))
    return s


@lru_cache(maxsize=1)
def tile_marble() -> pygame.Surface:
    """Polished marble floor for elevator lobby."""
    s = _surf(alpha=False)
    s.fill(_c("#d8d0c8"))
    vein = _c("#b0a898")
    # Subtle veining
    for d in range(12):
        _px(s, 4 + d, 8 + (d % 3), vein)
        _px(s, 20 - d, 22 - (d % 4), vein)
        _px(s, 10 + d, 16 + (d % 2), vein)
    # Polish highlights
    _px(s, 8, 4, _c("#e8e0d8"))
    _px(s, 20, 12, _c("#e8e0d8"))
    _px(s, 14, 24, _c("#e8e0d8"))
    # Tile seam
    _hline(s, 0, T - 1, T - 1, _c("#c0b8b0"))
    _vline(s, T - 1, 0, T - 1, _c("#c0b8b0"))
    return s


@lru_cache(maxsize=1)
def tile_office_wall() -> pygame.Surface:
    """Light gray drywall partition."""
    s = _surf(alpha=False)
    s.fill(_c("#c0c0c0"))
    # Subtle texture
    for y in range(0, T, 6):
        for x in range(0, T, 5):
            _px(s, x, y, _c("#b8b8b8"))
    # Baseboard at bottom
    _fr(s, 0, T - 3, T, 3, _c("#808080"))
    _hline(s, 0, T - 1, T - 4, _c("#aaaaaa"))
    return s


@lru_cache(maxsize=1)
def tile_office_door() -> pygame.Surface:
    """Frosted glass door with brushed metal frame."""
    s = _surf(alpha=False)
    s.fill(_c("#c0c0c0"))  # wall background
    # Door frame (brushed metal)
    frame = _c("#999999")
    _vline(s, 4, 0, T - 1, frame)
    _vline(s, 27, 0, T - 1, frame)
    _hline(s, 4, 27, 0, frame)
    # Frosted glass panel
    _fr(s, 5, 1, 22, 28, _c("#a8c0d0"))
    # Frosting effect — horizontal streaks
    for y in range(2, 28, 3):
        _hline(s, 6, 26, y, _c("#b0c8d8"))
    # Handle (brass)
    _fr(s, 24, 12, 2, 6, _c("#d4aa44"))
    # Floor threshold
    _fr(s, 0, 29, T, 3, _c("#808080"))
    return s


@lru_cache(maxsize=1)
def tile_conf_carpet() -> pygame.Surface:
    """Dark navy conference room carpet."""
    s = _surf(alpha=False)
    s.fill(_c("#1a2040"))
    dark, light = _c("#141830"), _c("#242850")
    # Subtle diamond pattern
    for y in range(0, T, 8):
        for x in range(0, T, 8):
            _px(s, x + 4, y, dark)
            _px(s, x, y + 4, dark)
            _px(s, x + 4, y + 4, light)
    return s


@lru_cache(maxsize=1)
def tile_elevator() -> pygame.Surface:
    """Brushed steel elevator doors (closed)."""
    s = _surf(alpha=False)
    s.fill(_c("#888888"))
    # Door panels
    _fr(s, 2, 2, 13, 28, _c("#909090"))
    _fr(s, 17, 2, 13, 28, _c("#909090"))
    # Center seam
    _vline(s, 15, 2, 29, _c("#666666"))
    _vline(s, 16, 2, 29, _c("#666666"))
    # Brushed metal streaks
    for y in range(4, 28, 3):
        _hline(s, 3, 14, y, _c("#989898"))
        _hline(s, 18, 29, y, _c("#989898"))
    # Frame
    _fr(s, 0, 0, T, 2, _c("#707070"))
    _fr(s, 0, 30, T, 2, _c("#707070"))
    _vline(s, 0, 0, T - 1, _c("#707070"))
    _vline(s, T - 1, 0, T - 1, _c("#707070"))
    # Floor indicator above
    _fr(s, 12, 0, 8, 2, _c("#222222"))
    _px(s, 15, 0, _c("#ff4444"))  # floor number LED
    return s


@lru_cache(maxsize=1)
def tile_elevator_door() -> pygame.Surface:
    """Open elevator entrance — visible dark interior."""
    s = _surf(alpha=False)
    s.fill(_c("#d8d0c8"))  # marble surround
    # Dark elevator interior
    _fr(s, 4, 2, 24, 28, _c("#1a1a1a"))
    # Parted metal doors (slightly open)
    _fr(s, 4, 2, 3, 28, _c("#808080"))
    _fr(s, 25, 2, 3, 28, _c("#808080"))
    # Door streaks
    for y in range(4, 28, 4):
        _px(s, 5, y, _c("#909090"))
        _px(s, 26, y, _c("#909090"))
    # Frame
    _vline(s, 3, 0, T - 1, _c("#999999"))
    _vline(s, 28, 0, T - 1, _c("#999999"))
    _hline(s, 3, 28, 1, _c("#999999"))
    # Floor threshold
    _fr(s, 0, 30, T, 2, _c("#c0b8b0"))
    return s


# ═══════════════════ OFFICE DECORATIONS (32×32) ═══════════════════


@lru_cache(maxsize=4)
def sprite_desk(variant: int = 0) -> pygame.Surface:
    """Top-down L-shaped desk with monitor and keyboard."""
    colors = [_c("#c8a060"), _c("#e0e0e0"), _c("#4a3020"), _c("#808080")]
    top = colors[variant % len(colors)]
    s = _surf()
    # L-shaped desk surface
    _fr(s, 2, 4, 28, 14, top)          # main surface
    _fr(s, 2, 18, 10, 10, top)         # side return
    # Desk edge (darker)
    dk = (max(0, top[0] - 40), max(0, top[1] - 40),
          max(0, top[2] - 40), 255)
    _hline(s, 2, 29, 4, dk)
    _hline(s, 2, 29, 17, dk)
    _hline(s, 2, 11, 27, dk)
    _vline(s, 2, 4, 27, dk)
    _vline(s, 29, 4, 17, dk)
    _vline(s, 11, 18, 27, dk)
    # Monitor
    _fr(s, 14, 5, 12, 8, _c("#1a1a2a"))   # screen
    _fr(s, 15, 6, 10, 6, _c("#2040a0"))   # display
    _px(s, 16, 7, _c("#40a0ff"))           # cursor/glow
    _fr(s, 18, 13, 6, 2, _c("#333333"))   # stand
    # Keyboard
    _fr(s, 4, 9, 9, 4, _c("#333333"))
    _fr(s, 5, 10, 7, 2, _c("#444444"))    # key area
    # Mouse
    _fr(s, 5, 15, 3, 4, _c("#222222"))
    _px(s, 6, 15, _c("#444444"))           # scroll wheel
    # Papers on side return
    _fr(s, 4, 20, 5, 6, _c("#f0f0e8"))
    _hline(s, 5, 8, 22, _c("#aaaaaa"))
    _hline(s, 5, 8, 24, _c("#aaaaaa"))
    return s


@lru_cache(maxsize=1)
def sprite_office_chair() -> pygame.Surface:
    """Rolling office chair top-down view."""
    s = _surf()
    # Star base (chrome)
    for dx, dy in [(-6, 0), (6, 0), (0, -6), (0, 6), (-4, -4), (4, 4)]:
        _fr(s, 14 + dx, 14 + dy, 4, 4, _c("#888888"))
    # Seat (black)
    _fr(s, 8, 8, 16, 16, _c("#2a2a2a"))
    # Seat cushion
    _fr(s, 10, 10, 12, 12, _c("#333333"))
    # Armrests
    _fr(s, 6, 10, 3, 12, _c("#3a3a3a"))
    _fr(s, 23, 10, 3, 12, _c("#3a3a3a"))
    # Center wheel hub
    _fr(s, 14, 14, 4, 4, _c("#666666"))
    return s


@lru_cache(maxsize=4)
def sprite_potted_plant(variant: int = 0) -> pygame.Surface:
    """Indoor plant: 0=fern, 1=palm, 2=succulent."""
    s = _surf()
    # Pot
    pot_c = _c("#a0522d") if variant != 2 else _c("#e0e0e0")
    _fr(s, 10, 22, 12, 8, pot_c)
    _fr(s, 8, 21, 16, 2, pot_c)
    # Dirt
    _fr(s, 11, 21, 10, 2, _c("#4a3020"))
    if variant == 0:  # fern — spreading fronds
        green = _c("#2a8a3a")
        dk = _c("#1a6a2a")
        for dx in range(-8, 9, 2):
            for dy in range(0, 14):
                spread = abs(dx) * dy // 8
                px_x = 16 + dx + spread // 4
                px_y = 20 - dy
                if 0 <= px_x < T and 0 <= px_y < T:
                    _px(s, px_x, px_y, green if dy % 3 else dk)
    elif variant == 1:  # palm — tall trunk with top fronds
        _fr(s, 14, 8, 4, 14, _c("#5a3a1a"))
        for angle in range(5):
            for d in range(8):
                dx = (angle - 2) * 3 + (d if angle > 2 else -d)
                dy = -(8 - d)
                px_x = 16 + dx
                px_y = 8 + dy + abs(d * d) // 12
                if 0 <= px_x < T and 0 <= px_y < T:
                    _px(s, px_x, px_y, _c("#2a7a2a"))
    else:  # succulent — compact rosette
        for ring in range(6, 0, -1):
            c = _c("#3a9a4a") if ring % 2 else _c("#2a7a3a")
            for dx in range(-ring, ring + 1):
                for dy in range(-ring, ring + 1):
                    if abs(dx) + abs(dy) <= ring:
                        px_x, px_y = 16 + dx, 16 + dy
                        if 0 <= px_x < T and 0 <= px_y < T:
                            _px(s, px_x, px_y, c)
    return s


@lru_cache(maxsize=1)
def sprite_whiteboard() -> pygame.Surface:
    """Wall-mounted whiteboard with marker scribbles."""
    s = _surf()
    # Frame
    _fr(s, 2, 2, 28, 22, _c("#888888"))
    # White surface
    _fr(s, 3, 3, 26, 20, _c("#f0f0f0"))
    # Marker scribbles
    for d in range(10):
        _px(s, 5 + d, 6 + (d % 3), _c("#cc2222"))
    for d in range(8):
        _px(s, 8 + d, 12 + (d % 2), _c("#2222cc"))
    for d in range(12):
        _px(s, 4 + d, 18, _c("#22aa22"))
    # Marker tray
    _fr(s, 4, 24, 24, 3, _c("#555555"))
    # Markers in tray
    _fr(s, 6, 24, 4, 2, _c("#cc2222"))
    _fr(s, 12, 24, 4, 2, _c("#2222cc"))
    _fr(s, 18, 24, 4, 2, _c("#22aa22"))
    return s


@lru_cache(maxsize=1)
def sprite_coffee_machine() -> pygame.Surface:
    """Espresso machine on small counter."""
    s = _surf()
    # Counter
    _fr(s, 4, 18, 24, 12, _c("#c8a060"))
    _hline(s, 4, 27, 18, _c("#a08040"))
    # Machine body
    _fr(s, 8, 4, 16, 16, _c("#2a2a2a"))
    # Silver top
    _fr(s, 8, 4, 16, 3, _c("#888888"))
    # Display panel
    _fr(s, 10, 8, 12, 5, _c("#444444"))
    # Power LED
    _px(s, 22, 14, _c("#ff2222"))
    # Drip tray
    _fr(s, 10, 16, 12, 3, _c("#555555"))
    # Coffee cup
    _fr(s, 3, 20, 4, 6, _c("#e0e0e0"))
    _fr(s, 4, 20, 2, 2, _c("#4a2010"))  # coffee inside
    return s


@lru_cache(maxsize=1)
def sprite_water_cooler() -> pygame.Surface:
    """Standard water cooler with blue jug."""
    s = _surf()
    # Stand body (white)
    _fr(s, 10, 14, 12, 16, _c("#e0e0e0"))
    # Spigot area
    _fr(s, 12, 18, 8, 4, _c("#cccccc"))
    _fr(s, 15, 20, 2, 3, _c("#888888"))  # spigot
    # Water jug on top (blue)
    _fr(s, 11, 2, 10, 13, _c("#4488cc"))
    _fr(s, 12, 0, 8, 3, _c("#3377bb"))
    # Water level inside
    _fr(s, 12, 4, 8, 9, _c("#5599dd"))
    _px(s, 14, 5, _c("#77bbee"))  # highlight
    # Cup dispenser on side
    _fr(s, 22, 16, 4, 6, _c("#cccccc"))
    _fr(s, 23, 17, 2, 4, _c("#f0f0f0"))  # cups
    return s


@lru_cache(maxsize=1)
def sprite_server_rack() -> pygame.Surface:
    """Server rack with blinking LEDs."""
    s = _surf()
    # Rack frame (dark metal)
    _fr(s, 4, 0, 24, 32, _c("#1a1a1a"))
    # Frame edges
    _vline(s, 4, 0, 31, _c("#333333"))
    _vline(s, 27, 0, 31, _c("#333333"))
    # Server units (horizontal)
    for i in range(6):
        y = 2 + i * 5
        _fr(s, 5, y, 22, 4, _c("#2a2a2a"))
        _hline(s, 5, 26, y, _c("#333333"))
        # Front panel
        _fr(s, 6, y + 1, 18, 2, _c("#222222"))
        # LEDs
        _px(s, 25, y + 1, _c("#00ff44"))  # green power
        _px(s, 25, y + 2, _c("#4488ff") if i % 2 == 0 else _c("#44ff88"))
        _px(s, 23, y + 1, _c("#ff8800") if i == 2 else _c("#00ff44"))
    return s


@lru_cache(maxsize=1)
def sprite_bookshelf() -> pygame.Surface:
    """Bookshelf with colored book spines."""
    s = _surf()
    # Shelf frame (dark wood)
    wood = _c("#5a3a1a")
    _fr(s, 2, 0, 28, 32, wood)
    # Shelves — 4 rows
    book_colors = [
        _c("#cc3333"), _c("#3333cc"), _c("#33aa33"), _c("#8b4513"),
        _c("#dd8800"), _c("#aa33aa"), _c("#338888"), _c("#cccc33"),
    ]
    for row in range(4):
        sy = 1 + row * 8
        _fr(s, 3, sy, 26, 7, _c("#3a2010"))  # shelf back
        # Books
        bx = 3
        for bi in range(5):
            bw = 4 + (bi % 2)
            bh = 5 + ((bi + row) % 3)
            c = book_colors[(bi + row * 3) % len(book_colors)]
            _fr(s, bx, sy + (7 - bh), bw, bh, c)
            # Spine highlight
            _px(s, bx + 1, sy + (7 - bh) + 1, (min(255, c[0] + 40),
                 min(255, c[1] + 40), min(255, c[2] + 40), 255))
            bx += bw + 1
            if bx > 26:
                break
        # Shelf board
        _hline(s, 2, 29, sy + 7, _c("#4a2a10"))
    return s


# Bronx agent visual configs
_CHAR_CONFIGS = {
    "FORGE":  {"skin": "#a0785a", "hair": "#1a1a1a", "outfit": "#2a4a8a",
               "accent": "#e87830", "hat": "cap"},
    "VIAL":   {"skin": "#f0d8c0", "hair": "#4a0080", "outfit": "#e0e0e0",
               "accent": "#8b5cf6", "hat": "goggles"},
    "WARD":   {"skin": "#8a6540", "hair": "#1a1a2a", "outfit": "#1a2a5a",
               "accent": "#3b82f6", "hat": "cap"},
    "LORE":   {"skin": "#f0d0b0", "hair": "#c49a3c", "outfit": "#2a6a3a",
               "accent": "#10b981", "hat": "glasses"},
    "SPARK":  {"skin": "#e8c098", "hair": "#ff6600", "outfit": "#8a6a20",
               "accent": "#f59e0b", "hat": "beanie"},
    "TIDE":   {"skin": "#6a4a30", "hair": "#1a3a5a", "outfit": "#e87830",
               "accent": "#06b6d4", "hat": "hardhat"},
    "ORACLE": {"skin": "#e8d0c0", "hair": "#c0c0d0", "outfit": "#5a2060",
               "accent": "#ec4899", "hat": "headphones"},
    "SAGE":   {"skin": "#d0b898", "hair": "#2a2a3a", "outfit": "#4a3080",
               "accent": "#9333ea", "hat": "glasses"},
}


@lru_cache(maxsize=128)
def character_sprite(name: str, direction: int = 0,
                     frame: int = 0) -> pygame.Surface:
    cfg = _CHAR_CONFIGS.get(name, _CHAR_CONFIGS["FORGE"])
    s = _surf(CH_W, CH_H)
    _draw_character(s, cfg["skin"], cfg["hair"], cfg["outfit"],
                    cfg["accent"], direction, frame, cfg["hat"])
    return s


# ═══════════════════ STATUS ICONS (12×12) ═══════════════════


@lru_cache(maxsize=8)
def status_icon(icon_type: str) -> pygame.Surface:
    s = _surf(12, 12)
    if icon_type == "zzz":
        _fr(s, 1, 1, 4, 2, _c("#aaaaff"))
        _fr(s, 3, 4, 4, 2, _c("#8888dd"))
        _fr(s, 5, 7, 4, 2, _c("#6666bb"))
    elif icon_type == "working":
        for dx, dy in [(6, 0), (5, 1), (4, 2), (3, 3), (4, 4),
                        (5, 5), (6, 6), (5, 7), (4, 8)]:
            _px(s, dx, dy, "#ffd700")
            _px(s, dx + 1, dy, "#ffd700")
    elif icon_type == "alert":
        _fr(s, 4, 0, 4, 8, _c("#ff4444"))
        _fr(s, 4, 9, 4, 3, _c("#ff4444"))
    elif icon_type == "heart":
        for dx, dy in [(2, 2), (3, 1), (4, 2), (5, 1), (6, 0),
                        (7, 1), (8, 2), (9, 1), (5, 4), (6, 5),
                        (5, 3), (7, 3), (3, 3), (9, 3), (4, 4),
                        (8, 4), (6, 3), (7, 5)]:
            _px(s, dx, dy, "#ff4466")
    elif icon_type == "levelup":
        _fr(s, 5, 0, 2, 2, _c("#ffd700"))
        _fr(s, 3, 2, 6, 2, _c("#ffd700"))
        _fr(s, 1, 4, 10, 1, _c("#ffd700"))
        _fr(s, 5, 5, 2, 6, _c("#ffd700"))
    return s
