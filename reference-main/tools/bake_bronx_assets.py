# SPDX-License-Identifier: AGPL-3.0-only
"""Bake QUE Guild office sprites to PNG + atlas.json for Godot.

Usage: python tools/bake_bronx_assets.py

Writes to godot/bronx-rpg/assets/.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

pygame.init()
pygame.display.set_mode((1, 1))

from src.rpg.sprite_factory import (  # noqa: E402
    tile_hardwood, tile_carpet, tile_glass_wall, tile_marble,
    tile_office_wall, tile_office_door, tile_conf_carpet,
    tile_elevator, tile_elevator_door,
    sprite_desk, sprite_office_chair, sprite_potted_plant,
    sprite_whiteboard, sprite_coffee_machine, sprite_water_cooler,
    sprite_server_rack, sprite_bookshelf,
    character_sprite, status_icon,
    T, CH_W, CH_H,
)

OUT = _ROOT / "godot" / "bronx-rpg" / "assets"
TILES = OUT / "tiles"
DECOS = OUT / "decos"
CHARS = OUT / "characters"
ICONS = OUT / "icons"


def _ensure():
    for d in (TILES, DECOS, CHARS, ICONS):
        d.mkdir(parents=True, exist_ok=True)


def _save(surf: pygame.Surface, path: Path) -> dict:
    pygame.image.save(surf, str(path))
    return {
        "path": str(path.relative_to(OUT)).replace("\\", "/"),
        "w": surf.get_width(),
        "h": surf.get_height(),
    }


def bake_tiles() -> dict:
    out = {}
    # hardwood variants 0..7
    for v in range(8):
        out[f"hardwood_{v}"] = _save(tile_hardwood(v), TILES / f"hardwood_{v}.png")
    # carpet variants 0..7
    for v in range(8):
        out[f"carpet_{v}"] = _save(tile_carpet(v), TILES / f"carpet_{v}.png")
    # single-surface tiles
    out["glass_wall"] = _save(tile_glass_wall(), TILES / "glass_wall.png")
    out["marble"] = _save(tile_marble(), TILES / "marble.png")
    out["office_wall"] = _save(tile_office_wall(), TILES / "office_wall.png")
    out["office_door"] = _save(tile_office_door(), TILES / "office_door.png")
    out["conf_carpet"] = _save(tile_conf_carpet(), TILES / "conf_carpet.png")
    out["elevator"] = _save(tile_elevator(), TILES / "elevator.png")
    out["elevator_door"] = _save(tile_elevator_door(), TILES / "elevator_door.png")
    return out


def bake_decos() -> dict:
    out = {}
    for v in range(4):
        out[f"desk_{v}"] = _save(sprite_desk(v), DECOS / f"desk_{v}.png")
    out["office_chair"] = _save(sprite_office_chair(), DECOS / "office_chair.png")
    for v in range(3):
        out[f"potted_plant_{v}"] = _save(sprite_potted_plant(v), DECOS / f"potted_plant_{v}.png")
    out["whiteboard"] = _save(sprite_whiteboard(), DECOS / "whiteboard.png")
    out["coffee_machine"] = _save(sprite_coffee_machine(), DECOS / "coffee_machine.png")
    out["water_cooler"] = _save(sprite_water_cooler(), DECOS / "water_cooler.png")
    out["server_rack"] = _save(sprite_server_rack(), DECOS / "server_rack.png")
    out["bookshelf"] = _save(sprite_bookshelf(), DECOS / "bookshelf.png")
    return out


def bake_characters() -> dict:
    out = {}
    names = ["FORGE", "VIAL", "WARD", "LORE", "SPARK", "TIDE", "ORACLE", "SAGE"]
    for name in names:
        frames = {}
        for direction in range(4):
            for frame in range(2):
                surf = character_sprite(name, direction, frame)
                path = CHARS / f"{name.lower()}_d{direction}_f{frame}.png"
                frames[f"d{direction}_f{frame}"] = _save(surf, path)
        out[name] = frames
    return out


def bake_icons() -> dict:
    out = {}
    for key in ("zzz", "working", "alert", "heart", "levelup"):
        out[key] = _save(status_icon(key), ICONS / f"{key}.png")
    return out


def main():
    _ensure()
    atlas = {
        "tile_size": T,
        "char_size": {"w": CH_W, "h": CH_H},
        "tiles": bake_tiles(),
        "decos": bake_decos(),
        "characters": bake_characters(),
        "icons": bake_icons(),
    }
    atlas_path = OUT / "atlas.json"
    atlas_path.write_text(json.dumps(atlas, indent=2), encoding="utf-8")
    total = (
        len(atlas["tiles"])
        + len(atlas["decos"])
        + sum(len(v) for v in atlas["characters"].values())
        + len(atlas["icons"])
    )
    print(f"Baked {total} sprites to {OUT}")
    print(f"Atlas: {atlas_path}")


if __name__ == "__main__":
    main()
