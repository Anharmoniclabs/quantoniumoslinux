# SPDX-License-Identifier: AGPL-3.0-only
"""Export QUE Guild office TileMap to world.json for Godot.

Usage: python tools/export_bronx_world.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.rpg.world import (  # noqa: E402
    TileMap, MAP_W, MAP_H, TILE_SIZE,
    HARDWOOD, CARPET, GLASS_WALL, MARBLE,
    OFFICE_WALL, OFFICE_DOOR, CONF_CARPET, ELEVATOR, ELEV_DOOR,
    DESK, OFFICE_CHAIR, POTTED_PLANT, WHITEBOARD,
    COFFEE_MACHINE, WATER_COOLER, SERVER_RACK, BOOKSHELF,
    ROOMS, AGENT_HOMES, MEETING_POINT, agent_home_pos,
)

_TILE_NAMES = {
    HARDWOOD: "hardwood", CARPET: "carpet",
    GLASS_WALL: "glass_wall", MARBLE: "marble",
    OFFICE_WALL: "office_wall", OFFICE_DOOR: "office_door",
    CONF_CARPET: "conf_carpet", ELEVATOR: "elevator",
    ELEV_DOOR: "elevator_door",
}
_DECO_NAMES = {
    DESK: "desk", OFFICE_CHAIR: "office_chair",
    POTTED_PLANT: "potted_plant", WHITEBOARD: "whiteboard",
    COFFEE_MACHINE: "coffee_machine", WATER_COOLER: "water_cooler",
    SERVER_RACK: "server_rack", BOOKSHELF: "bookshelf",
}


def main():
    tm = TileMap()

    tiles: list[list[dict]] = []
    for gy in range(MAP_H):
        row = []
        for gx in range(MAP_W):
            tid = tm.tiles[gy][gx]
            name = _TILE_NAMES.get(tid, "hardwood")
            cell: dict = {"t": name}

            if tid in (HARDWOOD, CARPET):
                cell["v"] = (gx * 7 + gy * 13) % 8
            # other tiles have no variant

            row.append(cell)
        tiles.append(row)

    decorations = []
    for (dx, dy), did in tm.decorations.items():
        decorations.append({
            "x": dx, "y": dy,
            "type": _DECO_NAMES.get(did, "desk"),
            "variant": tm.deco_variants.get((dx, dy), 0),
        })

    buildings = []
    for rkey, (rx, ry, rw, rh) in ROOMS.items():
        # Door position for west offices (col 7), east offices (col 23)
        if rx < 15:
            door_x = rx + rw
        else:
            door_x = rx - 1
        buildings.append({
            "key": rkey,
            "x": rx, "y": ry, "w": rw, "h": rh,
            "label": rkey.replace("_", " ").upper(),
            "door": {"x": door_x, "y": ry + rh // 2},
        })

    agent_homes = {}
    for name in AGENT_HOMES:
        hx, hy = agent_home_pos(name)
        agent_homes[name] = {"x": hx, "y": hy}

    out = {
        "tile_size": TILE_SIZE,
        "map_w": MAP_W,
        "map_h": MAP_H,
        "village_square": {"x": MEETING_POINT[0], "y": MEETING_POINT[1]},
        "tiles": tiles,
        "decorations": decorations,
        "buildings": buildings,
        "agent_homes": agent_homes,
    }

    dest = _ROOT / "godot" / "bronx-rpg" / "data" / "world.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Exported {MAP_W}x{MAP_H} office world to {dest}")
    print(f"  {len(decorations)} decorations, {len(buildings)} rooms")


if __name__ == "__main__":
    main()
