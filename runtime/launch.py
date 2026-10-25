#!/usr/bin/env python3
"""
QUE Guild Launcher — boots the agent dashboard from repo root.

Usage:
    python launch.py                                # Phaser browser dashboard (default)
    python launch.py --pack deepseek-r1-7b.rftmwpk  # dashboard + R1 native engine
    python launch.py --headless --pack deepseek-r1-7b.rftmwpk  # headless + R1
    python launch.py --godot    # legacy Godot visualization
    python launch.py --headless # headless tick loop only (no UI)
    python launch.py --port 9001  # custom port for dashboard
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import webbrowser
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
GODOT_PROJECT = REPO_ROOT / "godot" / "bronx-rpg"

# Known Godot paths to check
_GODOT_SEARCH = [
    os.environ.get("GODOT", ""),
    str(Path.home() / "Downloads" / "Godot_v4.6.2-stable_win64.exe" / "Godot_v4.6.2-stable_win64.exe"),
    str(Path.home() / "Downloads" / "Godot_v4.6.2-stable_win64.exe" / "Godot_v4.6.2-stable_win64_console.exe"),
    str(Path.home() / "scoop" / "apps" / "godot" / "current" / "godot.exe"),
    r"C:\Program Files\Godot\godot.exe",
    r"C:\Program Files (x86)\Godot\godot.exe",
]


def find_godot() -> str | None:
    godot_on_path = shutil.which("godot")
    if godot_on_path:
        return godot_on_path
    for candidate in _GODOT_SEARCH:
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def launch_dashboard(port: int = 9000, pack: str | None = None) -> int:
    """Launch the Phaser browser dashboard via WebSocket bridge."""
    print("QUE Guild — Agent Command Center")
    print(f"  Dashboard:  http://localhost:{port}")
    print(f"  WebSocket:  ws://localhost:{port}/ws")
    if pack:
        print(f"  Engine:     native C++ via {pack}")
    print()
    print("  8 agents will start patrolling automatically.")
    print("  Press Ctrl+C to shut down.")
    print()

    # Open browser after a short delay
    def _open_browser():
        time.sleep(1.5)
        webbrowser.open(f"http://localhost:{port}")

    import threading
    threading.Thread(target=_open_browser, daemon=True).start()

    # Start the bridge (blocking)
    bridge_path = REPO_ROOT / "src" / "dashboard" / "ws_bridge.py"
    cmd = [sys.executable, str(bridge_path), "--port", str(port)]
    if pack:
        cmd.extend(["--pack", pack])
    try:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
        return proc.returncode
    except KeyboardInterrupt:
        print("\nShutting down.")
        return 0


def launch_headless(pack: str | None = None) -> int:
    """Run the headless tick loop with no UI."""
    print("QUE Guild — Headless Mode")
    if pack:
        print(f"  Engine: native C++ via {pack}")
    print("  Agents will tick via stdout JSON. Ctrl+C to quit.")
    print()

    headless_path = REPO_ROOT / "src" / "rpg_headless.py"
    cmd = [sys.executable, str(headless_path)]
    if pack:
        cmd.extend(["--pack", pack])
    try:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
        return proc.returncode
    except KeyboardInterrupt:
        print("\nShutting down.")
        return 0


def launch_godot(editor: bool = False) -> int:
    """Legacy: launch via Godot."""
    godot = find_godot()
    if godot is None:
        print("ERROR: Cannot find Godot executable.")
        print("  Set the GODOT env var: set GODOT=C:\\path\\to\\Godot.exe")
        return 1

    if not GODOT_PROJECT.is_dir():
        print(f"ERROR: Godot project not found at {GODOT_PROJECT}")
        return 1

    print(f"QUE Guild — Godot Mode")
    print(f"  Godot:   {godot}")
    print(f"  Project: {GODOT_PROJECT}")
    print()

    # Bake assets if needed
    atlas = GODOT_PROJECT / "assets" / "atlas.json"
    if not atlas.exists():
        bake = REPO_ROOT / "tools" / "bake_bronx_assets.py"
        if bake.exists():
            subprocess.run([sys.executable, str(bake)], check=True)

    # Export world if needed
    world_json = GODOT_PROJECT / "data" / "world.json"
    if not world_json.exists():
        export = REPO_ROOT / "tools" / "export_bronx_world.py"
        if export.exists():
            subprocess.run([sys.executable, str(export)], check=True)

    cmd = [godot, "--path", str(GODOT_PROJECT)]
    if editor:
        cmd.append("--editor")
        print("Launching Godot editor...")
    else:
        print("Launching Godot visualization...")

    try:
        proc = subprocess.run(cmd)
        return proc.returncode
    except KeyboardInterrupt:
        print("\nShutting down.")
        return 0


def main() -> int:
    args = sys.argv[1:]

    # Parse --pack <path>
    pack = None
    if "--pack" in args:
        idx = args.index("--pack")
        if idx + 1 < len(args):
            pack = args[idx + 1]
        else:
            print("ERROR: --pack requires a .rftmwpk path")
            return 1

    if "--godot" in args:
        return launch_godot(editor="--editor" in args)
    elif "--headless" in args:
        return launch_headless(pack=pack)
    else:
        port = 9000
        if "--port" in args:
            idx = args.index("--port")
            if idx + 1 < len(args):
                try:
                    port = int(args[idx + 1])
                except ValueError:
                    print("ERROR: --port requires an integer")
                    return 1
        return launch_dashboard(port, pack=pack)


if __name__ == "__main__":
    raise SystemExit(main())
