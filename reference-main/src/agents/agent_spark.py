# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""SPARK — The Artificer. Smoke-tests UI, validates assets, and generates UI health reports."""

from __future__ import annotations
import hashlib
import json
import re
import time
from pathlib import Path
from .agent_base import BaseAgent, _PROJECT_ROOT
from .quest_board import Quest, QuestBoard
from .skill_library import SkillLibrary


class SparkAgent(BaseAgent):

    def __init__(self, quest_board: QuestBoard, skill_library: SkillLibrary):
        super().__init__("SPARK", quest_board, skill_library)

    def _register_skills(self):
        self.skill_library.register(
            "asset_check", "Verify UI assets exist with sizes and hashes",
            "SPARK", self._skill_asset_check, category="ui")
        self.skill_library.register(
            "app_import_check", "Syntax-validate chatbox and RPG modules",
            "SPARK", self._skill_import_check, category="ui")
        self.skill_library.register(
            "theme_validate", "Validate theme color palettes and contrast",
            "SPARK", self._skill_theme_validate, category="ui")
        self.skill_library.register(
            "rpg_integrity", "Validate RPG sprite/world/renderer consistency",
            "SPARK", self._skill_rpg_integrity, category="ui")
        self.skill_library.register(
            "ui_report", "Generate comprehensive UI health report",
            "SPARK", self._skill_ui_report, category="ui")

    def execute_quest(self, quest: Quest) -> str:
        title_lower = quest.title.lower()
        if "asset" in title_lower or "icon" in title_lower:
            return self._skill_asset_check()
        elif "import" in title_lower or "smoke" in title_lower or "syntax" in title_lower:
            return self._skill_import_check()
        elif "theme" in title_lower or "color" in title_lower:
            return self._skill_theme_validate()
        elif "rpg" in title_lower or "sprite" in title_lower:
            return self._skill_rpg_integrity()
        elif "report" in title_lower:
            return self._skill_ui_report()
        else:
            return self._skill_asset_check()

    def idle_action(self):
        thought = self.ollama_think(
            "I'm SPARK the Artificer. I smoke-test UI and validate assets. "
            "Should I: check assets, validate imports, or check RPG integrity?"
        )
        if thought.startswith("Error"):
            return self._skill_import_check()
        self._emit("idle_action", {"result": f"Thinking: {thought[:150]}"})
        tl = thought.lower()
        if "asset" in tl:
            return self._skill_asset_check()
        elif "rpg" in tl or "sprite" in tl:
            return self._skill_rpg_integrity()
        elif "theme" in tl:
            return self._skill_theme_validate()
        return self._skill_import_check()

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _sha256_short(path: Path) -> str:
        h = hashlib.sha256()
        try:
            h.update(path.read_bytes())
        except Exception:
            return "ERR"
        return h.hexdigest()[:12]

    # ── Skills ────────────────────────────────────────────────────

    def _skill_asset_check(self) -> str:
        ui_dir = _PROJECT_ROOT / "ui"
        findings = []

        expected = ["que_icon.png", "que_icon.ico"]
        for name in expected:
            p = ui_dir / name
            if p.exists():
                kb = p.stat().st_size / 1024
                sha = self._sha256_short(p)
                findings.append(f"{name}: {kb:.1f}KB sha={sha} ✅")
            else:
                findings.append(f"{name}: MISSING ❌")

        if ui_dir.exists():
            all_assets = [f for f in ui_dir.iterdir() if f.is_file()]
            total_kb = sum(f.stat().st_size for f in all_assets) / 1024
            findings.append(f"Total: {len(all_assets)} assets, {total_kb:.0f}KB")

            self.memory.add("asset_check",
                            f"{len(all_assets)} assets, {total_kb:.0f}KB",
                            importance=3)
        else:
            findings.append("ui/ directory: MISSING ❌")

        return " | ".join(findings)

    def _skill_import_check(self) -> str:
        apps_dir = _PROJECT_ROOT / "src" / "apps"
        rpg_dir = _PROJECT_ROOT / "src" / "rpg"
        results = []
        errors_found = 0

        # Chatbox modules
        chatbox_mods = ["chatbox_theme", "chatbox_widgets", "chatbox_network", "chatbox_agent"]
        for mod in chatbox_mods:
            mod_file = apps_dir / f"{mod}.py"
            if mod_file.exists():
                content = mod_file.read_text(encoding="utf-8", errors="replace")
                try:
                    compile(content, str(mod_file), "exec")
                    lines = len(content.splitlines())
                    results.append(f"apps/{mod}: OK ({lines}L) ✅")
                except SyntaxError as e:
                    errors_found += 1
                    results.append(f"apps/{mod}: SYNTAX ERROR line {e.lineno} ❌")
            else:
                results.append(f"apps/{mod}: NOT FOUND ❌")

        # RPG modules
        if rpg_dir.exists():
            for py in sorted(rpg_dir.glob("*.py")):
                if py.name.startswith("__"):
                    continue
                content = py.read_text(encoding="utf-8", errors="replace")
                try:
                    compile(content, str(py), "exec")
                    results.append(f"rpg/{py.name}: OK ✅")
                except SyntaxError as e:
                    errors_found += 1
                    results.append(f"rpg/{py.name}: LINE {e.lineno} ❌")

        if errors_found:
            self.memory.add("import_check",
                            f"{errors_found} syntax errors found",
                            importance=8)
        return "\n".join(results)

    def _skill_theme_validate(self) -> str:
        theme_file = _PROJECT_ROOT / "src" / "apps" / "chatbox_theme.py"
        if not theme_file.exists():
            return "Error: chatbox_theme.py not found"

        content = theme_file.read_text(encoding="utf-8", errors="replace")

        has_dark = "DARK" in content
        has_light = "LIGHT" in content

        hex_colors = sorted(set(re.findall(r'#[0-9a-fA-F]{6}', content)))

        # Check for very low contrast pairs (both very dark or both very light)
        dark_colors = [c for c in hex_colors if int(c[1:3], 16) < 40]
        light_colors = [c for c in hex_colors if int(c[1:3], 16) > 220]

        results = [
            f"DARK palette: {'✅' if has_dark else '❌'}",
            f"LIGHT palette: {'✅' if has_light else '❌'}",
            f"Unique colors: {len(hex_colors)}",
            f"Very dark (<40): {len(dark_colors)}",
            f"Very light (>220): {len(light_colors)}",
        ]

        return " | ".join(results)

    def _skill_rpg_integrity(self) -> str:
        rpg_dir = _PROJECT_ROOT / "src" / "rpg"
        if not rpg_dir.exists():
            return "Error: src/rpg/ not found"

        issues = []

        # Check sprite_factory.py for tile size
        sf = rpg_dir / "sprite_factory.py"
        if sf.exists():
            content = sf.read_text(encoding="utf-8", errors="replace")
            tile_match = re.search(r'TILE\s*=\s*(\d+)', content)
            if tile_match:
                tile_size = int(tile_match.group(1))
                if tile_size not in (16, 32, 64):
                    issues.append(f"Unusual TILE size: {tile_size}")
            else:
                issues.append("No TILE constant found in sprite_factory")
        else:
            issues.append("sprite_factory.py missing")

        # Check world.py for grid dimensions
        wf = rpg_dir / "world.py"
        if wf.exists():
            content = wf.read_text(encoding="utf-8", errors="replace")
            cols_match = re.search(r'COLS\s*=\s*(\d+)', content)
            rows_match = re.search(r'ROWS\s*=\s*(\d+)', content)
            ts_match = re.search(r'TILE_SIZE\s*=\s*(\d+)', content)
            if cols_match and rows_match and ts_match:
                cols, rows, ts = int(cols_match.group(1)), int(rows_match.group(1)), int(ts_match.group(1))
                w = cols * ts
                h = rows * ts
                if w > 1000 or h > 540:
                    issues.append(f"World {w}×{h} exceeds viewport 1000×540")
            else:
                issues.append("Missing grid constants in world.py")
        else:
            issues.append("world.py missing")

        # Check renderer exists
        rf = rpg_dir / "renderer.py"
        if rf.exists():
            content = rf.read_text(encoding="utf-8", errors="replace")
            has_scale = "SCALE" in content
            has_map = "MapRenderer" in content
            if not has_map:
                issues.append("MapRenderer class missing from renderer")
        else:
            issues.append("renderer.py missing")

        if issues:
            self.memory.add("rpg_integrity",
                            f"{len(issues)} issues: {'; '.join(issues[:3])}",
                            importance=6)
            return "⚠️ RPG integrity issues:\n  " + "\n  ".join(issues)
        return "✅ RPG modules consistent (tiles, grid, renderer)"

    def _skill_ui_report(self) -> str:
        # Asset inventory
        ui_dir = _PROJECT_ROOT / "ui"
        asset_count = 0
        total_kb = 0
        if ui_dir.exists():
            assets = [f for f in ui_dir.iterdir() if f.is_file()]
            asset_count = len(assets)
            total_kb = sum(f.stat().st_size for f in assets) / 1024

        # Syntax check across all UI modules
        syntax_errors = 0
        module_count = 0
        for d in [_PROJECT_ROOT / "src" / "apps", _PROJECT_ROOT / "src" / "rpg"]:
            if not d.exists():
                continue
            for py in d.glob("*.py"):
                if py.name.startswith("__"):
                    continue
                module_count += 1
                try:
                    compile(py.read_text(errors="replace"), str(py), "exec")
                except SyntaxError:
                    syntax_errors += 1

        health = "GREEN"
        if syntax_errors > 0:
            health = "RED"
        elif asset_count == 0:
            health = "YELLOW"

        report = {
            "agent": "SPARK",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "health": health,
            "ui_assets": asset_count,
            "assets_total_kb": round(total_kb, 1),
            "modules_checked": module_count,
            "syntax_errors": syntax_errors,
        }

        result = self.safe_write(
            "data/agent_reports/spark_ui_report.json",
            json.dumps(report, indent=2)
        )
        if result.startswith("OK"):
            return (f"UI report: {health} — {asset_count} assets, {module_count} modules, "
                    f"{syntax_errors} syntax errors")
        return result
