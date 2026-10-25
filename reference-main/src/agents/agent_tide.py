# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""TIDE — The Navigator. Validates mobile/frontend code and generates frontend health reports."""

from __future__ import annotations
import json
import re
import time
from pathlib import Path
from .agent_base import BaseAgent, _PROJECT_ROOT
from .quest_board import Quest, QuestBoard
from .skill_library import SkillLibrary


class TideAgent(BaseAgent):

    def __init__(self, quest_board: QuestBoard, skill_library: SkillLibrary):
        super().__init__("TIDE", quest_board, skill_library)

    def _register_skills(self):
        self.skill_library.register(
            "mobile_deps_check", "Validate mobile deps and run npm audit",
            "TIDE", self._skill_deps_check, category="frontend")
        self.skill_library.register(
            "mobile_structure", "Verify mobile project structure and file inventory",
            "TIDE", self._skill_structure_check, category="frontend")
        self.skill_library.register(
            "ts_syntax_scan", "Deep-scan TS/JS files for issues",
            "TIDE", self._skill_ts_scan, category="frontend")
        self.skill_library.register(
            "ts_todo_scan", "Find TODO/FIXME/HACK markers in frontend code",
            "TIDE", self._skill_todo_scan, category="frontend")
        self.skill_library.register(
            "frontend_report", "Generate comprehensive frontend health report",
            "TIDE", self._skill_frontend_report, category="frontend")

    def execute_quest(self, quest: Quest) -> str:
        title_lower = quest.title.lower()
        if "dep" in title_lower or "package" in title_lower or "audit" in title_lower:
            return self._skill_deps_check()
        elif "structure" in title_lower:
            return self._skill_structure_check()
        elif "typescript" in title_lower or "syntax" in title_lower or "scan" in title_lower:
            return self._skill_ts_scan()
        elif "todo" in title_lower or "fixme" in title_lower:
            return self._skill_todo_scan()
        elif "report" in title_lower:
            return self._skill_frontend_report()
        else:
            return self._skill_structure_check()

    def idle_action(self):
        thought = self.ollama_think(
            "I'm TIDE the Navigator. I validate mobile/frontend code. "
            "Should I: check dependencies, verify structure, scan TypeScript, or find TODOs?"
        )
        if thought.startswith("Error"):
            return self._skill_ts_scan()
        self._emit("idle_action", {"result": f"Thinking: {thought[:150]}"})
        tl = thought.lower()
        if "dep" in tl or "package" in tl:
            return self._skill_deps_check()
        elif "structure" in tl:
            return self._skill_structure_check()
        elif "todo" in tl or "fixme" in tl:
            return self._skill_todo_scan()
        return self._skill_ts_scan()

    # ── Helpers ───────────────────────────────────────────────────

    def _get_ts_files(self):
        mobile_dir = _PROJECT_ROOT / "mobile"
        if not mobile_dir.exists():
            return []
        files = list(mobile_dir.rglob("*.ts")) + list(mobile_dir.rglob("*.tsx"))
        return [f for f in files if "node_modules" not in str(f)]

    # ── Skills ────────────────────────────────────────────────────

    def _skill_deps_check(self) -> str:
        pkg_path = _PROJECT_ROOT / "mobile" / "package.json"
        if not pkg_path.exists():
            return "No mobile/package.json found"

        try:
            pkg = json.loads(pkg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            return f"Error: invalid package.json — {e}"

        deps = pkg.get("dependencies", {})
        dev_deps = pkg.get("devDependencies", {})
        name = pkg.get("name", "unknown")

        node_modules = _PROJECT_ROOT / "mobile" / "node_modules"
        installed = node_modules.exists()

        parts = [f"📦 {name}: {len(deps)} deps, {len(dev_deps)} devDeps. "
                 f"node_modules: {'present' if installed else 'MISSING'}"]

        # Try npm audit if node_modules present
        if installed:
            rc, output = self.safe_subprocess("npm audit --json")
            if rc == 0:
                try:
                    audit = json.loads(output)
                    vulns = audit.get("metadata", {}).get("vulnerabilities", {})
                    total_v = sum(vulns.get(sev, 0) for sev in ["high", "critical"])
                    parts.append(f"npm audit: {total_v} high/critical vulnerabilities")
                    if total_v > 0:
                        self.memory.add("deps_check",
                                        f"{total_v} high/critical vulns",
                                        importance=8)
                except (json.JSONDecodeError, KeyError):
                    parts.append("npm audit: ran (output not JSON)")
            else:
                parts.append("npm audit: skipped (not available or errored)")

        # Check for lock file presence
        lock_files = ["package-lock.json", "yarn.lock", "pnpm-lock.yaml"]
        found_lock = [l for l in lock_files if (pkg_path.parent / l).exists()]
        if found_lock:
            parts.append(f"Lock: {', '.join(found_lock)}")
        else:
            parts.append("⚠️ No lock file found")

        return " | ".join(parts)

    def _skill_structure_check(self) -> str:
        mobile_dir = _PROJECT_ROOT / "mobile"
        if not mobile_dir.exists():
            alt = _PROJECT_ROOT / "quantonium_os_src" / "frontend"
            if alt.exists():
                ts_count = len(list(alt.rglob("*.ts")))
                return f"Mobile code at quantonium_os_src/frontend/ ({ts_count} .ts files)"
            return "No mobile/ directory found"

        checks = {
            "package.json": (mobile_dir / "package.json").exists(),
            "app/": (mobile_dir / "app").exists(),
            "tsconfig": (mobile_dir / "tsconfig.json").exists(),
            "components/": (mobile_dir / "components").exists() or (mobile_dir / "app" / "components").exists(),
        }

        ts_files = self._get_ts_files()
        total_lines = 0
        largest = ("", 0)
        for f in ts_files:
            try:
                lines = len(f.read_text(errors="replace").splitlines())
                total_lines += lines
                if lines > largest[1]:
                    largest = (f.name, lines)
            except Exception:
                pass

        status = ", ".join(f"{k}:{'✅' if v else '❌'}" for k, v in checks.items())
        return (f"Mobile structure: {status} | {len(ts_files)} TS/TSX files, "
                f"~{total_lines:,} lines. Largest: {largest[0]} ({largest[1]}L)")

    def _skill_ts_scan(self) -> str:
        ts_files = self._get_ts_files()
        if not ts_files:
            return "No TS/TSX files found (mobile/ missing?)"

        issues = []
        for tf in ts_files[:50]:
            try:
                content = tf.read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()
                rel = tf.name

                # :any abuse
                any_count = content.count(": any")
                if any_count > 3:
                    issues.append(f"  {rel}: {any_count} uses of ': any'")

                # console.log left behind
                log_count = content.count("console.log")
                if log_count > 2:
                    issues.append(f"  {rel}: {log_count} console.log calls")

                # Unused imports (very rough: import line with name not used elsewhere)
                for line in lines:
                    m = re.match(r"import\s+(\w+)\s+from", line)
                    if m:
                        name = m.group(1)
                        # Count occurrences beyond the import line
                        if content.count(name) == 1:
                            issues.append(f"  {rel}: possibly unused import '{name}'")

                # Very long files
                if len(lines) > 500:
                    issues.append(f"  {rel}: {len(lines)} lines (consider splitting)")

            except Exception:
                pass

        if issues:
            self.memory.add("ts_scan",
                            f"{len(issues)} issues in {len(ts_files)} files",
                            importance=5)
            return f"⚠️ {len(issues)} issues in {len(ts_files)} TS files:\n" + "\n".join(issues[:15])
        return f"✅ {len(ts_files)} TS/TSX files scanned, no major issues"

    def _skill_todo_scan(self) -> str:
        ts_files = self._get_ts_files()
        if not ts_files:
            return "No TS/TSX files found"

        findings = []
        for tf in ts_files[:50]:
            try:
                for i, line in enumerate(tf.read_text(errors="replace").splitlines(), 1):
                    for marker in ("TODO", "FIXME", "HACK", "XXX"):
                        if marker in line:
                            findings.append(f"  {tf.name}:{i}: {line.strip()[:80]}")
                            break
            except Exception:
                pass

        if findings:
            return f"📝 {len(findings)} TODO/FIXME markers:\n" + "\n".join(findings[:20])
        return f"✅ No TODO/FIXME markers in {len(ts_files)} frontend files"

    def _skill_frontend_report(self) -> str:
        ts_files = self._get_ts_files()
        total_lines = 0
        any_count = 0
        log_count = 0
        for f in ts_files:
            try:
                content = f.read_text(errors="replace")
                total_lines += len(content.splitlines())
                any_count += content.count(": any")
                log_count += content.count("console.log")
            except Exception:
                pass

        # Dep info
        pkg_path = _PROJECT_ROOT / "mobile" / "package.json"
        dep_count = 0
        if pkg_path.exists():
            try:
                pkg = json.loads(pkg_path.read_text())
                dep_count = len(pkg.get("dependencies", {}))
            except Exception:
                pass

        health = "GREEN"
        if any_count > 20 or log_count > 20:
            health = "YELLOW"
        if not ts_files:
            health = "YELLOW"

        report = {
            "agent": "TIDE",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "health": health,
            "ts_files": len(ts_files),
            "total_lines": total_lines,
            "any_type_uses": any_count,
            "console_log_calls": log_count,
            "dependencies": dep_count,
        }

        result = self.safe_write(
            "data/agent_reports/tide_frontend_report.json",
            json.dumps(report, indent=2)
        )
        if result.startswith("OK"):
            return (f"Frontend report: {health} — {len(ts_files)} files, {total_lines:,} lines, "
                    f"{any_count} :any, {log_count} console.log")
        return result
