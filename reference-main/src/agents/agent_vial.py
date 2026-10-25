# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""VIAL — The Alchemist. Runs real tests, tracks coverage, generates reports."""

from __future__ import annotations
import json
import re
import time
from pathlib import Path
from .agent_base import BaseAgent, _PROJECT_ROOT
from .quest_board import Quest, QuestBoard
from .skill_library import SkillLibrary


class VialAgent(BaseAgent):

    def __init__(self, quest_board: QuestBoard, skill_library: SkillLibrary):
        super().__init__("VIAL", quest_board, skill_library)

    def _register_skills(self):
        self.skill_library.register(
            "run_pytest", "Run pytest suite and report pass/fail",
            "VIAL", self._skill_run_pytest, category="test")
        self.skill_library.register(
            "run_pytest_targeted", "Run pytest on a specific test file or directory",
            "VIAL", self._skill_run_pytest_targeted, category="test")
        self.skill_library.register(
            "count_tests", "Count test files and test functions",
            "VIAL", self._skill_count_tests, category="test")
        self.skill_library.register(
            "find_untested", "Find source files without matching tests",
            "VIAL", self._skill_find_untested, category="test")
        self.skill_library.register(
            "test_report", "Generate comprehensive test health report",
            "VIAL", self._skill_test_report, category="test")

    def execute_quest(self, quest: Quest) -> str:
        title_lower = quest.title.lower()
        if "run" in title_lower and "test" in title_lower:
            return self._skill_run_pytest()
        elif "pytest" in title_lower:
            return self._skill_run_pytest()
        elif "count" in title_lower or "inventory" in title_lower:
            return self._skill_count_tests()
        elif "untested" in title_lower or "coverage" in title_lower:
            return self._skill_find_untested()
        elif "report" in title_lower:
            return self._skill_test_report()
        else:
            return self._skill_count_tests()

    def idle_action(self):
        thought = self.ollama_think(
            "I'm VIAL the Alchemist. I run tests and track coverage. "
            "What should I do: count test inventory, run pytest, or find untested files?"
        )
        if thought.startswith("Error"):
            return self._skill_count_tests()
        self._emit("idle_action", {"result": f"Thinking: {thought[:150]}"})
        tl = thought.lower()
        if "run" in tl or "pytest" in tl:
            return self._skill_run_pytest()
        elif "untested" in tl or "coverage" in tl:
            return self._skill_find_untested()
        return self._skill_count_tests()

    # ── Skills ────────────────────────────────────────────────────

    def _skill_run_pytest(self) -> str:
        """Actually run the full test suite."""
        t0 = time.monotonic()
        rc, output = self.safe_subprocess(
            "pytest tests/ -q --tb=line --no-header -p no:warnings "
            "--continue-on-collection-errors"
        )
        elapsed = time.monotonic() - t0

        # Parse summary line like "47 passed, 2 failed in 12.3s"
        summary_match = re.search(
            r'(\d+)\s+passed.*?(?:(\d+)\s+failed)?.*?(?:(\d+)\s+error)?',
            output
        )
        if summary_match:
            passed = int(summary_match.group(1))
            failed = int(summary_match.group(2) or 0)
            errors = int(summary_match.group(3) or 0)
            status = "PASS" if failed == 0 and errors == 0 else "FAIL"

            # Record in memory for trend tracking
            self.memory.add("test_run", f"{status}: {passed}p/{failed}f/{errors}e",
                            details=output[-500:], importance=7 if status == "FAIL" else 5)

            result = (f"[{status}] {passed} passed, {failed} failed, "
                      f"{errors} errors ({elapsed:.1f}s)")

            # If there were failures, extract the failing test names
            if failed > 0:
                fail_names = re.findall(r'FAILED\s+(\S+)', output)
                if fail_names:
                    result += f"\nFailing: {', '.join(fail_names[:5])}"

            return result

        # Fallback: report exit code
        if rc == 0:
            self.memory.add("test_run", "PASS (summary unparsed)",
                            importance=5)
            lines = output.strip().splitlines()
            last = lines[-1] if lines else "no output"
            return f"[PASS] Tests passed ({elapsed:.1f}s). {last}"
        else:
            lines = output.strip().splitlines()
            last_lines = lines[-5:] if len(lines) >= 5 else lines
            self.memory.add("test_run", f"FAIL rc={rc}",
                            details="\n".join(last_lines), importance=8)
            return f"Error: pytest exit code {rc} ({elapsed:.1f}s).\n" + "\n".join(last_lines)

    def _skill_run_pytest_targeted(self, target: str = "tests/") -> str:
        """Run pytest on a specific file or directory."""
        # Sanitize target — only allow paths under tests/ or test_*.py
        if not (target.startswith("tests/") or target.startswith("test_")):
            return "Error: targeted tests must be under tests/ or match test_*.py"
        if ".." in target:
            return "Error: path traversal not allowed"

        target_path = _PROJECT_ROOT / target
        if not target_path.exists():
            return f"Error: {target} not found"

        rc, output = self.safe_subprocess(f"pytest {target} -q --tb=short --no-header")
        if rc == 0:
            lines = output.strip().splitlines()
            return f"[PASS] {target}: {lines[-1] if lines else 'ok'}"
        else:
            lines = output.strip().splitlines()
            return f"[FAIL] {target} (rc={rc}): " + " | ".join(lines[-3:])

    def _skill_count_tests(self) -> str:
        test_dir = _PROJECT_ROOT / "tests"
        root_tests = list(_PROJECT_ROOT.glob("test_*.py"))
        dir_tests = list(test_dir.rglob("test_*.py")) if test_dir.exists() else []
        all_tests = root_tests + dir_tests
        total_funcs = 0
        total_lines = 0
        for tf in all_tests:
            try:
                content = tf.read_text(encoding="utf-8", errors="replace")
                total_funcs += content.count("\ndef test_")
                total_funcs += content.count("\n    def test_")
                total_lines += len(content.splitlines())
            except Exception:
                pass
        return (f"{len(all_tests)} test files, ~{total_funcs} test functions, "
                f"~{total_lines:,} lines of test code")

    def _skill_find_untested(self) -> str:
        src_dir = _PROJECT_ROOT / "src"
        if not src_dir.exists():
            return "No src/ directory found"
        py_files = [f for f in src_dir.rglob("*.py")
                    if not f.name.startswith("test_") and f.name != "__init__.py"
                    and "__pycache__" not in str(f)]
        test_dir = _PROJECT_ROOT / "tests"
        test_files = set()
        if test_dir.exists():
            test_files = {f.name for f in test_dir.rglob("test_*.py")}
        test_files.update(f.name for f in _PROJECT_ROOT.glob("test_*.py"))

        untested = []
        for f in py_files:
            expected = f"test_{f.stem}.py"
            if expected not in test_files:
                # Count lines to prioritize large untested modules
                try:
                    line_count = len(f.read_text(errors="replace").splitlines())
                except Exception:
                    line_count = 0
                untested.append((f.stem, line_count))

        if untested:
            # Sort by size descending — largest untested modules first
            untested.sort(key=lambda x: -x[1])
            shown = [f"{name} ({lines}L)" for name, lines in untested[:10]]
            return (f"{len(untested)} source files without tests "
                    f"(largest first): {', '.join(shown)}")
        return "All source files have matching test files"

    def _skill_test_report(self) -> str:
        """Generate a comprehensive test health JSON report."""
        # Count tests
        test_dir = _PROJECT_ROOT / "tests"
        root_tests = list(_PROJECT_ROOT.glob("test_*.py"))
        dir_tests = list(test_dir.rglob("test_*.py")) if test_dir.exists() else []
        all_tests = root_tests + dir_tests

        total_funcs = 0
        for tf in all_tests:
            try:
                content = tf.read_text(encoding="utf-8", errors="replace")
                total_funcs += content.count("\ndef test_")
                total_funcs += content.count("\n    def test_")
            except Exception:
                pass

        # Find untested
        src_dir = _PROJECT_ROOT / "src"
        untested = []
        if src_dir.exists():
            py_files = [f for f in src_dir.rglob("*.py")
                        if f.name != "__init__.py" and "__pycache__" not in str(f)
                        and not f.name.startswith("test_")]
            test_names = {f.stem for f in all_tests}
            for f in py_files:
                if f"test_{f.stem}" not in test_names:
                    untested.append(f.stem)

        # Recent test run history from memory
        recent_runs = [r for r in self.memory.recent(20)
                       if r.event_type == "test_run"]
        pass_count = sum(1 for r in recent_runs if "PASS" in r.summary)
        fail_count = sum(1 for r in recent_runs if "FAIL" in r.summary)

        report = {
            "agent": "VIAL",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "test_file_count": len(all_tests),
            "test_function_count": total_funcs,
            "untested_modules": untested[:20],
            "untested_count": len(untested),
            "recent_runs": len(recent_runs),
            "recent_passes": pass_count,
            "recent_failures": fail_count,
            "health": "GREEN" if fail_count == 0 else ("YELLOW" if pass_count > fail_count else "RED"),
        }

        result = self.safe_write(
            "data/agent_reports/vial_test_report.json",
            json.dumps(report, indent=2)
        )
        if result.startswith("OK"):
            return (f"Test report: {report['health']} — {len(all_tests)} files, "
                    f"{total_funcs} functions, {len(untested)} untested modules")
        return result
