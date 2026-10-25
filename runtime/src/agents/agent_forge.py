# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""FORGE — The Kernel Smith. Actually builds native C++/ASM code and benchmarks it."""

from __future__ import annotations
import json
import os
import time
from pathlib import Path
from .agent_base import BaseAgent, _PROJECT_ROOT
from .quest_board import Quest, QuestBoard
from .skill_library import SkillLibrary


class ForgeAgent(BaseAgent):

    def __init__(self, quest_board: QuestBoard, skill_library: SkillLibrary):
        super().__init__("FORGE", quest_board, skill_library)

    def _register_skills(self):
        self.skill_library.register(
            "check_native_build", "Verify rftmw_native C++ module compiles",
            "FORGE", self._skill_check_build, category="build")
        self.skill_library.register(
            "run_cmake_build", "Actually run cmake build for native module",
            "FORGE", self._skill_run_build, category="build")
        self.skill_library.register(
            "check_build_artifacts", "Check native build output files exist",
            "FORGE", self._skill_check_artifacts, category="build")
        self.skill_library.register(
            "scan_asm_kernels", "Count and list ASM kernel source files",
            "FORGE", self._skill_scan_asm, category="build")
        self.skill_library.register(
            "build_report", "Generate build health report with metrics",
            "FORGE", self._skill_build_report, category="build")

    def execute_quest(self, quest: Quest) -> str:
        title_lower = quest.title.lower()
        if "cmake" in title_lower or "compile" in title_lower:
            return self._skill_run_build()
        elif "build" in title_lower and "report" in title_lower:
            return self._skill_build_report()
        elif "build" in title_lower:
            return self._skill_check_build()
        elif "artifact" in title_lower:
            return self._skill_check_artifacts()
        elif "asm" in title_lower or "kernel" in title_lower:
            return self._skill_scan_asm()
        elif "benchmark" in title_lower:
            return self._skill_build_report()
        else:
            return self._skill_check_build()

    def idle_action(self):
        # Ask Ollama what to focus on
        artifacts = self._count_artifacts()
        thought = self.ollama_think(
            f"I'm FORGE the Kernel Smith. I have {artifacts} build artifacts. "
            f"My build dir is src/rftmw_native/. What should I check or work on next? "
            f"Pick one: check build, scan ASM, or generate report."
        )
        if thought.startswith("Error"):
            # Fallback to deterministic behavior if Ollama is down
            if artifacts == 0:
                return self._skill_run_build()
            return self._skill_check_artifacts()

        self._emit("idle_action", {"result": f"Thinking: {thought[:150]}"})

        thought_lower = thought.lower()
        if "report" in thought_lower:
            return self._skill_build_report()
        elif "asm" in thought_lower or "kernel" in thought_lower:
            return self._skill_scan_asm()
        elif "build" in thought_lower or "cmake" in thought_lower:
            return self._skill_check_build()
        elif artifacts == 0:
            return self._skill_run_build()
        return self._skill_check_artifacts()

    # ── Skills ────────────────────────────────────────────────────

    def _count_artifacts(self) -> int:
        native_dir = _PROJECT_ROOT / "src" / "rftmw_native"
        if not native_dir.exists():
            return 0
        return len(list(native_dir.rglob("*.pyd")) + list(native_dir.rglob("*.so")))

    def _skill_check_build(self) -> str:
        build_dir = _PROJECT_ROOT / "src" / "rftmw_native" / "build_asm"
        if not build_dir.exists():
            build_dir = _PROJECT_ROOT / "src" / "rftmw_native" / "build"
        if not build_dir.exists():
            return "No build directory found — native module needs initial setup"

        cmake_cache = build_dir / "CMakeCache.txt"
        if cmake_cache.exists():
            content = self.safe_read(str(cmake_cache.relative_to(_PROJECT_ROOT)), max_lines=50)
            avx2 = "AVX2" in content
            asm = "ASM" in content
            artifacts = self._count_artifacts()
            return (f"Build dir: {build_dir.name}. "
                    f"AVX2={'ON' if avx2 else 'OFF'}, ASM={'ON' if asm else 'OFF'}. "
                    f"Artifacts: {artifacts}. CMakeCache present.")
        return f"Build dir {build_dir.name} exists but no CMakeCache.txt"

    def _skill_run_build(self) -> str:
        """Actually run cmake --build to compile the native module."""
        native_dir = _PROJECT_ROOT / "src" / "rftmw_native"
        if not native_dir.exists():
            return "Error: src/rftmw_native/ not found"

        # Find or create build directory
        build_dir = native_dir / "build_asm"
        if not build_dir.exists():
            build_dir = native_dir / "build"

        cmake_lists = native_dir / "CMakeLists.txt"
        if not cmake_lists.exists():
            return "Error: no CMakeLists.txt found in src/rftmw_native/"

        # If no build dir, configure first
        if not build_dir.exists():
            rc, output = self.safe_subprocess(
                f"cmake -S src/rftmw_native -B src/rftmw_native/build_asm"
            )
            if rc != 0:
                return f"Error: cmake configure failed (rc={rc}): {output[-300:]}"
            build_dir = native_dir / "build_asm"

        # Build
        t0 = time.monotonic()
        rc, output = self.safe_subprocess(
            f"cmake --build {build_dir.relative_to(_PROJECT_ROOT)}"
        )
        elapsed = time.monotonic() - t0

        if rc == 0:
            artifacts = self._count_artifacts()
            result = (f"Build succeeded in {elapsed:.1f}s. "
                      f"{artifacts} artifact(s) produced.")
            self.memory.add("build_success", result, importance=7)
            return result
        else:
            # Extract error lines
            lines = output.strip().splitlines()
            error_lines = [l for l in lines if "error" in l.lower()][:5]
            summary = "\n".join(error_lines) if error_lines else lines[-3:] if lines else "no output"
            return f"Error: build failed (rc={rc}, {elapsed:.1f}s):\n{summary}"

    def _skill_check_artifacts(self) -> str:
        native_dir = _PROJECT_ROOT / "src" / "rftmw_native"
        pyd_files = list(native_dir.rglob("*.pyd"))
        so_files = list(native_dir.rglob("*.so"))
        lib_files = list(native_dir.rglob("*.a")) + list(native_dir.rglob("*.lib"))
        artifacts = pyd_files + so_files + lib_files
        if artifacts:
            details = []
            for f in artifacts[:8]:
                size_kb = f.stat().st_size / 1024
                details.append(f"{f.name} ({size_kb:.0f}KB)")
            return f"Native artifacts: {', '.join(details)} ({len(artifacts)} total)"
        return "No native artifacts found — build needed"

    def _skill_scan_asm(self) -> str:
        asm_dir = _PROJECT_ROOT / "src" / "rftmw_native"
        asm_files = list(asm_dir.rglob("*.asm")) + list(asm_dir.rglob("*.s"))
        c_files = list(asm_dir.rglob("*.c")) + list(asm_dir.rglob("*.cpp"))
        h_files = list(asm_dir.rglob("*.h")) + list(asm_dir.rglob("*.hpp"))

        # Count total lines of assembly
        total_asm_lines = 0
        for f in asm_files:
            try:
                total_asm_lines += len(f.read_text(errors="replace").splitlines())
            except Exception:
                pass

        return (f"ASM kernels: {len(asm_files)} files ({total_asm_lines:,} lines), "
                f"C/C++: {len(c_files)} sources, {len(h_files)} headers")

    def _skill_build_report(self) -> str:
        """Generate a JSON build health report."""
        native_dir = _PROJECT_ROOT / "src" / "rftmw_native"

        asm_files = list(native_dir.rglob("*.asm")) + list(native_dir.rglob("*.s"))
        c_files = list(native_dir.rglob("*.c")) + list(native_dir.rglob("*.cpp"))
        artifacts = (list(native_dir.rglob("*.pyd")) + list(native_dir.rglob("*.so"))
                     + list(native_dir.rglob("*.a")))

        build_dir = native_dir / "build_asm"
        has_cache = (build_dir / "CMakeCache.txt").exists() if build_dir.exists() else False

        report = {
            "agent": "FORGE",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "asm_kernel_count": len(asm_files),
            "c_cpp_source_count": len(c_files),
            "artifact_count": len(artifacts),
            "artifacts": [f.name for f in artifacts[:10]],
            "cmake_configured": has_cache,
            "build_dir": str(build_dir.name) if build_dir.exists() else None,
            "health": "GREEN" if artifacts else ("YELLOW" if has_cache else "RED"),
        }

        result = self.safe_write(
            "data/agent_reports/forge_build_report.json",
            json.dumps(report, indent=2)
        )
        if result.startswith("OK"):
            return f"Build report: {report['health']} — {len(artifacts)} artifacts, {len(asm_files)} ASM kernels"
        return result
