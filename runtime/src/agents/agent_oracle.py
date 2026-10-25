# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""ORACLE — The Coordinator. Smart dispatch, cross-agent correlation, guild reports."""

from __future__ import annotations
import datetime
import json
import time
from pathlib import Path
from .agent_base import BaseAgent, _PROJECT_ROOT
from .quest_board import Quest, QuestBoard, QuestPriority
from .skill_library import SkillLibrary
from .router_config import route_task, get_agent_model, get_token_budget


# Standard quest templates for each agent
PATROL_QUESTS = [
    ("FORGE", "Check native build", "Verify rftmw_native build artifacts exist", 25),
    ("FORGE", "Scan ASM kernels", "Count and validate ASM kernel source files", 20),
    ("FORGE", "Build report", "Generate build health JSON report", 30),
    ("VIAL",  "Count test inventory", "Count test files and test functions", 20),
    ("VIAL",  "Run pytest suite", "Execute full test suite and log results", 35),
    ("VIAL",  "Test health report", "Generate test health JSON report", 30),
    ("WARD",  "Secret scan", "Scan repo for hardcoded secrets and keys", 30),
    ("WARD",  "Dependency audit", "Check requirements.txt for issues", 25),
    ("WARD",  "Code smell scan", "Scan Python files for dangerous patterns", 30),
    ("WARD",  "Security report", "Generate composite security JSON report", 35),
    ("LORE",  "Doc link check", "Validate internal markdown links", 25),
    ("LORE",  "Doc inventory", "Count and catalog documentation files", 20),
    ("LORE",  "Doc quality scan", "Check docs for TODOs, empty sections", 25),
    ("LORE",  "Doc health report", "Generate documentation health JSON report", 30),
    ("SPARK", "Asset integrity check", "Verify UI assets exist and are valid", 25),
    ("SPARK", "App syntax check", "Verify chatbox and RPG modules compile", 25),
    ("SPARK", "RPG integrity check", "Validate RPG sprite/world/renderer consistency", 25),
    ("SPARK", "UI health report", "Generate UI health JSON report", 30),
    ("TIDE",  "Mobile structure check", "Verify mobile project structure", 25),
    ("TIDE",  "Mobile deps check", "Validate mobile package.json dependencies", 25),
    ("TIDE",  "Frontend TODO scan", "Find TODO/FIXME markers in frontend code", 20),
    ("TIDE",  "Frontend health report", "Generate frontend health JSON report", 30),
    ("SAGE",  "Architecture analysis", "Map project structure and module inventory", 25),
    ("SAGE",  "Module deep-read", "Analyze a Python module's design and patterns", 30),
    ("SAGE",  "Cross-reference scan", "Find cross-module dependencies and coupling", 25),
    ("SAGE",  "Run experiment", "Run a safe experiment to test a codebase hypothesis", 35),
    ("SAGE",  "Research report", "Generate comprehensive research findings report", 30),
    # ── RFT / RFTMW-specific quests ─────────────────────────────────────────────
    ("FORGE", "Build rftmw_native", "Run cmake build for the rftmw_native C++/ASM module", 40),
    ("FORGE", "Benchmark Q6K matmul", "Benchmark the Q6K matrix-multiply kernel and log results", 35),
    ("FORGE", "Verify pack integrity", "Verify checksums for all .rftmwpk pack files", 30),
    ("VIAL",  "Test RFT inference", "Run test_e2e_inference.py and log results", 35),
    ("VIAL",  "Test pack loading", "Run test_pack_index.py and test_native_load.py", 30),
    ("VIAL",  "Compare numpy vs CPP", "Run test_cpp_vs_numpy.py and report diffs", 35),
    ("SAGE",  "Profile RFT layers", "Profile rftmw_numpy_engine.py layers for bottlenecks", 40),
    ("SAGE",  "RFT optimization plan", "Propose 3 concrete optimizations for the RFTMW pipeline", 45),
    ("SAGE",  "Compression ratio audit", "Measure pack-file compression ratios vs baseline GGUF", 35),
]


class OracleAgent(BaseAgent):

    def __init__(self, quest_board: QuestBoard, skill_library: SkillLibrary):
        super().__init__("ORACLE", quest_board, skill_library)
        self._last_patrol = ""

    def _register_skills(self):
        self.skill_library.register(
            "dispatch_patrol", "Post standard patrol quests for all agents",
            "ORACLE", self._skill_dispatch_patrol, category="coordination")
        self.skill_library.register(
            "smart_dispatch", "Read agent reports and post targeted follow-up quests",
            "ORACLE", self._skill_smart_dispatch, category="coordination")
        self.skill_library.register(
            "status_dashboard", "Generate guild status dashboard",
            "ORACLE", self._skill_status_dashboard, category="coordination")
        self.skill_library.register(
            "guild_report", "Aggregate all agent reports into guild health report",
            "ORACLE", self._skill_guild_report, category="coordination")
        self.skill_library.register(
            "route_quest", "Route a task to the best agent using AGENT_ROUTER_CONFIG.yaml",
            "ORACLE", self._skill_route_quest, category="coordination")

    def execute_quest(self, quest: Quest) -> str:
        title_lower = quest.title.lower()
        if "patrol" in title_lower or "dispatch" in title_lower:
            return self._skill_dispatch_patrol()
        elif "smart" in title_lower or "follow" in title_lower:
            return self._skill_smart_dispatch()
        elif "status" in title_lower or "dashboard" in title_lower:
            return self._skill_status_dashboard()
        elif "report" in title_lower or "guild" in title_lower:
            return self._skill_guild_report()
        else:
            return self._skill_status_dashboard()

    def idle_action(self):
        active = self.quest_board.active_quests()
        pending_count = len([q for q in active if q.status == "pending"])
        in_progress = len([q for q in active if q.status == "in_progress"])

        thought = self.ollama_think(
            f"I'm ORACLE, guild coordinator. Currently {pending_count} pending quests, "
            f"{in_progress} in-progress. Should I: dispatch new patrols, "
            f"read agent reports for smart follow-ups, or generate a guild status dashboard?"
        )

        if thought.startswith("Error"):
            # Fallback
            if pending_count < 3:
                return self._skill_dispatch_patrol()
            return self._skill_smart_dispatch()

        self._emit("idle_action", {"result": f"Thinking: {thought[:150]}"})

        thought_lower = thought.lower()
        if "patrol" in thought_lower or "dispatch" in thought_lower:
            return self._skill_dispatch_patrol()
        elif "report" in thought_lower or "smart" in thought_lower:
            return self._skill_smart_dispatch()
        elif "status" in thought_lower or "dashboard" in thought_lower:
            return self._skill_status_dashboard()
        elif pending_count < 3:
            return self._skill_dispatch_patrol()
        return self._skill_smart_dispatch()

    # ── Skills ────────────────────────────────────────────────────

    def _skill_dispatch_patrol(self) -> str:
        active = self.quest_board.active_quests()
        pending_count = len([q for q in active if q.status in ("pending", "in_progress")])
        # Allow re-dispatch whenever the board falls below 15 quests so agents
        # stay perpetually busy; post_quest deduplicates titles internally.
        if pending_count >= 15:
            return f"Board healthy — {pending_count} quests still pending"

        posted = 0
        for agent, title, desc, xp in PATROL_QUESTS:
            active = self.quest_board.pending_for(agent)
            already = any(q.title == title for q in active)
            if not already:
                self.quest_board.post_quest(
                    title=title, description=desc, assigned_to=agent,
                    priority=QuestPriority.NORMAL, xp_reward=xp,
                    triggered_by="ORACLE",
                )
                posted += 1

        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
        self._last_patrol = now
        self.memory.add("dispatch", f"Posted {posted} patrol quests at {now}", importance=6)
        return f"Dispatched {posted} patrol quests across the guild"

    def _skill_smart_dispatch(self) -> str:
        """Read agent reports and post HIGH-priority follow-up quests based on findings."""
        reports_dir = _PROJECT_ROOT / "data" / "agent_reports"
        if not reports_dir.exists():
            return "No agent reports directory — run patrol first"

        posted = 0
        findings = []

        # Read each agent's report and react
        report_map = {
            "ward_security_report.json": self._react_ward,
            "vial_test_report.json": self._react_vial,
            "lore_doc_report.json": self._react_lore,
            "spark_ui_report.json": self._react_spark,
            "tide_frontend_report.json": self._react_tide,
            "forge_build_report.json": self._react_forge,
            "sage_research_report.json": self._react_sage,
        }

        for filename, reactor in report_map.items():
            report_path = reports_dir / filename
            if not report_path.exists():
                continue
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                result = reactor(report)
                if result:
                    posted += result[0]
                    findings.extend(result[1])
            except (json.JSONDecodeError, KeyError):
                pass

        if posted:
            self.memory.add("smart_dispatch",
                            f"Posted {posted} targeted quests: {'; '.join(findings[:3])}",
                            importance=7)
        return f"Smart dispatch: read {len(report_map)} reports, posted {posted} targeted quests"

    def _react_ward(self, report: dict) -> tuple[int, list[str]] | None:
        posted = 0
        findings = []
        if report.get("health") in ("YELLOW", "RED"):
            # Safely extract integer from reports that may have text prefixes
            def _safe_int(val):
                if isinstance(val, int):
                    return val
                if isinstance(val, str):
                    import re
                    m = re.search(r'\d+', val)
                    return int(m.group()) if m else 0
                return 0
            secrets = _safe_int(report.get("secrets_found", 0))
            smells = _safe_int(report.get("code_smells", 0))
            if secrets > 0:
                self.quest_board.post_quest(
                    title="Secret scan — URGENT",
                    description=f"WARD found {secrets} hardcoded secrets. Re-scan and verify.",
                    assigned_to="WARD", priority=QuestPriority.HIGH, xp_reward=50,
                    triggered_by="ORACLE",
                )
                posted += 1
                findings.append(f"WARD: {secrets} secrets")
            if smells > 5:
                self.quest_board.post_quest(
                    title="Code smell deep scan",
                    description=f"WARD found {smells} code smells. Focus on eval/exec patterns.",
                    assigned_to="WARD", priority=QuestPriority.HIGH, xp_reward=40,
                    triggered_by="ORACLE",
                )
                posted += 1
                findings.append(f"WARD: {smells} smells")
        return (posted, findings) if posted else None

    def _react_vial(self, report: dict) -> tuple[int, list[str]] | None:
        posted = 0
        findings = []
        if report.get("health") in ("YELLOW", "RED"):
            failures = int(report.get("recent_failures", 0))
            if failures > 0:
                self.quest_board.post_quest(
                    title="Run pytest suite — failures detected",
                    description=f"VIAL reports {failures} recent failures. Re-run and diagnose.",
                    assigned_to="VIAL", priority=QuestPriority.HIGH, xp_reward=45,
                    triggered_by="ORACLE",
                )
                posted += 1
                findings.append(f"VIAL: {failures} failures")
        return (posted, findings) if posted else None

    def _react_lore(self, report: dict) -> tuple[int, list[str]] | None:
        posted = 0
        findings = []
        broken = int(report.get("broken_links", 0))
        stale = int(report.get("stale_references", 0))
        if broken > 0:
            self.quest_board.post_quest(
                title="Doc link check — broken links",
                description=f"LORE found {broken} broken links. Verify and catalog.",
                assigned_to="LORE", priority=QuestPriority.NORMAL, xp_reward=30,
                triggered_by="ORACLE",
            )
            posted += 1
            findings.append(f"LORE: {broken} broken links")
        if stale > 3:
            self.quest_board.post_quest(
                title="Stale reference scan",
                description=f"LORE found {stale} stale file refs in docs.",
                assigned_to="LORE", priority=QuestPriority.NORMAL, xp_reward=25,
                triggered_by="ORACLE",
            )
            posted += 1
            findings.append(f"LORE: {stale} stale refs")
        return (posted, findings) if posted else None

    def _react_spark(self, report: dict) -> tuple[int, list[str]] | None:
        posted = 0
        findings = []
        if int(report.get("syntax_errors", 0)) > 0:
            n = int(report["syntax_errors"])
            self.quest_board.post_quest(
                title="App syntax check — errors found",
                description=f"SPARK found {n} syntax errors in UI modules.",
                assigned_to="SPARK", priority=QuestPriority.HIGH, xp_reward=40,
                triggered_by="ORACLE",
            )
            posted += 1
            findings.append(f"SPARK: {n} syntax errors")
        return (posted, findings) if posted else None

    def _react_tide(self, report: dict) -> tuple[int, list[str]] | None:
        posted = 0
        findings = []
        any_uses = int(report.get("any_type_uses", 0))
        if any_uses > 20:
            self.quest_board.post_quest(
                title="TypeScript scan — excessive :any",
                description=f"TIDE found {any_uses} uses of ': any'. Deep scan needed.",
                assigned_to="TIDE", priority=QuestPriority.NORMAL, xp_reward=25,
                triggered_by="ORACLE",
            )
            posted += 1
            findings.append(f"TIDE: {any_uses} :any")
        return (posted, findings) if posted else None

    def _react_forge(self, report: dict) -> tuple[int, list[str]] | None:
        posted = 0
        findings = []
        if report.get("health") in ("YELLOW", "RED"):
            self.quest_board.post_quest(
                title="Check native build — issues detected",
                description="FORGE reports build health issues. Investigate.",
                assigned_to="FORGE", priority=QuestPriority.HIGH, xp_reward=40,
                triggered_by="ORACLE",
            )
            posted += 1
            findings.append("FORGE: build issues")
        return (posted, findings) if posted else None

    def _react_sage(self, report: dict) -> tuple[int, list[str]] | None:
        posted = 0
        findings = []
        research_findings = int(report.get("research_findings", 0))
        if research_findings < 3:
            self.quest_board.post_quest(
                title="Architecture analysis — more data needed",
                description="SAGE has few findings. Run architecture and cross-reference scans.",
                assigned_to="SAGE", priority=QuestPriority.NORMAL, xp_reward=30,
                triggered_by="ORACLE",
            )
            posted += 1
            findings.append(f"SAGE: only {research_findings} findings")
        return (posted, findings) if posted else None

    def _skill_status_dashboard(self) -> str:
        active = self.quest_board.active_quests()
        recent = self.quest_board.recent_completed(10)

        pending = [q for q in active if q.status == "pending"]
        in_progress = [q for q in active if q.status == "in_progress"]
        completed = [q for q in recent if q.status == "completed"]
        failed = [q for q in recent if q.status == "failed"]

        return (f"Guild Status: {len(pending)} pending, {len(in_progress)} in-progress, "
                f"{len(completed)} completed, {len(failed)} failed (recent)")

    def _skill_guild_report(self) -> str:
        """Aggregate all agent reports into a single guild health report."""
        reports_dir = _PROJECT_ROOT / "data" / "agent_reports"
        agent_reports = {}

        report_files = [
            "forge_build_report.json",
            "vial_test_report.json",
            "ward_security_report.json",
            "lore_doc_report.json",
            "spark_ui_report.json",
            "tide_frontend_report.json",
            "sage_research_report.json",
        ]

        for fname in report_files:
            path = reports_dir / fname if reports_dir.exists() else None
            if path and path.exists():
                try:
                    agent_reports[fname.split("_")[0].upper()] = json.loads(
                        path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, KeyError):
                    agent_reports[fname.split("_")[0].upper()] = {"health": "UNKNOWN"}

        # Compute overall health
        healths = [r.get("health", "UNKNOWN") for r in agent_reports.values()]
        if "RED" in healths:
            overall = "RED"
        elif "YELLOW" in healths:
            overall = "YELLOW"
        elif healths and all(h == "GREEN" for h in healths):
            overall = "GREEN"
        else:
            overall = "UNKNOWN"

        # Quest stats
        recent = self.quest_board.recent_completed(50)
        completed = len([q for q in recent if q.status == "completed"])
        failed = len([q for q in recent if q.status == "failed"])

        report = {
            "agent": "ORACLE",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "overall_health": overall,
            "agent_health": {k: v.get("health", "UNKNOWN") for k, v in agent_reports.items()},
            "reports_available": len(agent_reports),
            "quests_completed_recent": completed,
            "quests_failed_recent": failed,
            "agent_summaries": agent_reports,
        }

        result = self.safe_write(
            "data/agent_reports/oracle_guild_report.json",
            json.dumps(report, indent=2)
        )
        if result.startswith("OK"):
            self.memory.add("guild_report",
                            f"Overall: {overall}, {len(agent_reports)} agents reporting",
                            importance=7)
            return (f"Guild report: {overall} — {len(agent_reports)} agents reporting, "
                    f"{completed} quests done, {failed} failed")
        return result

    def _skill_route_quest(self, task: str = "", file_path: str = "") -> str:
        """Route a task to the best agent using AGENT_ROUTER_CONFIG.yaml rules."""
        if not task:
            return "Error: No task provided to route"
        
        agent, priority_boost, reason = route_task(task, file_path if file_path else None)
        model = get_agent_model(agent)
        budget = get_token_budget(agent)
        
        # Post the quest to the routed agent
        self.quest_board.post_quest(
            title=task[:60],
            description=task,
            assigned_to=agent,
            priority=QuestPriority.HIGH if priority_boost >= 10 else QuestPriority.NORMAL,
            xp_reward=25 + priority_boost,
            triggered_by="ORACLE",
        )
        
        self.memory.add("route_quest",
                        f"Routed to {agent}: {task[:50]}",
                        details=reason,
                        importance=6)
        
        return (f"Routed to {agent} (model: {model}, budget: {budget} tokens)\n"
                f"Reason: {reason}")

    def route_and_dispatch(self, task: str, file_path: str = "") -> tuple[str, str]:
        """Route a task and return (agent_name, reason). Used by external code."""
        agent, priority_boost, reason = route_task(task, file_path if file_path else None)
        return (agent, reason)
