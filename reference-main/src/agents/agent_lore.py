# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""LORE — The Chronicler. Validates docs, fixes broken links, and generates doc health reports."""

from __future__ import annotations
import json
import re
import time
from pathlib import Path
from .agent_base import BaseAgent, _PROJECT_ROOT
from .quest_board import Quest, QuestBoard
from .skill_library import SkillLibrary


class LoreAgent(BaseAgent):

    def __init__(self, quest_board: QuestBoard, skill_library: SkillLibrary):
        super().__init__("LORE", quest_board, skill_library)

    def _register_skills(self):
        self.skill_library.register(
            "link_check", "Validate internal markdown links",
            "LORE", self._skill_link_check, category="docs")
        self.skill_library.register(
            "doc_inventory", "Count and list documentation files",
            "LORE", self._skill_doc_inventory, category="docs")
        self.skill_library.register(
            "stale_doc_scan", "Find references to missing files in docs",
            "LORE", self._skill_stale_scan, category="docs")
        self.skill_library.register(
            "doc_quality_scan", "Check docs for quality issues (TODOs, empty sections)",
            "LORE", self._skill_quality_scan, category="docs")
        self.skill_library.register(
            "doc_report", "Generate comprehensive documentation health report",
            "LORE", self._skill_doc_report, category="docs")

    def execute_quest(self, quest: Quest) -> str:
        title_lower = quest.title.lower()
        if "link" in title_lower:
            return self._skill_link_check()
        elif "inventory" in title_lower or "count" in title_lower:
            return self._skill_doc_inventory()
        elif "stale" in title_lower:
            return self._skill_stale_scan()
        elif "quality" in title_lower or "todo" in title_lower:
            return self._skill_quality_scan()
        elif "report" in title_lower:
            return self._skill_doc_report()
        else:
            return self._skill_doc_inventory()

    def idle_action(self):
        thought = self.ollama_think(
            "I'm LORE the Chronicler. I validate documentation. "
            "Should I: check links, scan for TODOs, or do a doc inventory?"
        )
        if thought.startswith("Error"):
            return self._skill_quality_scan()
        self._emit("idle_action", {"result": f"Thinking: {thought[:150]}"})
        tl = thought.lower()
        if "link" in tl:
            return self._skill_link_check()
        elif "inventory" in tl or "count" in tl:
            return self._skill_doc_inventory()
        return self._skill_quality_scan()

    # ── Helpers ───────────────────────────────────────────────────

    def _get_all_md(self):
        md_files = list(_PROJECT_ROOT.glob("*.md"))
        docs_dir = _PROJECT_ROOT / "docs"
        if docs_dir.exists():
            md_files.extend(docs_dir.rglob("*.md"))
        return md_files

    # ── Skills ────────────────────────────────────────────────────

    def _skill_link_check(self) -> str:
        md_files = self._get_all_md()
        total_links = 0
        broken = []
        valid = 0

        for md in md_files:
            try:
                content = md.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            links = re.findall(r'\[([^\]]*)\]\(([^)]+)\)', content)
            for text, href in links:
                if href.startswith("http") or href.startswith("#") or href.startswith("mailto:"):
                    continue
                total_links += 1
                clean = href.split("#")[0]
                if not clean:
                    continue
                target = (md.parent / clean).resolve()
                if target.exists():
                    valid += 1
                else:
                    rel_md = md.relative_to(_PROJECT_ROOT)
                    broken.append({
                        "source": str(rel_md),
                        "link_text": text,
                        "target": clean,
                    })

        if broken:
            self.memory.add("link_check",
                            f"{len(broken)} broken links found",
                            details=json.dumps(broken[:5]),
                            importance=7)

            lines = [f"⚠️ {len(broken)} broken links (of {total_links} checked):"]
            for b in broken[:10]:
                lines.append(f"  {b['source']}: [{b['link_text']}]({b['target']})")
            return "\n".join(lines)

        self.memory.add("link_check", f"All {total_links} links valid", importance=4)
        return f"✅ All {total_links} internal doc links valid"

    def _skill_doc_inventory(self) -> str:
        md_root = list(_PROJECT_ROOT.glob("*.md"))
        docs_dir = _PROJECT_ROOT / "docs"
        md_docs = list(docs_dir.rglob("*.md")) if docs_dir.exists() else []
        all_md = md_root + md_docs

        total_lines = 0
        largest = ("", 0)
        smallest = ("", float("inf"))
        for f in all_md:
            try:
                lines = len(f.read_text(encoding="utf-8", errors="replace").splitlines())
                total_lines += lines
                if lines > largest[1]:
                    largest = (f.name, lines)
                if lines < smallest[1]:
                    smallest = (f.name, lines)
            except Exception:
                pass

        return (f"{len(all_md)} markdown files ({len(md_root)} root, {len(md_docs)} in docs/), "
                f"~{total_lines:,} total lines. "
                f"Largest: {largest[0]} ({largest[1]}L), "
                f"Smallest: {smallest[0]} ({smallest[1]}L)")

    def _skill_stale_scan(self) -> str:
        md_files = list(_PROJECT_ROOT.glob("*.md"))
        stale = []

        for md in md_files:
            try:
                content = md.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            # Find backtick-quoted filenames
            refs = re.findall(r'`([a-zA-Z_/][a-zA-Z0-9_/.]*\.(?:py|ts|tsx|js|json|md|yml|yaml))`', content)
            for ref in refs:
                target = _PROJECT_ROOT / ref
                if not target.exists():
                    stale.append({"doc": md.name, "ref": ref})

        if stale:
            self.memory.add("stale_scan",
                            f"{len(stale)} stale file references",
                            details=json.dumps(stale[:10]),
                            importance=6)
            lines = [f"⚠️ {len(stale)} stale references:"]
            for s in stale[:10]:
                lines.append(f"  {s['doc']}: references `{s['ref']}` (not found)")
            return "\n".join(lines)
        return "✅ No stale file references found in root docs"

    def _skill_quality_scan(self) -> str:
        """Scan docs for quality issues: TODOs, empty sections, very short docs."""
        md_files = self._get_all_md()
        issues = []

        for md in md_files:
            try:
                content = md.read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()
            except Exception:
                continue

            rel = md.relative_to(_PROJECT_ROOT)

            # TODOs
            todo_count = sum(1 for l in lines if "TODO" in l or "FIXME" in l or "HACK" in l)
            if todo_count > 0:
                issues.append(f"  {rel}: {todo_count} TODO/FIXME markers")

            # Empty sections (heading followed immediately by another heading)
            for i, line in enumerate(lines[:-1]):
                if line.startswith("#") and i + 1 < len(lines):
                    next_nonempty = ""
                    for j in range(i + 1, min(i + 3, len(lines))):
                        if lines[j].strip():
                            next_nonempty = lines[j]
                            break
                    if next_nonempty.startswith("#"):
                        issues.append(f"  {rel}: empty section '{line.strip()[:40]}'")
                        break  # one per file

            # Very short docs (< 5 non-empty lines)
            nonempty = [l for l in lines if l.strip()]
            if len(nonempty) < 5 and len(nonempty) > 0:
                issues.append(f"  {rel}: very short ({len(nonempty)} lines)")

        if issues:
            return f"⚠️ {len(issues)} doc quality issues:\n" + "\n".join(issues[:15])
        return f"✅ {len(md_files)} docs scanned, no quality issues found"

    def _skill_doc_report(self) -> str:
        """Generate comprehensive doc health JSON report."""
        md_files = self._get_all_md()

        # Link check
        total_links = 0
        broken_count = 0
        for md in md_files:
            try:
                content = md.read_text(encoding="utf-8", errors="replace")
                links = re.findall(r'\[([^\]]*)\]\(([^)]+)\)', content)
                for _, href in links:
                    if href.startswith("http") or href.startswith("#"):
                        continue
                    total_links += 1
                    clean = href.split("#")[0]
                    if clean and not (md.parent / clean).resolve().exists():
                        broken_count += 1
            except Exception:
                pass

        # Stale refs
        stale_count = 0
        for md in _PROJECT_ROOT.glob("*.md"):
            try:
                content = md.read_text(encoding="utf-8", errors="replace")
                refs = re.findall(r'`([a-zA-Z_/][a-zA-Z0-9_/.]*\.py)`', content)
                for ref in refs:
                    if not (_PROJECT_ROOT / ref).exists():
                        stale_count += 1
            except Exception:
                pass

        # Total lines
        total_lines = sum(
            len(f.read_text(errors="replace").splitlines())
            for f in md_files if f.exists()
        )

        health = "GREEN"
        if broken_count > 0 or stale_count > 3:
            health = "YELLOW"
        if broken_count > 5 or stale_count > 10:
            health = "RED"

        report = {
            "agent": "LORE",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "health": health,
            "doc_count": len(md_files),
            "total_lines": total_lines,
            "total_internal_links": total_links,
            "broken_links": broken_count,
            "stale_references": stale_count,
        }

        result = self.safe_write(
            "data/agent_reports/lore_doc_report.json",
            json.dumps(report, indent=2)
        )
        if result.startswith("OK"):
            return (f"Doc report: {health} — {len(md_files)} docs, {total_lines:,} lines, "
                    f"{broken_count} broken links, {stale_count} stale refs")
        return result
