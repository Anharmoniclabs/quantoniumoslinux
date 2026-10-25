# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""WARD — The Guardian. Real security scans, audits, and remediation reports."""

from __future__ import annotations
import json
import re
import time
from pathlib import Path
from .agent_base import BaseAgent, _PROJECT_ROOT
from .quest_board import Quest, QuestBoard
from .skill_library import SkillLibrary

# Patterns that indicate potential secrets
_SECRET_PATTERNS = [
    (r'(?i)(api[_-]?key|secret|token|password|passwd)\s*[=:]\s*["\'][^"\']{8,}',
     "Possible hardcoded secret", "HIGH"),
    (r'(?i)-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----', "Private key found", "CRITICAL"),
    (r'sk-[a-zA-Z0-9]{20,}', "OpenAI API key pattern", "CRITICAL"),
    (r'ghp_[a-zA-Z0-9]{36}', "GitHub personal access token", "CRITICAL"),
    (r'(?i)aws[_-]?(access|secret)[_-]?key\s*[=:]\s*["\'][^"\']{16,}',
     "AWS key pattern", "CRITICAL"),
    (r'(?i)bearer\s+[a-zA-Z0-9._~+/=-]{20,}', "Bearer token", "HIGH"),
]

# Dangerous patterns in Python code
_CODE_SMELL_PATTERNS = [
    (r'\beval\s*\(', "eval() call — potential code injection"),
    (r'\bexec\s*\(', "exec() call — potential code injection"),
    (r'subprocess\..*shell\s*=\s*True', "subprocess with shell=True"),
    (r'pickle\.loads?\s*\(', "pickle.load — deserialization risk"),
    (r'__import__\s*\(', "Dynamic import — potential injection"),
    (r'os\.system\s*\(', "os.system — use subprocess instead"),
]


class WardAgent(BaseAgent):

    def __init__(self, quest_board: QuestBoard, skill_library: SkillLibrary):
        super().__init__("WARD", quest_board, skill_library)

    def _register_skills(self):
        self.skill_library.register(
            "secret_scan", "Scan repo for hardcoded secrets/keys",
            "WARD", self._skill_secret_scan, category="security")
        self.skill_library.register(
            "dep_audit", "Check requirements.txt for known issues",
            "WARD", self._skill_dep_audit, category="security")
        self.skill_library.register(
            "safety_check", "Validate agent safety provenance",
            "WARD", self._skill_safety_check, category="security")
        self.skill_library.register(
            "code_smell_scan", "Scan for dangerous code patterns",
            "WARD", self._skill_code_smell_scan, category="security")
        self.skill_library.register(
            "security_report", "Generate full security audit report",
            "WARD", self._skill_security_report, category="security")

    def execute_quest(self, quest: Quest) -> str:
        title_lower = quest.title.lower()
        if "secret" in title_lower or "key" in title_lower:
            return self._skill_secret_scan()
        elif "dep" in title_lower or "audit" in title_lower or "require" in title_lower:
            return self._skill_dep_audit()
        elif "safety" in title_lower or "provenance" in title_lower:
            return self._skill_safety_check()
        elif "smell" in title_lower or "dangerous" in title_lower:
            return self._skill_code_smell_scan()
        elif "report" in title_lower:
            return self._skill_security_report()
        else:
            return self._skill_secret_scan()

    def idle_action(self):
        thought = self.ollama_think(
            "I'm WARD the Guardian. I do security scans. "
            "What should I focus on: scan for secrets, audit dependencies, or check for code smells?"
        )
        if thought.startswith("Error"):
            return self._skill_code_smell_scan()
        self._emit("idle_action", {"result": f"Thinking: {thought[:150]}"})
        tl = thought.lower()
        if "secret" in tl or "key" in tl:
            return self._skill_secret_scan()
        elif "dep" in tl or "audit" in tl:
            return self._skill_dep_audit()
        return self._skill_code_smell_scan()

    # ── Skills ────────────────────────────────────────────────────

    def _scan_files(self):
        """Yield (path, content) for scannable files."""
        skip_dirs = {".git", "__pycache__", "node_modules", ".venv",
                     "build", "build_asm", "build_cpp", "build_cpp_nolto",
                     ".egg-info", "quantoniumos.egg-info"}
        skip_exts = {".pyc", ".pyd", ".so", ".dll", ".exe", ".bin",
                     ".gguf", ".rftmwpk", ".png", ".jpg", ".ico",
                     ".whl", ".tar", ".gz", ".zip", ".a", ".lib",
                     ".o", ".obj"}

        for f in _PROJECT_ROOT.rglob("*"):
            if f.is_dir():
                continue
            if any(d in f.parts for d in skip_dirs):
                continue
            if f.suffix.lower() in skip_exts:
                continue
            if f.stat().st_size > 500_000:
                continue
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                yield f, content
            except Exception:
                pass

    def _skill_secret_scan(self) -> str:
        findings = []
        scanned = 0

        for py_file, content in self._scan_files():
            scanned += 1
            for pattern, desc, severity in _SECRET_PATTERNS:
                matches = re.findall(pattern, content)
                if matches:
                    rel = py_file.relative_to(_PROJECT_ROOT)
                    findings.append({
                        "file": str(rel),
                        "issue": desc,
                        "severity": severity,
                        "count": len(matches),
                    })
            if len(findings) >= 50:
                break

        if findings:
            # Log findings to memory with high importance
            self.memory.add("secret_scan",
                            f"ALERT: {len(findings)} potential secrets found",
                            details=json.dumps(findings[:5]),
                            importance=9)

            lines = [f"⚠️ {len(findings)} potential secret(s) found ({scanned} files scanned):"]
            for f in findings[:10]:
                lines.append(f"  [{f['severity']}] {f['file']}: {f['issue']} ({f['count']}x)")

            return "\n".join(lines)

        self.memory.add("secret_scan", f"Clean: scanned {scanned} files", importance=4)
        return f"✅ No hardcoded secrets detected ({scanned} files scanned)"

    def _skill_dep_audit(self) -> str:
        req_file = _PROJECT_ROOT / "requirements.txt"
        if not req_file.exists():
            return "No requirements.txt found"

        content = req_file.read_text(encoding="utf-8", errors="replace")
        lines = [l.strip() for l in content.splitlines()
                 if l.strip() and not l.strip().startswith("#")]

        unpinned = [l for l in lines if ">=" not in l and "==" not in l and "<" not in l]
        pinned = [l for l in lines if "==" in l]
        loose = [l for l in lines if ">=" in l and "==" not in l]

        results = [f"Total dependencies: {len(lines)}"]
        results.append(f"Pinned (==): {len(pinned)}")
        results.append(f"Range (>=): {len(loose)}")
        results.append(f"Unpinned: {len(unpinned)}")

        if unpinned:
            results.append(f"⚠️ Unpinned deps: {', '.join(unpinned[:5])}")
            self.memory.add("dep_audit",
                            f"{len(unpinned)} unpinned dependencies",
                            importance=6)

        # Try running pip-audit if available
        try:
            rc, output = self.safe_subprocess("pip-audit --format json -r requirements.txt")
            if rc == 0:
                try:
                    audit_data = json.loads(output)
                    vulns = [v for v in audit_data if v.get("vulns")]
                    if vulns:
                        results.append(f"🔴 {len(vulns)} packages with known vulnerabilities")
                        self.memory.add("dep_audit",
                                        f"VULN: {len(vulns)} vulnerable packages",
                                        importance=9)
                    else:
                        results.append("✅ No known vulnerabilities (pip-audit)")
                except json.JSONDecodeError:
                    pass
        except PermissionError:
            pass  # pip-audit not available — that's OK

        return " | ".join(results)

    def _skill_safety_check(self) -> str:
        agent_file = _PROJECT_ROOT / "src" / "apps" / "chatbox_agent.py"
        if not agent_file.exists():
            return "chatbox_agent.py not found"

        content = self.safe_read("src/apps/chatbox_agent.py", max_lines=200)
        checks = {
            "sandbox_boundary": "_PROJECT_ROOT" in content,
            "path_traversal": "_safe_path" in content or "safe_path" in content,
            "rate_limit": "rate_limit" in content.lower() or "MAX_TOOL_CALLS" in content,
            "injection_filter": "INJECTION" in content or "_sanitize_output" in content,
            "audit_log": "AuditLog" in content or "audit" in content.lower(),
            "blocklist": "_PYTHON_BLOCKED" in content or "blocklist" in content.lower(),
        }
        passed = sum(1 for v in checks.values() if v)
        total = len(checks)
        details = ", ".join(f"{k}:{'✅' if v else '❌'}" for k, v in checks.items())

        self.memory.add("safety_check", f"{passed}/{total} checks passed",
                        importance=8 if passed < total else 5)

        return f"Safety provenance: {passed}/{total} checks passed. {details}"

    def _skill_code_smell_scan(self) -> str:
        """Scan Python files for dangerous code patterns."""
        findings = []
        scanned = 0

        for py_file, content in self._scan_files():
            if py_file.suffix != ".py":
                continue
            scanned += 1
            for pattern, desc in _CODE_SMELL_PATTERNS:
                matches = re.findall(pattern, content)
                if matches:
                    rel = py_file.relative_to(_PROJECT_ROOT)
                    findings.append(f"  {rel}: {desc} ({len(matches)}x)")
            if len(findings) >= 30:
                break

        if findings:
            self.memory.add("code_smell", f"{len(findings)} code smells found",
                            importance=7)
            return (f"⚠️ {len(findings)} code smell(s) in {scanned} Python files:\n"
                    + "\n".join(findings[:10]))
        return f"✅ No dangerous code patterns in {scanned} Python files"

    def _skill_security_report(self) -> str:
        """Generate a comprehensive security JSON report."""
        # Run all scans
        secret_result = self._skill_secret_scan()
        dep_result = self._skill_dep_audit()
        safety_result = self._skill_safety_check()
        smell_result = self._skill_code_smell_scan()

        has_secrets = "⚠️" in secret_result
        has_dep_issues = "Unpinned" in dep_result or "🔴" in dep_result
        has_safety_gaps = "❌" in safety_result
        has_smells = "⚠️" in smell_result

        issues = sum([has_secrets, has_dep_issues, has_safety_gaps, has_smells])
        health = "GREEN" if issues == 0 else ("YELLOW" if issues <= 1 else "RED")

        report = {
            "agent": "WARD",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "health": health,
            "issues_found": issues,
            "secret_scan": secret_result[:500],
            "dep_audit": dep_result[:500],
            "safety_check": safety_result[:500],
            "code_smells": smell_result[:500],
        }

        result = self.safe_write(
            "data/agent_reports/ward_security_report.json",
            json.dumps(report, indent=2)
        )
        if result.startswith("OK"):
            return f"Security report: {health} — {issues} issue area(s)"
        return result
