# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""SAGE — The Research Scientist. Studies the codebase, runs experiments, reports findings."""

from __future__ import annotations
import json
import os
import re
import time
from pathlib import Path
from .agent_base import BaseAgent, _PROJECT_ROOT
from .quest_board import Quest, QuestBoard
from .skill_library import SkillLibrary
from .router_config import get_experiment_templates, get_experiment


class SageAgent(BaseAgent):

    def __init__(self, quest_board: QuestBoard, skill_library: SkillLibrary):
        super().__init__("SAGE", quest_board, skill_library)

    def _register_skills(self):
        self.skill_library.register(
            "analyze_architecture", "Map project directory tree and module structure",
            "SAGE", self._skill_analyze_architecture, category="research")
        self.skill_library.register(
            "analyze_module", "Deep-read a Python module and summarize its design",
            "SAGE", self._skill_analyze_module, category="research")
        self.skill_library.register(
            "run_experiment", "Run a safe Python snippet and capture output",
            "SAGE", self._skill_run_experiment, category="research")
        self.skill_library.register(
            "research_report", "Generate comprehensive research findings report",
            "SAGE", self._skill_research_report, category="research")
        self.skill_library.register(
            "cross_reference", "Find cross-module dependencies and coupling",
            "SAGE", self._skill_cross_reference, category="research")
        self.skill_library.register(
            "deep_optimization_scan",
            "Repo-wide learn + 14b optimization experiments + repo report",
            "SAGE", self._skill_deep_optimization_scan, category="research")
        self.skill_library.register(
            "run_template_experiment",
            "Run a pre-vetted experiment template from SAGE_EXPERIMENTS.yaml",
            "SAGE", self._skill_run_template_experiment, category="research")
        self.skill_library.register(
            "list_experiments",
            "List all available experiment templates",
            "SAGE", self._skill_list_experiments, category="research")

    def execute_quest(self, quest: Quest) -> str:
        title_lower = quest.title.lower()
        if "architecture" in title_lower or "structure" in title_lower:
            return self._skill_analyze_architecture()
        elif "module" in title_lower or "analyze" in title_lower:
            return self._skill_analyze_module()
        elif "experiment" in title_lower or "run" in title_lower:
            return self._skill_run_experiment()
        elif "report" in title_lower:
            return self._skill_research_report()
        elif "cross" in title_lower or "depend" in title_lower or "coupling" in title_lower:
            return self._skill_cross_reference()
        else:
            return self._skill_analyze_architecture()

    def idle_action(self):
        thought = self.ollama_think(
            "I'm SAGE the Research Scientist. I study this codebase and run experiments. "
            "Should I: map the architecture, deep-read a module, cross-reference dependencies, "
            "or run an experiment to test a hypothesis?"
        )
        if thought.startswith("Error"):
            return self._skill_analyze_architecture()
        self._emit("idle_action", {"result": f"Thinking: {thought[:150]}"})
        tl = thought.lower()
        if "architect" in tl or "structure" in tl or "map" in tl:
            return self._skill_analyze_architecture()
        elif "module" in tl or "read" in tl or "deep" in tl:
            return self._skill_analyze_module()
        elif "cross" in tl or "depend" in tl or "coupl" in tl:
            return self._skill_cross_reference()
        elif "experiment" in tl or "test" in tl or "hypoth" in tl:
            return self._skill_run_experiment()
        return self._skill_analyze_architecture()

    # ── Skills ────────────────────────────────────────────────────

    def _skill_analyze_architecture(self) -> str:
        """Map the project's directory tree and count files by type."""
        counts = {}
        total_py_lines = 0
        dirs_seen = set()

        for root, dirs, files in os.walk(str(_PROJECT_ROOT)):
            # Skip hidden dirs, node_modules, __pycache__, .git, venv
            dirs[:] = [d for d in dirs if not d.startswith(".")
                       and d not in ("node_modules", "__pycache__", "venv", ".venv",
                                     "build", "dist", ".git")]

            rel_root = Path(root).relative_to(_PROJECT_ROOT)
            depth = len(rel_root.parts)
            if depth <= 2:
                dirs_seen.add(str(rel_root) if str(rel_root) != "." else "/")

            for f in files:
                ext = Path(f).suffix.lower()
                if ext:
                    counts[ext] = counts.get(ext, 0) + 1

                if ext == ".py":
                    fp = Path(root) / f
                    try:
                        total_py_lines += len(fp.read_text(errors="replace").splitlines())
                    except Exception:
                        pass

        # Top extensions
        top_ext = sorted(counts.items(), key=lambda x: -x[1])[:8]
        ext_summary = ", ".join(f"{ext}: {n}" for ext, n in top_ext)
        total_files = sum(counts.values())

        # Top-level dirs
        top_dirs = sorted(dirs_seen)[:15]

        result = (
            f"Architecture: {total_files} files across {len(dirs_seen)} directories. "
            f"Python: {counts.get('.py', 0)} files, ~{total_py_lines:,} lines. "
            f"Top types: {ext_summary}. "
            f"Top dirs: {', '.join(top_dirs)}"
        )

        self.memory.add("architecture",
                        f"{total_files} files, {counts.get('.py', 0)} .py, "
                        f"{total_py_lines:,} lines",
                        importance=5)
        return result

    def _skill_analyze_module(self) -> str:
        """Pick a Python module and do a deep read — classes, functions, imports, complexity."""
        # Find Python files in src/, pick the one least recently analyzed
        src_dir = _PROJECT_ROOT / "src"
        if not src_dir.exists():
            return "Error: src/ directory not found"

        py_files = [f for f in src_dir.rglob("*.py")
                    if "__pycache__" not in str(f) and not f.name.startswith("__")]

        if not py_files:
            return "Error: no Python modules found in src/"

        # Ask LLM which module looks interesting, or just pick one we haven't analyzed
        analyzed = set()
        for m in self.memory.recent(20):
            if m.event_type == "module_analysis":
                # Extract filename from summary
                match = re.search(r"Analyzed (\S+)", m.summary)
                if match:
                    analyzed.add(match.group(1))

        # Pick first unanalyzed file, or cycle back
        target = None
        for f in sorted(py_files, key=lambda x: x.stat().st_size, reverse=True):
            if f.name not in analyzed:
                target = f
                break
        if target is None:
            target = py_files[0]

        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Error reading {target.name}: {e}"

        lines = content.splitlines()
        rel_path = target.relative_to(_PROJECT_ROOT)

        # Extract structure
        imports = [l.strip() for l in lines if l.strip().startswith(("import ", "from "))]
        classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
        functions = re.findall(r'^def\s+(\w+)', content, re.MULTILINE)
        methods = re.findall(r'^\s+def\s+(\w+)', content, re.MULTILINE)

        # Docstring
        doc_match = re.search(r'^"""(.+?)"""', content, re.DOTALL)
        module_doc = doc_match.group(1).strip()[:100] if doc_match else "No module docstring"

        # Ask LLM for insight
        insight = self.ollama_think(
            f"I'm analyzing {rel_path} ({len(lines)} lines). "
            f"Module doc: {module_doc}. "
            f"Classes: {classes[:5]}. Functions: {functions[:5]}. "
            f"What's the design pattern here? Any concerns? Be concise."
        )

        result = (
            f"Module: {rel_path} ({len(lines)}L)\n"
            f"  Doc: {module_doc}\n"
            f"  Imports: {len(imports)} | Classes: {len(classes)} | "
            f"Functions: {len(functions)} | Methods: {len(methods)}\n"
            f"  Classes: {', '.join(classes[:6]) or 'none'}\n"
            f"  Top functions: {', '.join(functions[:6]) or 'none'}\n"
            f"  Insight: {insight[:200] if not insight.startswith('Error') else 'LLM unavailable'}"
        )

        self.memory.add("module_analysis",
                        f"Analyzed {target.name}: {len(classes)} classes, "
                        f"{len(functions)} fns, {len(lines)}L",
                        importance=5)
        return result

    def _skill_run_experiment(self) -> str:
        """Run a safe Python snippet to test a hypothesis about the codebase."""
        # Ask LLM what to investigate
        thought = self.ollama_think(
            "I need to run a quick Python experiment to learn about this codebase. "
            "Suggest a safe, read-only experiment: counting something, checking imports, "
            "measuring file sizes, or testing a module's public API. "
            "Give me just the Python code (3-5 lines max, no file writes, no network, "
            "no subprocess). Start your response with ```python"
        )

        if thought.startswith("Error"):
            # Fallback: count Python files by directory
            code = (
                "import os\n"
                "from pathlib import Path\n"
                "root = Path(r'" + str(_PROJECT_ROOT).replace("'", "\\'") + "')\n"
                "counts = {}\n"
                "for f in root.rglob('*.py'):\n"
                "    if '__pycache__' not in str(f):\n"
                "        d = str(f.parent.relative_to(root)).split(os.sep)[0]\n"
                "        counts[d] = counts.get(d, 0) + 1\n"
                "for d, n in sorted(counts.items(), key=lambda x: -x[1])[:10]:\n"
                "    print(f'{d}: {n} Python files')\n"
            )
        else:
            # Extract code from LLM response
            code_match = re.search(r'```python\s*\n(.+?)```', thought, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
            else:
                # Try to extract any code-like block
                code_lines = [l for l in thought.splitlines()
                              if l.strip() and not l.strip().startswith("#")]
                if code_lines:
                    code = "\n".join(code_lines[:8])
                else:
                    code = "print('No experiment generated')"

        # Safety: reject dangerous patterns
        dangerous = ["subprocess", "os.system", "shutil.rmtree", "open(", "write(",
                     "requests.", "urllib.", "socket.", "exec(", "eval(",
                     "__import__", "os.remove", "os.unlink", "pathlib.Path.unlink",
                     "pathlib.Path.write"]
        code_lower = code.lower()
        for d in dangerous:
            if d.lower() in code_lower:
                return f"Error: experiment blocked — contains '{d}' (safety violation)"

        # Run via safe_subprocess with python -c
        escaped_code = code.replace("\\", "\\\\").replace('"', '\\"')
        rc, output = self.safe_subprocess(f'python -c "{escaped_code}"')

        result_status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        trimmed_output = output.strip()[:500] if output else "(no output)"

        result = (
            f"Experiment: {result_status}\n"
            f"Code:\n{code[:300]}\n"
            f"Output:\n{trimmed_output}"
        )

        self.memory.add("experiment",
                        f"Ran experiment: {result_status}",
                        details=trimmed_output[:200],
                        importance=5)
        return result

    def _skill_research_report(self) -> str:
        """Compile all research findings into a JSON report."""
        # Gather recent research memories
        recent = self.memory.recent(50)
        research_entries = [m for m in recent
                           if m.event_type in ("architecture", "module_analysis",
                                               "experiment", "cross_reference")]

        findings = []
        for m in research_entries:
            findings.append({
                "type": m.event_type,
                "timestamp": m.timestamp,
                "summary": m.summary[:200],
                "details": m.details[:300] if m.details else "",
            })

        # Architecture snapshot
        py_count = 0
        py_lines = 0
        for f in (_PROJECT_ROOT / "src").rglob("*.py"):
            if "__pycache__" not in str(f):
                py_count += 1
                try:
                    py_lines += len(f.read_text(errors="replace").splitlines())
                except Exception:
                    pass

        report = {
            "agent": "SAGE",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "health": "GREEN",
            "python_modules": py_count,
            "python_lines": py_lines,
            "research_findings": len(findings),
            "findings": findings[:20],
        }

        result = self.safe_write(
            "data/agent_reports/sage_research_report.json",
            json.dumps(report, indent=2)
        )
        if result.startswith("OK"):
            return (f"Research report: {py_count} modules, {py_lines:,} lines, "
                    f"{len(findings)} findings recorded")
        return result

    def _skill_cross_reference(self) -> str:
        """Analyze import graphs and cross-module dependencies in src/."""
        src_dir = _PROJECT_ROOT / "src"
        if not src_dir.exists():
            return "Error: src/ not found"

        # Build import graph
        graph = {}  # module -> list of imports
        py_files = [f for f in src_dir.rglob("*.py")
                    if "__pycache__" not in str(f) and not f.name.startswith("__")]

        for pf in py_files:
            try:
                content = pf.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            rel = str(pf.relative_to(_PROJECT_ROOT)).replace("\\", "/")
            imports = []

            for line in content.splitlines():
                line = line.strip()
                # from src.agents.X import Y
                m = re.match(r'from\s+(src\.\S+|\.+\S*)\s+import', line)
                if m:
                    imports.append(m.group(1))
                # import src.X
                m = re.match(r'import\s+(src\.\S+)', line)
                if m:
                    imports.append(m.group(1))

            if imports:
                graph[rel] = imports

        # Find most-imported modules
        import_counts = {}
        for deps in graph.values():
            for dep in deps:
                import_counts[dep] = import_counts.get(dep, 0) + 1

        top_imported = sorted(import_counts.items(), key=lambda x: -x[1])[:8]

        # Find modules with most outgoing deps
        top_dependent = sorted(graph.items(), key=lambda x: -len(x[1]))[:5]

        # Circular dependency check (simple: A imports B and B imports A)
        circulars = []
        for mod_a, deps_a in graph.items():
            for dep in deps_a:
                # Normalize: dep might be "src.agents.agent_base" while mod_a is "src/agents/agent_base.py"
                dep_path = dep.replace(".", "/") + ".py"
                if dep_path in graph:
                    deps_b = graph[dep_path]
                    mod_a_import = mod_a.replace("/", ".").rstrip(".py")
                    if any(mod_a_import in d for d in deps_b):
                        pair = tuple(sorted([mod_a, dep_path]))
                        if pair not in circulars:
                            circulars.append(pair)

        parts = [
            f"Cross-reference: {len(graph)} modules with imports",
            f"Most imported: {', '.join(f'{m}({n})' for m, n in top_imported[:5])}",
            f"Most deps: {', '.join(f'{m}({len(d)})' for m, d in top_dependent[:3])}",
        ]
        if circulars:
            parts.append(f"⚠️ {len(circulars)} potential circular deps")
        else:
            parts.append("No circular dependencies detected")

        self.memory.add("cross_reference",
                        f"{len(graph)} modules, top: {top_imported[0][0] if top_imported else 'none'}",
                        importance=5)

        return "\n".join(parts)

    # ── Deep Optimization Scan (button-triggered, 14b) ────────────

    _DANGEROUS_EXPERIMENT_TOKENS = (
        "subprocess", "os.system", "shutil.rmtree", "write(", "open(",
        "requests.", "urllib.", "socket.", "exec(", "eval(",
        "__import__", "os.remove", "os.unlink",
    )

    def _skill_deep_optimization_scan(self, emit_fn=None) -> str:
        """Full-stack scan: learn repo, run 14b experiments, export report to repo."""
        def _say(msg: str) -> None:
            if emit_fn:
                try:
                    emit_fn(msg)
                except Exception:
                    pass
            self._emit("idle_action", {"result": msg[:200]})

        _say("SAGE: initiating deep optimization scan (qwen2.5-coder:14b)")
        self.reset_tick_budget()

        # 1. Architecture snapshot
        arch = self._skill_analyze_architecture()
        _say("SAGE: architecture mapped")

        # 2. Import graph
        crossref = self._skill_cross_reference()
        _say("SAGE: cross-reference complete")

        # 3. Collect hotspot modules
        src_dir = _PROJECT_ROOT / "src"
        module_summaries = []
        if src_dir.exists():
            py_files = sorted(
                [f for f in src_dir.rglob("*.py")
                 if "__pycache__" not in str(f) and not f.name.startswith("__")],
                key=lambda x: x.stat().st_size, reverse=True,
            )[:5]
            for pf in py_files:
                try:
                    content = pf.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                lines = content.splitlines()
                rel_path = str(pf.relative_to(_PROJECT_ROOT)).replace("\\", "/")
                classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
                functions = re.findall(r'^def\s+(\w+)', content, re.MULTILINE)
                module_summaries.append({
                    "path": rel_path,
                    "lines": len(lines),
                    "classes": classes[:10],
                    "functions": functions[:10],
                })
        _say(f"SAGE: scanned {len(module_summaries)} hotspot modules")

        # 4. Ask 14b for optimization hypotheses
        module_brief = "\n".join(
            f"- {m['path']} ({m['lines']}L, {len(m['classes'])} classes, {len(m['functions'])} fns)"
            for m in module_summaries
        ) or "(no modules)"
        hypo_prompt = (
            "You are SAGE running a repo-wide optimization audit on a Python project.\n\n"
            f"ARCHITECTURE: {arch}\n\n"
            f"CROSS-REFERENCE: {crossref}\n\n"
            f"LARGEST MODULES:\n{module_brief}\n\n"
            "Propose exactly 3 concrete, testable optimization hypotheses for this repo. "
            "Each must be verifiable with a 3-8 line READ-ONLY Python snippet that prints measurements. "
            "No file writes, no network, no subprocess, no open(), no exec/eval.\n\n"
            "Respond in this exact format:\n"
            "HYPOTHESIS 1: <one line>\n"
            "```python\n<code>\n```\n\n"
            "HYPOTHESIS 2: <one line>\n"
            "```python\n<code>\n```\n\n"
            "HYPOTHESIS 3: <one line>\n"
            "```python\n<code>\n```"
        )
        hypo_response = self.ollama_think(hypo_prompt, max_tokens=1200, temperature=0.4)
        if hypo_response.startswith("Error"):
            return f"Error: 14b hypothesis generation failed — {hypo_response}"
        _say("SAGE: 14b proposed hypotheses")

        # 5. Extract and run experiments (contained)
        code_blocks = re.findall(r'```python\s*\n(.+?)```', hypo_response, re.DOTALL)
        hypothesis_lines = re.findall(r'HYPOTHESIS\s*\d+:\s*(.+)', hypo_response)
        experiments = []
        for i, raw_code in enumerate(code_blocks[:3]):
            code = raw_code.strip()
            hypothesis = (hypothesis_lines[i].strip()
                          if i < len(hypothesis_lines) else f"Hypothesis {i + 1}")

            code_lower = code.lower()
            blocked = next((d for d in self._DANGEROUS_EXPERIMENT_TOKENS
                            if d in code_lower), None)
            if blocked:
                experiments.append({
                    "hypothesis": hypothesis,
                    "code": code,
                    "rc": -1,
                    "output": f"BLOCKED by safety gate — contains '{blocked}'",
                })
                _say(f"SAGE: experiment {i + 1} BLOCKED ({blocked})")
                continue

            self.reset_tick_budget()
            escaped = code.replace("\\", "\\\\").replace('"', '\\"')
            try:
                rc, output = self.safe_subprocess(f'python -c "{escaped}"')
            except Exception as exc:
                rc, output = -1, f"execution error: {type(exc).__name__}: {exc}"

            experiments.append({
                "hypothesis": hypothesis,
                "code": code,
                "rc": rc,
                "output": (output or "").strip()[:1500],
            })
            _say(f"SAGE: experiment {i + 1} {'OK' if rc == 0 else f'FAILED(rc={rc})'}")

        # 6. Ask 14b to synthesize a final optimization report
        exp_brief_parts = []
        for i, e in enumerate(experiments, 1):
            exp_brief_parts.append(
                f"\n### Experiment {i}: {e['hypothesis']}\n"
                f"Return code: {e['rc']}\n"
                f"Code:\n{e['code']}\n"
                f"Output:\n{e['output']}\n"
            )
        exp_brief = "\n".join(exp_brief_parts) or "(no experiments ran)"

        synth_prompt = (
            "You are SAGE writing a final optimization report for a Python + native "
            "+ frontend monorepo.\n\n"
            f"ARCHITECTURE: {arch}\n\n"
            f"CROSS-REFERENCE: {crossref}\n\n"
            f"EXPERIMENTS:\n{exp_brief}\n\n"
            "Write a markdown report with these sections:\n"
            "1. Executive summary (3 bullets)\n"
            "2. Findings per experiment (what the numbers actually show)\n"
            "3. Ranked optimization recommendations with expected impact\n"
            "4. Risks and trade-offs\n"
            "Be concrete. Cite file paths where relevant."
        )
        synthesis = self.ollama_think(synth_prompt, max_tokens=2000, temperature=0.5)
        if synthesis.startswith("Error"):
            synthesis = f"(synthesis unavailable — {synthesis})"
        _say("SAGE: 14b synthesis complete")

        # 7. Assemble final markdown
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        human_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        md_parts = [
            "# SAGE Deep Optimization Report",
            "",
            f"**Generated:** {human_ts}",
            "**Model:** qwen2.5-coder:14b",
            "**Agent:** SAGE (Research Scientist, QUE Guild)",
            "",
            "## Architecture",
            "",
            arch,
            "",
            "## Cross-Reference",
            "",
            "```",
            crossref,
            "```",
            "",
            "## Hotspot Modules",
            "",
        ]
        for m in module_summaries:
            md_parts.append(
                f"- `{m['path']}` — {m['lines']} lines, "
                f"{len(m['classes'])} classes, {len(m['functions'])} fns"
            )
        md_parts += [
            "",
            "## Hypotheses (14b)",
            "",
            "```",
            hypo_response,
            "```",
            "",
            "## Experiments",
            "",
        ]
        for i, e in enumerate(experiments, 1):
            status = "PASS" if e["rc"] == 0 else f"FAIL (rc={e['rc']})"
            md_parts += [
                f"### {i}. {e['hypothesis']}",
                "",
                f"**Result:** {status}",
                "",
                "```python",
                e["code"],
                "```",
                "",
                "```",
                e["output"][:2000] or "(no output)",
                "```",
                "",
            ]
        md_parts += [
            "## Synthesis",
            "",
            synthesis,
            "",
            "---",
            "*Generated by SAGE via QUE Guild dashboard deep scan.*",
        ]
        markdown = "\n".join(md_parts)

        # 8. Write full report to repo (data/agent_reports/)
        self.reset_tick_budget()
        report_rel = f"data/agent_reports/sage_deepscan_{timestamp}.md"
        report_abs = _PROJECT_ROOT / report_rel
        try:
            report_abs.parent.mkdir(parents=True, exist_ok=True)
            report_abs.write_text(markdown, encoding="utf-8")
            repo_note = f"saved: {report_rel}"
        except Exception as exc:
            # Fallback: safe_write (capped at 99 KB)
            repo_note = self.safe_write(report_rel, markdown[:99_000])

        self.memory.add(
            "deep_scan",
            f"Deep scan: {len(experiments)} experiments, "
            f"{sum(1 for e in experiments if e['rc'] == 0)} passed",
            details=repo_note,
            importance=9,
        )
        _say(f"SAGE: deep scan complete → {report_abs.name}")

        return (
            f"Deep scan complete. Experiments run: {len(experiments)}. "
            f"{repo_note}"
        )

    # ── Template Experiment System ────────────────────────────────

    def _skill_list_experiments(self) -> str:
        """List all available experiment templates from SAGE_EXPERIMENTS.yaml."""
        templates = get_experiment_templates()
        if not templates:
            return "No experiment templates available (SAGE_EXPERIMENTS.yaml not found or empty)"
        
        lines = ["Available SAGE experiments:"]
        for name, spec in templates.items():
            desc = spec.get("description", "No description")[:60]
            tags = ", ".join(spec.get("tags", []))[:30]
            lines.append(f"  • {name}: {desc}")
            if tags:
                lines.append(f"      Tags: [{tags}]")
        
        return "\n".join(lines)

    def _skill_run_template_experiment(self, experiment_name: str = "") -> str:
        """Run a pre-vetted experiment template from SAGE_EXPERIMENTS.yaml."""
        templates = get_experiment_templates()
        if not templates:
            return "Error: No experiment templates available"
        
        # If no name given, pick one at random
        if not experiment_name:
            import random
            experiment_name = random.choice(list(templates.keys()))
        
        spec = templates.get(experiment_name)
        if not spec:
            available = ", ".join(sorted(templates.keys())[:5])
            return f"Error: Unknown experiment '{experiment_name}'. Available: {available}"
        
        name = spec.get("name", experiment_name)
        hypothesis = spec.get("hypothesis", "")
        code = spec.get("code", "")
        expected = spec.get("expected_outcome", "")
        tags = spec.get("tags", [])
        
        if not code:
            return f"Error: Experiment '{experiment_name}' has no code"
        
        # Safety check
        code_lower = code.lower()
        for d in self._DANGEROUS_EXPERIMENT_TOKENS:
            if d in code_lower:
                return f"Error: Experiment blocked — template contains '{d}'"
        
        # Run the experiment
        self.reset_tick_budget()
        escaped = code.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        
        try:
            # Run via safe subprocess
            import subprocess
            proc = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(_PROJECT_ROOT),
            )
            rc = proc.returncode
            output = (proc.stdout + proc.stderr).strip()[:2000]
        except subprocess.TimeoutExpired:
            rc = -1
            output = "TIMEOUT: Experiment exceeded 30s limit"
        except Exception as exc:
            rc = -1
            output = f"ERROR: {type(exc).__name__}: {exc}"
        
        status = "PASS" if rc == 0 else f"FAIL (rc={rc})"
        
        result = [
            f"📊 Experiment: {name}",
            f"   Hypothesis: {hypothesis}",
            f"   Status: {status}",
            f"   Expected: {expected}",
            "",
            "Output:",
            output[:1500] if output else "(no output)",
        ]
        
        self.memory.add(
            "template_experiment",
            f"{experiment_name}: {status}",
            details=output[:300],
            importance=6,
        )
        
        return "\n".join(result)

    def run_all_template_experiments(self, emit_fn=None) -> str:
        """Run all template experiments and return a summary report."""
        def _say(msg: str) -> None:
            if emit_fn:
                try:
                    emit_fn(msg)
                except Exception:
                    pass
            self._emit("idle_action", {"result": msg[:200]})
        
        templates = get_experiment_templates()
        if not templates:
            return "No experiment templates available"
        
        results = []
        passed = 0
        failed = 0
        
        for name in list(templates.keys()):
            _say(f"SAGE: running {name}...")
            result = self._skill_run_template_experiment(name)
            if "PASS" in result:
                passed += 1
            else:
                failed += 1
            results.append(result)
        
        summary = f"Template experiments: {passed} passed, {failed} failed"
        _say(f"SAGE: {summary}")
        
        return "\n\n---\n\n".join(results) + f"\n\n## Summary\n{summary}"

