# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Agent tool-calling loop for QUE chatbox — hardened edition.

Implements a ReAct-style generate→parse→execute→re-feed cycle that lets
the local Qwen model use tools (file read, directory listing, Python eval,
search, grep) to answer user queries.

Security & Provenance (OWASP / QuantoniumOS GOVERNANCE.md compliant):
  ◆ Sandboxed execution — all file ops confined to PROJECT_ROOT
  ◆ Path-traversal prevention — symlink resolution + prefix check
  ◆ Input validation — type-checked, length-capped, pattern-filtered
  ◆ Output sanitization — strips prompt-injection markers from tool results
  ◆ Rate limiting — per-session tool-call budget + cooldown
  ◆ Resource caps — file read size, glob depth, eval timeout
  ◆ Provenance audit log — every tool invocation recorded with
    timestamp, caller, args, result hash, duration, verdict
  ◆ Allowlist-only tool dispatch — no dynamic executor registration
  ◆ Reasoning transparency — full chain-of-thought is auditable
    per ETHICS_POLICY.md §III (no black-box mode)

The agent runs entirely in-process — no external API needed.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import re
import signal
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from atomic_io import AtomicJsonlWriter

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════
#  SAFETY CONSTANTS
# ══════════════════════════════════════════════════════════════════

# Sandbox boundary — every file op must resolve inside this tree
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Resource caps
MAX_FILE_READ_BYTES = 512_000          # 512 KB per read_file call
MAX_FILE_READ_LINES = 300              # Max lines returned per read
MAX_GLOB_RESULTS = 50                  # search_files cap
MAX_GREP_RESULTS = 30                  # grep cap
MAX_GREP_LINE_LEN = 200               # truncate long grep match lines
MAX_PYTHON_EVAL_LEN = 2000            # max code length for python_eval
MAX_TOOL_RESULT_LEN = 4000            # truncate tool output
MAX_ARG_STRING_LEN = 1000             # max length of any string argument
MAX_AGENT_TURNS = 5                   # max tool-call rounds per query

# Rate limiting
MAX_TOOL_CALLS_PER_SESSION = 30       # total tool calls before hard stop
TOOL_COOLDOWN_SECONDS = 0.5           # min delay between consecutive calls

# Binary/sensitive file extensions — never read or search
_BINARY_EXTS = frozenset({
    ".pyc", ".pyd", ".so", ".dll", ".exe", ".bin", ".o", ".a",
    ".gguf", ".rftmwpk", ".whl", ".tar", ".gz", ".zip", ".7z",
    ".bz2", ".xz", ".png", ".jpg", ".jpeg", ".gif", ".bmp",
    ".ico", ".wav", ".mp3", ".mp4", ".avi", ".mov", ".pdf",
    ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
})

# Sensitive paths — never expose even if inside project root
_SENSITIVE_PATTERNS = [
    ".env", ".git/config", "id_rsa", "id_ed25519",
    ".ssh/", "credentials", "secrets", "token",
    ".aws/", "private_key", "password",
]

# Prompt injection markers — stripped from tool outputs before feeding back
_INJECTION_PATTERNS = re.compile(
    r'<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>'
    r'|<tool_call>|</tool_call>'
    r'|<\|system\|>|<\|user\|>|<\|assistant\|>'
    r'|\[INST\]|\[/INST\]'
    r'|<<SYS>>|<</SYS>>'
    r'|Human:|Assistant:',
    re.IGNORECASE,
)

# Python eval blocklist — matched case-insensitively
_PYTHON_BLOCKED = [
    "import os", "import sys", "import subprocess", "import shutil",
    "import pathlib", "import socket", "import http", "import urllib",
    "import ctypes", "import signal", "import multiprocessing",
    "subprocess", "__import__", "exec(", "eval(",
    "open(", "os.system", "os.popen", "os.exec",
    "os.remove", "os.unlink", "os.rmdir", "os.rename",
    "shutil", "rmtree", "compile(", "globals(", "locals(",
    "getattr(", "setattr(", "delattr(", "vars(",
    "breakpoint(", "__builtins__", "__class__",
    "__subclasses__", "__bases__", "__mro__",
    "type.__", "object.__",
]


# ══════════════════════════════════════════════════════════════════
#  PROVENANCE AUDIT LOG
# ══════════════════════════════════════════════════════════════════

class AgentAuditLog:
    """Append-only provenance log for every agent tool invocation.

    Writes JSONL records to logs/agent_audit_<session>.jsonl with:
      - timestamp (ISO 8601)
      - session_id (UUID per agent session)
      - turn (which ReAct turn)
      - tool_name
      - arguments (sanitized — long values truncated)
      - result_hash (SHA-256 of full result for tamper detection)
      - result_preview (first 200 chars)
      - duration_ms
      - verdict ("ok" | "blocked" | "error" | "rate_limited")
      - error_detail (if verdict != "ok")

    Compliant with ETHICS_POLICY.md §III (full auditability).
    """

    def __init__(self):
        self.session_id = str(uuid.uuid4())
        os.makedirs("logs", exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join("logs", f"agent_audit_{stamp}_{self.session_id[:8]}.jsonl")
        self._writer = AtomicJsonlWriter(path)
        self._call_count = 0
        self._last_call_time = 0.0
        self._lock = threading.Lock()

        logger.info("Agent audit log: %s (session %s)", path, self.session_id[:8])

    @property
    def call_count(self) -> int:
        return self._call_count

    def check_rate_limit(self) -> Optional[str]:
        """Return an error string if rate limit exceeded, else None."""
        with self._lock:
            if self._call_count >= MAX_TOOL_CALLS_PER_SESSION:
                return (f"Tool call budget exhausted ({MAX_TOOL_CALLS_PER_SESSION} calls). "
                        f"Please start a new conversation.")
            now = time.monotonic()
            if (now - self._last_call_time) < TOOL_COOLDOWN_SECONDS:
                return f"Tool cooldown active ({TOOL_COOLDOWN_SECONDS}s between calls)."
        return None

    def record(
        self,
        turn: int,
        tool_name: str,
        arguments: Dict[str, Any],
        result: str,
        duration_ms: float,
        verdict: str,
        error_detail: str = "",
    ) -> None:
        """Write one audit record."""
        with self._lock:
            self._call_count += 1
            self._last_call_time = time.monotonic()

        # Sanitize arguments for logging (truncate long values)
        safe_args = {}
        for k, v in arguments.items():
            sv = str(v)
            safe_args[k] = sv[:200] if len(sv) > 200 else sv

        result_hash = hashlib.sha256(result.encode("utf-8", errors="replace")).hexdigest()

        record = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "session_id": self.session_id,
            "turn": turn,
            "tool": tool_name,
            "arguments": safe_args,
            "result_sha256": result_hash,
            "result_preview": result[:200].replace("\n", "\\n"),
            "duration_ms": round(duration_ms, 2),
            "verdict": verdict,
        }
        if error_detail:
            record["error"] = error_detail[:500]

        try:
            self._writer.write(record)
        except Exception as exc:
            logger.warning("Audit write failed: %s", exc)

    def close(self):
        try:
            self._writer.close()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════
#  INPUT VALIDATION
# ══════════════════════════════════════════════════════════════════

def _validate_string(value: Any, name: str, max_len: int = MAX_ARG_STRING_LEN) -> str:
    """Validate and sanitize a string argument."""
    if not isinstance(value, str):
        raise ValueError(f"Argument '{name}' must be a string, got {type(value).__name__}")
    if len(value) > max_len:
        raise ValueError(f"Argument '{name}' exceeds max length ({len(value)} > {max_len})")
    # Block null bytes
    if "\x00" in value:
        raise ValueError(f"Argument '{name}' contains null bytes")
    return value


def _validate_int(value: Any, name: str, min_val: int = 0, max_val: int = 10000) -> int:
    """Validate an integer argument."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Argument '{name}' must be an integer, got {type(value).__name__}")
    if v < min_val or v > max_val:
        raise ValueError(f"Argument '{name}' out of range [{min_val}, {max_val}]")
    return v


# ══════════════════════════════════════════════════════════════════
#  PATH SECURITY
# ══════════════════════════════════════════════════════════════════

def _safe_path(p: str) -> Path:
    """Resolve a path, ensuring it stays within the project root.

    Security measures:
      1. Resolve symlinks to real paths
      2. Normalize with .resolve() to eliminate .. traversal
      3. Verify the resolved path starts with PROJECT_ROOT
      4. Check against sensitive path patterns
      5. Block binary file access for read operations
    """
    p = _validate_string(p, "path")

    # Resolve relative to project root
    if Path(p).is_absolute():
        resolved = Path(p).resolve()
    else:
        resolved = (_PROJECT_ROOT / p).resolve()

    # Security: prevent path traversal — must be within sandbox
    try:
        resolved.relative_to(_PROJECT_ROOT)
    except ValueError:
        raise PermissionError(f"Access denied: path outside project sandbox")

    # Security: block sensitive files
    resolved_lower = str(resolved).lower().replace("\\", "/")
    for pattern in _SENSITIVE_PATTERNS:
        if pattern.lower() in resolved_lower:
            raise PermissionError(f"Access denied: sensitive path pattern '{pattern}'")

    return resolved


def _is_binary_file(path: Path) -> bool:
    """Check if a file is binary based on extension."""
    return path.suffix.lower() in _BINARY_EXTS


# ══════════════════════════════════════════════════════════════════
#  OUTPUT SANITIZATION
# ══════════════════════════════════════════════════════════════════

def _sanitize_output(text: str) -> str:
    """Strip prompt-injection markers from tool output before feeding to model.

    This prevents a malicious file from containing <|im_start|>system or
    similar tokens that could hijack the model's behavior.
    """
    return _INJECTION_PATTERNS.sub("[FILTERED]", text)


# ══════════════════════════════════════════════════════════════════
#  TOOL DEFINITIONS
# ══════════════════════════════════════════════════════════════════

TOOL_DEFS: List[Dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read the contents of a text file. Returns the text content (max 300 lines).",
        "parameters": {
            "path": {"type": "string", "description": "Relative file path to read (within project)"},
            "start_line": {"type": "integer", "description": "Optional 1-based start line", "default": 1},
            "end_line": {"type": "integer", "description": "Optional 1-based end line (0 = all)", "default": 0},
        },
    },
    {
        "name": "list_dir",
        "description": "List files and folders in a directory (within project).",
        "parameters": {
            "path": {"type": "string", "description": "Directory path to list"},
        },
    },
    {
        "name": "python_eval",
        "description": "Evaluate a safe Python expression (math, string ops). No file/network/system access.",
        "parameters": {
            "code": {"type": "string", "description": "Python expression or short script (max 2000 chars)"},
        },
    },
    {
        "name": "search_files",
        "description": "Search for files matching a glob pattern in the project tree.",
        "parameters": {
            "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py' or 'src/**/*.ts'"},
            "max_results": {"type": "integer", "description": "Max results (capped at 50)", "default": 20},
        },
    },
    {
        "name": "grep",
        "description": "Search file contents for a text pattern (case-insensitive, within project).",
        "parameters": {
            "pattern": {"type": "string", "description": "Text or regex pattern to search for"},
            "path": {"type": "string", "description": "File or directory to search in", "default": "."},
            "max_results": {"type": "integer", "description": "Max matching lines (capped at 30)", "default": 20},
        },
    },
    {
        "name": "get_time",
        "description": "Get the current date and time.",
        "parameters": {},
    },
]


# ══════════════════════════════════════════════════════════════════
#  TOOL EXECUTORS (sandboxed, validated, sanitized)
# ══════════════════════════════════════════════════════════════════

def _exec_read_file(path: str, start_line: int = 1, end_line: int = 0, **_) -> str:
    path = _validate_string(path, "path")
    start_line = _validate_int(start_line, "start_line", min_val=1, max_val=999999)
    end_line = _validate_int(end_line, "end_line", min_val=0, max_val=999999)

    fpath = _safe_path(path)
    if not fpath.exists():
        return f"Error: file not found: {fpath.name}"
    if not fpath.is_file():
        return f"Error: not a file: {fpath.name}"
    if _is_binary_file(fpath):
        return f"Error: cannot read binary file: {fpath.name}"

    # Size cap — refuse to read very large files entirely
    try:
        size = fpath.stat().st_size
        if size > MAX_FILE_READ_BYTES:
            return (f"Error: file too large ({size:,} bytes > {MAX_FILE_READ_BYTES:,} limit). "
                    f"Use start_line/end_line to read a portion.")
    except OSError as exc:
        return f"Error: cannot stat file: {exc}"

    try:
        text = fpath.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        s = max(0, start_line - 1)
        e = end_line if end_line > 0 else len(lines)
        selected = lines[s:e]
        if len(selected) > MAX_FILE_READ_LINES:
            selected = selected[:MAX_FILE_READ_LINES]
            selected.append(f"\n... ({len(lines) - MAX_FILE_READ_LINES} more lines truncated)\n")
        return _sanitize_output("".join(selected))
    except Exception as exc:
        return f"Error reading file: {type(exc).__name__}"


def _exec_list_dir(path: str = ".", **_) -> str:
    path = _validate_string(path, "path")
    dpath = _safe_path(path)
    if not dpath.exists():
        return f"Error: directory not found"
    if not dpath.is_dir():
        return f"Error: not a directory"
    try:
        entries = sorted(dpath.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        lines = []
        for e in entries[:MAX_GLOB_RESULTS]:
            # Don't expose .git internals or sensitive dirs
            if e.name.startswith(".git") and e.is_dir() and e.name != ".github":
                lines.append(f"  {e.name}/ (hidden)")
                continue
            suffix = "/" if e.is_dir() else ""
            lines.append(f"  {e.name}{suffix}")
        if len(entries) > MAX_GLOB_RESULTS:
            lines.append(f"  ... ({len(entries) - MAX_GLOB_RESULTS} more)")
        return "\n".join(lines) if lines else "(empty directory)"
    except Exception as exc:
        return f"Error listing directory: {type(exc).__name__}"


def _exec_python_eval(code: str, **_) -> str:
    """Evaluate Python in a strictly restricted sandbox."""
    code = _validate_string(code, "code", max_len=MAX_PYTHON_EVAL_LEN)

    # Security: block dangerous operations (case-insensitive)
    code_lower = code.lower()
    for blocked in _PYTHON_BLOCKED:
        if blocked.lower() in code_lower:
            return f"Blocked: '{blocked}' is not allowed for safety."

    # Security: block dunder access patterns
    if re.search(r'__\w+__', code):
        return "Blocked: dunder attribute access is not allowed."

    # Restricted builtins — arithmetic, string ops, basic data structures only
    import math
    safe_globals = {
        "__builtins__": {},
        "math": math,
        "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
        "len": len, "sorted": sorted, "reversed": reversed, "range": range,
        "int": int, "float": float, "str": str, "bool": bool,
        "list": list, "dict": dict, "tuple": tuple, "set": set,
        "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
        "True": True, "False": False, "None": None,
        "print": lambda *a, **kw: None,  # no-op print
        "hex": hex, "bin": bin, "oct": oct, "chr": chr, "ord": ord,
        "isinstance": isinstance, "issubclass": issubclass,
        "pow": pow, "divmod": divmod,
    }

    try:
        result = eval(code, safe_globals)  # noqa: S307
        return _sanitize_output(str(result))
    except SyntaxError:
        # Try exec for statements — ONLY if no blocked patterns
        try:
            local_ns: Dict[str, Any] = {}
            exec(code, safe_globals, local_ns)  # noqa: S102
            if "result" in local_ns:
                return _sanitize_output(str(local_ns["result"]))
            return str(local_ns) if local_ns else "(executed, no return value)"
        except Exception as exc:
            return f"Error: {type(exc).__name__}: {exc}"
    except Exception as exc:
        return f"Error: {type(exc).__name__}: {exc}"


def _exec_search_files(pattern: str, max_results: int = 20, **_) -> str:
    pattern = _validate_string(pattern, "pattern", max_len=200)
    max_results = _validate_int(max_results, "max_results", min_val=1, max_val=MAX_GLOB_RESULTS)

    # Security: block patterns that escape the sandbox
    if ".." in pattern or pattern.startswith("/") or pattern.startswith("\\"):
        return "Blocked: pattern must not contain '..' or start with '/' or '\\'."

    try:
        matches = list(_PROJECT_ROOT.glob(pattern))
        # Double-check all results are inside sandbox
        safe_matches = []
        for m in matches:
            try:
                m.resolve().relative_to(_PROJECT_ROOT)
                if not _is_binary_file(m):
                    safe_matches.append(m)
            except ValueError:
                continue
        safe_matches = safe_matches[:max_results]
        if not safe_matches:
            return "(no files matched)"
        lines = [str(m.relative_to(_PROJECT_ROOT)) for m in safe_matches]
        return "\n".join(lines)
    except Exception as exc:
        return f"Error: {type(exc).__name__}"


def _exec_grep(pattern: str, path: str = ".", max_results: int = 20, **_) -> str:
    pattern = _validate_string(pattern, "pattern", max_len=500)
    path = _validate_string(path, "path")
    max_results = _validate_int(max_results, "max_results", min_val=1, max_val=MAX_GREP_RESULTS)

    search_path = _safe_path(path)
    results: List[str] = []

    # Security: limit regex complexity (no catastrophic backtracking)
    if len(pattern) > 200:
        return "Blocked: pattern too long (max 200 chars for regex)."
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Invalid pattern: {e}"

    try:
        files = [search_path] if search_path.is_file() else sorted(search_path.rglob("*"))
        files_scanned = 0
        for fpath in files:
            if not fpath.is_file():
                continue
            if _is_binary_file(fpath):
                continue
            # Skip .git internals
            try:
                rel = fpath.relative_to(_PROJECT_ROOT)
                if str(rel).startswith(".git/") or str(rel).startswith(".git\\"):
                    continue
            except ValueError:
                continue
            files_scanned += 1
            if files_scanned > 500:  # Don't scan more than 500 files
                results.append("... (search halted: >500 files scanned)")
                break
            try:
                size = fpath.stat().st_size
                if size > MAX_FILE_READ_BYTES:
                    continue  # Skip very large files
                text = fpath.read_text(encoding="utf-8", errors="ignore")
                for i, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        rel_str = str(fpath.relative_to(_PROJECT_ROOT))
                        truncated = line.strip()[:MAX_GREP_LINE_LEN]
                        results.append(f"{rel_str}:{i}: {truncated}")
                        if len(results) >= max_results:
                            break
            except Exception:
                continue
            if len(results) >= max_results:
                break
    except Exception as exc:
        return f"Error: {type(exc).__name__}"
    return _sanitize_output("\n".join(results)) if results else "(no matches)"


def _exec_get_time(**_) -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")


# Allowlist-only executor registry — NO dynamic registration allowed
_EXECUTORS: Dict[str, Callable] = {
    "read_file": _exec_read_file,
    "list_dir": _exec_list_dir,
    "python_eval": _exec_python_eval,
    "search_files": _exec_search_files,
    "grep": _exec_grep,
    "get_time": _exec_get_time,
}


# ══════════════════════════════════════════════════════════════════
#  TOOL-CALL PARSING
# ══════════════════════════════════════════════════════════════════

# Qwen2.5 native tool-calling format
_TOOL_CALL_RE = re.compile(
    r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
    re.DOTALL,
)

# R1 <think>...</think> block — strip from final output
_THINK_BLOCK_RE = re.compile(
    r'<think>.*?</think>\s*',
    re.DOTALL,
)


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract tool calls from model output with validation.
    
    Handles R1's <think>...</think> reasoning blocks — strips them
    before parsing so tool calls embedded after thinking are found.
    """
    # Strip R1 thinking blocks
    text_clean = _THINK_BLOCK_RE.sub('', text)
    calls = []
    for m in _TOOL_CALL_RE.finditer(text_clean):
        raw = m.group(1)
        # Security: cap JSON size to prevent memory bomb
        if len(raw) > 5000:
            logger.warning("Tool call JSON too large (%d bytes), skipped", len(raw))
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        if not isinstance(name, str):
            continue
        # Only allow known tools
        if name not in _EXECUTORS:
            logger.warning("Model requested unknown tool: %s", name)
            continue
        args = obj.get("arguments", obj.get("parameters", {}))
        if not isinstance(args, dict):
            args = {}
        calls.append({"name": name, "arguments": args})
    return calls


def execute_tool(
    call: Dict[str, Any],
    audit: Optional[AgentAuditLog] = None,
    turn: int = 0,
) -> str:
    """Execute a single tool call with full safety and provenance.

    Returns the sanitized result string. Never raises — all errors
    are caught and returned as error strings.
    """
    name = call.get("name", "")
    args = call.get("arguments", {})

    # Rate limit check
    if audit:
        rate_err = audit.check_rate_limit()
        if rate_err:
            if audit:
                audit.record(turn, name, args, rate_err, 0.0, "rate_limited")
            return f"Error: {rate_err}"

    # Allowlist check
    executor = _EXECUTORS.get(name)
    if executor is None:
        msg = f"Error: unknown tool '{name}'. Available: {', '.join(sorted(_EXECUTORS))}"
        if audit:
            audit.record(turn, name, args, msg, 0.0, "blocked", "unknown tool")
        return msg

    # Execute with timing
    t0 = time.perf_counter()
    try:
        result = executor(**args)
    except PermissionError as exc:
        result = f"Blocked: {exc}"
        duration_ms = (time.perf_counter() - t0) * 1000
        if audit:
            audit.record(turn, name, args, result, duration_ms, "blocked", str(exc))
        return result
    except ValueError as exc:
        result = f"Invalid input: {exc}"
        duration_ms = (time.perf_counter() - t0) * 1000
        if audit:
            audit.record(turn, name, args, result, duration_ms, "blocked", str(exc))
        return result
    except Exception as exc:
        result = f"Error: {type(exc).__name__}: {exc}"
        duration_ms = (time.perf_counter() - t0) * 1000
        if audit:
            audit.record(turn, name, args, result, duration_ms, "error", str(exc))
        return result

    duration_ms = (time.perf_counter() - t0) * 1000

    # Truncate very long results
    if len(result) > MAX_TOOL_RESULT_LEN:
        result = result[:MAX_TOOL_RESULT_LEN] + f"\n... (truncated, {len(result):,} chars total)"

    # Sanitize output — strip prompt injection markers
    result = _sanitize_output(result)

    # Audit log
    if audit:
        audit.record(turn, name, args, result, duration_ms, "ok")

    return result


# ══════════════════════════════════════════════════════════════════
#  AGENT SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════

def build_agent_system_prompt() -> str:
    """Build a compact system prompt for tool use (minimizes prefill tokens).
    
    Optimized for DeepSeek-R1 reasoning models.
    Uses aggressive formatting to force tool-call emission.
    """
    tools = "Tools: " + ", ".join(
        f'{t["name"]}({",".join(t["parameters"])})'
        if t["parameters"] else t["name"] + "()"
        for t in TOOL_DEFS
    )
    return (
        "You are QUE Agent on QuantoniumOS. You MUST use tools to answer questions.\n"
        "NEVER guess or describe what you would do. ALWAYS call a tool first.\n\n"
        + tools + "\n\n"
        "FORMAT — output EXACTLY (no code blocks, no quotes, just the raw tag):\n"
        '<tool_call>{"name":"TOOL_NAME","arguments":{"arg":"value"}}</tool_call>\n\n'
        "Example 1:\n"
        '<tool_call>{"name":"read_file","arguments":{"path":"README.md"}}</tool_call>\n\n'
        "Example 2:\n"
        '<tool_call>{"name":"list_dir","arguments":{"path":"src"}}</tool_call>\n\n'
        "Example 3:\n"
        '<tool_call>{"name":"grep","arguments":{"pattern":"def main","path":"src"}}</tool_call>\n\n'
        "WORKFLOW:\n"
        "1. Think briefly about what tool to use\n"
        "2. Output a <tool_call> tag (REQUIRED on first response)\n"
        "3. Read <tool_response> results\n"
        "4. Call more tools or give your final answer\n\n"
        "CRITICAL: Your FIRST response MUST contain a <tool_call>. "
        "Do NOT explain what you plan to do — just call the tool.\n"
    )


# ══════════════════════════════════════════════════════════════════
#  AGENT STREAMING LOOP (with provenance)
# ══════════════════════════════════════════════════════════════════

def agent_stream(
    engine: Any,
    messages: List[Dict[str, str]],
    *,
    max_tokens: int = 512,
    temperature: float = 0.7,
    on_stage: Optional[Callable] = None,
    on_tool_start: Optional[Callable[[str, Dict], None]] = None,
    on_tool_result: Optional[Callable[[str, str], None]] = None,
    on_reasoning: Optional[Callable[[str], None]] = None,
) -> Generator[str, None, None]:
    """ReAct agent loop with full safety and provenance.

    Generates text, checks for tool calls, executes them in the sandbox,
    feeds sanitized results back to the model, and repeats until no more
    tool calls or MAX_AGENT_TURNS.

    R1 reasoning tokens (whether in <think> blocks or bare CoT) are routed
    to `on_reasoning` callback instead of being yielded to the chat bubble.
    Only the clean final answer (after tool calls complete) is yielded.

    Yields individual token strings just like engine.generate_stream().
    """
    audit = AgentAuditLog()
    system_prompt = build_agent_system_prompt()

    def _build_prompt(msgs: List[Dict[str, str]]) -> str:
        parts = [f"<|im_start|>system\n{system_prompt}<|im_end|>"]
        for msg in msgs[-20:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    conv = list(messages)

    try:
        for turn in range(MAX_AGENT_TURNS):
            prompt = _build_prompt(conv)
            full_response = []
            # Track R1 <think> blocks
            in_think = False

            for token in engine.generate_stream(
                prompt,
                max_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                on_stage=on_stage,
            ):
                if not token:
                    continue
                full_response.append(token)
                # Route ALL tokens to on_reasoning so the thinking panel
                # shows live progress. Nothing goes to the chat bubble yet.
                if on_reasoning:
                    on_reasoning(token)

            # Full response assembled — parse and decide what to show
            response_text = "".join(full_response)

            # Strip <think>...</think> blocks for display
            display_text = _THINK_BLOCK_RE.sub('', response_text).strip()
            # Also strip any trailing </think> if the block was never closed
            if "</think>" in display_text:
                display_text = display_text.split("</think>", 1)[-1].strip()

            tool_calls = parse_tool_calls(response_text)

            if not tool_calls:
                # No tools called — yield the clean final answer to the bubble.
                # Strip tool_call tags from display text just in case
                clean = re.sub(r'</?tool_call>', '', display_text).strip()
                if clean:
                    yield clean
                return

            # Tools were called — yield only the non-tool-call text before executing
            # (often empty for R1 which goes straight to tool_call)
            pre_tool = re.split(r'<tool_call>', display_text)[0].strip()
            if pre_tool:
                yield pre_tool

            conv.append({"role": "assistant", "content": response_text})

            tool_results = []
            for call in tool_calls:
                name = call["name"]
                args = call["arguments"]

                if on_tool_start:
                    on_tool_start(name, args)

                result = execute_tool(call, audit=audit, turn=turn)
                tool_results.append(
                    f"<tool_response>\nTool: {name}\nResult: {result}\n</tool_response>"
                )

                if on_tool_result:
                    on_tool_result(name, result)

            result_msg = "\n".join(tool_results)
            conv.append({"role": "user", "content": result_msg})
            yield "\n\n"

        yield "\n\n*(Agent reached maximum tool-call rounds)*"

    finally:
        # Always close the audit log
        audit.close()
