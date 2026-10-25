# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""Local-only tools for an agentic chat loop.

Deliberately restricted:
- No network access.
- No arbitrary shell execution.
- File reads are limited to within the repo root.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[2]


def _within_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(REPO_ROOT)
        return True
    except Exception:
        return False


def list_files(rel_dir: str = ".", max_items: int = 200) -> tuple[bool, str]:
    base = (REPO_ROOT / rel_dir).resolve()
    if not _within_repo(base):
        return False, "Path is outside repo."
    if not base.exists():
        return False, "Path does not exist."
    if not base.is_dir():
        return False, "Path is not a directory."

    out: List[str] = []
    for p in base.rglob("*"):
        if len(out) >= max_items:
            break
        if p.is_file():
            rel = p.resolve().relative_to(REPO_ROOT)
            out.append(str(rel).replace("\\", "/"))
    return True, "\n".join(out)


def read_file(rel_path: str, max_chars: int = 8000) -> tuple[bool, str]:
    p = (REPO_ROOT / rel_path).resolve()
    if not _within_repo(p):
        return False, "Path is outside repo."
    if not p.exists() or not p.is_file():
        return False, "File not found."

    # Avoid dumping huge binaries.
    try:
        size = p.stat().st_size
    except Exception:
        size = None
    if size is not None and size > 5_000_000:
        return False, f"File too large to read here ({size} bytes)."

    try:
        data = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return False, f"Read failed: {type(e).__name__}: {e}"
    if len(data) > max_chars:
        data = data[:max_chars] + "\n... (truncated)\n"
    return True, data


def search_repo(query: str, max_results: int = 50) -> tuple[bool, str]:
    query = (query or "").strip()
    if not query:
        return False, "Empty query."

    # Prefer ripgrep if available.
    rg = "rg.exe" if os.name == "nt" else "rg"
    try:
        proc = subprocess.run(
            [rg, "-n", "--max-count", str(max_results), query, str(REPO_ROOT)],
            capture_output=True,
            text=True,
            check=False,
        )
        out = (proc.stdout or proc.stderr or "").strip()
        return True, out if out else "No matches."
    except FileNotFoundError:
        return False, "ripgrep (rg) not installed."
    except Exception as e:
        return False, f"Search failed: {type(e).__name__}: {e}"
