# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""Minimal Ollama client (local OSS model server).

Ollama runs a local HTTP server, typically at:
  http://127.0.0.1:11434

This client is intentionally tiny and dependency-free.

Env vars (optional)
- QUANTONIUM_OLLAMA_BASE_URL (default: http://127.0.0.1:11434)
- QUANTONIUM_OLLAMA_MODEL (default: llama3.2:3b)

Notes
- This is for local inference only. It does not download models itself.
- Use `ollama pull <model>` outside Python to fetch weights.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple


def _base_url() -> str:
    return os.getenv("QUANTONIUM_OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")


def _default_model() -> str:
    return os.getenv("QUANTONIUM_OLLAMA_MODEL", "qwen2.5-coder:3b")


def _http_post_json(url: str, payload: Dict[str, Any], timeout_s: float = 25.0) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    # Ollama sometimes streams newline-delimited JSON; accept either.
    raw = raw.strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        return json.loads(raw)
    last_obj: Optional[Dict[str, Any]] = None
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            last_obj = json.loads(line)
        except Exception:
            continue
    return last_obj or {}


def ollama_chat(
    user_text: str,
    history: Optional[List[Tuple[str, str]]] = None,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> str:
    """Chat via Ollama /api/chat."""

    msgs: List[Dict[str, str]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt.strip()})

    if history:
        for u, a in history[-6:]:
            if u:
                msgs.append({"role": "user", "content": u.strip()})
            if a:
                msgs.append({"role": "assistant", "content": a.strip()})

    msgs.append({"role": "user", "content": user_text.strip()})

    payload: Dict[str, Any] = {
        "model": model or _default_model(),
        "messages": msgs,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            # Ollama doesn't guarantee max tokens option for every model backend,
            # but it's accepted by many.
            "num_predict": int(max_tokens),
        },
    }

    out = _http_post_json(f"{_base_url()}/api/chat", payload)
    msg = out.get("message") or {}
    content = msg.get("content")
    return (content or "").strip()


def ollama_chat_stream(
    user_text: str,
    on_chunk: Callable[[str], None],
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 256,
    timeout_s: float = 180.0,
) -> str:
    """Streaming variant of ollama_chat.

    Calls on_chunk(delta) for every incoming token/fragment and returns the
    full concatenated response at the end. Used by Oracle discourse so the
    UI can render partial answers as they arrive instead of blocking.
    """
    msgs: List[Dict[str, str]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt.strip()})
    msgs.append({"role": "user", "content": user_text.strip()})

    payload: Dict[str, Any] = {
        "model": model or _default_model(),
        "messages": msgs,
        "stream": True,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        },
    }

    url = f"{_base_url()}/api/chat"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    full: List[str] = []
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            msg = obj.get("message") or {}
            delta = msg.get("content") or ""
            if delta:
                full.append(delta)
                try:
                    on_chunk(delta)
                except Exception:
                    pass
            if obj.get("done"):
                break
    return "".join(full).strip()


def ollama_tags(timeout_s: float = 10.0) -> Dict[str, Any]:
    """Return model list metadata from /api/tags (if available)."""
    url = f"{_base_url()}/api/tags"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)
