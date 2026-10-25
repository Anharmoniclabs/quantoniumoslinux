# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""Training guardrails for local corpora.

Goal: prevent accidental fine-tuning on secrets/PII or on strongly tool-instructional
content that would push the assistant toward agentic behavior.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


@dataclasses.dataclass(frozen=True)
class GuardrailPolicy:
    drop_secrets: bool = True
    redact_pii: bool = True
    drop_tool_instructions: bool = True
    max_chars: int = 32_000


@dataclasses.dataclass
class GuardrailStats:
    rows_in: int = 0
    rows_out: int = 0
    dropped_secrets: int = 0
    dropped_tool_instructions: int = 0
    truncated: int = 0
    redacted_emails: int = 0
    redacted_phones: int = 0
    redacted_ssn: int = 0
    redacted_private_keys: int = 0


_RE_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_RE_PHONE = re.compile(r"\b(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b")
_RE_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

_RE_PRIVATE_KEY_BLOCK = re.compile(r"-----BEGIN [A-Z0-9 ]+ PRIVATE KEY-----", re.IGNORECASE)
_RE_AWS_KEY = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_RE_GITHUB_TOKEN = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")
_RE_OPENAI_KEY = re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")

_RE_TOOLISH = re.compile(
    r"\b("
    r"run\s+this\s+command|execute\s+this|open\s+a\s+terminal|powershell|cmd\.exe|bash|zsh|"
    r"sudo\s+|apt(-get)?\s+install|brew\s+install|pip\s+install|conda\s+install|"
    r"curl\s+https?://|wget\s+https?://|git\s+clone|Invoke-WebRequest|Start-Process"
    r")\b",
    re.IGNORECASE,
)


def _iter_jsonl(path: str) -> Iterator[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def sanitize_text(text: str, policy: GuardrailPolicy, stats: GuardrailStats) -> Optional[str]:
    raw = (text or "").strip()
    if not raw:
        return None

    if policy.drop_secrets:
        if (
            _RE_PRIVATE_KEY_BLOCK.search(raw)
            or _RE_AWS_KEY.search(raw)
            or _RE_GITHUB_TOKEN.search(raw)
            or _RE_OPENAI_KEY.search(raw)
        ):
            stats.dropped_secrets += 1
            return None

    if policy.drop_tool_instructions and _RE_TOOLISH.search(raw):
        stats.dropped_tool_instructions += 1
        return None

    if policy.redact_pii:
        raw, n = _RE_EMAIL.subn("[REDACTED_EMAIL]", raw)
        stats.redacted_emails += n

        raw, n = _RE_PHONE.subn("[REDACTED_PHONE]", raw)
        stats.redacted_phones += n

        raw, n = _RE_SSN.subn("[REDACTED_SSN]", raw)
        stats.redacted_ssn += n

        raw, n = _RE_PRIVATE_KEY_BLOCK.subn("[REDACTED_PRIVATE_KEY_BLOCK]", raw)
        stats.redacted_private_keys += n

    if policy.max_chars > 0 and len(raw) > policy.max_chars:
        stats.truncated += 1
        raw = raw[: policy.max_chars].rstrip()

    return raw


def sanitize_corpus_rows(
    rows: Iterable[Dict[str, object]],
    *,
    policy: GuardrailPolicy,
) -> Tuple[List[Dict[str, object]], GuardrailStats]:
    out: List[Dict[str, object]] = []
    stats = GuardrailStats()
    for row in rows:
        stats.rows_in += 1
        text = row.get("text")
        if not isinstance(text, str):
            continue
        cleaned = sanitize_text(text, policy, stats)
        if cleaned is None:
            continue
        out.append({**row, "text": cleaned})
        stats.rows_out += 1
    return out, stats


def sanitize_corpus_jsonl(
    in_path: str,
    out_path: str,
    *,
    policy: GuardrailPolicy,
    report_path: Optional[str] = None,
) -> GuardrailStats:
    rows = list(_iter_jsonl(in_path))
    cleaned, stats = sanitize_corpus_rows(rows, policy=policy)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in cleaned:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        payload = dataclasses.asdict(stats)
        payload["in_path"] = os.path.normpath(in_path)
        payload["out_path"] = os.path.normpath(out_path)
        payload["policy"] = dataclasses.asdict(policy)
        Path(report_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return stats


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--no_drop_secrets", action="store_true")
    ap.add_argument("--no_redact_pii", action="store_true")
    ap.add_argument("--no_drop_tool_instructions", action="store_true")
    ap.add_argument("--max_chars", type=int, default=32_000)
    args = ap.parse_args()

    policy = GuardrailPolicy(
        drop_secrets=not args.no_drop_secrets,
        redact_pii=not args.no_redact_pii,
        drop_tool_instructions=not args.no_drop_tool_instructions,
        max_chars=max(0, int(args.max_chars)),
    )
    stats = sanitize_corpus_jsonl(
        args.in_path,
        args.out_path,
        policy=policy,
        report_path=args.report,
    )
    print(json.dumps(dataclasses.asdict(stats), indent=2))


if __name__ == "__main__":
    main()

