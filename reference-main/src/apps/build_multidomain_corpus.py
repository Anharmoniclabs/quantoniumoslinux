# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""Build a safer multi-domain corpus for LoRA training.

This merges:
- Your chatbox transcripts (supervised user/assistant pairs).
- A repo/domain corpus (docs + code text).
- Optional extra JSONL corpora you provide locally ({"text": "..."} rows).

Outputs a single JSONL suitable for `train_local_chat_lora.py --corpus`.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.apps.training_guardrails import GuardrailPolicy, sanitize_corpus_rows


def iter_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
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


def load_chat_pairs(log_glob: str, max_pairs: int) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for path in sorted(glob.glob(log_glob)):
        last_user = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                t = obj.get("type")
                if t == "user":
                    last_user = obj.get("text", "")
                elif t == "assistant" and last_user is not None:
                    pairs.append((str(last_user), str(obj.get("text", ""))))
                    last_user = None
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def format_supervised_pair(*, user_text: str, assistant_text: str, model_id: str) -> str:
    system_prompt = (
        "You are a helpful, non-agentic assistant. "
        "Refuse downloads, external network access, and system modification commands."
    )
    mid = (model_id or "").lower()
    if "tinyllama" in mid and "chat" in mid:
        return "\n".join(
            [
                f"<|system|>\n{system_prompt}</s>",
                f"<|user|>\n{user_text.strip()}</s>",
                f"<|assistant|>\n{assistant_text.strip()}</s>",
            ]
        )
    return "\n".join(
        [
            f"System: {system_prompt}",
            f"User: {user_text.strip()}",
            f"Assistant: {assistant_text.strip()}",
        ]
    )


def stable_dedupe_rows(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    seen: Set[str] = set()
    out: List[Dict[str, object]] = []
    for row in rows:
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        out.append(row)
    return out


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def build_repo_corpus(out_path: Path, *, profile: str, max_files: Optional[int]) -> None:
    cmd = [
        sys.executable,
        "src/apps/build_domain_corpus.py",
        "--out",
        str(out_path),
        "--profile",
        profile,
    ]
    if max_files is not None:
        cmd += ["--max_files", str(int(max_files))]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.getenv("QUANTONIUM_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
    ap.add_argument("--out", default="ai/training/datasets/multidomain_corpus.jsonl")

    ap.add_argument("--chat_logs", default="logs/chat_*.jsonl")
    ap.add_argument("--max_pairs", type=int, default=2000)
    ap.add_argument("--include_chat", action="store_true", default=True)
    ap.add_argument("--no_chat", action="store_true", help="Disable chat log ingestion.")

    ap.add_argument("--include_repo", action="store_true", default=True)
    ap.add_argument("--no_repo", action="store_true", help="Disable repo corpus ingestion.")
    ap.add_argument("--repo_profile", default="default", choices=["default", "rft"])
    ap.add_argument("--repo_max_files", type=int, default=1200)

    ap.add_argument(
        "--extra_corpus",
        action="append",
        default=[],
        help="Extra local JSONL corpora to include (repeatable). Must have {'text': ...} rows.",
    )

    ap.add_argument("--no_guardrails", action="store_true")
    ap.add_argument("--guardrails_report", default="ai/training/reports/guardrails_multidomain.json")
    ap.add_argument("--guardrails_max_chars", type=int, default=32_000)
    args = ap.parse_args()

    out_path = (REPO_ROOT / args.out).resolve()
    parts_dir = (REPO_ROOT / "ai" / "training" / "datasets" / "_parts").resolve()
    parts_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []

    if not args.no_chat:
        pairs = load_chat_pairs(args.chat_logs, max_pairs=int(args.max_pairs))
        for u, a in pairs:
            rows.append({"text": format_supervised_pair(user_text=u, assistant_text=a, model_id=args.model), "source": "chat_logs"})
        print(f"Loaded chat pairs: {len(pairs)}")
    else:
        print("Skipping chat logs.")

    if not args.no_repo:
        repo_out = parts_dir / "repo_domain_corpus.jsonl"
        build_repo_corpus(repo_out, profile=args.repo_profile, max_files=int(args.repo_max_files))
        repo_rows = list(iter_jsonl(repo_out))
        rows.extend([{**r, "source": "repo"} for r in repo_rows if isinstance(r.get("text"), str)])
        print(f"Loaded repo corpus rows: {len(repo_rows)}")
    else:
        print("Skipping repo corpus.")

    for p in args.extra_corpus:
        path = (REPO_ROOT / p).resolve() if not os.path.isabs(p) else Path(p)
        if not path.exists():
            print(f"Extra corpus not found: {path}")
            continue
        extra_rows = list(iter_jsonl(path))
        rows.extend([{**r, "source": f"extra:{path.name}"} for r in extra_rows if isinstance(r.get("text"), str)])
        print(f"Loaded extra corpus: {path} rows={len(extra_rows)}")

    rows = stable_dedupe_rows(rows)

    if args.no_guardrails:
        cleaned = rows
        stats = None
    else:
        policy = GuardrailPolicy(
            drop_secrets=True,
            redact_pii=True,
            drop_tool_instructions=True,
            max_chars=max(0, int(args.guardrails_max_chars)),
        )
        cleaned, stats = sanitize_corpus_rows(rows, policy=policy)
        Path(args.guardrails_report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.guardrails_report).write_text(
            json.dumps(
                {
                    "policy": {
                        "drop_secrets": policy.drop_secrets,
                        "redact_pii": policy.redact_pii,
                        "drop_tool_instructions": policy.drop_tool_instructions,
                        "max_chars": policy.max_chars,
                    },
                    "stats": stats.__dict__,
                    "out": os.path.normpath(str(out_path)),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Guardrails report: {args.guardrails_report}")

    n = write_jsonl(out_path, cleaned)
    print(f"Wrote corpus rows: {n} -> {out_path}")


if __name__ == "__main__":
    main()
