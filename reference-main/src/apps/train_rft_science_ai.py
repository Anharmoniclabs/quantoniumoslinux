# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""End-to-end pipeline for a stronger local chat model:
1) Build open-science corpus from arXiv.
2) Optionally mix with local repo/domain corpus.
3) Train LoRA with RFT pre-stage compression.

This wraps existing scripts in this repository and keeps outputs reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.apps.training_guardrails import GuardrailPolicy, sanitize_corpus_jsonl


def run_cmd(cmd: List[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
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


def merge_corpora(paths: List[Path], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: Set[str] = set()
    n = 0
    with out_path.open("w", encoding="utf-8") as out:
        for path in paths:
            if not path.exists():
                continue
            for row in iter_jsonl(path):
                text = (row.get("text") or "").strip()
                if not text:
                    continue
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--local_only", action="store_true")
    ap.add_argument("--skip_scrape", action="store_true")
    ap.add_argument("--skip_repo_corpus", action="store_true")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--space_audit", action="store_true")

    ap.add_argument("--science_out", default="ai/training/datasets/open_science_arxiv.jsonl")
    ap.add_argument("--science_meta_out", default="ai/training/datasets/open_science_arxiv_meta.jsonl")
    ap.add_argument("--science_categories", default="cs.AI,cs.LG,cs.CL,stat.ML,cs.CR,quant-ph,eess.SP,math.NA")
    ap.add_argument(
        "--science_terms",
        default="all:quantum OR all:machine learning OR all:signal processing OR all:cryptography",
    )
    ap.add_argument("--science_max_results", type=int, default=1200)
    ap.add_argument("--science_since_days", type=int, default=3650)
    ap.add_argument("--science_require_license", action="store_true")

    ap.add_argument("--repo_corpus_out", default="ai/training/datasets/repo_rft_corpus.jsonl")
    ap.add_argument("--repo_max_files", type=int, default=1200)
    ap.add_argument("--repo_profile", default="rft", choices=["default", "rft"])

    ap.add_argument("--merged_out", default="ai/training/datasets/science_plus_repo_corpus.jsonl")
    ap.add_argument("--merged_sanitized_out", default="ai/training/datasets/science_plus_repo_corpus.sanitized.jsonl")
    ap.add_argument("--adapter_out", default="ai/training/models/science_rft_lora")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_seq_len", type=int, default=768)

    ap.add_argument("--rft_layer_limit", type=int, default=24)
    ap.add_argument("--rft_entropy_threshold", type=float, default=0.40)
    ap.add_argument("--rft_keep_ratio", type=float, default=0.30)
    ap.add_argument("--rft_max_error", type=float, default=0.08)
    ap.add_argument("--rft_report", default="ai/training/reports/rft_prestage_science.json")
    ap.add_argument("--train_report", default="ai/training/reports/trainability_science.json")
    ap.add_argument("--space_audit_model", default=None, help="Model id/path for memory-space audit; defaults to --model.")
    ap.add_argument("--space_audit_report", default="ai/training/reports/rftmw_space_audit.json")
    ap.add_argument("--space_audit_layer_limit", type=int, default=24)
    ap.add_argument("--space_budgets_gb", default="8,16,24,32,48,64,96,128")
    ap.add_argument("--no_guardrails", action="store_true")
    ap.add_argument("--guardrails_report", default="ai/training/reports/guardrails_science.json")
    ap.add_argument("--guardrails_max_chars", type=int, default=32_000)
    args = ap.parse_args()

    science_out = REPO_ROOT / args.science_out
    science_meta_out = REPO_ROOT / args.science_meta_out
    repo_corpus_out = REPO_ROOT / args.repo_corpus_out
    merged_out = REPO_ROOT / args.merged_out

    if not args.skip_scrape:
        scrape_cmd = [
            sys.executable,
            "src/apps/build_open_science_corpus.py",
            "--out",
            args.science_out,
            "--meta_out",
            args.science_meta_out,
            "--categories",
            args.science_categories,
            "--terms",
            args.science_terms,
            "--max_results",
            str(args.science_max_results),
            "--since_days",
            str(args.science_since_days),
        ]
        if args.science_require_license:
            scrape_cmd.append("--require_license")
        run_cmd(scrape_cmd)
    else:
        print("Skipping science scrape.")

    corpus_parts = [science_out]

    if not args.skip_repo_corpus:
        repo_cmd = [
            sys.executable,
            "src/apps/build_domain_corpus.py",
            "--out",
            args.repo_corpus_out,
            "--profile",
            args.repo_profile,
            "--max_files",
            str(args.repo_max_files),
        ]
        run_cmd(repo_cmd)
        corpus_parts.append(repo_corpus_out)
    else:
        print("Skipping repo corpus.")

    merged_rows = merge_corpora(corpus_parts, merged_out)
    print(f"Merged corpus rows: {merged_rows} -> {merged_out}")

    train_corpus = merged_out
    if not args.no_guardrails:
        policy = GuardrailPolicy(
            drop_secrets=True,
            redact_pii=True,
            drop_tool_instructions=True,
            max_chars=max(0, int(args.guardrails_max_chars)),
        )
        sanitized_out = REPO_ROOT / args.merged_sanitized_out
        sanitize_corpus_jsonl(
            str(merged_out),
            str(sanitized_out),
            policy=policy,
            report_path=str(REPO_ROOT / args.guardrails_report),
        )
        train_corpus = sanitized_out
        print(f"Sanitized merged corpus -> {sanitized_out}")

    if args.skip_train:
        print("Skipping training.")
    else:
        train_cmd = [
            sys.executable,
            "src/apps/train_local_chat_lora.py",
            "--model",
            args.model,
            "--corpus",
            str(train_corpus),
            "--out",
            args.adapter_out,
            "--epochs",
            str(args.epochs),
            "--max_steps",
            str(args.max_steps),
            "--batch",
            str(args.batch),
            "--grad_accum",
            str(args.grad_accum),
            "--lr",
            str(args.lr),
            "--max_seq_len",
            str(args.max_seq_len),
            "--rft_prestage",
            "--rft_layer_limit",
            str(args.rft_layer_limit),
            "--rft_entropy_threshold",
            str(args.rft_entropy_threshold),
            "--rft_keep_ratio",
            str(args.rft_keep_ratio),
            "--rft_max_error",
            str(args.rft_max_error),
            "--rft_report",
            args.rft_report,
            "--train_report",
            args.train_report,
        ]
        if args.no_guardrails:
            train_cmd.append("--no_guardrails")
        if args.local_only:
            train_cmd.append("--local_only")
        run_cmd(train_cmd)

    if args.space_audit:
        audit_model = args.space_audit_model or args.model
        audit_cmd = [
            sys.executable,
            "src/apps/rftmw_space_audit.py",
            "--model",
            audit_model,
            "--out",
            args.space_audit_report,
            "--layer_limit",
            str(args.space_audit_layer_limit),
            "--entropy_threshold",
            str(args.rft_entropy_threshold),
            "--keep_ratio",
            str(args.rft_keep_ratio),
            "--max_rft_error",
            str(args.rft_max_error),
            "--memory_budgets_gb",
            args.space_budgets_gb,
        ]
        if args.local_only:
            audit_cmd.append("--local_only")
        run_cmd(audit_cmd)
    else:
        print("Skipping space audit.")

    print("Pipeline complete.")
    if not args.skip_train:
        print("Use the adapter with:")
        print(f"  set QUANTONIUM_MODEL_ID={args.model}")
        print(f"  set QUANTONIUM_LORA_PATH={args.adapter_out}")
        print("  set QUANTONIUM_LOCAL_BACKEND=rftmw")
        print("  set QUANTONIUM_LOCAL_LLM=1")
        print("  python src/apps/qshll_chatbox.py")


if __name__ == "__main__":
    main()
