#!/usr/bin/env python3
"""Reproducible side-by-side output check for two PhiState-MoE checkpoints.

Uses the SAME tokenizer, prompts, decode lengths, and sampling settings.
Does not use sealed test examples; generation quality is not inferred from BPT.
"""
import argparse
import ast
import json
from pathlib import Path

import sentencepiece as spm
import torch

from moe_train import ckpt_load

PROMPTS = [
    {"id": "python_fibonacci", "kind": "python", "prompt": "def fibonacci(n):\n"},
    {"id": "python_add", "kind": "python", "prompt": "def add(a, b):\n"},
    {"id": "operating_system", "kind": "text", "prompt": "The purpose of an operating system is"},
    {"id": "qa_ram_vs_storage", "kind": "text", "prompt": "User: What is the difference between RAM and storage?\nAssistant: "},
]


def generate(model, tokenizer, prompt, length, mode, seed, temperature=0.8, top_k=32):
    prompt_ids = tokenizer.encode(prompt, out_type=int)
    if not prompt_ids:
        raise ValueError("Tokenizer produced no prompt tokens")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    cache = None
    with torch.inference_mode():
        for token in prompt_ids:
            logits, cache = model.step(torch.tensor([[token]], dtype=torch.long), cache)
        produced = []
        for _ in range(length):
            next_logits = logits[0, -1] / max(temperature, 0.05)
            if mode == "greedy":
                token = int(torch.argmax(next_logits))
            else:
                values, indices = torch.topk(next_logits, min(top_k, next_logits.numel()))
                probabilities = torch.softmax(values, dim=-1)
                token = int(indices[torch.multinomial(probabilities, 1, generator=generator)])
            produced.append(token)
            logits, cache = model.step(torch.tensor([[token]], dtype=torch.long), cache)
    full_text = tokenizer.decode(prompt_ids + produced)
    unique_fraction = len(set(produced)) / max(1, len(produced))
    longest_identical_run = 0
    current_run = 0
    previous = None
    for token in produced:
        current_run = current_run + 1 if token == previous else 1
        longest_identical_run = max(longest_identical_run, current_run)
        previous = token
    return full_text, round(unique_fraction, 4), longest_identical_run


def python_syntax(text):
    try:
        ast.parse(text)
        return True, None
    except SyntaxError as error:
        return False, f"{error.msg} (line {error.lineno})"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="10,000-step checkpoint")
    parser.add_argument("--candidate", required=True, help="10,250-step checkpoint")
    parser.add_argument("--tokenizer", required=True, help="Original tokenizer.model")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--new-tokens", type=int, default=80)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    if args.new_tokens < 1 or args.threads < 1:
        parser.error("new-tokens and threads must be positive")
    torch.set_num_threads(args.threads)
    tokenizer = spm.SentencePieceProcessor(model_file=str(Path(args.tokenizer).resolve()))
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "comparison": "Identical prompt / deterministic seed / decoding configuration",
        "seed": args.seed,
        "new_tokens": args.new_tokens,
        "sampling": {"temperature": 0.8, "top_k": 32},
        "limitations": (
            "A small prompt sample, not an independently graded code or QA benchmark. "
            "Successful Python parsing is only a syntax check, not functional correctness. "
            "Truncation can make an otherwise promising function fail to parse."
        ),
        "checkpoints": {},
        "results": [],
    }
    fingerprint = config = None
    for label, path in (("10k", args.baseline), ("10250", args.candidate)):
        model, checkpoint = ckpt_load(str(path))
        if fingerprint is None:
            fingerprint = checkpoint["data_fingerprint"]
            config = checkpoint["config"]
        if checkpoint["data_fingerprint"] != fingerprint or checkpoint["config"] != config:
            raise RuntimeError("Checkpoint corpus fingerprint or architecture mismatch")
        if model.cfg.vocab != tokenizer.vocab_size():
            raise RuntimeError("Checkpoint vocabulary does not match supplied tokenizer")
        model.eval()
        report["checkpoints"][label] = {
            "path": str(Path(path).resolve()),
            "step": checkpoint["step"],
            "best_validation_bpt": checkpoint.get("best_validation_bpt"),
            "corpus_fingerprint": fingerprint,
        }
        del checkpoint
        for case in PROMPTS:
            for mode in ("greedy", "sample"):
                generated, diversity, longest_run = generate(
                    model, tokenizer, case["prompt"], args.new_tokens,
                    mode=mode, seed=args.seed
                )
                syntax_ok, syntax_error = (
                    python_syntax(generated) if case["kind"] == "python" else (None, None)
                )
                result = {
                    "model": label, "case": case["id"], "kind": case["kind"],
                    "prompt": case["prompt"], "decoding": mode,
                    "full_generated_text": generated,
                    "unique_generated_token_fraction": diversity,
                    "longest_identical_token_run": longest_run,
                    "python_source_parses": syntax_ok,
                    "syntax_error": syntax_error,
                }
                report["results"].append(result)
                print(f"{label} | {case['id']} | {mode} | diversity={diversity} | "
                      f"max_repeat={longest_run} | python_parses={syntax_ok}", flush=True)
        del model
    report_path = output / "quality_10k_vs_10250.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    lines = ["# PhiState: 10,000 vs. 10,250 steps", "",
             "Same tokenizer, prompt, seed and decoding configuration for each comparison.",
             f"Generated-token limit: {args.new_tokens}; sampling top-k=32, temperature=0.8.", "",
             "| Checkpoint | Step | Recorded best validation BPT |", "|---|---:|---:|"]
    for label, meta in report["checkpoints"].items():
        lines.append(f"| {label} | {meta['step']} | {meta['best_validation_bpt']} |")
    lines += ["", "The BPT values are recorded checkpoint metrics, not a new evaluation.",
              "Python syntax parsing does not establish code correctness. Full outputs follow.", ""]
    for case in PROMPTS:
        for mode in ("greedy", "sample"):
            lines.append(f"## {case['id']} — {mode}")
            lines.append("")
            for label in ("10k", "10250"):
                row = next(r for r in report["results"] if
                           r["model"] == label and r["case"] == case["id"] and r["decoding"] == mode)
                lines.append(f"**{label}** (unique token fraction {row['unique_generated_token_fraction']}, "
                             f"longest repeated-token run {row['longest_identical_token_run']}; "
                             f"Python parses: {row['python_source_parses']})")
                lines += ["", "~~~~text", row["full_generated_text"], "~~~~", ""]
    (output / "quality_10k_vs_10250.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print("COMPARISON SAVED:", report_path, flush=True)
    print("READABLE REPORT:", output / "quality_10k_vs_10250.md", flush=True)


if __name__ == "__main__":
    main()
