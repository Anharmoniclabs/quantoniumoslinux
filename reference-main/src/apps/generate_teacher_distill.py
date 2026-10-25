# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate distillation targets from an Ollama teacher model.

This script is intentionally separate from training so long teacher runs are
resumable and leave a progress file.  It uses an open-weight Ollama model as
the teacher and writes JSONL samples incrementally.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_PROMPTS = [
    "Solve this carefully and give a concise final answer: What is 2+2?",
    "Explain the difference between state-space attention and softmax attention.",
    "Write a small C function that clamps a float between two bounds.",
    "Explain why a recurrent SSM can behave like persistent memory.",
    "Debug this symptom: a language model emits random CJK characters after an SSM layer change.",
    "Describe how to benchmark tokens per second for a local inference engine.",
    "Explain what a tokenizer special token bug looks like in generation.",
    "Give a concise checklist for validating a native C inference path.",
]


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")


def load_prompts(source: Path | None, max_samples: int) -> List[str]:
    prompts: List[str] = []
    if source and source.exists():
        with source.open("r", encoding="utf-8") as f:
            for line in f:
                if len(prompts) >= max_samples:
                    break
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = obj.get("prompt") or obj.get("input") or obj.get("question")
                if prompt and prompt not in prompts:
                    prompts.append(str(prompt).strip())

    for prompt in DEFAULT_PROMPTS:
        if len(prompts) >= max_samples:
            break
        if prompt not in prompts:
            prompts.append(prompt)
    return prompts[:max_samples]


def existing_prompts(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = obj.get("prompt") or obj.get("input")
            if prompt:
                seen.add(str(prompt))
    return seen


def strip_reasoning(text: str) -> str:
    cleaned = text.strip()
    lower = cleaned.lower()
    if "</think>" in lower:
        idx = lower.rfind("</think>")
        tail = cleaned[idx + len("</think>") :].strip()
        if tail:
            cleaned = tail
    cleaned = cleaned.replace("<think>", "").replace("</think>", "").strip()
    if cleaned.startswith("**Final"):
        cleaned = cleaned.split("\n", 1)[-1].strip() if "\n" in cleaned else cleaned
    return cleaned


def teacher_response_text(result: Dict) -> str:
    response = result.get("response")
    if isinstance(response, str) and response.strip():
        return response

    message = result.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        thinking = message.get("thinking")
        if isinstance(thinking, str) and thinking.strip():
            return thinking

    thinking = result.get("thinking")
    if isinstance(thinking, str) and thinking.strip():
        return thinking
    return ""


def ollama_generate(
    teacher: str,
    prompt: str,
    num_predict: int,
    timeout: int,
    keep_alive: str,
    num_thread: int,
    num_gpu: int | None,
) -> Dict:
    system = (
        "You are the teacher for a local open-weight student model. "
        "Return a high-quality training target. Do not include hidden reasoning, "
        "chain-of-thought, analysis, scratchwork, or <think> blocks. "
        "Be correct, concise, and concrete."
    )
    user = f"Task:\n{prompt}\n\nTarget answer:"
    options = {
        "num_predict": num_predict,
        "temperature": 0.2,
        "top_p": 0.9,
        "num_ctx": 4096,
    }
    if num_thread > 0:
        options["num_thread"] = num_thread
    if num_gpu is not None:
        options["num_gpu"] = num_gpu

    payload = {
        "model": teacher,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "keep_alive": keep_alive,
        "think": False,
        "options": options,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def write_progress(path: Path, state: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate teacher distillation JSONL")
    ap.add_argument("--teacher", default="deepseek-r1:70b")
    ap.add_argument("--source", default="ai/training/datasets/combined_training.jsonl")
    ap.add_argument("--output", default="ai/training/datasets/distill_deepseek_r1_70b.jsonl")
    ap.add_argument("--max-samples", type=int, default=25)
    ap.add_argument("--num-predict", type=int, default=192)
    ap.add_argument("--num-thread", type=int, default=os.cpu_count() or 0)
    ap.add_argument("--num-gpu", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--keep-alive", default="30m")
    args = ap.parse_args()

    source = Path(args.source) if args.source else None
    output = Path(args.output)
    progress = output.with_suffix(output.suffix + ".progress.json")
    prompts = load_prompts(source, args.max_samples)
    done_prompts = existing_prompts(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    failures = 0
    completed = len(done_prompts)
    print(f"[distill] teacher={args.teacher}")
    print(f"[distill] output={output}")
    print(f"[distill] prompts={len(prompts)} already_done={completed}")

    with output.open("a", encoding="utf-8") as out:
        for index, prompt in enumerate(prompts, start=1):
            if prompt in done_prompts:
                continue
            t0 = time.time()
            print(f"[distill] {index}/{len(prompts)} {prompt[:80]}", flush=True)
            try:
                try:
                    result = ollama_generate(
                        args.teacher,
                        prompt,
                        args.num_predict,
                        args.timeout,
                        args.keep_alive,
                        args.num_thread,
                        args.num_gpu,
                    )
                except urllib.error.HTTPError:
                    if args.num_gpu is None:
                        raise
                    print("[distill] gpu offload request failed; retrying CPU path", flush=True)
                    result = ollama_generate(
                        args.teacher,
                        prompt,
                        args.num_predict,
                        args.timeout,
                        args.keep_alive,
                        args.num_thread,
                        None,
                    )
                target = strip_reasoning(teacher_response_text(result))
                if not target:
                    keys = ",".join(sorted(result.keys()))
                    reason = result.get("done_reason", "unknown")
                    raise RuntimeError(f"empty teacher response keys={keys} done_reason={reason}")
                row = {
                    "prompt": prompt,
                    "target": target,
                    "teacher": args.teacher,
                    "source": str(source) if source else "curated",
                    "teacher_eval_count": result.get("eval_count"),
                    "teacher_eval_duration": result.get("eval_duration"),
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                completed += 1
                done_prompts.add(prompt)
            except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
                failures += 1
                print(f"[distill] error: {exc}", flush=True)

            elapsed = time.time() - start
            state = {
                "teacher": args.teacher,
                "output": str(output),
                "completed": completed,
                "total": len(prompts),
                "failures": failures,
                "elapsed_sec": elapsed,
                "avg_sec_per_completed": elapsed / max(completed, 1),
                "last_prompt_index": index,
                "last_prompt": prompt,
                "last_step_sec": time.time() - t0,
            }
            write_progress(progress, state)

    print(f"[distill] done completed={completed} failures={failures}")


if __name__ == "__main__":
    main()
