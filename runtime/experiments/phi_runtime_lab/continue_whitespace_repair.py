#!/usr/bin/env python3
"""Explicit PhiState corpus migration and resumable, bounded repair training.

Preserves source weights, optimizer, total step and RNG. Records the corpus
change, measures a new validation baseline, and saves into a separate run.
SIGINT/SIGTERM or a STOP file requests a save after the current optimizer step.
No original checkpoint, corpus, or test-split evaluation is modified.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb
import torch
import torch.nn.functional as F

import moe_train
from repair_whitespace_corpus import require, sha, verify_output


def atomic_json(path, content):
    path = Path(path)
    pending = path.with_name(path.name + ".partial")
    pending.write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    pending.replace(path)


@contextmanager
def run_lock(directory):
    # Local flock works even when the destination is a Drive FUSE mount.
    key = hashlib.sha256(str(Path(directory).resolve()).encode()).hexdigest()
    lock = Path(tempfile.gettempdir()) / f"phistate_repair_{key}.lock"
    with lock.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This repair run already has an active worker") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def validate_migration(source, source_manifest, repaired, source_tokenizer, fixed_tokenizer):
    lineage = repaired.get("whitespace_repair", {})
    require(source.get("format") == "PhiStateMoE-v1", "Wrong checkpoint format")
    require(source.get("data_fingerprint") == source_manifest.get("data_fingerprint"),
            "Source checkpoint does not match --source-data")
    original_fp = lineage.get("original_data_fingerprint")
    require(original_fp and original_fp in (
        source_manifest.get("data_fingerprint"),
        source_manifest.get("extension", {}).get("base_data_fingerprint")),
        "Source corpus is unrelated to the repaired original corpus")
    require(sha(source_tokenizer) == lineage.get("original_tokenizer_sha256"),
            "Source tokenizer differs from the original tokenizer used for repair")
    require(sha(fixed_tokenizer) == lineage.get("tokenizer_sha256"), "Repaired tokenizer changed")
    a, b = pb.ModelProto(), pb.ModelProto()
    a.ParseFromString(Path(source_tokenizer).read_bytes())
    b.ParseFromString(Path(fixed_tokenizer).read_bytes())
    require([x.SerializeToString() for x in a.pieces] == [x.SerializeToString() for x in b.pieces],
            "Tokenizer vocabulary / token IDs changed")
    require(source["config"]["vocab"] == repaired["vocab"] == len(b.pieces), "Vocabulary mismatch")
    require(source.get("tokenizer") == repaired.get("tokenizer") == "bpe", "Wrong tokenizer type")
    require(lineage.get("all_documents_roundtrip_exact") and lineage.get("all_original_streams_accounted_for"),
            "Corpus lacks the completed repair proof")
    require(isinstance(source.get("optimizer_state"), dict), "Source has no resumable optimizer state")
    require(isinstance(source.get("train_args"), dict), "Source has no training arguments")
    for key in ("numpy_rng_state", "python_rng_state", "torch_rng_state"):
        require(key in source, f"Source missing {key}; refusing a silent RNG reset")


def generation_report(model, tokenizer, step, new_tokens):
    device = next(model.parameters()).device
    prompts = [("python_add", "code", "def add(a, b):\n"),
               ("python_fibonacci", "code", "def fibonacci(n):\n"),
               ("ram_vs_storage", "general", "User: What is the difference between RAM and storage?\nAssistant: "),
               ("operating_system", "general", "The purpose of an operating system is")]
    model.eval()
    rows = []
    with torch.inference_mode():
        for case, domain, prompt in prompts:
            model_input = f"<|{domain}|>\n" + prompt
            original_ids = tokenizer.encode(model_input)
            for mode in ("greedy", "sample"):
                rng = torch.Generator(device=device).manual_seed(20261027)
                state, produced = None, []
                for token in original_ids:
                    logits, state = model.step(torch.tensor([[token]], device=device), state)
                for _ in range(new_tokens):
                    scores = logits[0, -1].float()
                    require(torch.isfinite(scores).all().item(), "Nonfinite generation logits")
                    if mode == "greedy":
                        token = int(scores.argmax())
                    else:
                        values, choices = torch.topk(scores / 0.8, min(32, scores.numel()))
                        token = int(choices[torch.multinomial(values.softmax(-1), 1, generator=rng)].item())
                    if token == tokenizer.eos_id():
                        break
                    produced.append(token)
                    logits, state = model.step(torch.tensor([[token]], device=device), state)
                decoded = tokenizer.decode(original_ids + produced)
                header = f"<|{domain}|>\n"
                text = decoded[len(header):] if decoded.startswith(header) else decoded
                parses = None
                if domain == "code":
                    try:
                        ast.parse(text)
                        parses = True
                    except (SyntaxError, ValueError):
                        parses = False
                rows.append({"case": case, "mode": mode, "step": step, "text": text,
                             "python_syntax_parses": parses, "domain_marker": f"<|{domain}|>"})
    return {"step": step, "scope": "Fixed small output check; syntax is not code correctness",
            "max_new_tokens": new_tokens, "results": rows}


def run(a):
    source_path, source_data, data_dir, run_dir = map(
        lambda x: Path(x).expanduser().resolve(), (a.checkpoint, a.source_data, a.data_dir, a.run_dir))
    require(source_path.is_file(), f"Missing source checkpoint: {source_path}")
    require(run_dir not in (source_path.parent, source_data, data_dir), "Run must have its own directory")
    with run_lock(run_dir):
        return run_locked(a, source_path, source_data, data_dir, run_dir)


def run_locked(a, source_path, source_data, data_dir, run_dir):
    device = moe_train.resolve_device(a.device)
    moe_train.setup(a.threads, 20261027)
    source_manifest = json.loads((source_data / "manifest.json").read_text())
    manifest = json.loads((data_dir / "manifest.json").read_text())
    lineage = manifest.get("whitespace_repair", {})
    print("Verifying repaired corpus and source checkpoint...", flush=True)
    verify_output(data_dir, lineage.get("original_data_fingerprint"),
                  lineage.get("tokenizer_sha256"), lineage.get("raw_sha256"))
    source_sha = sha(source_path)
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    validate_migration(source, source_manifest, manifest,
                       source_data / "tokenizer.model", data_dir / "tokenizer.model")
    require(source["step"] == a.expected_source_step,
            f"Expected source step {a.expected_source_step}; found {source['step']}")
    require(a.target_steps >= source["step"], "Target is earlier than the source checkpoint")
    run_dir.mkdir(parents=True, exist_ok=True)
    latest, best_path = run_dir / "latest.pt", run_dir / "best.pt"
    stop_file = run_dir / "STOP"
    if stop_file.exists():
        if a.clear_stop:
            stop_file.unlink()
        else:
            print(f"STOP file exists. Use --clear-stop to intentionally resume: {stop_file}", flush=True)
            return
    resuming = latest.is_file()
    if not resuming:
        require(not best_path.exists() and not (run_dir / "migration.json").exists(),
                "Partial run already exists without latest.pt; preserve it and use a new run directory")
    state = torch.load(latest, map_location="cpu", weights_only=True) if resuming else source
    if resuming:
        migration = state.get("corpus_migration", {})
        require(migration.get("source_checkpoint_sha256") == source_sha, "Run has a different parent checkpoint")
        require(state.get("data_fingerprint") == manifest["data_fingerprint"], "Run has a different repaired corpus")
        require(state["config"] == source["config"], "Resumed architecture differs")
        require(a.eval_batches == migration["eval_batches"], "Keep eval-batches fixed within a run")
    else:
        migration = {
            "format": "PhiStateWhitespaceMigration-v1", "source_checkpoint": str(source_path),
            "source_checkpoint_sha256": source_sha, "source_step": source["step"],
            "source_data_fingerprint": source["data_fingerprint"],
            "repaired_data_fingerprint": manifest["data_fingerprint"],
            "tokenizer_sha256": lineage["tokenizer_sha256"], "optimizer_preserved": True,
            "previous_best_validation_bpt": source.get("best_validation_bpt"),
            "eval_batches": a.eval_batches,
            "scope": "Original corpus repaired; expanded-data additions are not included in this phase",
        }
    train_args = dict(state["train_args"])
    for key in ("context", "batch", "accum", "code_share", "optimizer", "lr", "weight_decay", "clip", "eval_batch"):
        require(key in train_args, f"Missing training setting: {key}")
    train_args.update(data_dir=str(data_dir), out=str(best_path), resume=str(latest),
                      device=str(device), steps=a.target_steps, eval_every=a.save_every,
                      eval_batches=a.eval_batches, threads=a.threads, save_optimizer=True)
    args = SimpleNamespace(**train_args)
    _, datasets, _ = moe_train.load_data(str(data_dir))
    model = moe_train.PhiStateMoELM(moe_train.MoEConfig(**state["config"]))
    model.load_state_dict(state["state_dict"], strict=True)
    model = model.to(device)
    optimizer = moe_train.optimizer_for(args, model)
    optimizer.load_state_dict(state["optimizer_state"])
    rng = np.random.default_rng()
    rng.bit_generator.state = state["numpy_rng_state"]
    choose = random.Random()
    choose.setstate(state["python_rng_state"])
    torch.set_rng_state(state["torch_rng_state"].cpu())
    if device.type == "cuda":
        require("cuda_rng_state_all" in state, "CUDA resume requires saved CUDA RNG state")
        require(len(state["cuda_rng_state_all"]) == torch.cuda.device_count(), "CUDA device count differs from checkpoint")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng_state_all"]])
    tokenizer = spm.SentencePieceProcessor(model_file=str(data_dir / "tokenizer.model"))
    step, processed, train_seconds = int(state["step"]), int(state.get("trained_tokens", 0)), float(state.get("train_seconds", 0))
    best = float(state["best_validation_bpt"]) if resuming else float("inf")
    del source, state
    stop_requested = [False]
    def request_stop(signum, frame):
        stop_requested[0] = True
    previous_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    def stopping():
        return stop_requested[0] or stop_file.exists()
    def checkpoint(validation, improved, status):
        payload = {
            "format": "PhiStateMoE-v1", "config": vars(model.cfg), "state_dict": model.state_dict(),
            "step": step, "best_validation_bpt": best, "data_fingerprint": manifest["data_fingerprint"],
            "tokenizer": "bpe", "train_args": train_args, "run_seed": args.seed,
            "trained_tokens": processed, "train_seconds": train_seconds,
            "numpy_rng_state": rng.bit_generator.state, "python_rng_state": choose.getstate(),
            "torch_rng_state": torch.get_rng_state(), "corpus_migration": migration,
            "last_validation": validation,
        }
        if device.type == "cuda":
            payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        # Save latest first so a failed best-file write cannot lose optimizer progress.
        payload["optimizer_state"] = optimizer.state_dict()
        moe_train.atomic_checkpoint(latest, payload)
        saved = torch.load(latest, map_location="cpu", weights_only=True)
        require(saved["step"] == step and saved["data_fingerprint"] == manifest["data_fingerprint"]
                and "optimizer_state" in saved, "Latest checkpoint readback failed")
        del saved
        if improved:
            inference = dict(payload)
            inference.pop("optimizer_state")
            moe_train.atomic_checkpoint(best_path, inference)
        atomic_json(run_dir / "progress.json", {
            "status": status, "saved_step": step, "source_step": migration["source_step"],
            "target_steps": a.target_steps, "best_validation_bpt": best,
            "last_validation": validation, "latest": str(latest), "best": str(best_path),
            "trained_tokens": processed, "data_fingerprint": manifest["data_fingerprint"],
        })
        print(f"SAVED verified latest: step {step} | {latest}", flush=True)
    try:
        if not resuming:
            print("Measuring the 60K weights on REPAIRED validation; old BPT is not comparable...", flush=True)
            validation = moe_train.evaluate_both(model, datasets, args.context, args.eval_batch, a.eval_batches)
            best = validation["val_mean_bpt"]
            require(math.isfinite(best), "Nonfinite repaired baseline")
            migration["repaired_baseline"] = validation
            checkpoint(validation, True, "prepared")
            atomic_json(run_dir / "migration.json", migration)
            print("REPAIRED BASELINE " + json.dumps(validation), flush=True)
        if not (run_dir / "generation_baseline.json").exists() and step == migration["source_step"]:
            atomic_json(run_dir / "generation_baseline.json", generation_report(model, tokenizer, step, a.generation_tokens))
        if a.prepare_only or step >= a.target_steps or stopping():
            print(f"READY / NO NEW STEPS: saved step {step}", flush=True)
            return
        print(json.dumps({"from_step": step, "target_steps": a.target_steps, "device": str(device),
                          "batch": args.batch, "accum": args.accum, "context": args.context,
                          "optimizer": args.optimizer, "lr": optimizer.param_groups[0]["lr"],
                          "save_every": a.save_every, "stop_file": str(stop_file)}), flush=True)
        start = step
        for next_step in range(step + 1, a.target_steps + 1):
            if stopping():
                break
            began = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            lm_sum = 0.0
            for _ in range(args.accum):
                domain = "code" if choose.random() < args.code_share else "general"
                x, y = moe_train.sample_batch(datasets[f"train_{domain}"], args.batch, args.context, rng)
                x, y = x.to(device), y.to(device)
                logits = model(x)
                lm = F.cross_entropy(logits.reshape(-1, model.cfg.vocab), y.reshape(-1))
                penalty, _ = model.router_penalty()
                loss = (lm + penalty) / args.accum
                require(torch.isfinite(loss).item(), f"Nonfinite loss at step {next_step}; last saved checkpoint retained")
                loss.backward()
                lm_sum += float(lm.detach()) / args.accum
                processed += y.numel()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip, error_if_nonfinite=True)
            optimizer.step()
            step = next_step
            train_seconds += time.perf_counter() - began
            if step % a.log_every == 0:
                print(f"TRAIN step {step}/{a.target_steps} | batch BPT {lm_sum / math.log(2):.4f}", flush=True)
            if step % a.save_every == 0 or step == start + 20 or step == a.target_steps or stopping():
                validation = moe_train.evaluate_both(model, datasets, args.context, args.eval_batch, a.eval_batches)
                score = validation["val_mean_bpt"]
                require(math.isfinite(score), "Nonfinite validation; previous saved checkpoint retained")
                improved, best = score < best, min(best, score)
                status = "stopped" if stopping() else "complete" if step == a.target_steps else "training"
                checkpoint(validation, improved, status)
                record = {"step": step, "train_language_bpt": lm_sum / math.log(2),
                          "clip_grad_norm": float(grad), "trained_tokens": processed, **validation}
                with (run_dir / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print("RESULT " + json.dumps(record), flush=True)
        # A stop can arrive between iterations; persist any remaining completed step.
        progress = json.loads((run_dir / "progress.json").read_text())
        if progress["saved_step"] != step:
            validation = moe_train.evaluate_both(model, datasets, args.context, args.eval_batch, a.eval_batches)
            require(math.isfinite(validation["val_mean_bpt"]), "Nonfinite final validation")
            improved, best = validation["val_mean_bpt"] < best, min(best, validation["val_mean_bpt"])
            checkpoint(validation, improved, "stopped")
        elif stopping():
            progress["status"] = "stopped"
            atomic_json(run_dir / "progress.json", progress)
        if not stopping():
            report = generation_report(model, tokenizer, step, a.generation_tokens)
            atomic_json(run_dir / f"generation_step_{step}.json", report)
            for row in report["results"]:
                print(f"\n{row['case']} | {row['mode']}\n{row['text']}", flush=True)
        print(f"{'STOPPED' if stopping() else 'FINISHED'}: verified saved step {step}", flush=True)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "source-data", "data-dir", "run-dir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--expected-source-step", type=int, default=60000)
    p.add_argument("--target-steps", type=int, default=61000)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--generation-tokens", type=int, default=64)
    p.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    p.add_argument("--clear-stop", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    a = p.parse_args()
    require(min(a.save_every, a.eval_batches, a.threads, a.log_every, a.generation_tokens) > 0, "Settings must be positive")
    run(a)


if __name__ == "__main__":
    main()
