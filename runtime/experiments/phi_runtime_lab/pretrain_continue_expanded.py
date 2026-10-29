#!/usr/bin/env python3
"""Continue a saved PhiState-MoE run on a separately expanded corpus.

Requires the original 8,192-token BPE tokenizer and optimizer checkpoint. It
rebases a temporary checkpoint copy onto the new corpus fingerprint, preserves
the saved weights/optimizer/RNG, and resets only the old validation-best score
because the validation set has grown. The source checkpoint is never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import torch

import moe_train


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def best_from_latest(latest: Path) -> Path:
    if not latest.name.endswith(".last.pt"):
        raise ValueError("Checkpoint path must end in .last.pt so its best/latest pair is unambiguous")
    return latest.with_name(latest.name[:-len(".last.pt")] + ".pt")


def save_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="Verified .last.pt checkpoint, including optimizer state")
    parser.add_argument("--data-dir", required=True, help="Expanded corpus directory")
    parser.add_argument("--out", required=True,
                        help="New or resumable output best checkpoint path, e.g. phi_60000.pt")
    parser.add_argument("--target-steps", type=int, default=60_000,
                        help="Total optimizer steps; 60,000 adds 10,000 after the 50K milestone")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--max-seconds", type=float, default=0)
    args = parser.parse_args()
    if min(args.target_steps, args.eval_every, args.eval_batches, args.threads) < 1:
        parser.error("step, evaluation, and thread settings must be positive")

    source = Path(args.checkpoint).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    destination = Path(args.out).expanduser().resolve()
    destination_last = destination.with_name(destination.stem + ".last" + destination.suffix)
    if not source.is_file():
        parser.error(f"Checkpoint not found: {source}")
    provenance_path = data_dir / "provenance.json"
    manifest_path = data_dir / "manifest.json"
    tokenizer_path = data_dir / "tokenizer.model"
    if not all(p.is_file() for p in (provenance_path, manifest_path, tokenizer_path)):
        parser.error("Expanded corpus must contain manifest.json, provenance.json, and tokenizer.model")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    source_fingerprint = manifest.get("extension", {}).get("base_data_fingerprint")
    if provenance.get("format") != "PhiMoECorpus-provenance-v1" or not source_fingerprint:
        parser.error("Missing or unrecognized extension provenance")
    expected_tokenizer_hash = manifest.get("extension", {}).get("sources", {}).get("tokenizer_sha256")
    if expected_tokenizer_hash and sha256_file(tokenizer_path) != expected_tokenizer_hash:
        parser.error("Tokenizer file hash differs from corpus provenance")

    model, payload = moe_train.ckpt_load(str(source))
    latest_fingerprint = payload.get("data_fingerprint")
    new_fingerprint = manifest.get("data_fingerprint")
    if not new_fingerprint or payload["config"].get("vocab") != int(manifest["vocab"]):
        parser.error("Checkpoint vocabulary does not match expanded corpus")
    if payload.get("tokenizer") != manifest.get("tokenizer"):
        parser.error("Checkpoint tokenizer type does not match expanded corpus")
    if "optimizer_state" not in payload:
        parser.error("Checkpoint has no optimizer state; refusing to silently reset it")
    if not isinstance(payload.get("train_args"), dict):
        parser.error("Checkpoint is missing saved training arguments")
    current_step = int(payload.get("step", -1))
    if current_step < 0:
        parser.error("Checkpoint has an invalid step")
    if current_step >= args.target_steps:
        print(f"ALREADY AT {current_step} steps; target {args.target_steps}. No training launched.")
        return

    if latest_fingerprint not in (source_fingerprint, new_fingerprint):
        parser.error("Checkpoint is not from the base corpus or this expanded corpus")
    best_source = best_from_latest(source)
    if not best_source.is_file():
        parser.error(f"Matching best checkpoint missing: {best_source}")
    _, best_payload = moe_train.ckpt_load(str(best_source))
    if (best_payload.get("data_fingerprint") != latest_fingerprint or
            best_payload.get("config") != payload.get("config")):
        parser.error("Best/latest checkpoint pair does not match")

    already_resuming_output = latest_fingerprint == new_fingerprint and source == destination_last
    if latest_fingerprint == source_fingerprint:
        if destination.exists() or destination_last.exists():
            parser.error("Destination already exists; use its .last.pt as --checkpoint to resume it")
    elif not already_resuming_output:
        parser.error("Expanded-corpus checkpoint must be resumed from the same output pair")

    manifest_loaded, _, _ = moe_train.load_data(str(data_dir))
    if manifest_loaded["data_fingerprint"] != new_fingerprint:
        parser.error("Expanded corpus failed its standard loader validation")

    print(json.dumps({
        "plan": "CONTINUE EXISTING WEIGHTS ON EXPANDED DATA",
        "from_step": current_step,
        "target_total_steps": args.target_steps,
        "new_steps": args.target_steps - current_step,
        "device": args.device,
        "original_data_fingerprint": source_fingerprint,
        "expanded_data_fingerprint": new_fingerprint,
        "tokenizer_vocab": int(manifest["vocab"]),
        "optimizer_state": "preserved",
        "output_best": str(destination),
        "output_latest": str(destination_last),
    }, indent=2), flush=True)

    train_args = dict(payload["train_args"])
    train_args.update({
        "action": "train",
        "data_dir": str(data_dir),
        "resume": str(source),
        "out": str(destination),
        "steps": args.target_steps,
        "eval_every": args.eval_every,
        "eval_batches": args.eval_batches,
        "threads": args.threads,
        "device": args.device,
        "max_seconds": args.max_seconds,
        "save_optimizer": True,
        "dry_run": False,
    })
    train_args = SimpleNamespace(**train_args)

    if latest_fingerprint == source_fingerprint:
        # The old best score belongs to the old validation set. Reset this metadata
        # only in temporary copies so the expanded validation is selected afresh.
        with tempfile.TemporaryDirectory(prefix="phi_expanded_rebase_") as temp:
            staged_latest = Path(temp) / "rebased.last.pt"
            rebased = dict(payload)
            rebased["data_fingerprint"] = new_fingerprint
            rebased["best_validation_bpt"] = float("inf")
            save_payload(staged_latest, rebased)
            train_args.resume = str(staged_latest)
            seed_best = dict(best_payload)
            seed_best["data_fingerprint"] = new_fingerprint
            seed_best["best_validation_bpt"] = float("inf")
            moe_train.atomic_checkpoint(destination, seed_best)
            # Keep the complete model and training state resident only once.
            del model, payload, best_payload, rebased, seed_best
            import gc
            gc.collect()
            moe_train.train(train_args)
    else:
        # Re-running after a Colab disconnect resumes from the last atomically saved
        # expanded-corpus checkpoint. The output .pt/.last.pt pair is the recovery point.
        if not already_resuming_output:
            parser.error("Refusing to write over an unrelated expanded-corpus checkpoint")
        del model, payload, best_payload
        import gc
        gc.collect()
        moe_train.train(train_args)


if __name__ == "__main__":
    main()
