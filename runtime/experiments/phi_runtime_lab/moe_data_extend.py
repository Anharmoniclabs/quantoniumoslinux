#!/usr/bin/env python3
"""Build a separate PhiState-MoE corpus with a small code + knowledge extension.

The original corpus, tokenizer, validation/test files, and checkpoint stay intact.
New examples are streamed, content-deduplicated, split by document hash, and
tokenized with the original SentencePiece model. See provenance.json for sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np


CODE_ID = "flytech/python-codes-25k"
GENERAL_ID = "wikimedia/wikipedia"
GENERAL_CONFIG = "20231101.en"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_text(row: dict) -> str:
    value = row.get("text")
    if isinstance(value, str) and value.strip():
        return value.strip()
    parts = []
    for key, label in (("instruction", "Instruction"), ("input", "Input"),
                       ("output", "Output")):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(f"{label}:\n{value.strip()}")
    return "\n\n".join(parts)


def wiki_text(row: dict) -> str:
    title = row.get("title")
    body = row.get("text")
    if not isinstance(body, str) or not body.strip():
        return ""
    title = title.strip() if isinstance(title, str) else ""
    return (f"Title: {title}\n\n" if title else "") + body.strip()


def add_stream(dataset, text_fn, domain: str, token_limit: int, split_pct: int,
               output: Path, manifest: dict, tokenizer, dtype, seen: set[str]) -> dict:
    train_key, val_key = f"train_{domain}", f"val_{domain}"
    train_path = output / manifest["bins"][train_key]["file"]
    val_path = output / manifest["bins"][val_key]["file"]
    counts = {"train": 0, "val": 0, "documents": 0, "duplicates": 0}
    content_digest = hashlib.sha256()

    with train_path.open("ab") as train_file, val_path.open("ab") as val_file:
        for row in dataset:
            text = text_fn(row)
            if not text or len(text) < 80:
                continue
            normalized = " ".join(text.split())
            text_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if text_hash in seen:
                counts["duplicates"] += 1
                continue
            seen.add(text_hash)
            ids = tokenizer.encode(text, out_type=int)
            if len(ids) < 12:
                continue
            # Use one hard aggregate token budget per domain. Keep documents whole.
            current = counts["train"] + counts["val"]
            if current >= token_limit:
                break
            token_bytes = np.asarray(ids, dtype=dtype).tobytes()
            is_val = int(text_hash[:8], 16) % 100 < split_pct
            destination = val_file if is_val else train_file
            destination.write(token_bytes)
            split = "val" if is_val else "train"
            counts[split] += len(ids)
            counts["documents"] += 1
            content_digest.update(bytes.fromhex(text_hash))
            content_digest.update(token_bytes)
        train_file.flush()
        os.fsync(train_file.fileno())
        val_file.flush()
        os.fsync(val_file.fileno())

    manifest["bins"][train_key]["tokens"] += counts["train"]
    manifest["bins"][val_key]["tokens"] += counts["val"]
    counts["content_sha256"] = content_digest.hexdigest()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Original prepared PhiMoE corpus")
    parser.add_argument("--out", required=True, help="New expanded corpus directory")
    parser.add_argument("--max-code-tokens", type=int, default=4_000_000)
    parser.add_argument("--max-general-tokens", type=int, default=4_000_000)
    parser.add_argument("--validation-percent", type=int, default=5)
    args = parser.parse_args()
    if min(args.max_code_tokens, args.max_general_tokens) < 100_000:
        parser.error("Each added domain needs at least 100,000 token budget")
    if not 1 <= args.validation_percent <= 20:
        parser.error("validation-percent must be in 1..20")

    base = Path(args.base).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    pending = out.with_name(out.name + ".building")
    if base == out or not (base / "manifest.json").is_file():
        parser.error("Base must be a prepared corpus and output must be a different directory")
    if out.exists() or pending.exists():
        parser.error(f"Output already exists; refusing to replace it: {out} (or {pending})")

    try:
        import sentencepiece as spm
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install dependencies first: pip install datasets sentencepiece") from exc

    original = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    if original.get("format") != "PhiMoECorpus-v1" or original.get("tokenizer") != "bpe":
        parser.error("Expected the original PhiMoECorpus-v1 BPE corpus")
    tokenizer_path = base / "tokenizer.model"
    if not tokenizer_path.is_file():
        parser.error("Original tokenizer.model is missing")
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if tokenizer.get_piece_size() != int(original["vocab"]):
        parser.error("Tokenizer vocabulary does not match the corpus manifest")

    dtype = np.uint16 if original["dtype"] == "uint16" else np.uint32
    shutil.copytree(base, pending)
    expanded = json.loads((pending / "manifest.json").read_text(encoding="utf-8"))
    tokenizer_hash = sha256_file(pending / "tokenizer.model")
    used = {
        "base_data_fingerprint": original["data_fingerprint"],
        "tokenizer_sha256": tokenizer_hash,
        "code_source": CODE_ID,
        "general_source": f"{GENERAL_ID}/{GENERAL_CONFIG}",
        "added": {},
    }

    try:
        print("Streaming MIT instructional Python examples:", CODE_ID, flush=True)
        code_ds = load_dataset(CODE_ID, split="train", streaming=True)
        used["added"]["code"] = add_stream(
            code_ds, code_text, "code", args.max_code_tokens,
            args.validation_percent, pending, expanded, tokenizer, dtype, set(),
        )
        print("Streaming English Wikipedia knowledge articles:", GENERAL_ID, GENERAL_CONFIG, flush=True)
        wiki_ds = load_dataset(GENERAL_ID, GENERAL_CONFIG, split="train", streaming=True)
        used["added"]["general"] = add_stream(
            wiki_ds, wiki_text, "general", args.max_general_tokens,
            args.validation_percent, pending, expanded, tokenizer, dtype, set(),
        )
        for domain in ("code", "general"):
            extra = used["added"][domain]
            if extra["train"] < 100_000 or extra["val"] < 10_000:
                raise RuntimeError(f"Too little {domain} data was added: {extra}")

        canonical = json.dumps(used, sort_keys=True, separators=(",", ":")).encode("utf-8")
        expanded["data_fingerprint"] = hashlib.sha256(canonical).hexdigest()
        expanded["extension"] = {
            "format": "PhiMoECorpus-extension-v1",
            "base_data_fingerprint": original["data_fingerprint"],
            "sources": used,
        }
        (pending / "manifest.json").write_text(
            json.dumps(expanded, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (pending / "provenance.json").write_text(json.dumps({
            "format": "PhiMoECorpus-provenance-v1",
            "sources": [
                {
                    "id": CODE_ID,
                    "purpose": "Instruction, input, and Python output examples for code training",
                    "license": "MIT (dataset card declaration)",
                    "url": "https://huggingface.co/datasets/flytech/python-codes-25k",
                },
                {
                    "id": f"{GENERAL_ID}/{GENERAL_CONFIG}",
                    "purpose": "English encyclopedic knowledge text for general-domain pretraining",
                    "license": "CC BY-SA 3.0 and GFDL; retain attribution/source information",
                    "url": "https://huggingface.co/datasets/wikimedia/wikipedia",
                },
            ],
            "selection": used,
            "limits": {
                "max_code_tokens": args.max_code_tokens,
                "max_general_tokens": args.max_general_tokens,
                "validation_percent": args.validation_percent,
                "split": "document hash modulo 100; added validation is separate from train",
            },
            "note": "This file records sources and licenses; it does not grant rights beyond those licenses.",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(pending, out)
    except Exception:
        shutil.rmtree(pending, ignore_errors=True)
        raise

    print(json.dumps({
        "status": "EXPANDED CORPUS READY",
        "base_data_fingerprint": original["data_fingerprint"],
        "expanded_data_fingerprint": expanded["data_fingerprint"],
        "vocab": expanded["vocab"],
        "added_code_tokens": used["added"]["code"]["train"] + used["added"]["code"]["val"],
        "added_general_tokens": used["added"]["general"]["train"] + used["added"]["general"]["val"],
        "output": str(out),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
