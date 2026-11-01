"""Verified next-token windows and matched validation probes for natural PhiState corpora."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F

FINGERPRINT = "5a45ae09dfc42c6021202838439e77bd5cb05ff38d673d23e3219a1ecbc6c1d3"
TOKENIZER_SHA = "9f704ff69de6cb60fd7466a26b70cbd0472bb1a192f1825a7c52cfc30179ea04"
DOMAINS = ("code", "general")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def whitespace_piece(piece):
    """Includes literal/escaped whitespace, including byte-fallback tab and newline."""
    byte = re.fullmatch(r"<0x([0-9a-fA-F]{2})>", piece)
    if byte:
        return int(byte[1], 16) in b" \t\n\r\v\f"
    return bool(piece) and all(c == "▁" or c.isspace() for c in piece)


@dataclass
class Corpus:
    manifest: dict
    tokenizer: object
    bins: dict
    whitespace: np.ndarray

    @property
    def fingerprint(self):
        return self.manifest["data_fingerprint"]

    @property
    def vocab(self):
        return self.manifest["vocab"]

    @property
    def tokenizer_sha256(self):
        return self.manifest["whitespace_repair"]["tokenizer_sha256"]


def open_corpus(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    fingerprint = digest({k: v for k, v in manifest.items() if k != "data_fingerprint"})
    if fingerprint != manifest.get("data_fingerprint") or fingerprint != FINGERPRINT:
        raise ValueError("Corpus does not match the verified repaired manifest.")
    if (manifest["format"], manifest["dtype"], manifest["tokenizer"], manifest["vocab"]) != (
        "PhiMoECorpus-v1",
        "uint16",
        "bpe",
        8192,
    ):
        raise ValueError("Unexpected repaired corpus format, dtype or vocabulary.")
    tokenizer_path = path / "tokenizer.model"
    if (
        sha_file(tokenizer_path) != TOKENIZER_SHA
        or manifest["whitespace_repair"]["tokenizer_sha256"] != TOKENIZER_SHA
    ):
        raise ValueError("Tokenizer fingerprint differs from the repaired corpus.")
    tok = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if tok.vocab_size() != manifest["vocab"]:
        raise ValueError("Tokenizer vocabulary size changed.")
    bins = {}
    for split in ("train", "val"):
        for domain in DOMAINS:
            key = f"{split}_{domain}"
            item = manifest["bins"][key]
            file = path / (key + ".bin")
            if (
                item["file"] != file.name
                or file.stat().st_size != 2 * item["tokens"]
                or sha_file(file) != item["sha256"]
            ):
                raise ValueError(f"Count or hash mismatch for {key}.")
            arr = np.memmap(file, mode="r", dtype="<u2")
            if np.max(arr) >= tok.vocab_size():
                raise ValueError(f"Out-of-range token ID in {key}.")
            bins[key] = arr
    whitespace = np.array(
        [whitespace_piece(tok.id_to_piece(i)) for i in range(tok.vocab_size())], dtype=bool
    )
    return Corpus(manifest, tok, bins, whitespace)


def _domain_plan(corpus, split, domain, count, context, seed):
    if type(context) is not int or context < 1 or count < 0:
        raise ValueError("Invalid token window size or count")
    rng = random.Random(digest(["natural-windows-v1", corpus.fingerprint, split, domain, seed, context]))
    tile = context + 1
    length = len(corpus.bins[f"{split}_{domain}"])
    start = rng.randrange(min(tile, max(1, length - tile + 1)))
    tiles = range((length - start) // tile)
    if count > len(tiles):
        raise ValueError(f"{split}_{domain} is too short for {count} disjoint windows of {tile} tokens.")
    return [
        {"split": split, "domain": domain, "offset": start + ix * tile, "context": context}
        for ix in rng.sample(tiles, count)
    ]


def schedule(corpus, steps, batch_size, context, seed, code_share=0.55):
    """An identical, bounded, nonoverlapping training deck for each compared model."""
    if (
        type(steps) is not int
        or steps < 1
        or type(batch_size) is not int
        or batch_size < 1
        or not 0 <= code_share <= 1
    ):
        raise ValueError("Invalid steps, batch size or domain fraction.")
    n = steps * batch_size

    # Balanced over ordered positions as well as over the full budget.
    def code_by(i):
        return math.floor(i * code_share + 0.5)

    domains = ["code" if code_by(i + 1) > code_by(i) else "general" for i in range(n)]
    streams = {
        d: iter(_domain_plan(corpus, "train", d, domains.count(d), context, seed)) for d in DOMAINS
    }
    rows = [next(streams[d]) for d in domains]
    result = {
        "version": "natural-windows-v1",
        "data_fingerprint": corpus.fingerprint,
        "seed": seed,
        "code_share": code_share,
        "batch_size": batch_size,
        "context": context,
        "steps": [rows[i : i + batch_size] for i in range(0, n, batch_size)],
    }
    result["fingerprint"] = digest(result)
    return result


def validation_plan(corpus, context=128, count=32, seed=20261102):
    if type(count) is not int or count < 1:
        raise ValueError("Validation requires at least one window in each domain.")
    return {d: _domain_plan(corpus, "val", d, count, context, seed) for d in DOMAINS}


def materialize(corpus, descriptors, device):
    """Return aligned long [batch,context] input and target tensors; never wrap across EOF."""
    if not descriptors:
        raise ValueError("Empty batch.")
    rows, width = [], descriptors[0]["context"] + 1
    for desc in descriptors:
        split, domain, start, context = (desc[k] for k in ("split", "domain", "offset", "context"))
        if split not in ("train", "val") or domain not in DOMAINS:
            raise ValueError("Only train and validation streams are available.")
        arr = corpus.bins[f"{split}_{domain}"]
        if (
            type(start) is not int
            or type(context) is not int
            or context <= 0
            or width != context + 1
            or start < 0
            or start + width > len(arr)
        ):
            raise ValueError(f"Invalid token window descriptor: {desc}")
        rows.append(arr[start : start + width])
    ids = torch.from_numpy(np.stack(rows).astype(np.int64)).to(device=device)
    return ids[:, :-1].contiguous(), ids[:, 1:].contiguous()


def _positions(width):
    # Up to eight evenly spaced indices in the later half; all have earlier context.
    start = max(1, width // 2)
    count = min(8, width - start)
    if count < 1:
        raise ValueError("Context probes need a window width of at least two.")
    return np.linspace(start, width - 1, num=count, dtype=np.int64).tolist()


@torch.inference_mode()
def evaluate(model, corpus, validation, device, batch_size=8):
    if batch_size < 1 or set(validation) != set(DOMAINS):
        raise ValueError("Invalid validation sets or batch size.")
    ws = torch.as_tensor(corpus.whitespace, device=device)
    was_training = model.training
    model.eval()
    result = {}
    try:
        for domain in DOMAINS:
            descriptors = validation[domain]
            if not descriptors or any(d["split"] != "val" or d["domain"] != domain for d in descriptors):
                raise ValueError("Expected fixed validation descriptors for each domain.")
            if len({d["context"] for d in descriptors}) != 1:
                raise ValueError("Matched validation requires one common context width.")
            width = descriptors[0]["context"]
            suffix = min(8, max(1, width // 2))
            sums = [0.0] * 7
            totals = [0] * 7
            for off in range(0, len(descriptors), batch_size):
                x, y = materialize(corpus, descriptors[off : off + batch_size], device)
                # Exact average NLL over all held-out targets.
                logits = model(x)
                nll = F.cross_entropy(
                    logits.float().flatten(0, 1), y.flatten(), reduction="none"
                ).reshape_as(y)
                del logits
                mask = ws[y]
                positions = _positions(x.size(1))
                sampled = nll[:, positions].reshape(-1)
                # Same immediate preceding tokens and targets as the full-prefix probes.
                one = model(x[:, positions].reshape(-1, 1))[:, 0]
                last = F.cross_entropy(one.float(), y[:, positions].reshape(-1), reduction="none")
                losses = [nll, nll[mask], nll[~mask], sampled, last]
                if len(descriptors) > 1:
                    # Equal sequence positions and same target. Replace only the earlier
                    # history with a disjoint same-domain validation window. The i-th
                    # probe within the retained suffix keeps i+1 real input tokens.
                    donors = [
                        descriptors[(j + 1) % len(descriptors)]
                        for j in range(off, min(off + batch_size, len(descriptors)))
                    ]
                    wrong, _ = materialize(corpus, donors, device)
                    wrong = torch.cat((wrong[:, :-suffix], x[:, -suffix:]), dim=1)
                    changed = model(wrong)[:, -suffix:]
                    changed_nll = F.cross_entropy(
                        changed.float().reshape(-1, changed.size(-1)),
                        y[:, -suffix:].reshape(-1),
                        reduction="none",
                    )
                    losses.extend((nll[:, -suffix:], changed_nll))
                for i, loss in enumerate(losses):
                    val = float(loss.double().sum())
                    if not math.isfinite(val):
                        raise FloatingPointError(f"Nonfinite validation loss in {domain}.")
                    sums[i] += val
                    totals[i] += loss.numel()
            all_bpt, white, other, full, last, prefix, wrong = [
                s / (n * math.log(2)) if n else None for s, n in zip(sums, totals)
            ]
            result[domain] = {
                "all_bpt": all_bpt,
                "whitespace_bpt": white,
                "other_bpt": other,
                "full_context_bpt": full,
                "last_only_bpt": last,
                "context_gain_bpt": last - full,
                "tokens": totals[0],
                "whitespace_tokens": totals[1],
                "other_tokens": totals[2],
                "context_probe_tokens": totals[3],
                "windows": len(descriptors),
                "history_matched_full_bpt": prefix,
                "wrong_history_bpt": wrong,
                "wrong_history_gain_bpt": wrong - prefix if prefix is not None else None,
                "wrong_history_tokens": totals[5],
                "wrong_history_retained_suffix": suffix,
            }
    finally:
        model.train(was_training)
    result["mean_bpt"] = sum(result[d]["all_bpt"] for d in DOMAINS) / len(DOMAINS)
    return result


@torch.inference_mode()
def completions(model, corpus, device, context, new_tokens=32):
    """Fixed domain-tagged natural prefixes; qualitative check outside checkpoint selection."""
    if context < 2 or not 1 <= new_tokens <= 128:
        raise ValueError("Invalid completion budget.")
    prompts = (("code", "def add(a, b):\n    "), ("general", "The purpose of an operating system is"))
    was_training = model.training
    model.eval()
    results = []
    try:
        for domain, text in prompts:
            ids = corpus.tokenizer.encode(f"<|{domain}|>\n{text}")
            output = []
            for _ in range(new_tokens):
                logits = model(torch.tensor([ids[-context:]], dtype=torch.long, device=device))[0, -1]
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Nonfinite completion logits.")
                token = int(logits.argmax())
                if token == corpus.tokenizer.eos_id():
                    break
                ids.append(token)
                output.append(token)
            results.append(
                {
                    "domain": domain,
                    "prompt": text,
                    "continuation": corpus.tokenizer.decode(output),
                    "generated_tokens": len(output),
                }
            )
    finally:
        model.train(was_training)
    return results
