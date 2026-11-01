"""Two fresh CPU controls for a query/assignment crossover diagnostic, independent of PhiState."""

import argparse
import json
import math
import re
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from phistate_binding_data import cases, make_bank, wording

SEED = 20261101
DENSE_STEPS = 640
SLOT_STEPS = 320
COPY_STEPS = 96
GOAL = 0.75


def tokenize(text):
    return re.findall(r"[A-Za-z]+|[0-9]|[^\w\s]", text)


def _prepared():
    bank = make_bank()
    rows = [
        (split, group["id"], r)
        for split, groups in bank.items()
        for group in groups
        for r in cases(group)
    ]
    # Vocabulary includes both surface forms, but novel-form examples never enter training.
    vocab = {"<pad>": 0}
    for _, _, row in rows:
        for regime in ("familiar", "new"):
            for word in tokenize(wording(row, regime)):
                if word not in vocab:
                    vocab[word] = len(vocab)
    pools = {}
    for split in bank:
        pool = [(gid, row) for s, gid, row in rows if s == split]
        for regime in ("familiar", "new"):
            x = torch.tensor([[vocab[t] for t in tokenize(wording(r, regime))] for _, r in pool])
            slots = torch.tensor(
                [
                    [
                        ord(r["key1"]) - 65,
                        int(r["value1"]),
                        ord(r["key2"]) - 65,
                        int(r["value2"]),
                        ord(r["query"]) - 65,
                    ]
                    for _, r in pool
                ]
            )
            gold = torch.tensor([int(r["answer"]) for _, r in pool])
            foil = torch.tensor([int(r["foil"]) for _, r in pool])
            pools[split, regime] = (x, slots, gold, foil)
    copies = {}
    for split in bank:
        edges = sorted(
            {(r["key1"], r["value1"]) for s, _, r in rows if s == split}
            | {(r["key2"], r["value2"]) for s, _, r in rows if s == split}
        )
        words = [f"Records: {k}={v}. Return the value of {k}. Reply with one digit." for k, v in edges]
        x = torch.tensor([[vocab[t] for t in tokenize(sent)] for sent in words])
        gold = torch.tensor([int(v) for k, v in edges])
        slots = torch.tensor([[ord(k) - 65, int(v), ord(k) - 65, int(v), ord(k) - 65] for k, v in edges])
        copies[split] = (x, slots, gold)
    return pools, len(vocab), copies


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.out = nn.Linear(width, width)
        self.ff = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width * 3), nn.GELU(), nn.Linear(width * 3, width)
        )

    def forward(self, x):
        q, k, v = (
            t.reshape(x.shape[0], x.shape[1], 2, -1).transpose(1, 2)
            for t in self.qkv(self.norm(x)).chunk(3, -1)
        )
        z = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape_as(x)
        x = x + self.out(z)
        return x + self.ff(x)


class Dense(nn.Module):
    """Causal attention over actual word/punctuation tokens; predicts one of the ten digits."""

    def __init__(self, vocab, width=32):
        super().__init__()
        self.word = nn.Embedding(vocab, width)
        self.position = nn.Embedding(64, width)
        self.blocks = nn.Sequential(Block(width), Block(width))
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 10))

    def forward(self, words, slots):
        h = self.word(words) + self.position(torch.arange(words.shape[1]))
        return self.out(self.blocks(h)[:, -1])


class LearnedSlots(nn.Module):
    """Positive control given two parsed key/value slots and the query. No access to the answer label."""

    def __init__(self, width=16):
        super().__init__()
        self.key = nn.Embedding(8, width)
        self.query = nn.Embedding(8, width)
        self.value = nn.Embedding(10, width)
        self.out = nn.Linear(width, 10)

    def forward(self, words, slots):
        k = self.key(slots[:, [0, 2]])
        v = self.value(slots[:, [1, 3]])
        q = self.query(slots[:, 4])
        attention = (k * q[:, None]).sum(-1) / math.sqrt(k.shape[-1])
        return self.out((v * attention.softmax(-1)[:, :, None]).sum(1))


@torch.inference_mode()
def score(model, sample):
    model.eval()
    words, slots, gold, foil = sample
    logits = model(words, slots)
    guess = logits.argmax(-1)
    exact = guess.eq(gold)
    good = exact.view(-1, 4)
    return {
        "items": int(exact.sum()),
        "total": len(guess),
        "all_four": int(good.all(1).sum()),
        "quartets": len(good),
        "quartet_rate": float(good.all(1).float().mean()),
        "two_choices": int((logits.gather(1, gold[:, None]) > logits.gather(1, foil[:, None])).sum()),
        "answer_bits": round(float(F.cross_entropy(logits, gold)) / math.log(2), 5),
    }


@torch.inference_mode()
def copy_score(model, samples):
    result = {}
    model.eval()
    for split, (x, slots, gold) in samples.items():
        correct = int((model(x, slots).argmax(-1) == gold).sum())
        result[split] = {"correct": correct, "total": len(gold)}
    return result


def fit(model, pools, seed, steps, copies=None):
    rng = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    train = pools["train", "familiar"]
    base = score(model, train)
    t = time.monotonic()
    copy_receipt = None
    if copies is not None:
        before = copy_score(model, copies)
        x, slots, gold = copies["train"]
        for _ in range(COPY_STEPS):
            model.train()
            logits = model(x, slots)
            loss = F.cross_entropy(logits, gold)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        copy_receipt = {
            "before": before,
            "after_copy": copy_score(model, copies),
            "updates": COPY_STEPS,
            "targets": COPY_STEPS * len(gold),
        }
    for _ in range(steps):
        model.train()
        # Always include all four counterfactuals in the same batch.
        groups = torch.randint(len(train[0]) // 4, (8,), generator=rng)
        idx = (groups[:, None] * 4 + torch.arange(4)).reshape(-1)
        words, slots, gold, _foil = (v[idx] for v in train)
        logits = model(words, slots)
        loss = F.cross_entropy(logits, gold)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    evaluation = {f"{s}/{regime}": score(model, sample) for (s, regime), sample in pools.items()}
    qualified = all(evaluation[k]["quartet_rate"] >= GOAL for k in ("train/familiar", "dev/familiar"))
    if copy_receipt:
        copy_receipt["after_binding"] = copy_score(model, copies)
    return {
        "qualified": qualified,
        "copy_curriculum": copy_receipt,
        "updates": steps,
        "target_tokens": steps * 32,
        "parameters": sum(p.numel() for p in model.parameters()),
        "elapsed_seconds": round(time.monotonic() - t, 3),
        "before_fit": base,
        "evaluation": evaluation,
    }


def run(seed=SEED):
    """Train both fixed-budget scratch references. Nothing is read from or written to a checkpoint."""
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            pools, vocab, copies = _prepared()
            result = {}
            for name, make, steps, first_copy in [
                ("dense_causal", lambda: Dense(vocab), DENSE_STEPS, False),
                ("copy_curriculum_dense", lambda: Dense(vocab), DENSE_STEPS, True),
                ("learned_slots", LearnedSlots, SLOT_STEPS, False),
            ]:
                torch.manual_seed(seed)
                result[name] = fit(make(), pools, seed, steps, copies if first_copy else None)
    finally:
        torch.set_num_threads(old_threads)
    return {
        "qualified": any(x["qualified"] for x in result.values()),
        "generic_dense_qualified": result["dense_causal"]["qualified"]
        or result["copy_curriculum_dense"]["qualified"],
        "models": result,
        "seed": seed,
        "minimum_train_and_novel_binding_quartet_rate": GOAL,
        "scope": "One-digit constrained-output toy references. Dense consumes word-tokenized prompts. "
        "Learned slots receives parsed keys and values, so wording is invisible to it. "
        "Qualification requires train/familiar and dev/familiar; new wording is reported separately. "
        "Neither reference uses the PhiState tokenizer, production weights or legacy final suite.",
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    report = run()
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
