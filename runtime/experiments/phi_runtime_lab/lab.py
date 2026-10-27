#!/usr/bin/env python3
"""Reproducible CPU-first PhiState hybrid-transformer research harness.

Run from this directory or anywhere: python lab.py --help
No web services, external tokenizers, or pretrained weights are needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from phistate import (
    CanonicalChannelMixer,
    GoldenState,
    ModelConfig,
    PhiStateLM,
    count_parameters,
    spectral_unitary,
)

HERE = Path(__file__).resolve().parent
EXTENSIONS = {".txt", ".md", ".py", ".rs", ".c", ".cpp", ".hpp", ".h", ".sh", ".toml"}
SKIP_DIRS = {".git", ".venv", "node_modules", "target", "build", "__pycache__", "private", "secrets", "models", "checkpoints", "dist", "data"}


def configure_cpu(threads: int, seed: int) -> None:
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def to_tensor(raw: bytes) -> torch.Tensor:
    return torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()).to(torch.int64)


def read_corpus(path: str, max_mb: float, seed: int) -> tuple[dict[str, torch.Tensor], dict]:
    location = Path(path).expanduser().resolve()
    if not location.exists():
        raise FileNotFoundError(f"No corpus at {location}")
    ceiling = int(max_mb * 1024 * 1024)
    if ceiling < 4096:
        raise ValueError("max-mb must be large enough for at least 4096 bytes")
    all_docs: list[tuple[str, bytes]] = []
    remaining = ceiling
    if location.is_file():
        content = location.read_bytes()[:ceiling]
        all_docs.append((str(location), content))
    else:
        candidates = sorted(p for p in location.rglob("*") if p.is_file()
                            and p.suffix.lower() in EXTENSIONS
                            and not set(p.relative_to(location).parts).intersection(SKIP_DIRS))
        for p in candidates:
            if remaining <= 0:
                break
            try:
                if p.stat().st_size < 128 or p.stat().st_size > 2_000_000:
                    continue
                contents = p.read_bytes()[:min(remaining, 256 * 1024)]
                if b"\0" in contents:
                    continue
                all_docs.append((str(p.relative_to(location)), contents))
                remaining -= len(contents)
            except (PermissionError, OSError):
                continue
    if not all_docs:
        raise RuntimeError("No usable text data (try --data file.txt or a directory with .md/.py/.txt)")
    # Exact-content dedupe, useful when datasets include redundant copies.
    unique = {}
    for name, raw in all_docs:
        digest = hashlib.sha256(raw).hexdigest()
        if digest not in unique:
            unique[digest] = (name, raw)
    docs = list(unique.values())
    if len(docs) >= 10:
        order = list(range(len(docs)))
        random.Random(seed + 790).shuffle(order)
        n_val = max(1, len(docs) // 10)
        n_test = max(1, len(docs) // 10)
        train_docs = order[:len(order)-n_val-n_test]
        val_docs = order[len(order)-n_val-n_test:len(order)-n_test]
        test_docs = order[len(order)-n_test:]
        groups = {"train": train_docs, "val": val_docs, "test": test_docs}
        chunks = {}
        for key, ids in groups.items():
            chunks[key] = b"\n\n".join(docs[i][1] for i in ids)
        split = "disjoint documents (exact duplicate files removed)"
    else:
        content = b"\n\n".join(raw for _, raw in docs)
        t1 = int(len(content) * 0.8)
        t2 = int(len(content) * 0.9)
        chunks = {"train": content[:t1], "val": content[t1:t2], "test": content[t2:]}
        split = "contiguous 80/10/10 byte split (few source documents)"
    lengths = {k: len(v) for k,v in chunks.items()}
    if min(lengths.values()) < 512:
        raise RuntimeError(f"Corpus too small for useful training: {lengths}; supply more text files")
    metadata = {
        "source": str(location), "sha256_fingerprints": [hashlib.sha256(c).hexdigest() for c in chunks.values()],
        "file_count": len(docs), "split_method": split, "bytes": lengths,
        "warning": "Check for near-duplicates; same project documents may share passages across splits."
    }
    return {k: to_tensor(v) for k, v in chunks.items()}, metadata


def sample_batch(corpus: torch.Tensor, batch: int, context: int, generator: torch.Generator):
    limit = len(corpus) - context - 1
    if limit < 0:
        raise RuntimeError(f"Split has {len(corpus)} bytes, shorter than context={context} + 1")
    starts = torch.randint(0, limit + 1, (batch,), generator=generator)
    idx = starts[:, None] + torch.arange(context + 1)[None, :]
    tokens = corpus[idx]
    return tokens[:, :-1], tokens[:, 1:]


@torch.inference_mode()
def eval_bpb(model: PhiStateLM, data: torch.Tensor, context: int, batch: int,
             eval_batches: int = 16):
    model.eval()
    max_start = len(data) - context - 1
    if max_start < 0:
        raise RuntimeError("Evaluation split too short for current context length")
    n_samples = batch * eval_batches
    starts = torch.linspace(0, max_start, steps=n_samples).round().long()
    total_loss, total_bytes = 0.0, 0
    t0 = time.perf_counter()
    for i in range(0, n_samples, batch):
        offsets = starts[i:i+batch, None] + torch.arange(context+1)[None, :]
        slab = data[offsets]
        x, y = slab[:, :-1], slab[:, 1:]
        logits = model(x)
        sum_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   y.reshape(-1), reduction="sum")
        total_loss += sum_loss.item()
        total_bytes += y.numel()
    dt = time.perf_counter() - t0
    nats = total_loss / max(total_bytes, 1)
    return {
        "nats_per_byte": round(nats, 6),
        "bits_per_byte": round(nats / math.log(2), 6),
        "per_byte_perplexity": round(math.exp(min(20,nats)), 5),
        "evaluated_bytes": total_bytes,
        "eval_bytes_per_sec": round(total_bytes / max(dt, 1e-9), 1),
    }


def filter_gradients(model: torch.nn.Module, basis: str, group: int, keep: float):
    """Ablation: truncate canonical-RFT/DFT coefficients of *computed* gradients.

    This does NOT remove backprop compute; it tests whether spectral gradient
    filtering improves sample efficiency. A faster custom optimizer requires
    separate implementation and validation.
    """
    if basis == "none":
        return 0
    U = spectral_unitary(group, basis)
    kept = max(1, math.ceil(group * keep))
    rows = 0
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None or p.ndim < 2 or p.shape[-1] % group:
                continue
            original = p.grad
            flat = original.reshape(-1, group).to(torch.complex64)
            coeffs = flat @ U.conj()
            if kept < group:
                ids = torch.topk(coeffs.abs(), kept, dim=1).indices
                mask = torch.zeros_like(coeffs, dtype=torch.bool)
                mask.scatter_(1, ids, True)
                coeffs = coeffs * mask
            recovered = (coeffs @ U.T).real
            p.grad.copy_(recovered.reshape_as(original))
            rows += flat.shape[0]
    return rows


def make_config(args) -> ModelConfig:
    return ModelConfig(
        mode=args.mode, mixer_basis=args.mix_basis, width=args.width, layers=args.layers,
        heads=args.heads, window=args.window,
        mix_group=args.mix_group, mlp_ratio=args.mlp_ratio,
    )


def train(args):
    configure_cpu(args.threads, args.seed)
    streams, md = read_corpus(args.data, args.max_mb, args.seed)
    cfg = make_config(args)
    cfg.validate()
    if any(len(v) <= args.context+1 for v in streams.values()):
        raise ValueError(f"context {args.context} too large for split {md['bytes']}")
    model = PhiStateLM(cfg)
    n_params = count_parameters(model)
    print(f"MODEL={cfg.mode} PARAMETERS={n_params:,} CORPUS={md['bytes']} SPLIT={md['split_method']}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed + 1234)
    checkpoint = Path(args.out).expanduser().resolve()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_val = math.inf
    t0 = time.perf_counter()
    tokens_seen = 0
    training_seconds = 0.0
    history = []
    results = None
    for step in range(1, args.steps + 1):
        step_started = time.perf_counter()
        model.train()
        x, y = sample_batch(streams["train"], args.batch, args.context, generator)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab), y.reshape(-1))
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        filtered = filter_gradients(model, args.grad_filter, args.mix_group, args.grad_keep)
        optimizer.step()
        training_seconds += time.perf_counter() - step_started
        tokens_seen += y.numel()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            score = eval_bpb(model, streams["val"], args.context, args.batch, args.eval_batches)
            elapsed = time.perf_counter() - t0
            history.append({"step":step,"train_bpb":round(loss.item()/math.log(2),6),"val_bpb":score["bits_per_byte"],"train_seconds":round(training_seconds,3)})
            print(f"step={step:>5} train_bpb={loss.item()/math.log(2):.4f} "
                  f"val_bpb={score['bits_per_byte']:.4f} "
                  f"train_bytes_per_sec={tokens_seen/max(training_seconds,1e-9):.1f} "
                  f"gradient_rows_filtered={filtered}", flush=True)
            if score["bits_per_byte"] < best_val:
                best_val = score["bits_per_byte"]
                payload = {
                    "format": "PhiStateLM-v1", "config": vars(cfg),
                    "state_dict": model.state_dict(), "best_step": step,
                    "best_val_bpb": best_val, "corpus": md,
                    "train": {
                        "seed": args.seed, "context": args.context,
                        "batch": args.batch, "lr": args.lr, "steps": args.steps,
                        "grad_filter": args.grad_filter,
                        "grad_keep": args.grad_keep,
                    },
                }
                torch.save(payload, checkpoint)
                results = {"mode":cfg.mode, "best_step":step, "best_val_bpb":best_val,
                           "parameter_count":n_params, "checkpoint":str(checkpoint),
                           "context":args.context, "train_seconds":round(elapsed,2),
                           "gradient_filter":args.grad_filter,"corpus":md,
                           "max_rss_mib":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,1),
                           "model_config":vars(cfg)}
    assert results is not None
    elapsed = time.perf_counter() - t0
    results["total_seconds"] = round(elapsed, 2)
    results["train_seconds"] = round(training_seconds, 2)
    results["train_bytes_per_sec"] = round(tokens_seen / max(training_seconds,1e-9),1)
    results["validation_history"] = history
    results["max_rss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,1)
    report_path = checkpoint.with_suffix(".json")
    report_path.write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
    print("SAVED "+str(checkpoint)+" REPORT "+str(report_path),flush=True)
    print("RESULT "+json.dumps({k:results[k] for k in ("mode","best_val_bpb","parameter_count","train_bytes_per_sec","max_rss_mib")}),flush=True)


def load_model(path: str):
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    if data.get("format") != "PhiStateLM-v1":
        raise ValueError("Incorrect checkpoint type")
    model = PhiStateLM.from_checkpoint_config(data["config"])
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model, data


def evaluate(args):
    configure_cpu(args.threads, args.seed)
    model, payload = load_model(args.checkpoint)
    streams, meta = read_corpus(args.data, args.max_mb, payload["train"]["seed"])
    if meta["sha256_fingerprints"] != payload["corpus"]["sha256_fingerprints"]:
        raise RuntimeError("Dataset fingerprint differs from training split; refusing incomparable evaluation")
    context = payload["train"]["context"]
    data = streams[args.split]
    score = eval_bpb(model, data, context, args.batch, args.eval_batches)
    output = {"mode":model.cfg.mode, "split":args.split, "best_val_bpb":payload["best_val_bpb"],
              "parameter_count":count_parameters(model), **score}
    print(json.dumps(output,indent=2))


@torch.inference_mode()
def generate(args):
    configure_cpu(args.threads, args.seed)
    model, payload = load_model(args.checkpoint)
    data = args.prompt.encode("utf-8")
    if not data:
        raise ValueError("Prompt must not be empty")
    state = None
    next_logits = None
    t0 = time.perf_counter()
    for byt in data:
        next_logits, state = model.step(torch.tensor([[byt]],dtype=torch.long), state)
    gen = torch.Generator().manual_seed(args.seed)
    result = bytearray(data)
    for i in range(args.new_bytes):
        logits = next_logits[0,-1] / max(0.05,args.temperature)
        if args.greedy:
            next_byte = int(logits.argmax())
        else:
            values, indices = torch.topk(logits, min(64, args.top_k))
            probs = torch.softmax(values, dim=-1)
            next_byte = int(indices[torch.multinomial(probs,1,generator=gen)])
        result.append(next_byte)
        if i + 1 < args.new_bytes:
            next_logits, state = model.step(torch.tensor([[next_byte]],dtype=torch.long),state)
    dt = time.perf_counter() - t0
    print("MODEL",model.cfg.mode,"PARAMS",count_parameters(model),
          "GENERATED_BYTES_PER_SEC",round(args.new_bytes/max(dt,1e-9),2))
    print(result.decode("utf-8",errors="replace"))


def compare(args):
    checkpoint_dir = Path(args.out_dir).expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    script = str(HERE / "lab.py")
    print("Train three variants in separate processes with the same seed, corpus, dimensions, and token budget.")
    print("The local baseline has fewer parameters: compare phi vs dft for equal-parameter tests.",flush=True)
    for mode in ("phi", "dft", "local"):
        cmd = [py, script, "train", "--data", args.data, "--mode", mode,
               "--out",str(checkpoint_dir / f"{mode}.pt"),
               "--width",str(args.width),"--layers",str(args.layers),
               "--heads",str(args.heads),"--window",str(args.window),
               "--mix-group",str(args.mix_group),"--mlp-ratio",str(args.mlp_ratio),
               "--mix-basis",args.mix_basis,
               "--steps",str(args.steps),"--batch",str(args.batch),
               "--context",str(args.context),"--threads",str(args.threads),
               "--lr",str(args.lr),"--seed",str(args.seed),
               "--max-mb",str(args.max_mb),
               "--eval-every",str(args.eval_every),
               "--eval-batches",str(args.eval_batches),
               "--grad-filter",args.grad_filter,"--grad-keep",str(args.grad_keep)]
        print("\nRUNNING:"," ".join(cmd),flush=True)
        subprocess.run(cmd,check=True)
    results=[]
    for mode in ("phi","dft","local"):
        result=json.loads((checkpoint_dir / f"{mode}.json").read_text(encoding="utf-8"))
        results.append(result)
    print("\nVALIDATION RESULTS (lower BPB better; not test set)")
    print(f"{'mode':<10}{'val BPB':>12}{'params':>12}{'bytes/s':>12}{'peak MiB':>12}")
    for r in results:
        print(f"{r['mode']:<10}{r['best_val_bpb']:>12.4f}{r['parameter_count']:>12,}"
              f"{r['train_bytes_per_sec']:>12.1f}{r['max_rss_mib']:>12.1f}")
    (checkpoint_dir / "comparison.json").write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
    phi,dft=results[0],results[1]
    delta=phi['best_val_bpb']-dft['best_val_bpb']
    print(f"Phi minus DFT validation BPB: {delta:+.5f}; negative is better for Phi.")
    print("Run `evaluate` on test ONLY after hyperparameters are finalized.")


def factorial(args):
    """Isolate golden recurrent frequency effects from canonical channel mixing."""
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    print("2x2 factorial: same token budget / parameters; varies memory init and mixer basis separately.", flush=True)
    for memory in ("phi", "dft"):
        for mixer in ("phi", "dft"):
            tag = memory + "_memory__" + mixer + "_mixer"
            cmd = [sys.executable, str(HERE / "lab.py"), "train",
                   "--data", args.data, "--mode", memory, "--mix-basis", mixer,
                   "--out",str(out / f"{tag}.pt"),
                   "--width",str(args.width),"--layers",str(args.layers),
                   "--heads",str(args.heads),"--window",str(args.window),
                   "--mix-group",str(args.mix_group),"--mlp-ratio",str(args.mlp_ratio),
                   "--steps",str(args.steps),"--batch",str(args.batch),
                   "--context",str(args.context),"--threads",str(args.threads),
                   "--lr",str(args.lr),"--seed",str(args.seed),
                   "--max-mb",str(args.max_mb),
                   "--eval-every",str(args.eval_every),
                   "--eval-batches",str(args.eval_batches),
                   "--grad-filter",args.grad_filter,"--grad-keep",str(args.grad_keep)]
            print("\nRUNNING",tag,flush=True)
            subprocess.run(cmd, check=True)
    print("\nFACTORIAL VALIDATION (lower BPB is better)")
    print(f"{'memory':<10}{'mixer':<10}{'val BPB':>12}{'params':>12}{'bytes/s':>12}")
    all_results = []
    for memory in ("phi", "dft"):
        for mixer in ("phi", "dft"):
            tag = memory + "_memory__" + mixer + "_mixer"
            result=json.loads((out / f"{tag}.json").read_text(encoding="utf-8"))
            all_results.append(result)
            print(f"{memory:<10}{mixer:<10}{result['best_val_bpb']:>12.4f}"
                  f"{result['parameter_count']:>12,}{result['train_bytes_per_sec']:>12.1f}")
    (out / "factorial_results.json").write_text(json.dumps(all_results,indent=2)+"\n",encoding="utf-8")
    print("Treat results as preliminary until replicated across several seeds and a larger held-out corpus.")


def make_demo(args):
    out=Path(args.out).expanduser().resolve();out.parent.mkdir(parents=True,exist_ok=True)
    rng=random.Random(83027)
    concepts=["processor","memory","kernel","cache","compiler","interrupt","scheduler","instruction","register","filesystem"]
    lines=[]
    for i in range(3000):
        x=rng.choice(concepts);y=rng.choice(concepts)
        a,b=rng.randint(1,99),rng.randint(1,99)
        lines.append(f"record {i%31}: the {x} monitors the {y}. {a} plus {b} equals {a+b}.\n")
    out.write_text("".join(lines), encoding="utf-8")
    print(f"Created SYNTHETIC SMOKE DATA ONLY: {out}, {out.stat().st_size} bytes")
    print("Never use this as proof of general AI or RFT superiority.")


def smoke(args):
    configure_cpu(args.threads,args.seed)
    for kind in ("phi","dft"):
        U = spectral_unitary(16,kind)
        err=(U.mH@U-torch.eye(16)).abs().max().item()
        assert err < 2e-5, f"unitarity failed: {kind} {err}"
        mixer=CanonicalChannelMixer(16,kind)
        x=torch.randn(2,9,32)
        assert (mixer(x)-x).abs().max()<2e-5, "RFT mixer identity failed"
        mem=GoldenState(32,kind)
        with torch.no_grad():
            full=mem(x)
            st=None; ys=[]
            for t in range(x.shape[1]):
                y,st=mem.step(x[:,t:t+1],st);ys.append(y)
            mismatch=(full-torch.cat(ys,dim=1)).abs().max().item()
            assert mismatch <2e-5, f"FFT vs recurrence mismatch: {kind}: {mismatch}"
        cfg=ModelConfig(mode=kind,width=32,layers=1,heads=4,window=8,mix_group=16)
        model=PhiStateLM(cfg).eval()
        toks=torch.randint(0,256,(2,12))
        with torch.no_grad():
            full=model(toks)
            states=None; streams=[]
            for i in range(toks.shape[1]):
                logit,states=model.step(toks[:,i:i+1],states)
                streams.append(logit)
            err2=(full-torch.cat(streams,dim=1)).abs().max().item()
            assert err2 <2e-5, f"full-vs-stream mismatch {kind}: {err2}"
            altered=toks.clone()
            altered[:,8:]=torch.randint(0,256,(2,4))
            causal=(full[:,:8]-model(altered)[:,:8]).abs().max().item()
            assert causal <2e-5, f"causal leakage {kind}: {causal}"
        print(f"PASS {kind} unitary={err:.3e} FFT_vs_step={mismatch:.3e} "
              f"LM_stream={err2:.3e} causal={causal:.3e}")
    cfg=ModelConfig(mode="phi",width=32,layers=1,heads=4,window=8,mix_group=16)
    model=PhiStateLM(cfg)
    opts=torch.optim.AdamW(model.parameters(),lr=1e-3)
    x=torch.randint(0,256,(2,10));y=torch.randint(0,256,(2,10))
    logits=model(x)
    loss=F.cross_entropy(logits.reshape(-1,256),y.reshape(-1))
    loss.backward()
    assert math.isfinite(loss.detach().item()),"nonfinite train loss"
    opts.step()
    print(f"PASS backward_gradients params={count_parameters(model):,} loss={loss.detach().item():.4f}")
    print("SMOKE PASS: architecture numerically consistent on this CPU.")


def add_model_flags(parser, include_mode=True):
    if include_mode:
        parser.add_argument("--mode",choices=["phi","dft","local"],default="phi")
    parser.add_argument("--mix-basis",choices=["auto","phi","dft"],default="auto",
                        help="Channel mixer choice; `auto` matches --mode. For 2x2 factorial tests, override it.")
    parser.add_argument("--width",type=int,default=96)
    parser.add_argument("--layers",type=int,default=2)
    parser.add_argument("--heads",type=int,default=4)
    parser.add_argument("--window",type=int,default=16)
    parser.add_argument("--mix-group",type=int,default=16)
    parser.add_argument("--mlp-ratio",type=int,default=2)


def add_train_flags(p):
    p.add_argument("--data",required=True,help="UTF-8 text file OR folder of .txt/.md/.py/.rs/.cpp etc")
    p.add_argument("--context",type=int,default=96)
    p.add_argument("--batch",type=int,default=8)
    p.add_argument("--steps",type=int,default=200)
    p.add_argument("--lr",type=float,default=4e-4)
    p.add_argument("--weight-decay",type=float,default=0.05)
    p.add_argument("--grad-clip",type=float,default=1.0)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--seed",type=int,default=20261027)
    p.add_argument("--max-mb",type=float,default=8.0)
    p.add_argument("--eval-every",type=int,default=50)
    p.add_argument("--eval-batches",type=int,default=8)
    p.add_argument("--grad-filter",choices=["none","phi","dft"],default="none",
                   help="Research ablation: project *computed* gradients into spectral top-k (extra compute)")
    p.add_argument("--grad-keep",type=float,default=0.50)


def main():
    a=argparse.ArgumentParser(description="PhiState: QuantoniumOS CPU-first hybrid spectral transformer research")
    sub=a.add_subparsers(dest="command",required=True)
    sp=sub.add_parser("smoke",help="Numerical correctness: unitarity, causal, FFT/recurrent, backward")
    sp.add_argument("--threads",type=int,default=4);sp.add_argument("--seed",type=int,default=20261027)
    sp=sub.add_parser("make-demo",help="Write synthetic smoke data; NOT general-language benchmark")
    sp.add_argument("--out",default=str(HERE/"demo.txt"))
    sp=sub.add_parser("train",help="Train new byte LM from scratch")
    add_train_flags(sp);add_model_flags(sp)
    sp.add_argument("--out",default=str(HERE/"checkpoints"/"phi.pt"))
    sp=sub.add_parser("compare",help="Fair-ish sequential phi vs dft vs local, equal data/seed")
    add_train_flags(sp);add_model_flags(sp,include_mode=False)
    sp.add_argument("--out-dir",default=str(HERE/"checkpoints"/"comparison"))
    sp=sub.add_parser("factorial",help="2x2 ablation: phi vs dft memory x phi vs dft channel mixing")
    add_train_flags(sp);add_model_flags(sp,include_mode=False)
    sp.add_argument("--out-dir",default=str(HERE/"checkpoints"/"factorial"))
    sp=sub.add_parser("evaluate",help="Held-out eval of an existing checkpoint")
    sp.add_argument("--checkpoint",required=True);sp.add_argument("--data",required=True)
    sp.add_argument("--split",choices=["val","test"],default="test")
    sp.add_argument("--max-mb",type=float,default=8.0)
    sp.add_argument("--batch",type=int,default=8)
    sp.add_argument("--eval-batches",type=int,default=16)
    sp.add_argument("--threads",type=int,default=4);sp.add_argument("--seed",type=int,default=20261027)
    sp=sub.add_parser("generate",help="Generate text with recurrent cache + sliding KV window")
    sp.add_argument("--checkpoint",required=True);sp.add_argument("--prompt",default="The kernel manages")
    sp.add_argument("--new-bytes",type=int,default=100)
    sp.add_argument("--temperature",type=float,default=0.8)
    sp.add_argument("--top-k",type=int,default=32)
    sp.add_argument("--greedy",action="store_true")
    sp.add_argument("--threads",type=int,default=4);sp.add_argument("--seed",type=int,default=20261027)
    args=a.parse_args()
    if hasattr(args,"threads") and args.threads<1: a.error("threads must be >=1")
    if hasattr(args,"grad_keep") and not 0<args.grad_keep<=1: a.error("grad-keep must be (0,1]")
    if hasattr(args,"steps") and (args.steps<1 or args.batch<1 or args.context<2 or args.eval_batches<1):
        a.error("steps, batch, eval-batches must be > 0 and context >= 2")
    if args.command=="smoke":smoke(args)
    elif args.command=="make-demo":make_demo(args)
    elif args.command=="train":train(args)
    elif args.command=="compare":compare(args)
    elif args.command=="factorial":factorial(args)
    elif args.command=="evaluate":evaluate(args)
    elif args.command=="generate":generate(args)


if __name__ == "__main__":
    main()
