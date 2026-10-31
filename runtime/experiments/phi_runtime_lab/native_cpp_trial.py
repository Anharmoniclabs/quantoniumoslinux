#!/usr/bin/env python3
"""Export PhiState MoE FP32 weights, verify native CPU logits, generate in C++.

Python/PyTorch only load/export the checkpoint and supply the parity reference.
All native model forward passes use phistate_moe_cli and RFTMW's FP32 kernel.
This is an inference trial, not a C++ training implementation.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time

import numpy as np
import torch

from moe_model import MoEConfig, PhiStateMoELM

LAB = Path(__file__).resolve().parent
NATIVE = LAB.parents[1] / "src" / "rftmw_native"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def compile_native(output):
    compiler = shutil.which("g++")
    if not compiler:
        raise RuntimeError("g++ is required; in Colab: !apt-get -qq install g++")
    subprocess.run([compiler, "-O3", "-march=native", "-std=c++17", "-pthread",
                    str(NATIVE / "phistate_moe_cli.cpp"), "-o", str(output)], check=True)


@torch.inference_mode()
def export_model(model, path):
    model = model.cpu().eval()
    c = model.cfg
    c.validate()
    with Path(path).open("wb") as f:
        f.write(b"PHIMOE1\n")
        f.write(struct.pack("<11i", c.width, c.layers, c.heads, c.window, c.mix_group,
                            c.vocab, c.experts, c.top_k, c.expert_ratio, c.shared_ratio,
                            int(c.mode != "local")))

        def write(t):
            a = t.detach().cpu().float().contiguous().numpy().astype("<f4", copy=False)
            if not np.isfinite(a).all():
                raise ValueError("Nonfinite exported tensor")
            f.write(a.tobytes())

        def expert(e):
            write(e.fc.weight)
            write(e.out.weight)

        write(model.embedding.weight)
        write(model.norm.weight)
        for b in model.blocks:
            write(b.norm1.weight)
            write(b.norm2.weight)
            write(b.attn.qkv.weight)
            write(b.attn.proj.weight)
            write(b.attn.bias)
            if b.memory is not None:
                m = b.memory
                write(m.input_proj.weight)
                write(m.input_gate.weight)
                write(m.output_proj.weight)
                a, readout = m.parameters_for_kernel()
                for t in (a.real, a.imag, readout.real, readout.imag):
                    write(t)
            write(b.ff.router.weight)
            for e in b.ff.experts:
                expert(e)
            if b.ff.shared is not None:
                expert(b.ff.shared)
            if b.mixer is not None:
                write(b.norm3.weight)
                # Fold the fixed checkpoint's spectral mixer into its equivalent
                # real matrix. C++ GEMV needs the transpose of row-vector x @ M.
                u = b.mixer.basis
                gain = 1 + 0.5 * torch.tanh(b.mixer.gain_raw)
                matrix = ((u.conj() * gain.unsqueeze(0)) @ u.T).real
                write(matrix.T)


def decode_arguments(decode_config=None):
    """Validate optional native decoding controls; None preserves the old CLI."""
    if decode_config is None:
        return []
    config = dict(temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.0, seed=0)
    if set(decode_config) - set(config):
        raise ValueError('Unknown native decoding option')
    config.update(decode_config)
    for key in ('temperature', 'top_p', 'repetition_penalty'):
        config[key] = float(config[key])
        if not math.isfinite(config[key]):
            raise ValueError('Nonfinite ' + key)
    if config['temperature'] < 0 or not 0 < config['top_p'] <= 1 or config['repetition_penalty'] < 1:
        raise ValueError('Invalid native decoding configuration')
    if (not isinstance(config['top_k'], int) or isinstance(config['top_k'], bool)
            or not 0 <= config['top_k'] <= 2**31-1):
        raise ValueError('top_k must be a nonnegative integer')
    if (not isinstance(config['seed'], int) or isinstance(config['seed'], bool)
            or not 0 <= config['seed'] <= 2**64-1):
        raise ValueError('seed must be an unsigned 64-bit integer')
    return [str(config[k]) for k in ('temperature', 'top_p', 'top_k', 'repetition_penalty', 'seed')]


def native_run(binary, weights, ids, vocab, directory, new_tokens=0, eos=-1, decode_config=None):
    directory = Path(directory)
    token_file, logit_file = directory / "tokens.u32", directory / "logits.f32"
    np.asarray(ids, dtype="<u4").tofile(token_file)
    started = time.perf_counter()
    result = subprocess.run([str(binary), str(weights), str(token_file), str(logit_file),
                             str(new_tokens), str(eos)] + decode_arguments(decode_config),
                            check=True, capture_output=True, text=True)
    stats = json.loads(result.stdout)
    stats["process_seconds_including_model_load"] = time.perf_counter() - started
    logits = np.fromfile(logit_file, dtype="<f4").reshape(len(ids), vocab)
    return logits, stats


@torch.inference_mode()
def parity(model, ids, native_logits):
    x = torch.tensor([ids], dtype=torch.long)
    start = time.perf_counter()
    cache, pieces = None, []
    for token in ids:
        y, cache = model.step(torch.tensor([[token]], dtype=torch.long), cache)
        pieces.append(y)
    elapsed = time.perf_counter() - start
    streamed = torch.cat(pieces, dim=1)[0].numpy()
    full = model(x)[0].numpy()
    error = np.abs(streamed - native_logits)
    full_error = np.abs(full - native_logits)
    # A mismatch in routing near a tied expert score is a real parity failure;
    # do not silently change the architecture or tolerance to hide it.
    close = bool(np.allclose(streamed, native_logits, rtol=1e-4, atol=1e-3))
    full_close = bool(np.allclose(full, native_logits, rtol=1e-4, atol=1e-3))
    agreement = float(np.mean(streamed.argmax(-1) == native_logits.argmax(-1)))
    return {"passed": close and full_close and agreement == 1.0,
            "tokens": len(ids), "max_logit_error": float(error.max()),
            "mean_logit_error": float(error.mean()), "full_max_error": float(full_error.max()),
            "top1_agreement": agreement, "torch_cpu_step_seconds": elapsed}


def run(args):
    import sentencepiece as spm
    torch.set_num_threads(args.threads)
    os.environ["RFTMW_THREADS"] = str(args.threads)
    data, checkpoint, output = Path(args.data_dir), Path(args.checkpoint), Path(args.output)
    if output.exists():
        raise RuntimeError(f"Output already exists; keep the old trial and choose a new output directory: {output}")
    manifest = json.loads((data / "manifest.json").read_text())
    checkpoint_hash = sha(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if sha(checkpoint) != checkpoint_hash:
        raise RuntimeError("Checkpoint changed during loading; retry with a completed checkpoint")
    if state.get("format") != "PhiStateMoE-v1" or state.get("data_fingerprint") != manifest["data_fingerprint"]:
        raise RuntimeError("Checkpoint and corpus do not match")
    tokenizer = spm.SentencePieceProcessor(model_file=str(data / "tokenizer.model"))
    expected_tokenizer = manifest.get("whitespace_repair", {}).get("tokenizer_sha256")
    if expected_tokenizer and sha(data / "tokenizer.model") != expected_tokenizer:
        raise RuntimeError("Tokenizer hash differs from repaired manifest")
    model = PhiStateMoELM(MoEConfig(**state["config"]))
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    if model.cfg.vocab != tokenizer.vocab_size():
        raise RuntimeError("Tokenizer vocabulary differs")
    step = int(state["step"])
    del state
    output.mkdir(parents=True)
    report = {"step": step, "checkpoint_sha256": checkpoint_hash,
              "data_fingerprint": manifest["data_fingerprint"],
              "config": vars(model.cfg), "threads": args.threads,
              "scope": "Native FP32 CPU inference only; no native backward pass or optimizer",
              "status": "checking", "parity": [], "generation": []}

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        with tempfile.TemporaryDirectory(prefix="phistate_native_") as tmp:
            tmp = Path(tmp)
            binary, weights = tmp / "phistate_moe_cli", tmp / "model.phimoe"
            print("Compiling C++ adapter with the existing RFTMW FP32 kernel...", flush=True)
            compile_native(binary)
            export_model(model, weights)
            report["native_model_sha256"] = sha(weights)
            cases = []
            for domain in ("code", "general"):
                spec = manifest["bins"][f"val_{domain}"]
                bin_path = data / spec["file"]
                if sha(bin_path) != spec["sha256"]:
                    raise RuntimeError(f"Validation bin hash mismatch: {domain}")
                tokens = np.memmap(bin_path, dtype=manifest["dtype"], mode="r")
                if len(tokens) < 128:
                    raise RuntimeError("Validation stream too short")
                cases.append((f"validation_{domain}", tokens[:128].astype(int).tolist()))
                del tokens
            prompts = [("python_add", "code", "def add(a, b):\n"),
                       ("python_fibonacci", "code", "def fibonacci(n):\n"),
                       ("ram_vs_storage", "general", "User: What is the difference between RAM and storage?\nAssistant: "),
                       ("operating_system", "general", "The purpose of an operating system is")]
            for name, domain, text in prompts:
                cases.append((name, tokenizer.encode(f"<|{domain}|>\n" + text)))
            for name, ids in cases:
                native, stats = native_run(binary, weights, ids, model.cfg.vocab, tmp)
                result = {"case": name, **parity(model, ids, native),
                          "native_seconds": stats["seconds"],
                          "native_process_seconds": stats["process_seconds_including_model_load"]}
                report["parity"].append(result)
                save()
                print(json.dumps(result), flush=True)
                if not result["passed"]:
                    raise RuntimeError("Native parity failed; generation withheld. See report.json")
            print("Parity passed. Generating with the C++ model...", flush=True)
            for name, domain, prompt in prompts:
                ids = tokenizer.encode(f"<|{domain}|>\n" + prompt)
                _, stats = native_run(binary, weights, ids, model.cfg.vocab, tmp,
                                      args.new_tokens, tokenizer.eos_id())
                generated = stats["generated_ids"]
                # Also compare logits along the native-generated trajectory.
                trajectory = ids + generated
                native, _ = native_run(binary, weights, trajectory, model.cfg.vocab, tmp)
                check = parity(model, trajectory, native)
                decoded = tokenizer.decode(trajectory)
                header = f"<|{domain}|>\n"
                text = decoded[len(header):] if decoded.startswith(header) else decoded
                report["generation"].append({"case": name, "text": text, **stats,
                                             "trajectory_parity": check})
                save()
                if not check["passed"]:
                    raise RuntimeError("Generated-trajectory parity failed; inspect report.json")
                print(f"\n{name} | C++ greedy\n{text}", flush=True)
            shutil.copy2(weights, output / "model.phimoe")
            shutil.copy2(binary, output / "phistate_moe_cli")
            shutil.copy2(data / "tokenizer.model", output / "tokenizer.model")
        report["status"] = "passed"
        save()
        print(f"NATIVE TRIAL COMPLETE: {output / 'report.json'}", flush=True)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        save()
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("checkpoint", "data-dir", "output"):
        p.add_argument("--" + flag, required=True)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--new-tokens", type=int, default=64)
    a = p.parse_args()
    if a.threads < 1 or not 1 <= a.new_tokens <= 4096:
        p.error("Invalid thread count or generation length")
    run(a)
