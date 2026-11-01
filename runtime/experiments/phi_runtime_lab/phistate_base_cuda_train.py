"""CUDA training on identical repaired next-token windows for a dense LM and PhiState."""

import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from chat_recovery_data import sha
from chat_recovery_train import atomic_json, run_lock
from coherence_recovery_train import evaluation_rng
from data_comparison_train import _interrupt_flag
from moe_model import MoEConfig, PhiStateMoELM
from moe_train import atomic_checkpoint

import phistate_base_windows as windows
from phistate_base_reference import DenseCausalLM

STEPS, BATCH, CONTEXT, LR = 512, 8, 128, 0.0003
ARMS = ("dense", "phistate", "warm")
BASE_SHA256 = "50f9bfe9b63082041b201bd6420531cbe6384cfc3ef5cce097588d8ec2396f46"


def require_a100():
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError(
            "Select an NVIDIA A100 GPU in Colab and connect the runtime before training."
        )
    return torch.device("cuda", 0)


def digest(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _cohort(name):
    for family, tag in [
        ("router", ".router."),
        ("experts", ".experts."),
        ("shared_expert", ".shared."),
        ("attention", "attn"),
        ("embedding", "embedding"),
        ("embedding", "embed"),
        ("recurrent", "memory"),
        ("recurrent", "ssm"),
    ]:
        if tag in name:
            return family
    return "rest"


@torch.no_grad()
def _displacement(model, origin):
    groups = {}
    for (name, p), start in zip(model.named_parameters(), origin):
        bucket = groups.setdefault(_cohort(name), [[], [], []])
        delta = (p - start).float().square().sum()
        bucket[0].append(delta)
        bucket[1].append(start.float().square().sum())
        bucket[2].append(delta > 0)
    totals = {
        key: [float(torch.stack(values).sum()) for values in entry] for key, entry in groups.items()
    }

    def metric(values):
        return {
            "relative_l2": math.sqrt(values[0] / max(values[1], 1e-30)),
            "changed_tensors": int(values[2]),
        }

    return {
        "all": metric([sum(elem[i] for elem in totals.values()) for i in range(3)]),
        "groups": {k: metric(v) for k, v in totals.items()},
    }


def _load_base(path, corpus):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    manifest = corpus.manifest
    if (
        checkpoint.get("format") != "PhiStateMoE-v1"
        or checkpoint.get("training_stage") is not None
        or checkpoint.get("sft_steps")
        or checkpoint.get("data_fingerprint") != manifest["data_fingerprint"]
        or checkpoint.get("tokenizer") != "bpe"
        or checkpoint["config"]["vocab"] != manifest["vocab"]
        or checkpoint.get("step") != 66000
        or checkpoint.get("train_args", {}).get("optimizer") != "adafactor"
        or not checkpoint.get("optimizer_state", {}).get("state")
    ):
        raise ValueError(
            "Require verified repaired 66,000-step PhiState base and saved Adafactor state."
        )
    return checkpoint


def _source_model(cfg, arm, context, source):
    if arm == "dense":
        return DenseCausalLM(
            vocab=cfg["vocab"],
            width=cfg["width"],
            layers=cfg["layers"],
            heads=cfg["heads"],
            context=context,
            hidden=6 * cfg["width"],
        )
    model = PhiStateMoELM(MoEConfig(**cfg))
    if arm == "warm":
        model.load_state_dict(source["state_dict"], strict=True)
    return model


def _make_optimizer(model, arm, source, device):
    if arm == "warm":
        saved = source["optimizer_state"]
        if len(saved["param_groups"]) != 1 or not saved["state"]:
            raise ValueError("Expected a complete single-group Adafactor source.")
        opt = torch.optim.Adafactor(
            model.parameters(), lr=float(saved["param_groups"][0]["lr"]), foreach=False
        )
        opt.load_state_dict(saved)
        return opt
    return torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        foreach=False,
        capturable=device.type == "cuda",
    )


def _optimizer_device(opt, device):
    state_tensors = [
        v
        for state in opt.state.values()
        for k, v in state.items()
        if torch.is_tensor(v) and k != "step"
    ]
    if not state_tensors or any(t.device != device for t in state_tensors):
        raise RuntimeError("Optimizer update state is missing or off the training device")
    return str(device)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _train(model, opt, corpus, plan, validation, out, req, device, source_meta, stop_path, owner):
    """Private implementation supports explicit miniature CPU tests; the public entry requires A100."""
    steps, batch, context = req["steps"], req["batch"], req["context"]
    arm = req["arm"]
    ck, progress = out / "latest.pt", out / "progress.json"
    if any(
        not p.requires_grad or p.device != device or p.dtype != torch.float32
        for p in model.parameters()
    ):
        raise RuntimeError(
            "Every model parameter must be trainable FP32 and resident on the training device"
        )
    origin = [p.detach().clone() for p in model.parameters()]
    count = sum(p.numel() for p in model.parameters())
    active = model.estimate_active_parameters() if isinstance(model, PhiStateMoELM) else count
    curve = []
    history = []
    phase = 0
    train_s = 0.0
    eval_s = 0.0
    peak = 0
    witness = {}
    if ck.exists():
        s = torch.load(ck, map_location="cpu", weights_only=True)
        if s["experiment_request"] != req or s["phase_step"] not in range(steps + 1):
            raise ValueError("Checkpoint has a different request or invalid phase")
        phase = s["phase_step"]
        if len(s["history"]) != phase or s["phase_trained_tokens"] != phase * batch * context:
            raise ValueError("Saved exposure disagrees with committed optimizer updates")
        model.load_state_dict(s["state_dict"], strict=True)
        opt.load_state_dict(s["optimizer_state"])
        torch.set_rng_state(s["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(s["cuda_rng_state"], device)
        curve, history = s["development_curve"], s["history"]
        train_s, eval_s = s["phase_train_seconds"], s["phase_eval_seconds"]
        peak = s["peak_bytes"]
        witness = s["device_witness"]
        del s
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def measure():
        nonlocal eval_s
        _sync(device)
        start = time.monotonic()
        with evaluation_rng(device), torch.inference_mode():
            model.eval()
            metrics = windows.evaluate(model, corpus, validation, device)
            delta = _displacement(model, origin)
        _sync(device)
        eval_s += time.monotonic() - start
        curve.append(
            {
                "step": phase,
                "trained_tokens": phase * batch * context,
                "metrics": metrics,
                "weight_update": delta,
            }
        )

    def record(status):
        nonlocal peak
        if device.type == "cuda":
            peak = max(peak, torch.cuda.max_memory_allocated(device))
        exposures = phase * batch * context
        return {
            "status": status,
            "completed_steps": phase,
            "trained_tokens": exposures,
            "target_steps": steps,
            "arm": arm,
            "seed": req["seed"],
            "curve": curve,
            "latest": str(ck),
            "request": req,
            "lifetime_step": source_meta.get("step", 0) + phase,
            "lifetime_trained_tokens": source_meta.get("trained_tokens", 0) + exposures,
            "device_receipt": {
                **witness,
                "device": str(device),
                "gpu": req["gpu"],
                "dtype": "float32",
                "parameters_total": count,
                "parameters_active_estimate": active,
                "peak_allocated_bytes": peak,
                "training_seconds": train_s,
                "validation_seconds": eval_s,
                "training_tokens_per_second": exposures / train_s if train_s else None,
                "parameter_matmul_flops_proxy": 6 * active * exposures,
                "proxy_scope": "6 times active parameters times targets; attention, spectral mixing, routing and dispatch costs are not measured by this proxy.",
            },
        }

    def save(status, *, write_weights=True):
        info = record(status)
        if write_weights:
            payload = {
                **source_meta,
                "format": "PhiStateMoE-v1" if arm != "dense" else "DenseCausalLM-v1",
                "config": req["model_config"],
                "state_dict": model.state_dict(),
                "step": info["lifetime_step"],
                "trained_tokens": info["lifetime_trained_tokens"],
                "data_fingerprint": req["data_fingerprint"],
                "optimizer_state": opt.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "experiment_request": req,
                "phase_step": phase,
                "phase_trained_tokens": phase * batch * context,
                "phase_train_seconds": train_s,
                "phase_eval_seconds": eval_s,
                "development_curve": curve,
                "history": history,
                "peak_bytes": peak,
                "device_witness": witness,
                "last_validation": curve[-1]["metrics"] if curve else None,
                "best_validation_bpt": min(
                    (v["metrics"]["mean_bpt"] for v in curve), default=float("inf")
                ),
            }
            if device.type == "cuda":
                payload["cuda_rng_state"] = torch.cuda.get_rng_state(device)
            if not all(torch.isfinite(p).all().item() for p in model.parameters()):
                raise FloatingPointError("Nonfinite parameters; last verified checkpoint retained.")
            owner()
            atomic_checkpoint(ck, payload)
        owner()
        atomic_json(progress, info)
        return info

    if phase == steps:
        if not curve or curve[-1]["step"] != steps:
            raise RuntimeError("Completed checkpoint lacks its endpoint evaluation")
        return save("complete", write_weights=False)

    def stop():
        return interrupted[0] or (out / "STOP").exists() or (stop_path and stop_path.exists())

    with _interrupt_flag() as interrupted:
        if stop():
            return save("stopped")
        if not curve:
            measure()
            save("training")
        for s in range(phase + 1, steps + 1):
            if stop():
                return save("stopped")
            owner()
            _sync(device)
            start = time.monotonic()
            x, y = windows.materialize(corpus, plan["steps"][s - 1], device)
            if (
                x.device != device
                or y.device != device
                or tuple(x.shape) != (batch, context)
                or x.shape != y.shape
            ):
                raise RuntimeError("Training inputs/targets are not the declared device and shape")
            model.train()
            opt.zero_grad(set_to_none=True)
            z = model(x)
            lm = F.cross_entropy(z.flatten(0, 1), y.flatten(), reduction="mean")
            router = None
            penalty = lm.new_zeros(())
            if isinstance(model, PhiStateMoELM):
                penalty, router = model.router_penalty()
            objective = lm + penalty
            if z.device != device or lm.device != device or not torch.isfinite(objective).item():
                raise FloatingPointError("Invalid training logits/objective; safe checkpoint retained.")
            objective.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not grads or any(g.device != device for g in grads):
                raise RuntimeError("Backward produced absent/off-device gradients")
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            opt_device = _optimizer_device(opt, device)
            witness = {
                "parameter_device": str(next(model.parameters()).device),
                "input_device": str(x.device),
                "target_device": str(y.device),
                "logit_device": str(z.device),
                "loss_device": str(lm.device),
                "grad_device": str(grads[0].device),
                "optimizer_moment_device": opt_device,
            }
            phase = s
            entry = {
                "step": s,
                "lm_bpt": float(lm.detach()) / math.log(2),
                "router_penalty": float(penalty.detach()),
                "preclip_gradient_norm": float(grad),
                "tokens": batch * context,
            }
            if router is not None:
                entry["router"] = router
            history.append(entry)
            del z, lm, objective, penalty, grads
            _sync(device)
            train_s += time.monotonic() - start
            if s in (max(1, steps // 2), steps):
                measure()
            if s == steps or s % 32 == 0 or stop() or curve[-1]["step"] == s:
                status = "complete" if s == steps else "stopped" if stop() else "training"
                info = save(status)
                print(
                    f"{device} {arm} seed {req['seed']}: {s}/{steps} updates | "
                    f"{info['trained_tokens']:,} next-token targets | "
                    f"{info['device_receipt']['training_tokens_per_second']:.1f} targets/s",
                    flush=True,
                )
                if status != "training":
                    return info
    raise RuntimeError("Training loop ended without a terminal record")


def run_arm(
    base_checkpoint,
    corpus,
    output,
    *,
    arm,
    seed,
    plan,
    validation,
    steps=STEPS,
    batch=BATCH,
    context=CONTEXT,
    stop_path=None,
    ownership_check=None,
):
    device = require_a100()
    if arm not in ARMS or not 1 <= steps <= 4096 or not 1 <= batch <= 64 or not 2 <= context <= 512:
        raise ValueError("Invalid arm or training budget")
    owner = ownership_check or (lambda: None)
    parent, out = Path(base_checkpoint).resolve(), Path(output).resolve()
    if (
        out == parent.parent
        or out == parent
        or len(plan["steps"]) != steps
        or any(len(g) != batch for g in plan["steps"])
    ):
        raise ValueError("Use an isolated output with a matching complete window schedule")
    identity = windows.digest({k: v for k, v in plan.items() if k != "fingerprint"})
    if (
        plan.get("fingerprint") != identity
        or plan.get("data_fingerprint") != corpus.fingerprint
        or plan.get("context") != context
        or plan.get("batch_size") != batch
        or plan.get("seed") != seed
        or any(
            d.get("split") != "train" or d.get("context") != context
            for group in plan["steps"]
            for d in group
        )
    ):
        raise ValueError(
            "Training windows differ from their declared seed, fingerprint, split or dimensions"
        )
    if set(validation) != set(windows.DOMAINS) or any(
        not rows
        or any(
            d.get("split") != "val" or d.get("domain") != key or d.get("context") != context
            for d in rows
        )
        for key, rows in validation.items()
    ):
        raise ValueError("Evaluation windows must be held-out windows from both natural domains")
    parent_hash = sha(parent)
    if parent_hash != BASE_SHA256:
        raise RuntimeError("Base checkpoint is not the archived repaired 66K source")
    source = _load_base(parent, corpus)
    if sha(parent) != parent_hash:
        raise RuntimeError("Base checkpoint changed while loading")
    cfg = source["config"]
    actual_cfg = (
        cfg
        if arm != "dense"
        else {
            "vocab": cfg["vocab"],
            "width": cfg["width"],
            "layers": cfg["layers"],
            "heads": cfg["heads"],
            "context": context,
            "hidden": 6 * cfg["width"],
        }
    )
    request = {
        "version": "PhiState-base-A100-arm-v1",
        "arm": arm,
        "seed": int(seed),
        "steps": steps,
        "batch": batch,
        "context": context,
        "data_fingerprint": corpus.manifest["data_fingerprint"],
        "base_sha256": parent_hash,
        "schedule_sha256": plan["fingerprint"],
        "validation_sha256": digest(validation),
        "model_config": actual_cfg,
        "source_step": source["step"] if arm == "warm" else 0,
        "optimizer": "saved Adafactor"
        if arm == "warm"
        else "fresh AdamW lr=.0003 betas=(.9,.95) wd=.01",
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "code": {
            p.name: sha(p)
            for p in (
                Path(__file__),
                Path(windows.__file__),
                Path(__file__).with_name("phistate_base_reference.py"),
            )
        },
    }
    stop_path = Path(stop_path) if stop_path is not None else None
    out.mkdir(exist_ok=True, parents=True)
    with run_lock(out), evaluation_rng(device):
        manifest = out / "request.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != request:
                raise RuntimeError("Saved arm request differs; refusing restart or extension")
        else:
            if any(p.name not in ("STOP", "request.partial") for p in out.iterdir()):
                raise RuntimeError("Unrecognized files in new arm folder")
            owner()
            atomic_json(manifest, request)
        prev_matmul = torch.backends.cuda.matmul.allow_tf32
        prev_cudnn = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            model = _source_model(cfg, arm, context, source).to(device)
            opt = _make_optimizer(model, arm, source, device)
            meta = (
                {
                    "step": source["step"],
                    "trained_tokens": source["trained_tokens"],
                    "tokenizer": "bpe",
                    "train_args": source["train_args"],
                    "corpus_migration": source.get("corpus_migration"),
                }
                if arm == "warm"
                else {}
            )
            del source
            info = _train(
                model, opt, corpus, plan, validation, out, request, device, meta, stop_path, owner
            )
            if sha(parent) != parent_hash:
                raise RuntimeError("Original repaired source changed externally during training")
            return info
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_matmul
            torch.backends.cudnn.allow_tf32 = prev_cudnn
