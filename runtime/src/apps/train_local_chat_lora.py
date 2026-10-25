# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""Train a small LoRA adapter on your local chat transcripts.

This is the simplest practical way to make the local model "smarter" for *your* domain
without retraining a full LLM.

Input
- Chatbox JSONL logs written by src/apps/qshll_chatbox.py in ./logs/chat_*.jsonl

Output
- A LoRA adapter directory (default: ai/training/models/eldrae_qwen3b_lora)

Usage
- python src/apps/train_local_chat_lora.py

After training, run chatbox with:
- QUANTONIUM_LORA_PATH=ai/training/models/eldrae_qwen3b_lora QUANTONIUM_MODEL_ID=Qwen/Qwen2.5-Coder-3B-Instruct python src/apps/qshll_chatbox.py

Notes
- Training is local and can be slow on CPU.
- Keep prompts non-agentic; this script does not add tool use.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.apps.training_guardrails import (
    GuardrailPolicy,
    GuardrailStats,
    sanitize_corpus_jsonl,
    sanitize_text,
)
from src.apps.smartweights import build_efficiency_block, build_lora_manifest, validate_manifest


def load_chat_pairs(log_glob: str, max_pairs: int = 2000) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for path in sorted(glob.glob(log_glob)):
        last_user = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                t = obj.get("type")
                if t == "user":
                    last_user = obj.get("text", "")
                elif t == "assistant" and last_user is not None:
                    pairs.append((last_user, obj.get("text", "")))
                    last_user = None
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def format_example(user_text: str, assistant_text: str) -> str:
    model_id = os.getenv("QUANTONIUM_MODEL_ID", "")
    mid = (model_id or "").lower()
    system_prompt = (
        "You are Eldrae, the seed local OS intelligence inside QuantoniumOS. "
        "You are fast, warm, concise, tool-aware, and honest about local limits."
    )
    if "qwen" in mid:
        return "\n".join(
            [
                f"<|im_start|>system\n{system_prompt}<|im_end|>",
                f"<|im_start|>user\n{user_text.strip()}<|im_end|>",
                f"<|im_start|>assistant\n{assistant_text.strip()}<|im_end|>",
            ]
        )
    if "tinyllama" in mid and "chat" in mid:
        # Provide a full supervised ChatML turn (prompt + target answer).
        return "\n".join(
            [
                f"<|system|>\n{system_prompt}</s>",
                f"<|user|>\n{user_text.strip()}</s>",
                f"<|assistant|>\n{assistant_text.strip()}</s>",
            ]
        )
    # Plain supervised pair.
    return "\n".join(
        [
            f"System: {system_prompt}",
            f"User: {user_text.strip()}",
            f"Assistant: {assistant_text.strip()}",
        ]
    )


TORCH_AVAILABLE: Optional[bool] = None


def _torch_import_healthcheck(timeout_s: float = 8.0) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import torch; print(getattr(torch,'__version__','ok'))"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode == 0:
            return True, (proc.stdout or "").strip()
        msg = (proc.stderr or proc.stdout or "").strip()
        return False, msg or f"torch import failed (exit {proc.returncode})"
    except subprocess.TimeoutExpired:
        return False, "torch import timed out"
    except Exception as e:
        return False, str(e)


def _lazy_import_training_stack():
    global TORCH_AVAILABLE
    if TORCH_AVAILABLE is True:
        import torch  # noqa: PLC0415
        from datasets import Dataset  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
        from peft import LoraConfig, get_peft_model  # noqa: PLC0415
        from trl import SFTTrainer, SFTConfig  # noqa: PLC0415

        return torch, Dataset, AutoModelForCausalLM, AutoTokenizer, LoraConfig, get_peft_model, SFTTrainer, SFTConfig

    ok, detail = _torch_import_healthcheck()
    if not ok:
        TORCH_AVAILABLE = False
        raise SystemExit(
            "PyTorch failed to import safely (subprocess check failed). "
            f"Details: {detail}"
        )

    import torch  # noqa: PLC0415
    from datasets import Dataset  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    from peft import LoraConfig, get_peft_model  # noqa: PLC0415
    from trl import SFTTrainer, SFTConfig  # noqa: PLC0415

    TORCH_AVAILABLE = True
    return torch, Dataset, AutoModelForCausalLM, AutoTokenizer, LoraConfig, get_peft_model, SFTTrainer, SFTConfig


def resolve_target_modules(model, mode: str) -> List[str]:
    """Pick LoRA target modules that match the loaded architecture."""
    if mode != "auto":
        return [m.strip() for m in mode.split(",") if m.strip()]

    module_names = [name for name, _ in model.named_modules()]

    # LLaMA/TinyLlama family.
    if any(name.endswith("q_proj") for name in module_names):
        return [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    # GPT-2/DialoGPT family.
    if any(name.endswith("c_attn") for name in module_names):
        return ["c_attn", "c_proj"]

    # Falcon family.
    if any(name.endswith("query_key_value") for name in module_names):
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

    # Conservative fallback.
    proj_like = sorted({
        name.split(".")[-1]
        for name in module_names
        if name.split(".")[-1].endswith("proj")
    })
    if proj_like:
        return proj_like
    return ["c_attn", "c_proj"]


def parse_layers_to_transform(raw: str | None) -> Optional[List[int]]:
    if not raw:
        return None
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError as exc:
            raise SystemExit(f"Invalid layer index in --layers_to_transform: {part}") from exc
    if not out:
        return None
    return sorted(set(out))


def resolve_layers_pattern(model, mode: str, layers_to_transform: Optional[List[int]]) -> Optional[str]:
    if not layers_to_transform:
        return None
    if mode and mode != "auto":
        return mode
    module_names = [name for name, _ in model.named_modules()]
    if any(".layers." in name for name in module_names):
        return "layers"
    if any(".h." in name for name in module_names):
        return "h"
    if any(".blocks." in name for name in module_names):
        return "blocks"
    return None


def count_parameters(model) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def run_rft_prestage(
    model,
    *,
    layer_limit: Optional[int],
    entropy_threshold: float,
    keep_ratio: float,
    max_rft_error: float,
    report_path: str,
) -> None:
    """Compress base weights with RFTMW and persist a report for reproducibility."""
    from quantonium_os_src.engine.rftmw_memory import RFTMWMemoryLayer

    mem = RFTMWMemoryLayer(
        entropy_threshold=entropy_threshold,
        weight_keep_ratio=keep_ratio,
        max_rft_error=max_rft_error,
    )
    report = mem.ingest_model(model, layer_limit=layer_limit, verbose=True)
    rows = mem.layer_report()

    payload = {
        "entropy_threshold": entropy_threshold,
        "weight_keep_ratio": keep_ratio,
        "max_rft_error": max_rft_error,
        "layer_limit": layer_limit,
        "summary": {
            "total_original_bytes": report.total_original_bytes,
            "total_compressed_bytes": report.total_compressed_bytes,
            "rft_layers": report.rft_layers,
            "int8_layers": report.int8_layers,
            "skip_layers": report.skip_layers,
            "compression_ratio": (
                report.total_original_bytes / max(report.total_compressed_bytes, 1)
            ),
        },
        "layers": rows,
    }

    out = Path(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved RFT pre-stage report: {out}")


def _apply_rftmw_quantized_base(
    model,
    *,
    layer_limit: Optional[int],
    entropy_threshold: float,
    keep_ratio: float,
    max_rft_error: float,
) -> Dict[str, object]:
    torch, *_rest = _lazy_import_training_stack()
    from quantonium_os_src.engine.rftmw_memory import RFTMWMemoryLayer

    mem = RFTMWMemoryLayer(
        entropy_threshold=entropy_threshold,
        weight_keep_ratio=keep_ratio,
        max_rft_error=max_rft_error,
    )
    mem.ingest_model(model, layer_limit=layer_limit, verbose=True)
    names = set(mem.weight_names())
    applied = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in names:
                continue
            arr = mem.get_weight(name)
            t = torch.from_numpy(arr).to(device=param.device, dtype=param.dtype)
            param.copy_(t.reshape(param.shape))
            applied += 1
    return {"applied_layers": applied, "layer_limit": layer_limit}


def cleanup_training_artifacts(output_dir: str) -> List[str]:
    out = Path(output_dir)
    removed: List[str] = []
    if not out.exists():
        return removed

    for checkpoint in sorted(out.glob("checkpoint-*")):
        shutil.rmtree(checkpoint, ignore_errors=True)
        removed.append(str(checkpoint))

    for name in ("trainer_state.json", "training_args.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        path = out / name
        if path.exists():
            path.unlink()
            removed.append(str(path))

    return removed


def _guardrails_filter_texts(texts: List[str], policy: GuardrailPolicy) -> Tuple[List[str], GuardrailStats]:
    stats = GuardrailStats()
    out: List[str] = []
    for t in texts:
        stats.rows_in += 1
        cleaned = sanitize_text(t, policy, stats)
        if cleaned is None:
            continue
        out.append(cleaned)
        stats.rows_out += 1
    return out, stats


def _repo_relative(path_text: str) -> str:
    path = Path(path_text)
    if not path.is_absolute():
        return path_text.replace("\\", "/")
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        raise SystemExit(f"SmartWeights paths must stay inside repo: {path_text}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default=os.getenv("QUANTONIUM_MODEL_ID", "Qwen/Qwen2.5-Coder-3B-Instruct"),
    )
    ap.add_argument("--logs", default="logs/chat_*.jsonl")
    ap.add_argument("--corpus", default=None, help="Path to a JSONL corpus with {'text': ...} rows")
    ap.add_argument("--out", default="ai/training/models/eldrae_qwen3b_lora")
    ap.add_argument("--max_pairs", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=-1, help="If >0, override epochs with a fixed number of optimizer steps.")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument(
        "--packing",
        action="store_true",
        help="Pack multiple short samples into each training sequence for better throughput.",
    )
    ap.add_argument(
        "--packing_strategy",
        default="bfd",
        help="TRL packing strategy to use when --packing is enabled.",
    )
    ap.add_argument("--local_only", action="store_true", help="Load model/tokenizer from local HF cache only.")
    ap.add_argument(
        "--target_modules",
        default="auto",
        help="LoRA target modules. Use 'auto' or comma-separated names (e.g., q_proj,v_proj,o_proj).",
    )
    ap.add_argument(
        "--layers_to_transform",
        default="",
        help="Optional comma-separated transformer layer indices to train (for microphase runs).",
    )
    ap.add_argument(
        "--layers_pattern",
        default="auto",
        help="PEFT layer container name when using --layers_to_transform. Use 'auto' to infer.",
    )
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--cleanup_checkpoints",
        action="store_true",
        help="Remove trainer checkpoints and transient state files after a successful save.",
    )
    ap.add_argument(
        "--rft_prestage",
        action="store_true",
        help="Run RFTMW compression on frozen base weights before LoRA training and save a report.",
    )
    ap.add_argument(
        "--rft_layer_limit",
        type=int,
        default=24,
        help="Max number of parameter tensors to RFT-compress in pre-stage (keeps runtime practical).",
    )
    ap.add_argument("--rft_entropy_threshold", type=float, default=0.40)
    ap.add_argument("--rft_keep_ratio", type=float, default=0.30)
    ap.add_argument("--rft_max_error", type=float, default=0.08)
    ap.add_argument(
        "--rft_report",
        default="ai/training/reports/rft_prestage.json",
        help="Where to write RFT pre-stage compression report JSON.",
    )
    ap.add_argument(
        "--train_report",
        default="ai/training/reports/trainability.json",
        help="Where to write parameter/trainability summary JSON.",
    )
    ap.add_argument(
        "--no_guardrails",
        action="store_true",
        help="Disable training guardrails (not recommended).",
    )
    ap.add_argument("--guardrails_report", default="ai/training/reports/guardrails.json")
    ap.add_argument("--guardrails_max_chars", type=int, default=32_000)
    ap.add_argument(
        "--rftmw_quantize_base",
        action="store_true",
        help="Apply RFTMW compression+decompression to base weights before LoRA training (trains in the same lossy space as RFTMW inference).",
    )
    ap.add_argument(
        "--smartweights_manifest",
        default="smartweights/packs/eldrae-core/smartweight.json",
        help="Where to write the SmartWeights manifest for the trained adapter.",
    )
    ap.add_argument("--smartweights_name", default="Eldrae Core SmartWeights")
    ap.add_argument("--smartweights_version", default=os.getenv("SMARTWEIGHTS_VERSION", "0.1.0"))
    ap.add_argument(
        "--no_smartweights_manifest",
        action="store_true",
        help="Do not write a SmartWeights manifest after training.",
    )
    args = ap.parse_args()

    # Ensure prompt formatting matches the model being trained.
    os.environ["QUANTONIUM_MODEL_ID"] = args.model

    torch, Dataset, AutoModelForCausalLM, AutoTokenizer, LoraConfig, get_peft_model, SFTTrainer, SFTConfig = _lazy_import_training_stack()

    if args.corpus:
        # Train on a multi-domain corpus (plain causal LM text)
        if not os.path.exists(args.corpus):
            raise SystemExit(f"Corpus not found: {args.corpus}")
        corpus_path = args.corpus
        if not args.no_guardrails:
            policy = GuardrailPolicy(
                drop_secrets=True,
                redact_pii=True,
                drop_tool_instructions=True,
                max_chars=max(0, int(args.guardrails_max_chars)),
            )
            src = Path(args.corpus)
            sanitized = Path("ai/training/datasets/_sanitized") / src.name
            report = Path(args.guardrails_report)
            corpus_report = report.with_name(report.stem + "_corpus.json")
            sanitize_corpus_jsonl(
                str(src),
                str(sanitized),
                policy=policy,
                report_path=str(corpus_report),
            )
            corpus_path = str(sanitized)
            print(f"Sanitized corpus -> {corpus_path} (report: {corpus_report})")

        ds = Dataset.from_json(corpus_path)
        if "text" not in ds.column_names:
            raise SystemExit("Corpus JSONL must contain a 'text' field")
        print(f"Loaded corpus: {corpus_path} rows={len(ds)}")
    else:
        # Train on your chat logs as instruction-style pairs
        pairs = load_chat_pairs(args.logs, max_pairs=args.max_pairs)
        if not pairs:
            raise SystemExit(f"No training pairs found in {args.logs}. Chat a bit first to create logs/chat_*.jsonl")
        texts = [format_example(u, a) for (u, a) in pairs]
        if args.no_guardrails:
            ds = Dataset.from_dict({"text": texts})
            print(f"Loaded chat pairs: {len(texts)}")
        else:
            policy = GuardrailPolicy(
                drop_secrets=True,
                redact_pii=True,
                drop_tool_instructions=True,
                max_chars=max(0, int(args.guardrails_max_chars)),
            )
            cleaned, stats = _guardrails_filter_texts(texts, policy)
            ds = Dataset.from_dict({"text": cleaned})
            print(f"Loaded chat pairs: {len(cleaned)} (from {len(texts)}; dropped={len(texts)-len(cleaned)})")
            Path(args.guardrails_report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.guardrails_report).write_text(
                json.dumps(
                    {
                        "policy": {
                            "drop_secrets": policy.drop_secrets,
                            "redact_pii": policy.redact_pii,
                            "drop_tool_instructions": policy.drop_tool_instructions,
                            "max_chars": policy.max_chars,
                        },
                        "stats": {
                            "rows_in": stats.rows_in,
                            "rows_out": stats.rows_out,
                            "dropped_secrets": stats.dropped_secrets,
                            "dropped_tool_instructions": stats.dropped_tool_instructions,
                            "truncated": stats.truncated,
                            "redacted_emails": stats.redacted_emails,
                            "redacted_phones": stats.redacted_phones,
                            "redacted_ssn": stats.redacted_ssn,
                            "redacted_private_keys": stats.redacted_private_keys,
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"Saved guardrails report: {args.guardrails_report}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=args.local_only)
    model.config.use_cache = False
    try:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
    except Exception:
        pass

    if args.rftmw_quantize_base:
        info = _apply_rftmw_quantized_base(
            model,
            layer_limit=args.rft_layer_limit,
            entropy_threshold=args.rft_entropy_threshold,
            keep_ratio=args.rft_keep_ratio,
            max_rft_error=args.rft_max_error,
        )
        print(f"Applied RFTMW-quantized base weights: {info}")

    if args.rft_prestage:
        run_rft_prestage(
            model,
            layer_limit=args.rft_layer_limit,
            entropy_threshold=args.rft_entropy_threshold,
            keep_ratio=args.rft_keep_ratio,
            max_rft_error=args.rft_max_error,
            report_path=args.rft_report,
        )

    target_modules = resolve_target_modules(model, args.target_modules)
    layers_to_transform = parse_layers_to_transform(args.layers_to_transform)
    layers_pattern = resolve_layers_pattern(model, args.layers_pattern, layers_to_transform)
    print(f"LoRA target modules: {target_modules}")
    if layers_to_transform:
        print(f"LoRA layers_to_transform: {layers_to_transform} pattern={layers_pattern}")

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        layers_to_transform=layers_to_transform,
        layers_pattern=layers_pattern,
    )

    model = get_peft_model(model, lora)
    total_params, trainable_params = count_parameters(model)
    pct = 100.0 * trainable_params / max(total_params, 1)
    print(f"Model parameters: total={total_params:,} trainable={trainable_params:,} ({pct:.3f}%)")

    train_report_path = Path(args.train_report)
    train_report_path.parent.mkdir(parents=True, exist_ok=True)
    train_report_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "target_modules": target_modules,
                "layers_to_transform": layers_to_transform,
                "layers_pattern": layers_pattern,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "trainable_percent": pct,
                "dataset_rows": len(ds),
                "max_seq_len": args.max_seq_len,
                "epochs": args.epochs,
                "max_steps": args.max_steps,
                "batch": args.batch,
                "grad_accum": args.grad_accum,
                "packing": args.packing,
                "packing_strategy": args.packing_strategy if args.packing else None,
                "learning_rate": args.lr,
                "local_only": args.local_only,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved trainability report: {train_report_path}")

    use_cuda = bool(torch.cuda.is_available())
    training_args = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        use_cpu=not use_cuda,
        optim="adamw_torch",
        fp16=use_cuda,
        gradient_checkpointing=False,
        report_to=[],
        dataset_text_field="text",
        max_length=args.max_seq_len,
        packing=args.packing,
        packing_strategy=args.packing_strategy,
        padding_free=False,
        use_cache=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=training_args,
    )

    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    trainer.model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    print(f"Saved LoRA adapter to: {args.out}")
    if args.cleanup_checkpoints:
        removed = cleanup_training_artifacts(args.out)
        if removed:
            print(f"Removed transient training artifacts: {len(removed)}")

    if not args.no_smartweights_manifest:
        datasets = [_repo_relative(args.corpus)] if args.corpus else [_repo_relative(args.logs)]
        seed_corpus = Path("smartweights/training/eldrae_seed.jsonl")
        if seed_corpus.exists() and seed_corpus.as_posix() not in datasets:
            datasets.append(seed_corpus.as_posix())
        reports = [_repo_relative(args.train_report)]
        if not args.no_guardrails:
            reports.append(_repo_relative(args.guardrails_report))
        if args.rft_prestage:
            reports.append(_repo_relative(args.rft_report))

        build_lora_manifest(
            out_path=_repo_relative(args.smartweights_manifest),
            adapter_path=_repo_relative(args.out),
            name=args.smartweights_name,
            version=args.smartweights_version,
            runtime_model="qwen2.5-coder:3b",
            training_model=args.model,
            datasets=datasets,
            reports=reports,
            capabilities=["chat", "memory", "retrieval", "tool_planning", "visual_programs"],
            efficiency=build_efficiency_block(
                model="qwen2.5-coder:3b",
                rank=args.lora_r,
                target_modules=target_modules,
                actual_trainable_parameters=trainable_params,
            ),
            optional_artifact=False,
            notes="Generated by train_local_chat_lora.py after saving the Eldrae adapter.",
        )
        errors, warnings = validate_manifest(args.smartweights_manifest, strict=True)
        for warning in warnings:
            print(f"SmartWeights warning: {warning}")
        if errors:
            raise SystemExit("SmartWeights manifest validation failed: " + "; ".join(errors))
        print(f"Saved SmartWeights manifest to: {args.smartweights_manifest}")


if __name__ == "__main__":
    main()
