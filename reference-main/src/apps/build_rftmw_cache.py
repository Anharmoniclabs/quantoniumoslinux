# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""Build an on-disk RFTMW compressed-weight cache for fast local startup.

This writes a single pack file (`*.rftmwpk`) that `QUANTONIUM_LOCAL_BACKEND=rftmw`
can reuse to skip re-compressing the base model on every launch.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Optional


def safe_model_name(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default=os.getenv("QUANTONIUM_RFTMW_MODEL_ID") or os.getenv("QUANTONIUM_MODEL_ID") or "",
        help="HF model id or local model path (default: env or auto)",
    )
    ap.add_argument("--cache_dir", default="ai/cache/rftmw")
    ap.add_argument("--cache_path", default="", help="Exact output pack path (overrides cache_dir).")
    ap.add_argument("--layer_limit", type=int, default=None)
    ap.add_argument("--entropy_threshold", type=float, default=0.40)
    ap.add_argument("--keep_ratio", type=float, default=0.30)
    ap.add_argument("--max_rft_error", type=float, default=0.08)
    ap.add_argument(
        "--max_rft_elems",
        type=int,
        default=2_000_000,
        help="Max tensor elements eligible for RFT (larger tensors fall back to INT8 for build-time practicality).",
    )
    ap.add_argument("--local_only", action="store_true", help="Do not allow HF downloads.")
    ap.add_argument(
        "--hf_home",
        default="",
        help="Optional HF_HOME to use for model files (recommended for offline cache: ai/hf_cache).",
    )
    ap.add_argument(
        "--lora_path",
        default="",
        help="Optional PEFT LoRA adapter dir to merge into the base model before compression.",
    )
    ap.add_argument("--force", action="store_true", help="Delete any existing cache first.")
    args = ap.parse_args()

    model_id = args.model.strip()
    if not model_id:
        # Prefer in-repo recovered HF model dir if present.
        if Path("ai/models/huggingface/DialoGPT-small").is_dir():
            model_id = "ai/models/huggingface/DialoGPT-small"
        else:
            model_id = "microsoft/DialoGPT-small"

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache_path.strip()) if args.cache_path.strip() else (cache_dir / f"{safe_model_name(model_id)}.rftmwpk")

    if args.force and cache_path.exists():
        cache_path.unlink()

    lora_path: Optional[str] = args.lora_path.strip() or None
    if lora_path:
        if not Path(lora_path).exists():
            raise SystemExit(f"--lora_path not found: {lora_path}")

    if args.hf_home.strip():
        os.makedirs(args.hf_home.strip(), exist_ok=True)
        os.environ["HF_HOME"] = args.hf_home.strip()

    # Tell the engine exactly where to store/read the cache.
    os.environ["QUANTONIUM_RFTMW_CACHE_PATH"] = str(cache_path)
    if args.local_only:
        os.environ["QUANTONIUM_LOCAL_ONLY"] = "1"
        # Force truly offline behavior (Transformers can otherwise attempt HEAD requests).
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    from quantonium_os_src.engine.rftmw_inference import CompressedInferenceEngine

    engine = CompressedInferenceEngine(
        model_name_or_path=model_id,
        entropy_threshold=args.entropy_threshold,
        weight_keep_ratio=args.keep_ratio,
        device="cpu",
        local_files_only=args.local_only,
        max_rft_elements=args.max_rft_elems,
    )
    engine.memory.max_rft_error = args.max_rft_error
    if lora_path:
        # Merge LoRA into base weights, then ingest that merged model into the pack.
        try:
            from peft import PeftModel
        except Exception as e:
            raise SystemExit(
                f"peft is required for --lora_path merge, but import failed: {type(e).__name__}: {e}"
            )

        engine._load_model()
        merged = PeftModel.from_pretrained(engine._model, lora_path)
        merged = merged.merge_and_unload()
        merged.eval()
        engine._model = merged

        # Recompute provenance for the merged weights.
        engine._provenance = engine._collect_provenance()

        # Bypass the cache-load path by temporarily hiding the file.
        # We want to always build a new pack from the merged model.
        if cache_path.exists():
            cache_path.unlink()

        import time as _time

        t0 = _time.perf_counter()
        engine.memory.ingest_model(engine._model, layer_limit=args.layer_limit, verbose=True)
        engine._compress_time = _time.perf_counter() - t0
        engine._compressed = True
        engine.memory.save_pack(
            str(cache_path),
            extra_meta={
                "model_name": model_id,
                "lora_path": str(lora_path),
                "fp32_size_bytes": engine._original_size_bytes,
                "provenance": dict(engine._provenance),
            },
        )
    else:
        engine.compress_model(layer_limit=args.layer_limit, verbose=True)

    prov = engine.provenance()
    print("")
    print(f"Cache pack: {cache_path}")
    if prov.get("state_dict_hash_sha256") not in (None, "model_not_loaded", "unavailable"):
        print(f"Model SHA-256: {prov['state_dict_hash_sha256']}")


if __name__ == "__main__":
    main()
