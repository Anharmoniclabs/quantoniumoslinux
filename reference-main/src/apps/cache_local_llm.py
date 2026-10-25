#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""Cache a HuggingFace model locally for fully-offline use.

Why this exists
- "Local LLM" in quantoniumos uses HuggingFace Transformers.
- Once a model is cached, you can set QUANTONIUM_LOCAL_ONLY=1 and run without
  network access (no API calls, no downloads).

Typical usage
- python src/apps/cache_local_llm.py --model distilgpt2 --cache-dir ai/hf_cache
- QUANTONIUM_LOCAL_ONLY=1 HF_HOME=ai/hf_cache python src/apps/cli_chat.py

Notes
- This does NOT "remove datacenters" globally; it removes *your runtime dependency*
  on remote inference/training for day-to-day use.
"""

from __future__ import annotations

import argparse
import os

from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.getenv("QUANTONIUM_MODEL_ID", "distilgpt2"))
    ap.add_argument(
        "--cache-dir",
        default=os.getenv("HF_HOME") or os.getenv("QUANTONIUM_HF_HOME"),
        help="Where to store the HuggingFace cache (sets HF_HOME for this process)",
    )
    ap.add_argument(
        "--snapshot",
        action="store_true",
        help="Use huggingface_hub.snapshot_download (no model load, low RAM).",
    )
    args = ap.parse_args()

    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = args.cache_dir

    print(f"Caching model locally: {args.model}")
    if args.cache_dir:
        print(f"HF_HOME set to: {os.environ['HF_HOME']}")

    if args.snapshot:
        try:
            from huggingface_hub import snapshot_download
        except Exception as e:
            raise SystemExit(
                f"--snapshot requested but huggingface_hub import failed: {type(e).__name__}: {e}"
            )

        # Download the full repo snapshot without instantiating the model.
        # Important: Transformers typically reads from HF_HOME/hub, while
        # huggingface_hub's cache_dir defaults to HF_HOME/hub as well. If the
        # user passed --cache-dir, ensure we use the "hub" subdir so both
        # libraries share a single cache layout.
        cache_dir = args.cache_dir
        if cache_dir:
            cache_dir = os.path.join(cache_dir, "hub")
            os.makedirs(cache_dir, exist_ok=True)
        snapshot_download(
            repo_id=args.model,
            cache_dir=cache_dir,
            local_files_only=False,
        )
        print("Snapshot download complete.")
        print("Tip: set HF_HOME to your cache dir for offline runs.")
        return

    # Force download now (so later we can run with local_files_only=True).
    # This loads weights into RAM, so prefer --snapshot for large models.
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=False)
    _ = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=False)

    # Touch some common optional assets too
    try:
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    except Exception:
        pass

    print("Done. You can now run offline with:")
    if args.cache_dir:
        print(f"  HF_HOME={args.cache_dir} QUANTONIUM_LOCAL_ONLY=1 python src/apps/cli_chat.py")
    else:
        print("  QUANTONIUM_LOCAL_ONLY=1 python src/apps/cli_chat.py")


if __name__ == "__main__":
    main()
