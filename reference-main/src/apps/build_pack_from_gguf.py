# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""Build a minimal .rftmwpk from a GGUF file.

The pack itself contains only the tensor index (names + shapes); the
engine reads actual weight bytes directly from the referenced GGUF at
inference time via the sidecar meta.json. This is what makes loads fast
and keeps the pack file small — the heavy lifting is delegated to the
already-on-disk GGUF.

Usage:
    python src/apps/build_pack_from_gguf.py \\
        --gguf "C:/Users/.../blobs/sha256-<hash>" \\
        --out qwen2.5-coder_7b.rftmwpk \\
        --model-name qwen2.5-coder:7b
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantonium_os_src.engine.rftmw_numpy_engine import (
    _parse_gguf_index,
    _read_gguf_metadata,
    _GGUF_TYPE_BLOCK_SIZE,
    Qwen25Config,
)


def _detect_generic_cfg(gguf_meta: dict):
    """Best-effort config extraction for non-qwen2 GGUF namespaces.

    This keeps pack-building usable for compatible Qwen-family exports that
    use different metadata prefixes (for example qwen35moe.*).
    """
    if not gguf_meta:
        return None

    # Find a namespace that exposes core transformer shape keys.
    namespaces = set()
    for k in gguf_meta.keys():
        if "." in k:
            namespaces.add(k.split(".", 1)[0])

    for ns in sorted(namespaces):
        n_layers = gguf_meta.get(f"{ns}.block_count")
        hidden = gguf_meta.get(f"{ns}.embedding_length")
        heads = gguf_meta.get(f"{ns}.attention.head_count")
        kv_heads = gguf_meta.get(f"{ns}.attention.head_count_kv", heads)
        if None in (n_layers, hidden, heads):
            continue

        ffn = (
            gguf_meta.get(f"{ns}.feed_forward_length")
            or gguf_meta.get(f"{ns}.expert_feed_forward_length")
            or gguf_meta.get(f"{ns}.expert_shared_feed_forward_length")
            or int(hidden) * 4
        )
        rope_theta = gguf_meta.get(f"{ns}.rope.freq_base", 1_000_000.0)
        rms_eps = gguf_meta.get(f"{ns}.attention.layer_norm_rms_epsilon", 1e-6)
        ctx_len = gguf_meta.get(f"{ns}.context_length", 32768)
        vocab = gguf_meta.get(f"{ns}.vocab_size") or gguf_meta.get("tokenizer.ggml.tokens", [])
        if not isinstance(vocab, int):
            vocab = len(vocab) if isinstance(vocab, list) else 151936

        return SimpleNamespace(
            hidden_size=int(hidden),
            num_heads=int(heads),
            num_kv_heads=int(kv_heads),
            head_dim=int(hidden) // int(heads),
            intermediate_size=int(ffn),
            num_layers=int(n_layers),
            vocab_size=int(vocab),
            rope_theta=float(rope_theta),
            rms_norm_eps=float(rms_eps),
            max_position=int(ctx_len),
            namespace=ns,
        )
    return None


def build_pack(gguf_path: Path, out_path: Path, model_name: str) -> None:
    if not gguf_path.exists():
        raise SystemExit(f"GGUF not found: {gguf_path}")

    print(f"[build_pack] GGUF: {gguf_path} ({gguf_path.stat().st_size / 1e9:.2f} GB)")

    # --- Read tensor index + metadata ---
    gguf_index, _data_start = _parse_gguf_index(gguf_path)
    gguf_meta = _read_gguf_metadata(gguf_path)
    print(f"[build_pack] Tensors: {len(gguf_index)}  KV entries: {len(gguf_meta)}")

    # --- Auto-detect model config ---
    cfg = Qwen25Config.from_gguf(gguf_path)
    if cfg is None:
        cfg = _detect_generic_cfg(gguf_meta)
        if cfg is None:
            raise SystemExit("Could not auto-detect model config from GGUF metadata")
        print(f"[build_pack] Generic config fallback namespace: {cfg.namespace}")
    print(f"[build_pack] Config: {cfg.num_layers} layers, hidden={cfg.hidden_size}, "
          f"heads={cfg.num_heads}/{cfg.num_kv_heads}, ffn={cfg.intermediate_size}, "
          f"vocab={cfg.vocab_size}")

    # --- Build weight index (empty blobs; real data lives in GGUF) ---
    weight_entries = []
    parameter_shapes = {}
    param_count = 0
    for name, (_offset, num_bytes, ttype, ne) in gguf_index.items():
        if num_bytes == 0 or ttype not in _GGUF_TYPE_BLOCK_SIZE:
            continue
        # GGUF stores ne as (in_features, out_features) for 2D; 1D is just (N,)
        original_shape = [int(d) for d in ne]
        n_elem = 1
        for d in ne:
            n_elem *= d
        param_count += n_elem
        parameter_shapes[name] = original_shape
        weight_entries.append({
            "name": name,
            "original_shape": original_shape,
            "original_dtype": "uint8" if ttype in (12, 14) else "float32",
            "original_bytes": int(num_bytes),
            "method": "NONE",
            "compressed_bytes": 0,
            "spectral_entropy": 1.0,
            "reconstruction_error": 0.0,
            "offset": 0,
            "length": 0,
        })

    header = {
        "format_version": 1,
        "entropy_threshold": 0.0,
        "weight_keep_ratio": 0.0,
        "kv_keep_ratio": 0.0,
        "max_rft_error": 0.0,
        "weights": weight_entries,
        "report": {},
    }

    # --- Write the pack ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(out_path, "wb") as f:
        f.write(b"RFTMWPK1")
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        # No blobs — engine reads from GGUF via sidecar
    print(f"[build_pack] Wrote {out_path} ({out_path.stat().st_size} bytes)")

    # --- Write sidecar meta.json ---
    print(f"[build_pack] Hashing GGUF for provenance…")
    h = hashlib.sha256()
    with open(gguf_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    gguf_sha = h.hexdigest()

    model_config = {
        "hidden_size": cfg.hidden_size,
        "num_heads": cfg.num_heads,
        "num_kv_heads": cfg.num_kv_heads,
        "head_dim": cfg.head_dim,
        "intermediate_size": cfg.intermediate_size,
        "num_layers": cfg.num_layers,
        "vocab_size": cfg.vocab_size,
        "rope_theta": cfg.rope_theta,
        "rms_norm_eps": cfg.rms_norm_eps,
        "max_position": cfg.max_position,
    }

    sidecar_data = {
        "cache_path": str(out_path),
        "provenance": {
            "model_name": model_name,
            "source_type": "direct_gguf",
            "gguf_path": str(gguf_path),
            "gguf_file_sha256": gguf_sha,
            "parameter_count": int(param_count),
            "parameter_shapes": parameter_shapes,
        },
        "model_config": model_config,
    }
    sidecar_path = out_path.with_suffix(".rftmwpk.meta.json")
    sidecar_path.write_text(json.dumps(sidecar_data, indent=2))
    print(f"[build_pack] Wrote sidecar {sidecar_path}")
    print(f"[build_pack] Parameters: {param_count / 1e9:.2f} B")
    print(f"[build_pack] Done.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="Path to source GGUF file")
    ap.add_argument("--out", required=True, help="Output .rftmwpk path")
    ap.add_argument("--model-name", required=True, help="Friendly name (e.g. qwen2.5-coder:7b)")
    args = ap.parse_args()
    build_pack(Path(args.gguf), Path(args.out), args.model_name)


if __name__ == "__main__":
    main()
