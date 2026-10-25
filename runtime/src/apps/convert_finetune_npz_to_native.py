#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Convert QSHLL fine-tune NPZ overlays into the native C++ sidecar format."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


MAGIC = b"RFTFTW1\0"


def convert(src: Path, dst: Path) -> None:
    data = np.load(src)
    names = sorted(data.files)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as out:
        out.write(MAGIC)
        out.write(struct.pack("<I", len(names)))
        for name in names:
            arr = np.asarray(data[name], dtype=np.float32, order="C")
            if arr.ndim != 2:
                raise ValueError(f"{name}: expected 2D matrix, got shape {arr.shape}")
            encoded = name.encode("utf-8")
            out.write(struct.pack("<I", len(encoded)))
            out.write(encoded)
            out.write(struct.pack("<II", arr.shape[0], arr.shape[1]))
            out.write(arr.tobytes(order="C"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="Input .finetuned_weights.npz")
    ap.add_argument("dst", type=Path, nargs="?", help="Output .rftft sidecar")
    args = ap.parse_args()
    dst = args.dst
    if dst is None:
        dst = args.src.with_suffix(".rftft")
    convert(args.src, dst)
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
