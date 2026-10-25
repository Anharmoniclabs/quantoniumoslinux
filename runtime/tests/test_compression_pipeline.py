#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
RFT Compression Pipeline — Reproducible Verification Test
==========================================================

End-to-end roundtrip test for the unified RFT compression pipeline.
Generates deterministic fixtures (seeded RNG), compresses, decompresses,
and asserts ratio >= 1x and relative error < 15% for every signal class.

Run::

    pytest tests/test_compression_pipeline.py -v
    # or standalone:
    python tests/test_compression_pipeline.py

Fixtures are created in a temporary directory and cleaned up automatically.
Nothing is written to the repo working tree.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

import numpy as np
import pytest

# Ensure repo root is importable when run standalone
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from quantonium_os_src.engine.rft_compress import compress, decompress

# ---------------------------------------------------------------------------
# Deterministic fixtures  (seed=42, always reproducible)
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

FIXTURES = {
    "sine_1d": np.sin(np.linspace(0, 20 * np.pi, 1024)).astype(np.float32),
    "ramp_1d": np.linspace(-1, 1, 1024).astype(np.float32),
    "noise_1d": RNG.standard_normal(1024).astype(np.float32),
    "random_2d": np.random.default_rng(42).standard_normal((256, 256)).astype(np.float32),
    "random_2d_lg": np.random.default_rng(42).standard_normal((512, 512)).astype(np.float32),
}

# Thresholds
MIN_RATIO = 1.0       # compressed must be smaller than original
MAX_REL_ERR = 0.15    # relative reconstruction error


# ---------------------------------------------------------------------------
# Parametrized roundtrip test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", FIXTURES.keys())
def test_roundtrip(name: str):
    """Compress -> decompress and check ratio + fidelity."""
    arr = FIXTURES[name]

    t0 = time.perf_counter()
    blob = compress(arr)
    t_enc = time.perf_counter() - t0

    t0 = time.perf_counter()
    back = decompress(blob, shape=arr.shape, dtype=arr.dtype)
    t_dec = time.perf_counter() - t0

    ratio = arr.nbytes / len(blob)
    rel_err = float(
        np.linalg.norm(arr.ravel() - back.ravel())
        / (np.linalg.norm(arr.ravel()) + 1e-12)
    )

    print(
        f"  {name:20s}  shape={str(arr.shape):14s}  "
        f"{arr.nbytes:>8d}B -> {len(blob):>8d}B  "
        f"ratio={ratio:.2f}x  err={rel_err:.4f}  "
        f"enc={1000*t_enc:.0f}ms  dec={1000*t_dec:.0f}ms"
    )

    assert ratio >= MIN_RATIO, f"{name}: ratio {ratio:.2f}x < {MIN_RATIO}x"
    assert rel_err < MAX_REL_ERR, f"{name}: rel_err {rel_err:.4f} >= {MAX_REL_ERR}"


# ---------------------------------------------------------------------------
# Blob persistence roundtrip  (write .rftb to disk, reload, verify)
# ---------------------------------------------------------------------------

def test_blob_persistence():
    """Write compressed blob to disk and reload — must survive serialization."""
    arr = FIXTURES["random_2d"]
    blob = compress(arr)

    with tempfile.NamedTemporaryFile(suffix=".rftb", delete=False) as f:
        f.write(blob)
        path = f.name

    try:
        with open(path, "rb") as f:
            blob2 = f.read()
        back = decompress(blob2, shape=arr.shape, dtype=arr.dtype)
        rel_err = float(
            np.linalg.norm(arr.ravel() - back.ravel())
            / (np.linalg.norm(arr.ravel()) + 1e-12)
        )
        assert rel_err < MAX_REL_ERR, f"persistence roundtrip err {rel_err:.4f}"
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Stress test  (512x512)
# ---------------------------------------------------------------------------

def test_stress_512x512():
    """512x512 tensor — must compress with ratio >= 1x under 60 s."""
    arr = FIXTURES["random_2d_lg"]

    t0 = time.perf_counter()
    blob = compress(arr)
    t_enc = time.perf_counter() - t0

    back = decompress(blob, shape=arr.shape, dtype=arr.dtype)
    ratio = arr.nbytes / len(blob)
    rel_err = float(
        np.linalg.norm(arr.ravel() - back.ravel())
        / (np.linalg.norm(arr.ravel()) + 1e-12)
    )

    print(f"  512x512  ratio={ratio:.2f}x  err={rel_err:.4f}  enc={1000*t_enc:.0f}ms")

    assert ratio >= MIN_RATIO
    assert rel_err < MAX_REL_ERR
    assert t_enc < 60.0, f"encoding took {t_enc:.1f}s — too slow"


# ---------------------------------------------------------------------------
# Standalone runner  (python tests/test_compression_pipeline.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("RFT Compression Pipeline — Verification")
    print("=" * 60)

    all_ok = True
    for name, arr in FIXTURES.items():
        t0 = time.perf_counter()
        blob = compress(arr)
        t1 = time.perf_counter()
        back = decompress(blob, shape=arr.shape, dtype=arr.dtype)
        t2 = time.perf_counter()

        ratio = arr.nbytes / len(blob)
        rel_err = float(
            np.linalg.norm(arr.ravel() - back.ravel())
            / (np.linalg.norm(arr.ravel()) + 1e-12)
        )
        ok = ratio >= MIN_RATIO and rel_err < MAX_REL_ERR
        if not ok:
            all_ok = False

        print(
            f"  {name:20s}  {str(arr.shape):14s}  "
            f"{arr.nbytes:>8d}B -> {len(blob):>8d}B  "
            f"ratio={ratio:.2f}x  err={rel_err:.4f}  "
            f"enc={1000*(t1-t0):.0f}ms  dec={1000*(t2-t1):.0f}ms  "
            f"[{'OK' if ok else 'FAIL'}]"
        )

    print()
    if all_ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED — see lines above")
        sys.exit(1)
