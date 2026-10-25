# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos

"""Unit tests for the D3 fast RFT operator Ψ = D_φ · C_σ · F.

This test targets the FFT-factorized unitary implementation in:
`algorithms.rft.core.fast_rft.FastRFT`.

It checks:
- energy preservation (Parseval) on random complex inputs
- round-trip accuracy via inverse(forward(x))
- inner-product preservation (unitarity) on random pairs
"""

from __future__ import annotations

import numpy as np
import pytest

from algorithms.rft.core.fast_rft import FastRFT


@pytest.mark.parametrize("N", [256, 1024, 4096])
def test_fast_rft_d3_is_unitary(N: int) -> None:
    rng = np.random.default_rng(1337 + N)
    fast = FastRFT(N)

    x1 = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    x2 = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    x1 = x1.astype(np.complex128, copy=False)
    x2 = x2.astype(np.complex128, copy=False)

    x1 /= np.linalg.norm(x1)
    x2 /= np.linalg.norm(x2)

    y1 = fast.forward(x1)
    y2 = fast.forward(x2)

    # Energy preservation.
    np.testing.assert_allclose(np.linalg.norm(y1), 1.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(np.linalg.norm(y2), 1.0, atol=1e-12, rtol=0.0)

    # Inner product preservation: <Ux,Uy> = <x,y>.
    ip_in = np.vdot(x1, x2)
    ip_out = np.vdot(y1, y2)
    np.testing.assert_allclose(ip_out, ip_in, atol=1e-12, rtol=0.0)

    # Round-trip.
    x1_rt = fast.inverse(y1)
    np.testing.assert_allclose(x1_rt, x1, atol=1e-10, rtol=0.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

