# quantoniumos Repository Organization

**Status:** cleaned and aligned with the current repo snapshot on 2026-04-16.

This document explains how the repository is organized **today**. It is intentionally conservative and avoids historical layouts that no longer exist.

## Organization principles

1. keep proofs and research status in `docs/`;
2. keep executable code in `algorithms/`, `quantonium_os_src/`, and `src/`;
3. keep tests under `tests/`;
4. keep hardware artifacts under `hardware/`;
5. keep exploratory work in `experiments/` and `benchmarks/`.

## Current top-level categories

| Category | Current paths |
| --- | --- |
| Math and algorithms | `algorithms/` |
| Desktop UI and engine | `quantonium_os_src/` |
| Native runtime and app utilities | `src/` |
| Tests and validation | `tests/` |
| Benchmarks and experiments | `benchmarks/`, `experiments/`, `examples/` |
| Docs and reports | `docs/` |
| Hardware and FPGA work | `hardware/` |
| Mobile workspace | `mobile/` |
| Tooling and assets | `tools/`, `ui/`, `data/`, `models/` |

## Where to look first

### If you need the math
- `docs/research/CURRENT_RESEARCH_POSITION.md`
- `docs/THEOREMS_RFT_IRONCLAD.md`
- `docs/PROOF_RESULTS.md`
- `docs/NON_CLAIMS.md`

### If you need inference code
- `quantonium_os_src/engine/`
- `src/rftmw_native/`
- `docs/RFTMW_INFERENCE_ENGINE_MANUAL.md`
- `docs/QUANTONIUMOS_INFERENCE_ARCHITECTURE.md`

### If you need validation evidence
- `tests/`
- `docs/reports/`
- `docs/research/PROVEN_EMPIRICAL_OPEN_MATRIX.md`

## Known legacy naming drift

The current tree uses:
- `mobile/` instead of `quantonium-mobile/`
- `quantonium_os_src/frontend/` instead of a root `frontend/`
- `src/rftmw_native/` and `quantonium_os_src/engine/` instead of `src/core/` or `src/rftmw_memory.py`

If a doc cites those old paths, treat it as historical unless it has been rewritten after the April 2026 audit.
