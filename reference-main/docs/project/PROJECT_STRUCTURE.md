# quantoniumos Project Structure

**Status:** verified against the current repository snapshot on 2026-04-16.

This page is the current layout reference. Older notes that mention `src/core/`, `assembly/kernel/`, or `quantonium-mobile/` do not match this snapshot.

## Top-level structure

| Path | Role |
| --- | --- |
| `algorithms/` | RFT and related algorithm work |
| `benchmarks/` | benchmark and comparison drivers |
| `docs/` | proofs, reports, manuals, and research notes |
| `experiments/` | exploratory research runs |
| `hardware/` | RTL, FPGA, synthesis, and hardware reports |
| `mobile/` | mobile application workspace |
| `quantonium_os_src/` | desktop apps, frontend, and engine modules |
| `src/` | native runtime, CLI/app utilities, and support code |
| `tests/` | proof, validation, native, medical, and integration tests |
| `tools/`, `ui/`, `data/`, `models/` | support assets and utilities |

## Key implementation zones

### `quantonium_os_src/`

Current subtrees:
- `apps/` — desktop applications such as notes, vault, validator, visualizer, simulator, and monitor
- `engine/` — inference and model-engine code
- `frontend/` — desktop shell entrypoints

### `src/`

Current subtrees:
- `rftmw_native/` — native inference implementation
- `apps/` — CLI, chat, training, and model utility scripts
- `agents/`, `dashboard/`, `rpg/` — support and demo areas

### `tests/`

The test suite is organized by purpose:
- `proofs/`
- `validation/`
- `native/`
- `medical/`
- `compression/`
- `integration/`
- `rft/`
- `transforms/`

## Main entrypoints

- `launch.py` — root launcher
- `run_demo.sh` — demo runner
- `verify_setup.sh` — environment verification
- `quantonium_os_src/frontend/quantonium_desktop.py` — desktop UI entry

## Documentation rule

For proof status and claim boundaries, prefer:
- `docs/DOCUMENTATION_STATUS.md`
- `docs/research/CURRENT_RESEARCH_POSITION.md`
- `docs/THEOREMS_RFT_IRONCLAD.md`
- `docs/NON_CLAIMS.md`

Keep this file limited to structure only.
