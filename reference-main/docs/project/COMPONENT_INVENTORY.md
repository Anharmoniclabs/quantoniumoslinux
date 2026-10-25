# Component Inventory

**Status:** current and conservative as of 2026-04-16.

This inventory lists the main components that are visibly present in the repo snapshot.

## Core code areas

| Path | What it contains |
| --- | --- |
| `algorithms/` | algorithm families including RFT-related work |
| `quantonium_os_src/apps/` | desktop apps such as notes, vault, validator, visualizer, simulator, and monitor |
| `quantonium_os_src/engine/` | Python inference engines and support logic |
| `quantonium_os_src/frontend/` | desktop shell entrypoints |
| `src/rftmw_native/` | native inference/runtime implementation |
| `src/apps/` | CLI, chat, training, packaging, and support utilities |

## Validation and research areas

| Path | What it contains |
| --- | --- |
| `tests/` | proof, validation, native, medical, compression, and integration tests |
| `benchmarks/` | benchmark drivers and evaluation scripts |
| `experiments/` | exploratory research runs |
| `docs/research/` | research position papers, limits, and open problems |
| `docs/reports/` | benchmark and validation writeups |

## Platform and support areas

| Path | What it contains |
| --- | --- |
| `hardware/` | FPGA, RTL, synthesis, and hardware documentation |
| `mobile/` | mobile app project |
| `tools/` | support scripts and utilities |
| `ui/` | shared UI assets |
| `data/` | datasets and generated artifacts |
| `models/` | model-related assets and helpers |

## Key root-level files

| File | Purpose |
| --- | --- |
| `README.md` | top-level project entry |
| `pyproject.toml` | packaging and dependency metadata |
| `requirements.txt` | runtime dependency list |
| `pytest.ini` | test configuration |
| `verify_setup.sh` | quick environment check |
| `run_demo.sh` | demo runner |
| `launch.py` | root launcher |

## Documentation authority

When there is a conflict between files, trust the following first:
- `docs/DOCUMENTATION_STATUS.md`
- `docs/research/CURRENT_RESEARCH_POSITION.md`
- `docs/THEOREMS_RFT_IRONCLAD.md`
- `docs/NON_CLAIMS.md`
