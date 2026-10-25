# AI Infrastructure Walkthrough — Current Snapshot

**Status:** rewritten during the 2026-04-16 documentation audit.

This page replaces an older walkthrough that referenced paths no longer present in the repo, including `src/rftmw_memory.py`, `src/rftmw_inference.py`, and `src/RFTMW.py`.

## Verified current AI/inference areas

The active code relevant to tokenization and inference in this snapshot is centered on:

- `quantonium_os_src/engine/rftmw_inference.py`
- `quantonium_os_src/engine/rftmw_numpy_engine.py`
- `quantonium_os_src/engine/rftmw_mmap_engine.py`
- `src/rftmw_native/`
- `tokenizer.json`

## Current flow

1. load or construct the tokenizer;
2. encode the prompt into token IDs;
3. run the NumPy or native inference path;
4. sample the next token;
5. decode tokens back to text.

## Current project value

The repo currently demonstrates:
- formal/discrete RFT mathematics,
- experimental compression and inference infrastructure,
- validation and benchmark scaffolding,
- desktop, mobile, and hardware-adjacent demos.

## Do not cite this page for

- removed or renamed source paths,
- theorem-strength claims outside the canonical proof documents,
- old model-asset inventories.

## Use these sources first

- `docs/DOCUMENTATION_STATUS.md`
- `docs/research/CURRENT_RESEARCH_POSITION.md`
- `docs/THEOREMS_RFT_IRONCLAD.md`
- `docs/QUANTONIUMOS_INFERENCE_ARCHITECTURE.md`
- `docs/RFTMW_INFERENCE_ENGINE_MANUAL.md`
