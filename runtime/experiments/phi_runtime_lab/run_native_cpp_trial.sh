#!/usr/bin/env bash
# Read the completed repair checkpoint and run an isolated native CPU trial.
set -euo pipefail
LAB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PHISTATE_REPAIR_ROOT="${PHISTATE_REPAIR_ROOT:-/content/drive/MyDrive/Quantonium/PhiState-L4/whitespace_repair_v1}"
PHISTATE_RUN_DIR="${PHISTATE_RUN_DIR:-$PHISTATE_REPAIR_ROOT/run_60k_to_61k}"
PHISTATE_DATA_DIR="${PHISTATE_DATA_DIR:-$PHISTATE_REPAIR_ROOT/corpus_repaired_v1}"
PHISTATE_OUTPUT_DIR="${PHISTATE_OUTPUT_DIR:-$PHISTATE_REPAIR_ROOT/native_cpp_trial_$(date -u +%Y%m%dT%H%M%S)}"
export PHISTATE_RUN_DIR
python - <<'PY'
import json, os
from pathlib import Path
run = Path(os.environ['PHISTATE_RUN_DIR'])
p = json.loads((run / 'progress.json').read_text())
if p['status'] != 'complete' or p['saved_step'] < 65000:
    raise SystemExit(f"Let the existing run finish first: status={p['status']}, saved step={p['saved_step']}")
if not (run / 'generation_step_65000.json').is_file():
    raise SystemExit('Checkpoint saved; wait for the existing 65K generation report to finish, then rerun.')
print(f"Testing saved checkpoint at step {p['saved_step']} using native C++ CPU inference.", flush=True)
PY
exec python -u "$LAB_DIR/native_cpp_trial.py" \
    --checkpoint "$PHISTATE_RUN_DIR/latest.pt" \
    --data-dir "$PHISTATE_DATA_DIR" \
    --output "$PHISTATE_OUTPUT_DIR" \
    --threads 2 --new-tokens 64
