#!/usr/bin/env bash
set -euo pipefail
LAB=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPAIR_ROOT=${PHISTATE_REPAIR_ROOT:-/content/drive/MyDrive/Quantonium/PhiState-L4/whitespace_repair_v1}
CHAT_ROOT=${PHISTATE_CHAT_ROOT:-$REPAIR_ROOT/chat_recovery_v1}
DATA=${PHISTATE_DATA_DIR:-$REPAIR_ROOT/corpus_repaired_v1}
SOURCE=${PHISTATE_SOURCE_CHECKPOINT:-$REPAIR_ROOT/run_60k_to_61k/latest.pt}
TARGET=${PHISTATE_SFT_STEPS:-2000}
exec 9>"${TMPDIR:-/tmp}/phistate_chat_recovery_pipeline.lock"
flock -n 9 || { echo 'A chat recovery pipeline is already running.'; exit 1; }
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'Select a GPU runtime in Colab; this training cell requires CUDA.'
print('TRAINING DEVICE:', torch.cuda.get_device_name(0), flush=True)
PY
mkdir -p "$CHAT_ROOT"
printf '%s\n' "$LAB" > "$CHAT_ROOT/code_directory.txt"
python -u "$LAB/chat_recovery_data.py" \
  --data-dir "$DATA" --output "$CHAT_ROOT/data" \
  --cache "$CHAT_ROOT/source_cache" --context 512
python -u "$LAB/chat_recovery_train.py" \
  --checkpoint "$SOURCE" --data-dir "$DATA" --sft-dir "$CHAT_ROOT/data" \
  --run-dir "$CHAT_ROOT/run" --steps "$TARGET" --device cuda
STATUS=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$CHAT_ROOT/run/progress.json")
if [[ "$STATUS" == complete || "$STATUS" == early_stopped || "$STATUS" == retention_stopped ]]; then
  python -u "$LAB/chat_recovery_cpp.py" export --run-dir "$CHAT_ROOT/run" --sft-dir "$CHAT_ROOT/data"
else
  echo "Training status: $STATUS. Saved checkpoints are in $CHAT_ROOT/run."
fi
echo "Progress: $CHAT_ROOT/run/progress.json"
echo "Chat command: python $LAB/chat_recovery_cpp.py chat --run-dir $CHAT_ROOT/run"
