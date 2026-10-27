#!/usr/bin/env bash
# Run from anywhere once this package is extracted at runtime/experiments/phi_runtime_lab.
# Usage: bash runtime/experiments/phi_runtime_lab/run_in_repo.sh smoke
#        bash runtime/experiments/phi_runtime_lab/run_in_repo.sh quick runtime/docs
#        bash runtime/experiments/phi_runtime_lab/run_in_repo.sh research runtime/docs
#        bash runtime/experiments/phi_runtime_lab/run_in_repo.sh gradients runtime/docs
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
ACTION="${1:-smoke}"
CORPUS="${2:-$ROOT/runtime/docs}"
PYTHON="${PYTHON:-python3}"
OUT="$HERE/checkpoints"
mkdir -p "$OUT"
case "$ACTION" in
  smoke) "$PYTHON" "$HERE/lab.py" smoke --threads 4 ;;
  quick)
    "$PYTHON" "$HERE/lab.py" compare --data "$CORPUS" \
      --width 48 --layers 1 --heads 4 --window 8 --mix-group 16 \
      --context 64 --batch 4 --steps 30 --eval-every 15 \
      --eval-batches 4 --threads 4 --out-dir "$OUT/quick" ;;
  research)
    "$PYTHON" "$HERE/lab.py" compare --data "$CORPUS" \
      --width 96 --layers 2 --heads 4 --window 16 --mix-group 16 \
      --context 128 --batch 8 --steps 500 --eval-every 50 \
      --eval-batches 16 --threads 4 --max-mb 8 \
      --out-dir "$OUT/research" ;;
  factorial)
    "$PYTHON" "$HERE/lab.py" factorial --data "$CORPUS" \
      --width 48 --layers 1 --heads 4 --window 8 --mix-group 16 \
      --context 64 --batch 4 --steps 100 --eval-every 25 \
      --eval-batches 6 --threads 4 --out-dir "$OUT/factorial" ;;
  gradients)
    # Same model, same seed, same data; ONLY optimizer-gradient treatment varies.
    for FILTER in none phi dft; do
      "$PYTHON" "$HERE/lab.py" train --data "$CORPUS" \
        --mode phi --width 48 --layers 1 --heads 4 --window 8 --mix-group 16 \
        --context 64 --batch 4 --steps 100 --eval-every 25 \
        --eval-batches 6 --threads 4 --grad-filter "$FILTER" --grad-keep 0.50 \
        --out "$OUT/gradients/phi_${FILTER}.pt"
    done ;;
  *) echo "Usage: $0 {smoke|quick|research|factorial|gradients} [corpus-file-or-directory]" >&2; exit 2 ;;
esac
