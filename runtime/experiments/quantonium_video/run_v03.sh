#!/usr/bin/env bash
# Run V0.3 with the existing background process supervisor in WSL Ubuntu.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export QVIDEO_TRAINER="$HERE/train_v03.py"
export QVIDEO_OUT="${QVIDEO_OUT:-$HOME/quantonium-video-runs/v03}"
export QVIDEO_STEPS="${QVIDEO_STEPS:-1000}"
export QVIDEO_THREADS="${QVIDEO_THREADS:-4}"
export QVIDEO_SAVE_EVERY="${QVIDEO_SAVE_EVERY:-200}"
exec bash "$HERE/run_5000.sh" "${1:-status}"
