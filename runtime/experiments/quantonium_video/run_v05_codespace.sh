#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="$HOME/quantonium-video-runs/v04_codespace/latest.pt"
OUT="$HOME/quantonium-video-runs/v05_codespace"
BIBLE="/workspaces/quantoniumoslinux/Heralds_of_the_Cemi_Full_Cast_Character_Bible-1.pdf"
PYTHON=python
case "$1" in
  status)
    ls -lh "$MODEL" "$BIBLE" 2>&1 || true
    ls -lh "$OUT/jayc_private/jayc_cutout.png" 2>&1 || true
    ;;
  prepare)
    test -f "$MODEL" || { echo "V0.4 weights missing"; exit 2; }
    test -f "$BIBLE" || { echo "Private Bible missing; supply path to jayc_assets_v05.py manually"; exit 2; }
    "$PYTHON" "$HERE/jayc_assets_v05.py" --bible "$BIBLE" --page 1 --out "$OUT/jayc_private"
    echo "REVIEW before rendering: $OUT/jayc_private/jayc_crop_review.png and jayc_cutout.png"
    ;;
  render)
    test -f "$MODEL" || { echo "V0.4 weights missing"; exit 2; }
    test -f "$OUT/jayc_private/jayc_cutout.png" || { echo "First run prepare and review images"; exit 2; }
    test "$QVIDEO_APPROVED_JC" = yes || { echo "Must approve character cutout first: QVIDEO_APPROVED_JC=yes"; exit 2; }
    "$PYTHON" "$HERE/jayc_preview_v05.py" --jayc "$OUT/jayc_private/jayc_cutout.png" --motion-checkpoint "$MODEL" --action dash --frames 48 --fps 24 --cel-hold 2 --width 960 --height 540 --out "$OUT"
    ;;
  *) echo "Usage: bash run_v05_codespace.sh status|prepare|render"; exit 2;;
esac
