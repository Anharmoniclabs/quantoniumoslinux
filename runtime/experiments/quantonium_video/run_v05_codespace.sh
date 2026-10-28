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
    ls -lh "$OUT/jayc_private/jayc_full_page_crop_grid.png" 2>&1 || true
    ls -lh "$OUT/jayc_private/jayc_approved.png" 2>&1 || true
    ;;
  prepare)
    test -f "$MODEL" || { echo "V0.4 weights missing"; exit 2; }
    test -f "$BIBLE" || { echo "Private Bible missing; supply path to jayc_assets_v05.py manually"; exit 2; }
    "$PYTHON" "$HERE/jayc_assets_v05.py" --bible "$BIBLE" --page 1 --preview-page --out "$OUT/jayc_private"
    echo "REVIEW entire page; pick a manual crop and isolate an approved JC PNG (see README)."
    ;;
  render)
    test -f "$MODEL" || { echo "V0.4 weights missing"; exit 2; }
    test -f "$OUT/jayc_private/jayc_approved.png" || { echo "QC FAILED: approved full-body transparent PNG not found. Do not render auto-extracted sheet, labels, or crop fragments."; exit 2; }
    test "${QVIDEO_APPROVED_JC:-no}" = yes || { echo "Must explicitly approve the cleaned PNG first: QVIDEO_APPROVED_JC=yes"; exit 2; }
    if [[ -z "${QVIDEO_POSES_DIR:-}" ]]; then
      echo "NO ARTICULATED ANIMATION: one PNG can only slide. Provide QVIDEO_POSES_DIR with >=2 approved pose PNGs." >&2
      exit 2
    fi
    test -d "$QVIDEO_POSES_DIR" || { echo "QVIDEO_POSES_DIR missing: $QVIDEO_POSES_DIR" >&2; exit 2; }
    pose_count="$(find "$QVIDEO_POSES_DIR" -maxdepth 1 -iname '*.png' -type f | wc -l)"
    ((pose_count>=2)) || { echo "Need >=2 distinct approved JC poses; found $pose_count" >&2; exit 2; }
    "$PYTHON" "$HERE/jayc_preview_v05.py" --jayc "$OUT/jayc_private/jayc_approved.png" --poses-dir "$QVIDEO_POSES_DIR" --motion-checkpoint "$MODEL" --action dash --frames 48 --fps 24 --cel-hold 2 --width 960 --height 540 --out "$OUT"
    ;;
  *) echo "Usage: bash run_v05_codespace.sh status|prepare|render"; exit 2;;
esac
