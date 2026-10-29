#!/usr/bin/env bash
# V0.6 JC action-proof assets are included on the public GitHub video branch.
# No rclone, Drive OAuth, user downloads, or separate asset transfers.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ASSETS="$HERE/assets/jc_v06_lossless"
POSES="$HOME/quantonium-anime-data/jc_v06/pose_assets"
MOTION="$HOME/quantonium-video-runs/v04_codespace/latest.pt"
OUTPUT="$HOME/quantonium-video-runs/v06_codespace"
for tool in python ffmpeg sha256sum; do
  command -v "$tool" >/dev/null || { echo "Missing command: $tool" >&2; exit 2; }
done
[[ -s "$MOTION" ]] || { echo "V0.4 trained checkpoint missing: $MOTION" >&2; exit 2; }
[[ -f "$ASSETS/SHA256SUMS" ]] || { echo "V0.6 tracked artwork is missing. Fetch and restore the video branch." >&2; exit 2; }
(cd "$ASSETS" && sha256sum -c SHA256SUMS)
mkdir -p "$POSES" "$OUTPUT"
python - "$ASSETS" "$POSES" <<'PY'
import sys
from pathlib import Path
from PIL import Image, ImageChops
assets=Path(sys.argv[1]); output=Path(sys.argv[2])
for n in range(1,9):
    source=assets/f'pose_{n:02d}.webp'
    dest=output/f'pose_{n:02d}.png'
    with Image.open(source) as img:
        rgba=img.convert('RGBA')
        rgba.save(dest,'PNG',optimize=True)
    with Image.open(dest) as check:
        assert ImageChops.difference(rgba,check.convert('RGBA')).getbbox() is None, str(dest)
    print('EXTRACTED',dest)
print('Eight lossless, visually identical JC poses verified.')
PY
python -u "$HERE/v06_pose_cels.py" \
  --poses "$POSES" --motion-checkpoint "$MOTION" \
  --out "$OUTPUT" --frames 48 --fps 24 \
  --width 960 --height 540
ls -lh "$OUTPUT/jc_action_cels_v06.mp4" "$OUTPUT/jc_pose_frame_review.png"
echo 'V0.6 render completed from GitHub-tracked poses; 5K-step V0.4 checkpoint untouched.'