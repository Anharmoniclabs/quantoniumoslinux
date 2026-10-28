#!/usr/bin/env bash
# Private JC cels: GitHub contains code only; Google Drive contains the artwork.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE="${QVIDEO_DRIVE_REMOTE:-gdrive}"
DRIVE_ASSET="Quantonium/JC_V06_Private/JC_V06_8_Private_Poses.zip"
ZIP_SHA256="bf22a5c1a42d1ed7083f7d6b23981b90e6f766e4f0e85c3e82cf2e292a7a9623"
ASSET_ROOT="$HOME/quantonium-anime-data/jc_v06"
ZIP="$ASSET_ROOT/JC_V06_8_Private_Poses.zip"
POSES="$ASSET_ROOT/pose_assets"
MOTION="$HOME/quantonium-video-runs/v04_codespace/latest.pt"
OUTPUT="$HOME/quantonium-video-runs/v06_codespace"

for command in python rclone unzip sha256sum; do
  command -v "$command" >/dev/null || {
    echo "Missing $command in Codespaces" >&2
    exit 2
  }
done
if ! rclone listremotes 2>/dev/null | grep -qx "$REMOTE:"; then
  cat >&2 <<'AUTH'
Private Google Drive isn't yet authorized in this Codespace.
Run: rclone config
Create a remote named gdrive using the drive backend.
Complete Google's OAuth authorization privately; do not paste tokens into chat.
Then rerun this script; no manual file download or upload is needed.
AUTH
  exit 3
fi
# An existing remote name is NOT evidence of valid OAuth. Test safely
# before transferring the creator-owned pose pack; never print the token.
if ! rclone lsf "$REMOTE:" --max-depth 1 >/dev/null 2>&1; then
  cat >&2 <<AUTH
Google Drive remote '$REMOTE' exists, but its authentication is missing,
expired or otherwise unusable.

Run in this Codespaces terminal:
    rclone config reconnect $REMOTE:

Follow the browser authorization prompts, then verify:
    rclone lsf $REMOTE:Quantonium/JC_V06_Private/

No ZIP download/upload on your computer is needed. Never paste OAuth
tokens into GitHub, a public terminal log, or chat.
AUTH
  exit 3
fi
if [[ ! -s "$MOTION" ]]; then
  echo "Missing trained V0.4 model: $MOTION" >&2
  exit 3
fi

mkdir -p "$ASSET_ROOT" "$OUTPUT"
if [[ ! -f "$ZIP" ]] || [[ "$(sha256sum "$ZIP" | cut -d' ' -f1)" != "$ZIP_SHA256" ]]; then
  echo "Fetching creator-owned poses from private Google Drive..."
  rm -f "$ZIP.part"
  rclone copyto "$REMOTE:$DRIVE_ASSET" "$ZIP.part" --progress
  echo "$ZIP_SHA256  $ZIP.part" | sha256sum -c - >/dev/null || {
    rm -f "$ZIP.part"
    echo "Transfer integrity mismatch" >&2
    exit 4
  }
  mv -f "$ZIP.part" "$ZIP"
fi

echo "$ZIP_SHA256  $ZIP" | sha256sum -c -
unzip -tq "$ZIP" >/dev/null
unzip -o -q "$ZIP" 'pose_assets/*' -d "$ASSET_ROOT"
for i in 01 02 03 04 05 06 07 08; do
  test -s "$POSES/pose_$i.png" || { echo "Missing pose_$i.png"; exit 5; }
done

python -u "$HERE/v06_pose_cels.py" \
  --poses "$POSES" \
  --motion-checkpoint "$MOTION" \
  --out "$OUTPUT" \
  --frames 48 --fps 24 --width 960 --height 540

echo "V0.6 VIDEO: $OUTPUT/jc_action_cels_v06.mp4"
echo "V0.6 QC: $OUTPUT/jc_pose_frame_review.png"
echo "This is timed original cels + learned root motion, NOT new pose generation."
