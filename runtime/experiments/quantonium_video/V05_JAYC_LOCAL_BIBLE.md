# Quantonium Anime Video V0.5 — Jayc local Bible workflow

V0.5 connects the trained V0.4 motion-controller checkpoint to your OWN
approved Jayc drawing from the local character Bible. Private PDF pages,
PNGs, checkpoints and output video stay on your OptiPlex. No original art
is committed or uploaded by these scripts.

Character lock: close dark Caesar waves/taper, large elongated eyes with
white sclera and slim pupils, brown skin, black streetwear, red/black
headphones and the separate gold pendant. Fine clean 1990s anime linework.
No redesign or generic substitute.

IMPORTANT: This stage is motion-controlled layered rendering, not new
AI pose generation. A single character illustration moves rigidly; to
animate real limbs, provide TWO OR MORE approved transparent pose drawings.

## Installation in WSL Ubuntu (not PowerShell)

    cd /mnt/c/Users/Anharoniclabs/quantoniumoslinux
    git pull --ff-only origin feat/quantonium-anime-video-v01
    source ~/.quantonium-video-env/bin/activate
    python -m pip install -U pymupdf

## Prepare Jayc from the LOCAL Bible

    python -u runtime/experiments/quantonium_video/jayc_assets_v05.py \
      --bible auto \
      --out ~/quantonium-video-runs/v05/jayc_private

The auto locator searches Windows Downloads/Documents/Desktop. Use
--bible "/mnt/c/Users/Anharoniclabs/Downloads/EXACT_FILE.pdf"
if it chooses the wrong file.

The 64-page Full Cast Bible uses page 1 for JC; the 99-page Complete
Character Image Bible uses page 8 for JC approved version 2. These
are different document layouts! Explicit --page and optional
--crop X0 Y0 X1 Y1 (normalized 0..1) can override the defaults.

Check the local output jayc_crop_review.png and jayc_cutout.png.
Do not proceed if another figure or Bible text is inside the crop, or
if Jayc's eyes, necklace, hands, shoes or hair were damaged.

    explorer.exe "$(wslpath -w ~/quantonium-video-runs/v05/jayc_private)"

## Render your ORIGINAL character (V0.4-trained path)

    python -u runtime/experiments/quantonium_video/jayc_preview_v05.py \
      --jayc ~/quantonium-video-runs/v05/jayc_private/jayc_cutout.png \
      --motion-checkpoint ~/quantonium-video-runs/v04/latest.pt \
      --action dash --frames 24 --fps 24 \
      --out ~/quantonium-video-runs/v05

Creates jayc_preview_latest.mp4 (or GIF without FFmpeg),
jayc_contact_latest.png and render_manifest.json.

## Approved 2D walk/fight pose pack

For legitimate articulated cel animation, add transparent, approved
Jayc pose images to a PRIVATE local folder:

    ~/quantonium-anime-data/Jayc_approved_poses/pose_000.png
    ~/quantonium-anime-data/Jayc_approved_poses/pose_001.png
    ~/quantonium-anime-data/Jayc_approved_poses/pose_002.png

Then run:

    python -u runtime/experiments/quantonium_video/jayc_preview_v05.py \
      --jayc ~/quantonium-video-runs/v05/jayc_private/jayc_cutout.png \
      --poses-dir ~/quantonium-anime-data/Jayc_approved_poses \
      --background "/path/to/approved_rooftop_background.png" \
      --action dash --frames 48 --fps 24 --cel-hold 2 \
      --width 960 --height 540 --out ~/quantonium-video-runs/v05

This shows existing creator-approved drawings with two-frame cel holds.
It does not hallucinate missing poses. New action synthesis requires
pose-labeled video training that we have not established yet.

Everything stays in WSL outside the repository. Output preview filenames
are reused rather than accumulated. V0.2, V0.3 and V0.4 weights are unchanged.
