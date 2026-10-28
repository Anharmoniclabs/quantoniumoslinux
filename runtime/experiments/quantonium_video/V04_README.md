# Quantonium Anime Video V0.4 — Layered motion

Research target: keep the character crisp and visible throughout a 12-frame
sequence, with foreground/background separation and camera parallax. This is
an original, **structured animation + learned motion** system, **not** a
general text-to-video foundation model and **not** ByteDance Seedance.

V0.3 minimized global frame error and often lost the small character. V0.4
uses ground-truth alpha masks, explicit layer compositing, character-weighted
image loss, and supervised motion trajectories. The MLP learns to interpolate
control paths (linear, ease, jump, dash) from scene metadata. Character and
scenery imagery is rendered from reusable layers, so the figure cannot be
deleted by the network. Limitation: movement/pose families and camera controls
are specified; the model **does not infer motion instructions from text** or
draw new limbs/faces. The 2D sprite leg cycle is scripted, not learned.

## OptiPlex WSL Ubuntu: install/pull/test

Do not run bash commands in Windows PowerShell. Start PowerShell with \`wsl\`,
then inside Ubuntu:

\`\`\`bash
cd /mnt/c/Users/Anharoniclabs/quantoniumoslinux
git pull --ff-only origin feat/quantonium-anime-video-v01
source ~/.quantonium-video-env/bin/activate

python -m py_compile runtime/experiments/quantonium_video/train_v04.py
python -u runtime/experiments/quantonium_video/train_v04.py \
  --out ~/quantonium-video-runs/v04 \
  --steps 20 --save-every 10 --threads 4
\`\`\`

Expect \`EVAL\`, \`SAVED\`, and \`PREVIEW\`, and \`FINISHED 20\`.
Checkpoint: \`~/quantonium-video-runs/v04/latest.pt\`.
Preview: \`preview_latest.mp4\` if FFmpeg installed, otherwise GIF.
Comparison grid: \`comparison_latest.png\` (columns in order:
**V0.4 learned / layered linear baseline / target**; rows at selected times).
The preview consists of 12 separately composited frames, not a diagnostic GIF.
Use \`explorer.exe "$(wslpath -w ~/quantonium-video-runs/v04)"\` to open it.

## Background training and resume

After the smoke test succeeds:

\`\`\`bash
bash runtime/experiments/quantonium_video/run_v04.sh start
bash runtime/experiments/quantonium_video/run_v04.sh status
bash runtime/experiments/quantonium_video/run_v04.sh logs
\`\`\`

Stops at **1,000 TOTAL steps**, resumes from 20, saves every 200 steps.
Stop safely with \`bash runtime/experiments/quantonium_video/run_v04.sh stop\`.
To run until **5,000 total steps** after the first run completes:

\`\`\`bash
QVIDEO_STEPS=5000 bash runtime/experiments/quantonium_video/run_v04.sh start
\`\`\`

Do not start two separate training jobs pointing to the same output folder.
The worker is detached from the terminal but NOT a reboot-proof service.
If WSL shuts down, restart using \`run_v04.sh start\`. The log is local and
trimmed on restart if it grows beyond 1 MiB. Checkpoint, best and previews
are overwritten rather than being accumulated.

## Evaluate honestly

Validation uses fixed unseen procedural seeds, with 12 held-out scenes at 3
timestamps per scene. Output metrics:

* \`model_l1\`: whole-image L1 for the learned layered compositing
* \`layered_linear_l1\`: the **important** fair layered-motion baseline using
  the same clean cutouts and backgrounds, but linearly interpolated paths
* \`crossfade_l1\`: naïve image dissolve baseline
* \`character_l1\`: foreground-weighted reconstruction error
* \`mask_iou\`: overlap with procedural ground-truth character masks
* \`position_mae_px\`: error in scene coordinates, pixel units

A low error or high mask IoU on this controlled synthetic generator does
**not** establish generalization to real anime, camera scenes, or human
movement. A small advantage over crossfading is not enough: compare against
the stronger \`layered_linear_l1\` baseline.

## Preview with your own original anime artwork

Supply an **original character PNG with transparent alpha** and optionally
one background plate PNG/JPG. Once you have trained and saved V0.4 weights:

\`\`\`bash
python -u runtime/experiments/quantonium_video/train_v04.py \
  --out ~/quantonium-video-runs/v04 --preview-only \
  --character-png "/path/to/my_character_cutout.png" \
  --background-png "/path/to/my_background.png"
\`\`\`

The character is composited as a rigid cutout that follows the learned
trajectory; this preserves its artwork but does not synthesize pose changes.
The background plate is a **flat** layer, not depth-aware Blender rendering.
To teach new poses and scene semantics, we will need supervised frame
sequences plus masks/poses and a further learned visual model. Do not
misrepresent a compositing result as free-form video generation.

## Storage and safety

Training output is under \`~/quantonium-video-runs/v04\` (WSL's Linux filesystem).
The completed V0.2 and V0.3 checkpoints remain untouched. The repository
contains only code; training artifacts and private artwork are never
automatically uploaded. Only use animation/assets you own or can lawfully use.
