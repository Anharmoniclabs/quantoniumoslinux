# Quantonium Video V0.5 — image sheet / motion correction

## 2026-10-28 quality-control decision

The earlier V0.5 output was **not approved**. Its PDF crop removed the top of JC's head and included the adjacent reference sheet's labels, palette chips and additional face. It then translated that static crop through a synthetic city backdrop. This is **not** character pose generation, Blender rigging, or an inference pass rebuilding limbs.

The V0.4 Codespaces checkpoint at `~/quantonium-video-runs/v04_codespace/latest.pt` finished 5,000 optimizer steps. Preserve it: the motion-controller training is valid for synthetic trajectory interpolation even though the V0.5 render failed visual QC.

## New safe procedure

1. Restore the video source from `origin/feat/quantonium-anime-video-v01`. Source changes do not modify existing weights.
2. Run `bash runtime/experiments/quantonium_video/run_v05_codespace.sh prepare`. It now renders **jayc_full_page_crop_grid.png** instead of making an unsafe automatic crop.
3. Open the full-page grid and select a normalized bounding box (X0,Y0,X1,Y1) covering one complete body, hair to shoes, *excluding all reference-sheet labels and extra faces*. No coordinates are pre-approved. Then run:
   ```bash
   python runtime/experiments/quantonium_video/jayc_assets_v05.py \
     --bible /workspaces/quantoniumoslinux/Heralds_of_the_Cemi_Full_Cast_Character_Bible-1.pdf \
     --page 1 --crop X0 Y0 X1 Y1 \
     --out ~/quantonium-video-runs/v05_codespace/jayc_private
   ```
   Replace X0..Y1 with carefully reviewed decimal values between 0 and 1. The default is intentionally disabled for the 64-page Bible.
4. Inspect `jayc_crop_review.png` and `jayc_cutout.png`. If they include cropped anatomy or adjacent detail art, repeat with a safer crop. A rectangular crop and exterior-white removal cannot necessarily isolate a full human silhouette from a detailed sheet; finish cleanup manually in an image editor. Save the separately curated, transparent, full-body PNG as `jayc_approved.png` in the same private directory.
5. For visible limb changes, provide **two or more distinct creator-approved pose PNGs** under a private pose directory, e.g. `~/quantonium-anime-data/Jayc_approved_poses/`. Every pose must preserve JC's real visual model, with matching scale/feet anchors. Then:
   ```bash
   export QVIDEO_APPROVED_JC=yes
   export QVIDEO_POSES_DIR=~/quantonium-anime-data/Jayc_approved_poses
   bash runtime/experiments/quantonium_video/run_v05_codespace.sh render
   ```
   The renderer now **refuses** to output a misleading rigid single-cutout video when pose images are absent.

## What this does NOT do

Even with a pose pack, V0.5 merely selects supplied pose PNGs with cel holds and uses V0.4's learned **x/y/camera** trajectories. No Blender armature, inverse kinematics, in-between pose synthesis, optical-flow training, or 2D anime generation is implemented. A true animation-learning V0.6 would require real sequence supervision (at least start/end key poses plus aligned in-betweens and character masks) and a separately designed model. Do not claim this stage has trained that capability.

Never commit original Bible PDFs, private user art, PNGs, or checkpoints to the public repository.
