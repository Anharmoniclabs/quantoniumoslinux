# Quantonium Video V0.6: authentic JC eight-pose kick proof

The V0.5 single-PNG experiment failed creator QC: head cropped, extra reference-sheet content, and rigid sliding cutout. **Do not call it an anime generation result.**

V0.6 instead uses the **eight distinct JC kick poses** on page 3 of the creator's *Full Cast Movement & In-Betweens Bible*. A new preview renderer advances through the action poses and optionally adds a *small* learned root-motion offset from the verified V0.4 5,000-step controller. No face or limbs are synthesized. This is not Blender rigging or trained diffusion/optical-flow inbetweening.

## Codespaces: use the private 8-pose asset pack

The creator receives the private `JC_V06_8_Private_Poses.zip` asset bundle through their ChatGPT conversation. This ZIP is **not in the public GitHub repo** because it contains creator-owned artwork.

1. In the Codespaces Explorer upload the ZIP into `/workspaces/quantoniumoslinux/JC_V06_8_Private_Poses.zip`. Keep it uncommitted.
2. Fetch **only source files** from the video branch:

```bash
cd /workspaces/quantoniumoslinux
source .phistate-env/bin/activate
git fetch origin feat/quantonium-anime-video-v01
git restore --source=origin/feat/quantonium-anime-video-v01 --worktree -- runtime/experiments/quantonium_video
```

3. Extract outside the repo and inspect the review PNG:

```bash
mkdir -p ~/quantonium-anime-data/jc_v06
unzip -o JC_V06_8_Private_Poses.zip -d ~/quantonium-anime-data/jc_v06
ls -lh ~/quantonium-anime-data/jc_v06/pose_assets/pose_*.png
```

4. Build the proof MP4 **using your actual trained V0.4 checkpoint**:

```bash
python -u runtime/experiments/quantonium_video/v06_pose_cels.py \
  --poses ~/quantonium-anime-data/jc_v06/pose_assets \
  --motion-checkpoint ~/quantonium-video-runs/v04_codespace/latest.pt \
  --out ~/quantonium-video-runs/v06_codespace \
  --frames 48 --fps 24 --width 960 --height 540

ls -lh ~/quantonium-video-runs/v06_codespace/jc_action_cels_v06.mp4 \
       ~/quantonium-video-runs/v06_codespace/jc_pose_frame_review.png
```

The MP4 is a **2-second, 24fps pose-changing proof**, with actual arm/leg silhouette changes and only small body root movement. If an approved background plate exists, add `--background /path/to/approved_background.png`. Without a supplied plate, it uses a deliberately simple studio-color background rather than an improvised video-game-looking city.

## Quality-control and scientific limits

- Review eyes, pendant, hair, fingers, shoes, crop alpha, relative character scale and the action timing. It is an **unapproved proof** until creator review.
- The entire V0.4 `latest.pt` and `best.pt` remain unchanged.
- Eight drawings on one sheet do **not** establish enough data for a generative video foundation model. To train a temporal pose-synthesis model, prepare time-aligned multi-frame action clips / pose labels and train/evaluate a separately specified architecture.
- No Blender scene or 2D bone rig is built here. A Blender rig or learned optical-flow/deformation network is a **separate V0.7 stage**. Don't claim it exists.
