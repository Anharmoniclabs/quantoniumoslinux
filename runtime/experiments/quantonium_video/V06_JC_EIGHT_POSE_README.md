# Quantonium Video V0.6 — JC eight-pose animation proof

The older V0.5 preview failed creator QC: cropped head, extraneous character-sheet callouts, and a rigid image sliding over a generated city background.

## Current workflow: GitHub push → Codespaces fetch → render

**Everything required for the V0.6 pose proof is now committed to `feat/quantonium-anime-video-v01`.**
All eight creator-owned JC kick poses are tracked in
`runtime/experiments/quantonium_video/assets/jc_v06_lossless/`
as **pixel-identical lossless WebP** files. The code converts these into PNGs
outside the repository during rendering. An MP4 proof, integrity checksums,
and the V0.6 renderer are also committed.

The creator explicitly approved making these pose assets available in the
public GitHub repository. This does not waive their copyright.

Run in the **existing GitHub Codespace**:

```bash
cd /workspaces/quantoniumoslinux
source .phistate-env/bin/activate
git fetch origin feat/quantonium-anime-video-v01
git restore --source=origin/feat/quantonium-anime-video-v01 \
  --worktree -- runtime/experiments/quantonium_video
python -m pip install -q pillow
bash runtime/experiments/quantonium_video/fetch_and_render_v06.sh
```

No Google Drive, rclone, local computer, uploaded ZIP, or OAuth setup is needed.

It reads the completed V0.4 motion checkpoint from
`~/quantonium-video-runs/v04_codespace/latest.pt`, which is never modified.

Outputs:
- `~/quantonium-video-runs/v06_codespace/jc_action_cels_v06.mp4`
- `~/quantonium-video-runs/v06_codespace/jc_pose_frame_review.png`
- `~/quantonium-video-runs/v06_codespace/v06_proof_manifest.json`
- `~/quantonium-anime-data/jc_v06/pose_assets/pose_01.png` through `pose_08.png`

These are the creator's eight body-changing kick poses, not a static cutout.
The trajectory comes from the trained V0.4 controller. The poses themselves
are selected existing drawings, **not generated in-between drawings**,
not Blender rigging, and not a new generative video model.

The 5,000-step V0.4 checkpoint remains intact. Review the new action proof
before approving visual continuity and timing.
