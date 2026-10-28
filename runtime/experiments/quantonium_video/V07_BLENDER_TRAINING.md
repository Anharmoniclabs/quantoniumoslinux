# V0.7: image-to-rig, Blender shot, guided in-between training

V0.6 selected eight fixed JC cels. V0.7 first fits image joints, builds and checks
a Blender motion shot, then trains a new RGBA interpolator using that shot's
projected controls. The existing V0.4 motion checkpoint is loaded read-only.

This implements the training path. It does **not** establish Seedance-level
quality, a faithful textured 3D JC reconstruction, or 1:1 identity in newly
generated poses. Eight illustrations do not supply hidden anatomy or every
intermediate drawing.

## Run in the existing Codespace

Use the Python environment that ran V0.4. Install Blender/FFmpeg once if absent:

```bash
cd /workspaces/quantoniumoslinux
git pull --ff-only origin feat/quantonium-anime-video-v01
sudo apt-get update
sudo apt-get install -y blender ffmpeg
QVIDEO_STEPS=5000 bash runtime/experiments/quantonium_video/run_v07.sh start
bash runtime/experiments/quantonium_video/run_v07.sh logs
```

The launcher requires `torch`, `numpy`, and `pillow` in `python`. Set
`QVIDEO_PYTHON=/path/to/your/environment/bin/python` if needed. There is no new
external service or paid generation. Blender is mandatory; a failed/missing
Blender prepass stops training instead of quietly using an unrelated renderer.

By default the motion input is
`~/quantonium-video-runs/v04_codespace/latest.pt`, with a fallback to
`~/quantonium-video-runs/v04/latest.pt`. Set `QVIDEO_MOTION` to override it.
V0.7 starts its **own** 5,000-total-step run under
`~/quantonium-video-runs/v07_codespace`; it never replaces V0.4/V0.6 outputs.

```bash
bash runtime/experiments/quantonium_video/run_v07.sh status
bash runtime/experiments/quantonium_video/run_v07.sh stop
# Repeat start to resume. Render the saved V0.7 weights without optimizer updates:
bash runtime/experiments/quantonium_video/run_v07.sh render
```

`start` survives terminal disconnects through nohup, not a Codespace shutdown.
One output directory accepts only one worker. Errors propagate to the log and
`exit_code`; interrupted training finishes its current update and saves.
Checkpoints are written atomically, with optimizer and RNG state. Data, shot,
model shape, and loss settings are checked on resume. Changed sources or shot
settings require a separate `QVIDEO_OUT` directory.

## What is connected

| Requested behavior | Implementation and current limit |
| --- | --- |
| Image makes the rig | `v07_image_rig.py` uses image pixel hashes to recover cached local pose detections and inspected corrections for the eight supplied cels. New images run the official local Keypoint R-CNN COCO_V1 detector; uncertain/multiple detections fail. Joints and hidden depth are estimates. |
| Blender goes first | `prepare_v07.py` calls Blender to create an armature, weighted volumetric proxy limbs, camera, courtyard and lighting. Original reference pixels are packed in the blend. This is a motion-blocking proxy, not a textured character reconstruction. |
| Accurate image registration | Source transforms, original image hashes and projected joints are recorded. The camera projection gate is 0.01 training pixels; evaluated bone endpoints must follow controls within 0.002 world units. These measure registration, not anatomical correctness. |
| Smooth frame-to-frame motion | Blender stages smooth joint transitions. Projected joint deformation supplies backward warp priors. Training adds temporal image/flow acceleration losses and learns warp corrections, visibility weights and RGBA residuals. |
| Ground contact | Source alpha determines the sole instead of crop height. V0.4 root travel is frozen during planted intervals. Training penalizes floor penetration and support-band drift. Generated planted frames receive an explicit contact correction; raw drift and correction are reported separately. |
| Background | The actual Blender courtyard plate is composited in training and the final preview. It is a staged scene, not learned background generation. The registered shot uses a fixed camera. |
| AI in-betweens | Six real middle-pose triples plus controlled affine examples supervise a new network. Training consumes start/target/end pose guides and rig warp fields. The learned fields render the original higher-resolution cels, preserving more detail than upscaling a low-resolution generated face. |

Every preview includes all eight source keys. They bypass neural synthesis;
their error against the **registered/rescaled source tensors** is zero before
compositing/encoding. Rescaling, antialiasing and lossy MP4 encoding mean the
final video is not byte/pixel-identical to the original WebP files. Generated
in-betweens have no 1:1 guarantee. The default 72 frames at 24 fps make a
three-second action, not a seamless loop.

## Take another image into Blender

The same fitting/prepass can build a static diagnostic armature from a new
single-person, full-body PNG/JPG. RGBA allows measured sole contact; for opaque
images the detector estimates it. This command makes a rig proof, not a newly
trained or automatically textured animated character:

```bash
# New-image detection requires a torchvision build matching your PyTorch version.
python -m pip install torchvision --index-url https://download.pytorch.org/whl/cpu
QVIDEO_IMAGE=/absolute/path/character.png \
  bash runtime/experiments/quantonium_video/run_v07.sh rig
```

The first unknown-image call downloads the official approximately 226 MiB
`keypointrcnn_resnet50_fpn_coco-fc266e95.pth` weights into PyTorch's local cache.
Loading uses `weights_only=True` and the published hash prefix. Images are not
uploaded. The default known-JC path needs no detector download.

Inspect `image_rig/joint_review_01.png` and the Blender proofs. Detection
confidence is not proof of correct anatomy: occluded limbs, unusual anime
proportions and ambiguous stances may need corrected landmarks. The known JC
landmark file retains the original detector points and the inspected estimated
controls; these are not labeled as creator-approved anatomy.

## Outputs and validation

- `blender_shot/motion_shot.blend`: packed source references, armature, weighted
  proxy geometry, staged shot and camera.
- `blender_shot/joint_review_*.png`, `rig_proof_*.png`, `shot_manifest.json`:
  source overlays, beginning/middle/end rig renders and per-frame projection checks.
- `latest.pt`, `best.pt`: V0.7 only; periodic and final saves.
- `preview_latest.mp4` (GIF if FFmpeg is absent), `comparison_latest.png`:
  generated action and selected review frames.
- `metrics_latest.json`, `preview_manifest.json`, `run_manifest.json`:
  model versus crossfade **and rig-warp** baselines, contact corrections,
  checkpoint steps, source identity and limitations.

Validation uses the same eight poses with a held-out augmentation seed. It is
not a held-out character/shot benchmark. A lower average pixel loss does not
prove clean hands, limb anatomy, facial identity or cinematic quality. Inspect
the full movie, especially takeoff, kick, landing and occlusion changes.

```bash
python -m unittest discover -s runtime/experiments/quantonium_video -p test_v07.py -v
```

Verified during implementation with Blender 4.2.9 LTS on CPU: image detector
execution, Blender prepass and evaluated rig projection, 300 guided optimizer
updates, source-key bypass, preview encoding and 13 regression tests. Resume
from step 300, duplicate-worker prevention, status and graceful stop/save at
step 301 also passed. The local
integration fixture uses a one-step V0.4 test checkpoint; it does not claim
access to or validation of the user's 5,000-step checkpoint.

| Local smoke check | Result |
| --- | --- |
| Step-300 model / rig warp / crossfade RGB L1 | 0.015007 / 0.015588 / 0.029447 |
| Blender maximum projection error | 0.0000958 training pixels |
| Maximum evaluated bone endpoint error | 0.000000322 world units |
| Preview | 960×540, 72 frames, 24 fps, 3 seconds |
| Registered source keys | 8 present; zero neural modification before encoding |
| Packed reference RGBA pixels | All 8 equal the original source pixels |

These are engineering smoke results on the small known-pose set. Visible
ghosting and shape changes remain in generated frames; this is not an animation
quality acceptance or evidence that a 5,000-step run will eliminate them.

Useful overrides: `QVIDEO_THREADS`, `QVIDEO_STEPS`, `QVIDEO_SAVE_EVERY`,
`QVIDEO_SIZE`, `QVIDEO_WIDTH`, `QVIDEO_BATCH`, `QVIDEO_FRAMES`,
`QVIDEO_RENDER_WIDTH`, `QVIDEO_RENDER_HEIGHT`, `QVIDEO_BLENDER`, `QVIDEO_OUT`.
Keep the same settings when resuming. No generated model weights, videos or
Blender binaries are committed.
