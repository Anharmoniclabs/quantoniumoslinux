# Quantonium Anime Video (CPU experiments)

**Current JC pipeline: [V0.7 Blender-first training](V07_BLENDER_TRAINING.md).**
Experimental [open-weight and movie-data integration](OPEN_VIDEO_TRAINING.md)
adds frozen RAFT motion guidance, licensed Blender-film samples, a separate
V0.8 fine-tuning path, and an optional Wan GPU generation backend.
Image-to-rig fitting and a verified Blender shot now precede guided RGBA
in-between training. The eight original cels and V0.4 checkpoint are preserved.
See that guide for the Codespaces launcher, exact registration limits, outputs
and the experimental quality status. Earlier experiments remain documented below.

## V0.1
This is **not ByteDance Seedance**, not a full text-to-video generator, and not trained on copyrighted anime. It is an original tiny 2D frame-inbetweening model intended as the first measurable building block toward cinematic animation, with a synthetic procedural training set produced on the fly.

## Quick start on the OptiPlex (WSL Ubuntu, Python >=3.10)
```bash
git fetch origin
git switch -c feat/quantonium-anime-video-v01 --track origin/feat/quantonium-anime-video-v01
python3 -m venv .video-env
source .video-env/bin/activate
pip install torch numpy pillow
python runtime/experiments/quantonium_video/train.py --steps 200 --threads 4
```
Trains at 64x64 on CPU, reports L1 interpolation loss and writes exactly one checkpoint (`runs/optiplex/latest.pt`) and one preview (`preview_latest.gif`). The procedural generator gives simplified cel-style character motion with camera-like parallax, not hand-drawn anime quality.

## Continuous training, restartable
```bash
source .video-env/bin/activate
python -u runtime/experiments/quantonium_video/train.py --steps 0 --save-every 200 --threads 4 2>&1 | tee -a video-train.log
```
Stop with Ctrl+C. Repeating the command resumes from `latest.pt`. For persistent unattended use run inside tmux or a user-managed systemd service. Keep the log rotated or redirect to a limited log file; the checkpoint and GIF are overwritten, not accumulated. Before starting, verify free storage and RAM.

## Your own licensed frame sequences (Blender or OpenToonz)
Set `--frames-dir /path/to/frames`. Each direct subdirectory should be an ordered set of at least three `.png` or `.jpg` frames of the **same contiguous shot**. Use filenames `0001.png`, `0002.png`, etc. They are resized to 64x64; adjacent triplets supply start/middle/end. When no valid sequences are found, training falls back to procedural data. Start with a few sequences you own or have permission to use. Build data split/evaluation and true motion metrics before making comparative claims.

## Wan generator adaptation (separate from interpolation)

For actual pretrained video-generator fine-tuning, use the
[Wan LoRA Colab notebook](Quantonium_Wan_LoRA_Colab.ipynb) and
[Wan training documentation](WAN_TRAINING.md). It trains Wan attention adapters,
saves optimizer/RNG state in Drive, resumes continuously, and loads those adapters
for video generation. Full-size GPU training remains subject to the notebook's
on-device verification; local checks use actual tiny Wan modules on CPU.

## Known limitations of the original minimal interpolator
* Minimal ~tens-of-thousands parameter convolutional residual interpolator; no temporal attention, diffusion, learned video tokenizer, text conditioning, pose-conditioning, or audio.
* Training uses generated midpoint supervision, not full-frame video generation.
* No video-to-video or text-to-video ability yet. The goal is to prove motion learning and checkpoint recovery on low-power hardware.
* The initial synthetic set is deterministic per seed, diverse only in simple color/shape/position.
* GPU acceleration and Blender render automation are future work.
* Keep all experiments and generated assets out of git.

## V0.2 OptiPlex: background run through 5,000 total steps

In **WSL Ubuntu** (not Windows PowerShell):

```bash
cd /mnt/c/Users/Anharoniclabs/quantoniumoslinux
git pull --ff-only origin feat/quantonium-anime-video-v01
bash runtime/experiments/quantonium_video/run_5000.sh start
bash runtime/experiments/quantonium_video/run_5000.sh status
bash runtime/experiments/quantonium_video/run_5000.sh logs
```

`start` resumes the existing `~/quantonium-video-runs/v02/latest.pt` checkpoint.
If that checkpoint reached 1,000 steps, it trains **4,000 more**, stopping
automatically at 5,000 total steps. Running `start` twice does not intentionally
start two workers. Logs are at `~/quantonium-video-runs/v02/train_5000.log`;
both checkpoint and preview GIF are overwritten, not accumulated.

To stop cleanly:

```bash
bash runtime/experiments/quantonium_video/run_5000.sh stop
```

The trainer catches SIGTERM, finishes the current update and saves its checkpoint.
On unexpected training failure, the supervisor retries up to three times.
The run survives normal terminal disconnection through `nohup`, but **not**
a Windows/WSL shutdown, sleep termination, or reboot. Run `start` again after
reboot; it resumes the last saved checkpoint. The training does not secretly
upload your frames or checkpoints.

Options: `QVIDEO_THREADS=3`, `QVIDEO_STEPS=5000`, `QVIDEO_OUT=/path`,
`QVIDEO_PYTHON=/path/to/python` can be passed before the launcher command.
On the current procedural dataset, this is an engineering study of
frame interpolation, not a Seedance-class video-generation system.

## V0.3: 12-frame, arbitrary-timestamp video generation (CPU)

V0.3 uses a separate U-Net-like temporal warp model. It predicts two displacement
fields, an occlusion/blending mask, and a small residual for any timestamp
between two anchor frames. The procedural scenes include camera pans and
layer parallax. This is still **frame interpolation**, not text-to-video or a
trained Seedance replacement. V0.2 weights are not compatible.

In **WSL Ubuntu** on the OptiPlex:

```bash
cd /mnt/c/Users/Anharoniclabs/quantoniumoslinux
git pull --ff-only origin feat/quantonium-anime-video-v01
source ~/.quantonium-video-env/bin/activate
sudo apt install -y ffmpeg  # optional if already present
python -u runtime/experiments/quantonium_video/train_v03.py \
    --out ~/quantonium-video-runs/v03 --steps 20 --save-every 10 --threads 4
```

For a separate background run to 1,000 TOTAL steps:

```bash
bash runtime/experiments/quantonium_video/run_v03.sh start
bash runtime/experiments/quantonium_video/run_v03.sh status
bash runtime/experiments/quantonium_video/run_v03.sh logs
bash runtime/experiments/quantonium_video/run_v03.sh stop
```

To continue the **same** run to 5,000 TOTAL steps:

```bash
QVIDEO_STEPS=5000 bash runtime/experiments/quantonium_video/run_v03.sh start
```

The runner writes under `~/quantonium-video-runs/v03` on WSL's
Linux filesystem. The checkpoint `latest.pt`, optional `best.pt`,
`preview_latest.mp4` (or GIF without ffmpeg), and bounded-on-restart
log are the only recurring outputs. The preview is a **12-frame sequence**
at 12 FPS (some frames are neural predictions; endpoints are anchors).
The model is trained from scratch and intentionally does not alter V0.2.

### Owned Blender / OpenToonz data

Use a folder with **at least two shot subfolders**, each with four or more
consecutive PNG/JPG frames, numbered in order. E.g.:
```text
/home/user/anime-shots/
  shot001/0001.png 0002.png 0003.png 0004.png ...
  shot002/0001.png 0002.png 0003.png 0004.png ...
```

```bash
python -u runtime/experiments/quantonium_video/train_v03.py \
  --out ~/quantonium-video-runs/v03-real \
  --frames-dir ~/anime-shots \
  --size 128 --frames 12 --steps 1000 --threads 4
```

Training/validation are separated **by shot folder**, not adjacent frames,
to avoid direct leakage. Use only animation you own or have rights to train
on. Training frames and model checkpoints remain local and out of Git.

### Evaluation

Every 200 steps, V0.3 reports `model_l1`, `crossfade_l1`, and
`edge_l1` on a fixed evaluation split. We want `model_l1` lower than
the crossfade baseline, along with clean continuous playback.
Benchmark values on synthetic data must not be interpreted as evidence of
real-anime generalization or Seedance-level quality. FFmpeg MP4 is optional:
the trainer falls back to an animated GIF if ffmpeg is not installed.
