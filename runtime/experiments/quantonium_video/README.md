# Quantonium Anime Video V0.1 (CPU experiment)
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

## Known limitations
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
