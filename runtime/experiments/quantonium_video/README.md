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
