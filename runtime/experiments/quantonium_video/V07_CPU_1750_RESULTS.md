# V0.7 CPU checkpoint at 1,750 steps

On October 29, 2026, the separate JC CPU experiment resumed from step 750 and
completed 1,000 additional optimizer updates. Both the latest checkpoint and
the best checkpoint record step **1,750**. The AdamW state also records 1,750
updates for every parameter; model and optimizer tensors are finite.

This 87,817-parameter run started from new V0.7 weights. It does not resume the
user's V0.4 Codespaces model. No V0.4 motion checkpoint was supplied, so root
travel is fixed. It uses the eight tracked JC cels and a verified Blender shot.

| Evaluation metric | Step 750 | Step 1,750 |
| --- | ---: | ---: |
| Model RGB L1 | 0.014222 | 0.012097 |
| Alpha-mask L1 | 0.027854 | 0.021529 |
| Contact loss | 0.026783 | 0.012973 |
| Rig-warp RGB L1 | 0.015481 | 0.015481 |
| Crossfade RGB L1 | 0.029438 | 0.029438 |

The final model RGB error is 14.94% below step 750 and 21.86% below the rig-warp
baseline. These are 12 evaluations on the **same eight poses with a held-out
augmentation seed**, not held-out motion, characters, or shots. They do not
measure an equivalent percentage improvement in perceived animation quality.

## Checks performed

- All 13 existing `test_v07.py` regression tests passed.
- The actual latest and best checkpoint states loaded strictly on CPU.
- The optimizer, RNG states, source hashes and Blender shot hashes validated.
- An incorrect expected step of 1,751 was rejected without changing weights.
- The MP4 decoded to 72 frames, 960×540, 24 fps, and 3 seconds.
- All eight registered source keys reported zero modification before encoding.
- All 72 frames were inspected as a contact sheet. Double outlines are reduced
  in some landing frames, but ghosting and shape changes remain during pose
  transitions. This is an experimental preview, not animation-quality approval.

Machine-readable evidence is in [results/v07_cpu_1750.json](results/v07_cpu_1750.json).
Training used source commit `9163835de8021541027b4383e52c7c4233a8f169`,
Python 3.12, PyTorch 2.6.0+cpu, NumPy 2.3.5, Pillow 12.3.0, four CPU threads,
128×128 training tensors, width 16, batch 2 and learning rate 0.0003.

The latest/best checkpoint SHA-256 is
`e211a2ea5ecae42931a152d38aca59758bbc49b1db2640c942dd8ff612fb3c24`.
Weights, optimizer state, Blender artifacts and generated media are supplied
separately in `QVideo_V07_CPU_Checkpoint.zip`; they are not committed to Git.

## Verify a restored run

Restore the complete `qvideo_v07_chat_run` directory from the checkpoint bundle.
With the video Python environment active and FFmpeg/ffprobe installed:

```bash
python runtime/experiments/quantonium_video/verify_v07_checkpoint.py \
  --run /path/to/qvideo_v07_chat_run --expected-step 1750
```

This verifier is read-only. It reloads the model and optimizer, validates data
and shot identity, reruns the fixed evaluation and checks the decoded preview.
Its `PASS` result describes integrity and measured behavior, not artistic quality.

## Continue this exact run

The following target is **2,000 total steps**, adding 250 to this checkpoint:

```bash
python -u runtime/experiments/quantonium_video/train_v07.py \
  --out /path/to/qvideo_v07_chat_run \
  --shot /path/to/qvideo_v07_chat_run/blender_shot \
  --size 128 --width 16 --batch 2 --threads 4 \
  --steps 2000 --save-every 100 --val-samples 12 \
  --lr 0.0003 --temporal-weight 0.15 \
  --frames 72 --fps 24 --render-width 960 --render-height 540
```

Keep the archived Blender shot and omit `--motion-checkpoint` for this run.
The standard Codespaces launcher expects a V0.4 checkpoint and is a different
configuration. Do not rebuild the shot or reset optimizer state to bypass a
resume mismatch. See the bundle's `RESUME.txt` for the full restore contract.
