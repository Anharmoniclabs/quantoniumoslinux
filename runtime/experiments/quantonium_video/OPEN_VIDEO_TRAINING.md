# Open weights and movie-data integration

V0.8 reuses the V0.7 `PoseInterpolator` architecture and strictly loads its
trained weights into a **separate fine-tuning run**. A frozen pretrained
RAFT-Small supplies endpoint motion guidance. Actual middle movie frames supply
supervision. Every update also rehearses the original JC poses with their
verified Blender guides, alpha and contact losses.

This is experimental frame interpolation, not Seedance or a replacement for the
existing character run. No new checkpoint is automatically promoted. The hourly
V0.7 task continues using its existing configuration, not this experimental run.

## Sources

`open_sources.json` records URLs, licenses, roles, full release hashes and an
immutable Wan revision. Media and weights stay outside Git.

| Source | Integration | Scope |
| --- | --- | --- |
| Torchvision RAFT-Small C_T_V2 | Loaded and frozen; cached bidirectional endpoint flow | 990,162 parameters; original RAFT code BSD-3-Clause; upstream training-data terms remain separate |
| Big Buck Bunny | RGB triplets for gradient updates | CC-BY-3.0; Blender Foundation attribution retained |
| Sintel trailer | Entire film held out from gradient updates | CC-BY-3.0; excludes opening/closing cards |
| Wan 2.1 T2V 1.3B Diffusers | Separate optional CUDA generation command | Apache-2.0; no direct weight merging with V0.7; GPU inference untested here |

Official sources: [RAFT weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.optical_flow.raft_small.html),
[RAFT license](https://github.com/princeton-vl/RAFT/blob/master/LICENSE),
[Big Buck Bunny](https://peach.blender.org/about/),
[Sintel](https://durian.blender.org/sharing/),
[Wan model](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers),
[Wan API](https://huggingface.co/docs/diffusers/api/pipelines/wan).
AnimeRun was investigated but is not auto-imported without an explicit dataset
license. No scraped anime footage is included.

## Prepare data and train

From this directory, in the existing Python CPU environment:

```bash
pip install -r requirements-open-cpu.txt --index-url https://download.pytorch.org/whl/cpu
python prepare_open_video.py --out ~/quantonium-video-data/open-v1 --samples 64
python train_open_v08.py \
  --init /path/to/original-v07/latest.pt \
  --shot /path/to/original-v07/blender_shot \
  --data ~/quantonium-video-data/open-v1 \
  --out ~/quantonium-video-runs/open-v08 --steps 100
```

If the parent used V0.4 motion, also pass its original `--motion-checkpoint`.
The rehearsal fingerprint and shot must match the parent. The first launch
starts a new AdamW optimizer at 3e-5; subsequent launches restore optimizer and
both RNG states. `--steps` counts **total updates in this new phase**. Retain an
immutable copy of the parent checkpoint: its hash is part of the resume contract.
Never use the original run's output directory.

The importer downloads about 69 MB total on the first launch, with bounded
downloads, exact release checksums, RGB-only extraction, and no audio. Each
triplet contains consecutive 12-fps frames; the model predicts the midpoint.
The cache stores frames, endpoint flow, source attribution, timestamps and hashes.
Review extracted triplets before larger runs: cut/blank filters are heuristics.
The checked-in result describes the reviewed cache used in this session;
regenerating a new cache can produce a different data fingerprint.

All movie samples are opaque RGB plates; they do not have character masks,
joint annotations or ground-contact labels. The first nine guide channels are
zero and contact is disabled for movie examples. Only endpoint frames enter
RAFT. The target frame never enters guidance, including during validation.
Scaled endpoint flows approximate intermediate backward motion; acceleration,
occlusion and disocclusion remain difficult. V0.7's zero-padding can also cause
border artifacts on opaque plates. Compare against crossfade and RAFT-only
baselines before claiming a quality gain.

## Render with the fine-tuned model

```bash
python interpolate_open_video.py --checkpoint ~/quantonium-video-runs/open-v08/best.pt \
  --first /path/start.png --last /path/end.png --out /path/new-preview
```

This uses the same endpoint-only guide path as training and writes a square RGB
MP4 plus frames and provenance. It does not invent JC rigs for arbitrary images.
Original JC rendering remains in `v07_preview.py`; its source-pose guarantees
and original checkpoint are preserved.

## Optional GPU generation

In a **separate CUDA PyTorch environment**, install `requirements-wan.txt`, then:

```bash
python generate_wan.py --prompt 'An original cel-shaded character practicing a slow kick, fixed camera' \
  --out ~/quantonium-video-runs/wan-shot001
```

This is the Wan model's generation, not newly learned text conditioning in the
tiny Quantonium interpolator. It checks CUDA before downloading; no remote paid
job is launched. It pins all components to one revision and uses safetensors.
Plan for multi-GB downloads, GPU VRAM and CPU offload memory. No GPU throughput,
memory fit or generation quality was validated here.

Outputs include numbered PNGs, MP4 and generation provenance. Reviewed generated
shots can enter the existing `train_v03.py --frames-dir` path (at least two shot
directories for its split). They are **not automatically mixed into V0.8's
licensed real-video evaluation**, and no Wan weights are merged into JC weights.

## Checks

```bash
python -m unittest test_v07 test_open_video -v
```

The integration tests cover flow direction, exact endpoints, cut filtering,
opaque-RGB conversion, checksum rejection and split leakage. The session also
executes real pretrained inference, fine-tuning, checkpoint resume and rendering.
See `results/open_video_integration.json` for actual measured results and limits.
