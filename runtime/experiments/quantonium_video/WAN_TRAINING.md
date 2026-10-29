# Wan general-generation training V2

[Open the Wan Colab notebook](https://colab.research.google.com/github/Anharmoniclabs/quantoniumoslinux/blob/feat/quantonium-anime-video-v01/runtime/experiments/quantonium_video/Quantonium_Wan_LoRA_Colab.ipynb).
Run sections 1–3 on L4; leave section 3 active. Stop once, wait for the
verified save, then use section 5 to generate with the trained adapter.

This adds actual LoRA optimization to the Wan 2.1 T2V 1.3B generation backend.
V2 targets broad prompt-to-video behavior rather than a JC-only pose model.
It is separate from V0.8 interpolation. The V0.8 checkpoint and 19,369 reported
phase steps are not convertible into Wan weights. A new Wan run starts at zero
on the pinned pretrained base, then resumes its own adapter, optimizer and RNG.

## Data and objective

`prepare_wan_training.py` verifies the eight original JC cels and downloads the
checksummed Blender film releases recorded in `open_sources.json`. JC cels are
single-frame identity examples and receive only 10% of the default sampling mix.
The remaining curriculum is 70% general video and 20% cinematic/anime video.

Additional owned or licensed clips can be placed in a folder with `index.csv`:

```csv
file,caption,group,split
clips/walk.mp4,"A person walks through a station, wide tracking shot",general,train
clips/fight.mp4,"Two animated fighters exchange kicks, orbiting camera",cinematic,train
clips/heldout.mp4,"A dog runs through grass, low tracking camera",general,validation
```

Allowed training groups are `general`, `cinematic`, and `identity`. Captions should
name the subject, action, environment, shot scale, and camera motion. Never add
unlicensed web scrapes or V0.8-generated frames as ground truth.
Big Buck Bunny supplies actual contiguous 17-frame clips at 8 fps; Sintel is
held out entirely. The existing three-frame RAFT cache and generated JC previews
are not repurposed as Wan training videos. Movie captions are generic; cut and
blank filtering are heuristics. Inspect the saved review sheet and source records.

The text encoder and VAE execute sequentially, then unload. Cached embeddings
use Wan's prompt cleanup, 512-token padding and masked zero embeddings. VAE
posterior means use `(z - latents_mean) / latents_std`. The transformer trains
with rectified-flow MSE: `x_t=(1-t)x+t*noise`, target `noise-x`, time `1000*t`,
and shift 3.0. Each update accumulates four batch-one samples with 50% probability
of each training group. Only rank-8 attention adapters receive gradients; base
weights, text encoder and VAE are frozen. Gradient checkpointing and BF16 are on.

This is a small adaptation experiment. Stills do not teach JC's temporal movement;
generic film clips do not establish accurate character motion. Text-to-video is
not exact reference-image/Blender conditioning, and there are no Seedance weights.

## Saves and recovery

`MyDrive/Quantonium/Video/Wan21_LoRA/run_general_v2` holds the new run. Saves happen at
step zero, after the first genuine GPU update, every 25 updates, and on clean
stop. Each complete snapshot contains Diffusers-compatible adapter safetensors,
optimizer state, random state, configuration and checksums. A receipt is written
last, and the files are read back before logging `WAN DRIVE SNAPSHOT VERIFIED`.
Failed saves stop the worker. Corrupt/incomplete newest saves fall back to a
verified older matching snapshot; failure of all snapshots refuses a fresh reset.

The trainer retains its latest three completed snapshots, best validation snapshot
and step zero. It rotates only its own older completed saves. Old interpolation
checkpoints and other Drive folders are untouched. Resume refuses changed data,
adapter shape, optimizer configuration or package versions. Change the run folder
for another experiment. The notebook guards against two video workers on one GPU.

`--steps 0` means continuous while Colab is alive; a positive value is a total
Wan optimizer-step target. Stops complete the current accumulated update before
saving. A sudden runtime loss can discard unsaved updates after the last verified
snapshot. A runtime restart requires rerunning notebook setup, data and Start.

## Commands

```bash
pip install -r requirements-wan-training.txt
python prepare_wan_training.py --out /path/wan-data --movies /path/movie-cache
python -u train_wan_lora.py --data /path/wan-data --out /path/wan-run \
  --steps 0 --save-every 25 --stop-file /path/WAN_STOP
python generate_wan.py --lora-run /path/wan-run --checkpoint latest \
  --prompt 'cemi_jc, 2D anime character in a gray jacket performing a controlled kick' \
  --out /path/new-preview --height 256 --width 256 --frames 17
```

Remove an old stop-request file deliberately before starting a new session. The
notebook does this after checking that no worker is active. Generator provenance
records the adapter step and hash. Omit `--lora-run` for a same-seed base comparison;
use `--checkpoint best` for the best fixed latent validation loss. That metric is
not generated-video quality and is not comparable to V0.8 image L1.

## Verification and limits

`python -m unittest test_wan_training -v` uses real tiny Wan modules without
pretrained downloads: gradient updates affect only adapters; interrupted/resumed
next updates match uninterrupted updates; exported adapters load into WanPipeline
and run latent generation; corrupt checkpoints and leaking/corrupt caches are
rejected; retention preserves the initial, best and recent snapshots.

The authoring environment has CPU only. Tiny Wan VAE one/17-frame encode/decode
checks passed; full pretrained caching, L4 memory fit, full-size training and
generated-video quality remain unverified until the notebook executes. The first
real GPU update must demonstrate nonzero adapter changes and finite gradients.
If the full-size model runs out of memory, previous saved checkpoints remain;
reduce the spatial/temporal cache settings and start a separate experiment.

Dependencies are pinned to the tested APIs in `requirements-wan-training.txt`.
The notebook isolates them and constrains pip to Colab's installed CUDA torch.
Source weights use an immutable revision and safetensors. No model/data blobs are
committed to Git; no external cloud job or automatic publishing is performed.

Sources: [Wan model](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers),
[Wan pipeline](https://huggingface.co/docs/diffusers/api/pipelines/wan),
[LoRA](https://huggingface.co/docs/diffusers/training/lora),
[Blender BBB](https://peach.blender.org/about/),
[Blender Sintel](https://durian.blender.org/sharing/).
