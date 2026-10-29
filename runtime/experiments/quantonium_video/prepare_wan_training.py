"""Cache Wan latents and captions: original JC stills plus real contiguous clips.

The eight cels remain SINGLE-FRAME appearance examples, never fake videos made
by repeating a pose or using the interpolator's imperfect generated frames.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import numpy as np
from PIL import Image, ImageOps
import torch

from prepare_open_video import download, reject_triplet, CATALOG
from train_v07 import DEFAULT_ASSETS
from wan_training import SOURCE, VERSION, digest, write_json, read_cache

JC_PROMPT = ('cemi_jc, original 2D anime young man with brown skin, short dark hair, '
             'huge white eyes with narrow dark pupils, gray jacket over black streetwear, '
             'headphones, gold necklace, white sneakers, full body action pose, fine clean outlines, '
             'flat cel shading, plain light gray background')
FILM_PROMPT = 'A scene from a stylized 3D animated short film, characters and scenery in motion.'


def extract_clip(path, stamp, size, frames=17, fps=8):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(stamp), '-i', str(path),
        '-an', '-vf', f'fps={fps},scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}',
        '-frames:v', str(frames), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-threads', '1', 'pipe:1'],
        capture_output=True, check=True, timeout=60).stdout
    if len(raw) != frames * size * size * 3:
        raise ValueError('Incomplete contiguous clip')
    return torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy().reshape(frames, size, size, 3)).permute(0, 3, 1, 2)


def original_still(path, size):
    with Image.open(path) as source:
        image = source.convert('RGBA')
        canvas = Image.new('RGBA', image.size, (224, 224, 224, 255))
        canvas.alpha_composite(image)
        rgb = ImageOps.pad(canvas.convert('RGB'), (size, size), color=(224, 224, 224))
    return torch.from_numpy(np.array(rgb)).permute(2, 0, 1).unsqueeze(0)


def movie_files(root):
    root.mkdir(parents=True, exist_ok=True)
    sources = json.loads(CATALOG.read_text())['videos']
    for name, spec in sources.items():
        archive = root / (name + ('.zip' if 'archive_member' in spec else '.mp4'))
        if not archive.exists():
            download(spec['url'], archive)
        if digest(archive) != spec['download_sha256']:
            raise ValueError('Movie release checksum differs: ' + name)
        video = root / (name + '.mp4')
        if 'archive_member' in spec:
            with zipfile.ZipFile(archive) as z:
                info = z.getinfo(spec['archive_member'])
                if info.file_size > 100_000_000:
                    raise ValueError('Unexpected movie size')
                with z.open(info) as src, video.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
        yield name, spec, video


def build_examples(args):
    assets = Path(args.assets)
    hashes = dict((line.split()[1].lstrip('*'), line.split()[0])
                  for line in (assets / 'SHA256SUMS').read_text().splitlines() if line.strip())
    examples = []
    for i in range(1, 9):
        path = assets / f'pose_{i:02d}.webp'
        if digest(path) != hashes[path.name]:
            raise ValueError('Original JC pose changed: ' + str(path))
        examples.append(dict(id=f'jc_{i:02d}', frames=original_still(path, args.size),
            caption=JC_PROMPT, source='original_jc_cels', source_sha256=digest(path),
            group='appearance', provenance='Original user-approved cel; appearance only; one frame'))
    for name, spec, video in movie_files(Path(args.movies)):
        count = args.clips if spec['split'] == 'train' else 32
        end = spec['end_seconds'] - args.frames / 8
        for i, stamp in enumerate(np.linspace(spec['start_seconds'], end, count)):
            frames = extract_clip(video, float(stamp), args.size, args.frames)
            # Reuse the conservative heuristic on every adjacent pair; not a semantic guarantee.
            if reject_triplet(frames):
                continue
            examples.append(dict(id=f'{name}_{i:03d}', frames=frames, caption=FILM_PROMPT,
                source=name, source_sha256=digest(video), start_seconds=float(stamp), fps=8,
                group='motion' if spec['split'] == 'train' else 'validation',
                provenance=spec))
    if not all(any(x['group'] == group for x in examples) for group in ('appearance', 'motion', 'validation')):
        raise ValueError('Filtering left an empty group')
    return examples


def prepare(args):
    root = Path(args.out)
    if (root / 'manifest.json').exists():
        groups, fingerprint = read_cache(root)
        manifest = json.loads((root / 'manifest.json').read_text())
        if manifest['settings'] != dict(size=args.size, frames=args.frames, clips=args.clips):
            raise ValueError('Cache settings changed; use a new cache and run folder')
        print('WAN CACHE VERIFIED', {k: len(v) for k, v in groups.items()}, fingerprint, flush=True)
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Wan preparation requires a BF16 CUDA GPU such as L4/A100')
    torch.set_num_threads(4)
    root.mkdir(parents=True, exist_ok=True)
    examples = build_examples(args)
    # Visual contact sheet for all extracted video candidates accepted by the heuristic.
    tile = 128
    review = Image.new('RGB', (6 * tile, ((len(examples) + 5) // 6) * tile), 'white')
    for i, row in enumerate(examples):
        mid = row['frames'][len(row['frames']) // 2].permute(1, 2, 0).numpy()
        review.paste(Image.fromarray(mid).resize((tile, tile)), (i % 6 * tile, i // 6 * tile))
    review.save(root / 'data_review.jpg')
    write_json(root / 'examples.json', [{k: v for k, v in row.items() if k != 'frames'} for row in examples])
    kwargs = dict(revision=SOURCE['revision'])
    from diffusers import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, UMT5EncoderModel
    tokenizer = AutoTokenizer.from_pretrained(SOURCE['repo'], subfolder='tokenizer', **kwargs)
    encoder = UMT5EncoderModel.from_pretrained(SOURCE['repo'], subfolder='text_encoder',
        torch_dtype=torch.bfloat16, use_safetensors=True, low_cpu_mem_usage=True, **kwargs).to('cuda').eval()
    encoder.requires_grad_(False)
    embeddings = {}
    with torch.no_grad():
        for caption in sorted(set(x['caption'] for x in examples)):
            tokens = tokenizer(prompt_clean(caption), padding='max_length', max_length=512,
                               truncation=True, add_special_tokens=True, return_tensors='pt')
            result = encoder(tokens.input_ids.cuda(), tokens.attention_mask.cuda()).last_hidden_state
            length = int(tokens.attention_mask.sum())
            result[:, length:] = 0
            embeddings[caption] = result.cpu().contiguous()
            del result
    del encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print('WAN TEXT CACHE READY; text encoder unloaded', flush=True)
    vae = AutoencoderKLWan.from_pretrained(SOURCE['repo'], subfolder='vae',
        torch_dtype=torch.float32, use_safetensors=True, **kwargs).to('cuda').eval()
    vae.requires_grad_(False)
    vae.enable_tiling()
    mean = torch.tensor(vae.config.latents_mean, device='cuda').view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device='cuda').view(1, -1, 1, 1, 1)
    records = []
    with torch.no_grad():
        for row in examples:
            pixels = row['frames'].permute(1, 0, 2, 3).unsqueeze(0).cuda().float() / 127.5 - 1
            # Deterministic posterior mean. Normalization is the inverse of Wan decode.
            latent = (vae.encode(pixels).latent_dist.mode() - mean) / std
            file = row['id'] + '.pt'
            temporary = root / (file + '.partial')
            torch.save(dict(latent=latent.cpu().to(torch.bfloat16).contiguous(),
                            prompt_embeds=embeddings[row['caption']]), temporary)
            temporary.replace(root / file)
            records.append({k: v for k, v in row.items() if k != 'frames'} |
                           dict(file=file, sha256=digest(root / file)))
            print('WAN LATENT CACHED', row['id'], list(latent.shape), flush=True)
    write_json(root / 'manifest.json', dict(version=VERSION, source=SOURCE,
        settings=dict(size=args.size, frames=args.frames, clips=args.clips), records=records,
        limitations='Generic film captions; cut filter is heuristic; JC stills teach appearance only.'))
    read_cache(root)
    print('WAN CACHE VERIFIED', len(records), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--movies', required=True)
    p.add_argument('--assets', default=str(DEFAULT_ASSETS))
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--frames', type=int, default=17)
    p.add_argument('--clips', type=int, default=64)
    args = p.parse_args()
    if args.size not in (128, 192, 256, 320) or args.frames not in (5, 9, 17) or not 8 <= args.clips <= 128:
        p.error('size 128/192/256/320; frames 5/9/17; clips 8..128')
    prepare(args)
