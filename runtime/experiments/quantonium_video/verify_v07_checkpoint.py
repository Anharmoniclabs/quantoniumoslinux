#!/usr/bin/env python3
"""Read-only validation of a restored V0.7 checkpoint, shot and MP4 preview."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import subprocess

import torch

from train_v07 import DATA_ID, DEFAULT_ASSETS, PoseData, PoseInterpolator, evaluate


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(args):
    torch.set_num_threads(args.threads)
    root = Path(args.run).expanduser().resolve()
    latest = root / 'latest.pt'
    state = torch.load(latest, map_location='cpu', weights_only=True)
    config = state['config']
    require(config['version'] == DATA_ID, 'Not a supported V0.7 checkpoint')
    step = state['step']
    require(isinstance(step, int) and step > 0, 'Invalid saved step')
    if args.expected_step is not None:
        require(step == args.expected_step,
                f'Expected step {args.expected_step}, found {step}')
    require(all(torch.isfinite(v).all() for v in state['model'].values()),
            'Nonfinite model tensor')
    random.Random().setstate(state['rng'])
    torch.Generator().set_state(state['torch_rng'])

    model = PoseInterpolator(config['size'], config['width'])
    model.load_state_dict(state['model'], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    optimizer.load_state_dict(state['optimizer'])
    for group in optimizer.param_groups:
        for param in group['params']:
            saved = optimizer.state[param]
            require(float(saved['step']) == step, 'Optimizer step does not match model step')
            for name in ('exp_avg', 'exp_avg_sq'):
                require(saved[name].shape == param.shape, 'Optimizer tensor shape mismatch')
                require(torch.isfinite(saved[name]).all(), 'Nonfinite optimizer tensor')

    best_path = root / 'best.pt'
    best = torch.load(best_path, map_location='cpu', weights_only=True)
    require(best['config'] == config, 'Best checkpoint config differs')
    require(0 < best['step'] <= step, 'Best checkpoint step is outside saved history')
    require(best['best'] == state['best'], 'Best metric does not match latest checkpoint')
    best_model = PoseInterpolator(config['size'], config['width'])
    best_model.load_state_dict(best['model'], strict=True)
    require(all(torch.isfinite(v).all() for v in best['model'].values()),
            'Nonfinite best-checkpoint tensor')

    data = PoseData(args.assets, config['size'], args.motion_checkpoint)
    data.attach_shot(root / 'blender_shot')
    require(data.fingerprint == config['data'], 'Checkpoint and source data differ')
    require(data.shot_hash == config['shot'], 'Checkpoint and Blender shot differ')
    metrics = evaluate(model, data, args.val_samples)

    preview = json.loads((root / 'preview_manifest.json').read_text())
    require(preview['step'] == step, 'Preview was rendered from another step')
    require(preview['blender_shot_sha256'] == data.shot_hash, 'Preview used another shot')
    require(preview['keyframe_count'] == 8, 'Preview does not contain all eight source keys')
    require(preview['max_registered_keyframe_error'] == 0,
            'Registered source keys were changed before encoding')
    video = root / 'preview_latest.mp4'
    probe = subprocess.run([
        'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration',
        '-of', 'json', str(video),
    ], check=True, capture_output=True, text=True)
    stream = json.loads(probe.stdout)['streams'][0]
    require(int(stream['nb_read_frames']) == preview['frames'], 'Decoded frame count differs')
    require(Fraction(stream['r_frame_rate']) == preview['fps'], 'Video frame rate differs')
    require([stream['width'], stream['height']] == preview['resolution'], 'Video size differs')
    require(abs(float(stream['duration']) - preview['frames'] / preview['fps']) < .01,
            'Video duration differs')
    return dict(
        status='PASS', step=step, best_step=best['step'], best_model_l1=state['best'],
        parameters=sum(p.numel() for p in model.parameters()), optimizer_step=step,
        checkpoint_sha256=sha256(latest), best_checkpoint_sha256=sha256(best_path),
        video_sha256=sha256(video), data_fingerprint=data.fingerprint,
        blender_shot_sha256=data.shot_hash, model_evaluation=metrics,
        video=stream, registered_source_keys=preview['keyframe_count'],
        max_registered_keyframe_error=preview['max_registered_keyframe_error'],
        scope='Same eight poses and held-out augmentation seed; not held-out motion or character evaluation',
        limitation='Passing these checks does not establish clean in-between anatomy or animation quality',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Restored run directory')
    parser.add_argument('--assets', default=str(DEFAULT_ASSETS))
    parser.add_argument('--motion-checkpoint', default='')
    parser.add_argument('--expected-step', type=int)
    parser.add_argument('--val-samples', type=int, default=12)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1 or args.val_samples < 1:
        parser.error('threads and val-samples must be positive')
    print(json.dumps(verify(args), indent=2))


if __name__ == '__main__':
    main()
