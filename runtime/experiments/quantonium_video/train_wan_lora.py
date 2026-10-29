"""Train Wan's generative transformer adapters, with continuous verified resume."""
import argparse
import fcntl
import json
from pathlib import Path
import signal
import time

import torch

from wan_training import (SOURCE, VERSION, TARGETS, add_adapter, flow_loss, read_cache,
                          find_checkpoint, restore_checkpoint, save_checkpoint, prune_checkpoints,
                          TRAIN_GROUPS)


def parse_mix(value):
    weights = {name: 0.0 for name in TRAIN_GROUPS}
    for item in value.split(','):
        name, raw = item.split('=', 1)
        if name not in weights:
            raise ValueError('Unknown training group: ' + name)
        weights[name] = float(raw)
    if any(v < 0 for v in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError('Training mix must contain nonnegative weights with a positive sum')
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def choose_group(groups, mix, rng):
    available = [(name, mix[name]) for name in TRAIN_GROUPS if groups[name] and mix[name] > 0]
    if not available:
        raise ValueError('No cached samples match the requested training mix')
    total = sum(weight for _, weight in available)
    draw = float(torch.rand((), generator=rng)) * total
    upto = 0.0
    for name, weight in available:
        upto += weight
        if draw <= upto:
            return groups[name]
    return groups[available[-1][0]]


@torch.no_grad()
def evaluate(model, samples, device, dtype):
    was_training = model.training
    model.eval()
    rng = torch.Generator().manual_seed(92001)
    values = [float(flow_loss(model, sample, rng, device, dtype)) for sample in samples]
    model.train(was_training)
    return sum(values) / len(values)


def train(args):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Use a BF16 CUDA GPU (L4/A100); CPU and T4 paths are not validated')
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _train(args, root)


def _train(args, root):
    from diffusers import WanTransformer3DModel
    import diffusers, transformers, peft
    torch.set_num_threads(4)
    torch.manual_seed(808)
    device, dtype = torch.device('cuda'), torch.bfloat16
    groups, fingerprint = read_cache(args.data)
    mix = parse_mix(args.mix)
    config = dict(version=VERSION, source=SOURCE, data=fingerprint, rank=args.rank,
        lr=args.lr, accumulation=args.accumulation, shift=3., targets=TARGETS,
        dtype='bfloat16', seed=808, training_mix=mix,
        diffusers=diffusers.__version__, transformers=transformers.__version__, peft=peft.__version__)
    saved = find_checkpoint(root, config)
    if args.require_resume and saved is None:
        raise FileNotFoundError('No saved Wan run; refusing a fresh run with --require-resume')
    print('WAN LOAD', SOURCE['repo'], SOURCE['revision'], flush=True)
    model = WanTransformer3DModel.from_pretrained(SOURCE['repo'], subfolder='transformer',
        revision=SOURCE['revision'], torch_dtype=dtype, use_safetensors=True).to(device)
    params = add_adapter(model, args.rank)
    model.enable_gradient_checkpointing()
    model.train()
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=.01)
    rng = torch.Generator().manual_seed(808)
    step, best_loss = 0, float('inf')
    if saved:
        state = restore_checkpoint(saved, model, optimizer, rng, config)
        step, best_loss = state['step'], state['best_loss']
        print('WAN RESUMED', step, '| model + optimizer + RNG restored', flush=True)
    else:
        print('WAN NEW LORA RUN 0 | pretrained generator base; V0.8 steps are separate', flush=True)
    print('WAN TRAINABLE PARAMETERS', sum(p.numel() for p in params), flush=True)
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    last_loss = None
    last_saved_step = step if saved else -1
    def save():
        nonlocal best_loss, last_saved_step
        value = evaluate(model, groups['validation'], device, dtype)
        if not torch.isfinite(torch.tensor(value)):
            raise FloatingPointError('Nonfinite validation loss; earlier checkpoint preserved')
        is_best = value < best_loss
        best_loss = min(best_loss, value)
        metrics = dict(step=step, train_flow_mse=last_loss, validation_flow_mse=value,
            best_validation_flow_mse=best_loss,
            scope='Fixed held-out film latent flow loss; not generated-video quality or V0.8 L1',
            peak_gpu_gib=round(torch.cuda.max_memory_allocated()/1024**3, 3))
        save_checkpoint(root, model, optimizer, rng, step, config, best_loss, metrics, is_best)
        prune_checkpoints(root)
        last_saved_step = step
        print(json.dumps(metrics), flush=True)
    if saved is None:
        save()  # Keep a reproducible step-zero baseline before the first update.
    first_update = True
    while (args.steps == 0 or step < args.steps) and not stopping:
        if args.stop_file and Path(args.stop_file).exists():
            stopping = True
            break
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        if first_update:
            before = [p.detach().clone() for p in params]
        loss_sum = 0.
        for _ in range(args.accumulation):
            group = choose_group(groups, mix, rng)
            sample = group[int(torch.randint(len(group), (1,), generator=rng))]
            loss = flow_loss(model, sample, rng, device, dtype)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite Wan loss; previous verified checkpoint preserved')
            (loss / args.accumulation).backward()
            loss_sum += float(loss.detach()) / args.accumulation
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        step += 1
        last_loss = loss_sum
        if first_update:
            change = sum(float((a - b).abs().sum()) for a, b in zip(before, params))
            if change <= 0 or float(norm) <= 0:
                raise RuntimeError('Wan adapter failed its real gradient/update smoke test')
            print('WAN GPU UPDATE VERIFIED', step, '| adapter delta:', change, flush=True)
            del before
            first_update = False
            save()  # First genuine GPU update is always durable, even before the next interval.
        elif step % args.save_every == 0:
            save()
        print('WAN STEP', step, '| flow_mse', round(last_loss, 6),
              '| seconds', round(time.monotonic() - started, 2), flush=True)
    if step != last_saved_step:
        save()
    print('WAN STOPPED' if stopping else 'WAN FINISHED', step, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--steps', type=int, default=0, help='TOTAL Wan updates; 0 = continuous')
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--save-every', type=int, default=25)
    p.add_argument('--mix', default='general=.70,cinematic=.20,identity=.10',
                   help='Sampling weights; missing/empty groups are safely renormalized')
    p.add_argument('--stop-file', default='')
    p.add_argument('--require-resume', action='store_true')
    args = p.parse_args()
    try:
        parse_mix(args.mix)
    except (ValueError, TypeError) as exc:
        p.error(str(exc))
    if args.steps < 0 or min(args.rank, args.accumulation, args.save_every) < 1 or not 0 < args.lr <= .001:
        p.error('Nonnegative steps; positive rank/accumulation/save interval; 0 < lr <= .001')
    train(args)
