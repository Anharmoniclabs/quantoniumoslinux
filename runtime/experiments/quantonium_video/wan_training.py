"""Wan 2.1 LoRA flow-matching and verified, resumable checkpoint storage."""
import hashlib
import json
from pathlib import Path
import time
import uuid

import torch

SOURCE = json.loads(Path(__file__).with_name('open_sources.json').read_text())['weights']['wan']
TARGETS = ['to_q', 'to_k', 'to_v', 'to_out.0']
VERSION = 'qvideo-wan21-general-lora-v2'
TRAIN_GROUPS = ('general', 'cinematic', 'identity')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def add_adapter(model, rank):
    from peft import LoraConfig
    model.requires_grad_(False)
    model.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights='gaussian',
                                 target_modules=TARGETS))
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'lora_' not in name:
                raise ValueError('Unexpected trainable base parameter: ' + name)
            param.data = param.data.float()
    return [p for p in model.parameters() if p.requires_grad]


def flow_inputs(clean, noise, sigma):
    """Wan rectified-flow convention: data at 0, noise at 1; v = noise-data."""
    sigma = sigma.reshape(-1, 1, 1, 1, 1).to(clean)
    return (1 - sigma) * clean + sigma * noise, noise - clean


def flow_loss(model, sample, rng, device, dtype, shift=3.):
    clean = sample['latent'].to(device=device, dtype=dtype)
    prompt = sample['prompt_embeds'].to(device=device, dtype=dtype)
    noise = torch.randn(clean.shape, generator=rng).to(device=device, dtype=dtype)
    u = torch.rand((clean.shape[0],), generator=rng)
    sigma = (shift * u / (1 + (shift - 1) * u)).to(device)
    noisy, target = flow_inputs(clean, noise, sigma)
    with torch.autocast(device_type='cuda', dtype=dtype, enabled=device.type == 'cuda'):
        prediction = model(hidden_states=noisy, timestep=sigma * 1000,
                           encoder_hidden_states=prompt, return_dict=False)[0]
    return (prediction.float() - target.float()).square().mean()


def verify_checkpoint(folder, config=None):
    from safetensors.torch import load_file
    folder = Path(folder)
    receipt = json.loads((folder / 'COMPLETE.json').read_text())
    if set(receipt['files']) != {'state.pt', 'pytorch_lora_weights.safetensors'}:
        raise ValueError('Unexpected Wan snapshot files')
    if config is not None and receipt['config'] != config:
        raise ValueError('Different Wan run configuration; choose a separate run folder')
    for name, sha in receipt['files'].items():
        if digest(folder / name) != sha:
            raise ValueError('Wan snapshot checksum mismatch: ' + name)
    state = torch.load(folder / 'state.pt', map_location='cpu', weights_only=True)
    if state['step'] != receipt['step'] or state['config'] != receipt['config']:
        raise ValueError('Wan snapshot step/config mismatch')
    weights = load_file(folder / 'pytorch_lora_weights.safetensors')
    if not weights or any(not k.startswith('transformer.') or 'lora_' not in k
                          or not torch.isfinite(v).all() for k, v in weights.items()):
        raise ValueError('Invalid Wan adapter weights')
    return state, weights


def find_checkpoint(root, config=None, best=False):
    """Inspect newest first; a partial/corrupt write never replaces a valid save."""
    root = Path(root)
    if best:
        pointer = json.loads((root / 'best.json').read_text())['folder']
        folder = root / 'snapshots' / pointer
        if folder.parent.resolve() != (root / 'snapshots').resolve():
            raise ValueError('Unsafe best checkpoint path')
        verify_checkpoint(folder, config)
        return folder
    candidates = sorted((root / 'snapshots').glob('step_*'), reverse=True)
    rejected = []
    for folder in candidates:
        try:
            verify_checkpoint(folder, config)
            if rejected:
                print('WAN recovery rejected:', '; '.join(rejected), flush=True)
            return folder
        except (ValueError, OSError, RuntimeError, EOFError, KeyError) as exc:
            rejected.append(folder.name + ': ' + str(exc))
    if candidates or (root / 'run_config.json').exists():
        raise ValueError('No valid Wan checkpoint; refusing a silent training reset. ' + '; '.join(rejected))
    return None


def save_checkpoint(root, model, optimizer, rng, step, config, best_loss, metrics, is_best):
    from diffusers import WanPipeline
    from peft import get_peft_model_state_dict
    root = Path(root)
    folder = root / 'snapshots' / f'step_{step:09d}_{time.time_ns()}_{uuid.uuid4().hex[:6]}'
    folder.mkdir(parents=True)
    adapter = {k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(model).items()}
    if any(not torch.isfinite(v).all() for v in adapter.values()):
        raise ValueError('Refusing nonfinite Wan weights')
    WanPipeline.save_lora_weights(folder, transformer_lora_layers=adapter)
    state = dict(step=step, config=config, optimizer=optimizer.state_dict(),
                 rng=rng.get_state(), torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                 best_loss=best_loss, metrics=metrics)
    torch.save(state, folder / 'state.pt')
    write_json(folder / 'COMPLETE.json', dict(step=step, config=config, files={
        name: digest(folder / name) for name in ('state.pt', 'pytorch_lora_weights.safetensors')}))
    verify_checkpoint(folder, config)
    write_json(root / 'run_config.json', config)
    write_json(root / 'latest.json', dict(folder=folder.name, step=step, metrics=metrics))
    if is_best:
        write_json(root / 'best.json', dict(folder=folder.name, step=step, loss=best_loss))
    print('WAN DRIVE SNAPSHOT VERIFIED', step, str(folder), flush=True)
    return folder


def restore_checkpoint(folder, model, optimizer, rng, config):
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    state, stored = verify_checkpoint(folder, config)
    adapter = {k.removeprefix('transformer.'): v for k, v in stored.items()}
    expected = get_peft_model_state_dict(model)
    if set(adapter) != set(expected) or any(adapter[k].shape != expected[k].shape for k in expected):
        raise ValueError('Adapter architecture mismatch')
    set_peft_model_state_dict(model, adapter)
    optimizer.load_state_dict(state['optimizer'])
    rng.set_state(state['rng'])
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state


def prune_checkpoints(root, keep_last=3):
    """Rotate only this trainer's completed saves; keep step zero and best too."""
    import shutil
    root = Path(root)
    best = json.loads((root / 'best.json').read_text())['folder']
    config = json.loads((root / 'run_config.json').read_text())
    completed = []
    for folder in sorted((root / 'snapshots').glob('step_*'), reverse=True):
        try:
            receipt = json.loads((folder / 'COMPLETE.json').read_text())
            if receipt['config'] == config:
                completed.append((folder, receipt['step']))
        except (OSError, ValueError, KeyError):
            continue
    for folder, step in completed[keep_last:]:
        if folder.name != best and step != 0:
            shutil.rmtree(folder)


def read_cache(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['source'] != SOURCE or manifest['version'] != VERSION:
        raise ValueError('Wrong Wan source/cache version')
    groups = {name: [] for name in (*TRAIN_GROUPS, 'validation')}
    source_splits = {}
    for row in manifest['records']:
        path = root / row['file']
        if path.parent.resolve() != root.resolve() or digest(path) != row['sha256']:
            raise ValueError('Corrupt or unsafe Wan cache')
        split = 'validation' if row['group'] == 'validation' else 'train'
        if source_splits.setdefault(row['source'], split) != split:
            raise ValueError('Source film leaks across Wan train/validation')
        sample = torch.load(path, map_location='cpu', weights_only=True)
        if (sample['latent'].ndim != 5 or sample['prompt_embeds'].ndim != 3
                or not all(torch.isfinite(v).all() for v in sample.values())):
            raise ValueError('Invalid cached Wan tensors')
        groups[row['group']].append(sample)
    if not groups['general'] or not groups['validation']:
        raise ValueError('Need general-video training and held-out validation samples')
    return groups, digest(root / 'manifest.json')
