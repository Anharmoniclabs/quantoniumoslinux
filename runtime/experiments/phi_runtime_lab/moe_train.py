#!/usr/bin/env python3
"""35M–100M sparse hybrid MoE research trainer for QuantoniumOS.

Trains from scratch on prepared, fixed-split coding + general-knowledge corpora.
Uses PyTorch CPU by default with optional CUDA training; native RFTMW MoE inference requires a new port.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import time

import numpy as np
import torch
import torch.nn.functional as F

from moe_model import PhiStateMoELM, MoEConfig, config_for
from phistate import count_parameters

ROOT = Path(__file__).resolve().parent
LN2 = math.log(2.0)


def setup(threads: int, seed: int):
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def resolve_device(choice: str) -> torch.device:
    if choice == 'auto':
        choice = 'cuda' if torch.cuda.is_available() else 'cpu'
    if choice == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA selected but torch.cuda.is_available() is False')
    return torch.device(choice)


def model_configuration(a, vocab: int) -> MoEConfig:
    return config_for(a.profile, vocab, mode=a.mode,
                      mixer_basis=a.mix_basis,
                      width=a.width, layers=a.layers, heads=a.heads, window=a.window,
                      experts=a.experts, top_k=a.top_k, expert_ratio=a.expert_ratio,
                      shared_ratio=a.shared_ratio,
                      router_aux_weight=a.router_aux_weight,
                      router_z_weight=a.router_z_weight)


def load_data(data_dir: str):
    path = Path(data_dir).expanduser().resolve()
    m_path = path / 'manifest.json'
    if not m_path.is_file():
        raise FileNotFoundError(f'No prepared corpus manifest: {m_path}; run moe_data.py first')
    manifest = json.loads(m_path.read_text(encoding='utf-8'))
    if manifest.get('format') != 'PhiMoECorpus-v1':
        raise ValueError('Unrecognized corpus format')
    dtype = np.uint16 if manifest['dtype'] == 'uint16' else np.uint32
    datasets = {}
    for key, spec in manifest['bins'].items():
        fp = path / spec['file']
        expected_bytes = spec['tokens'] * np.dtype(dtype).itemsize
        if fp.stat().st_size != expected_bytes:
            raise ValueError(f'Corrupted corpus: {fp} has unexpected length')
        datasets[key] = np.memmap(fp, dtype=dtype, mode='r')
    return manifest, datasets, path


def sample_batch(data: np.ndarray, batch: int, context: int, rng: np.random.Generator):
    if len(data) <= context + 1:
        raise ValueError(f'Corpus split length {len(data)} <= context {context}+1')
    starts = rng.integers(0, len(data) - context - 1, size=batch, endpoint=True)
    idx = starts[:, None] + np.arange(context + 1, dtype=np.int64)[None, :]
    slab = torch.from_numpy(np.asarray(data[idx], dtype=np.int64))
    return slab[:, :-1], slab[:, 1:]


@torch.inference_mode()
def evaluate_domain(model, data, context: int, batch: int, batches: int):
    model.eval()
    device = next(model.parameters()).device
    last = len(data) - context - 1
    if last < 0:
        raise ValueError('Validation split shorter than context')
    starts = np.linspace(0, last, num=batch * batches, dtype=np.int64)
    nll_sum = 0.0
    tokens = 0
    for j in range(0, len(starts), batch):
        idx = starts[j:j+batch, None] + np.arange(context + 1, dtype=np.int64)
        slab = torch.from_numpy(np.asarray(data[idx], dtype=np.int64))
        x, y = slab[:, :-1].to(device), slab[:, 1:].to(device)
        logits = model(x)
        nll_sum += float(F.cross_entropy(logits.reshape(-1, model.cfg.vocab),
                                        y.reshape(-1), reduction='sum').item())
        tokens += y.numel()
    return {'nll': nll_sum, 'tokens': tokens, 'nats_per_token': nll_sum / tokens,
            'bits_per_token': nll_sum / (tokens * LN2)}


def evaluate_both(model, datasets, context, batch, batches):
    t0 = time.perf_counter()
    code = evaluate_domain(model, datasets['val_code'], context, batch, batches)
    knowledge = evaluate_domain(model, datasets['val_general'], context, batch, batches)
    nll = code['nll'] + knowledge['nll']
    count = code['tokens'] + knowledge['tokens']
    return {
        'val_code_bpt': round(code['bits_per_token'], 5),
        'val_general_bpt': round(knowledge['bits_per_token'], 5),
        'val_mean_bpt': round(nll / (count * LN2), 5),
        'val_count_tokens': count,
        'val_elapsed_seconds': round(time.perf_counter() - t0, 3),
    }


def optimizer_for(a, model):
    if a.optimizer == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=a.lr,
                                 weight_decay=a.weight_decay, foreach=False)
    if a.optimizer == 'adafactor':
        if not hasattr(torch.optim, 'Adafactor'):
            raise RuntimeError('PyTorch does not provide Adafactor in this version; use --optimizer adamw')
        return torch.optim.Adafactor(model.parameters(), lr=a.lr,
                                    weight_decay=a.weight_decay, foreach=False)
    raise ValueError(a.optimizer)


def atomic_checkpoint(target: Path, content: dict):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + '.partial')
    try:
        torch.save(content, tmp)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def ckpt_load(filename: str):
    data = torch.load(filename, map_location='cpu', weights_only=True)
    if data.get('format') != 'PhiStateMoE-v1':
        raise ValueError('Not a PhiState-MoE checkpoint (do not pass old PhiStateLM-v1 checkpoint)')
    cfg = MoEConfig(**data['config'])
    model = PhiStateMoELM(cfg)
    model.load_state_dict(data['state_dict'], strict=True)
    model.eval()
    return model, data


def overview(model, manifest: dict | None = None):
    total = count_parameters(model)
    active = model.estimate_active_parameters()
    print(json.dumps({
        'parameters_total': total, 'parameters_activated_per_token_estimate': active,
        'active_parameter_fraction': round(active / total, 3),
        'trainable_weights_fp32_mib': round(total * 4 / 2**20, 1),
        'adamw_weights_grads_optimizer_minimum_mib': round(total * 16 / 2**20, 1),
        'note': ('Estimated parameter memory only; attention activations, OS/PyTorch runtime, '
                 'tokenizer/corpus and expert dispatch also consume memory. Sparse forward does not '
                 'remove the need to store every expert and optimizer state.'),
        'config': vars(model.cfg),
        'corpus': manifest.get('data_fingerprint') if manifest else None,
    }, indent=2), flush=True)


def train(a):
    setup(a.threads, a.seed)
    device = resolve_device(getattr(a, 'device', 'cpu'))
    print('TRAIN DEVICE', str(device),
          torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU', flush=True)
    manifest, datasets, path = load_data(a.data_dir)
    cfg = model_configuration(a, manifest['vocab'])
    cfg.validate()
    if not 0 <= a.code_share <= 1:
        raise ValueError('--code-share must lie in [0,1]')
    for split in ('train', 'val'):
        for domain in ('code', 'general'):
            if len(datasets[f'{split}_{domain}']) <= a.context + 1:
                raise ValueError(f'{split}_{domain} is shorter than context {a.context}')
    if sum(manifest['bins'][f'train_{domain}']['tokens'] for domain in ('code','general')) < 1_000_000:
        print('WARNING: Less than 1M training tokens! This can test the training loop, '
              'but it cannot teach a 35M/100M model useful general coding and knowledge.', flush=True)
    model = PhiStateMoELM(cfg)
    overview(model, manifest)
    if a.dry_run:
        print('DRY-RUN COMPLETE: architecture built, no training executed.', flush=True)
        return

    model = model.to(device)
    opt = optimizer_for(a, model)
    start_step = 0
    best = float('inf')
    rng = np.random.default_rng(a.seed + 425)
    choose = random.Random(a.seed + 101)
    previously_processed = 0
    previous_training_seconds = 0.0
    if a.resume:
        previous_model, previous = ckpt_load(a.resume)
        if vars(previous_model.cfg) != vars(cfg):
            raise RuntimeError('Checkpoint config differs; refusing resume')
        if previous['data_fingerprint'] != manifest['data_fingerprint']:
            raise RuntimeError('Checkpoint corpus fingerprint differs; refusing resume')
        model.load_state_dict(previous_model.state_dict())
        start_step = int(previous['step'])
        best = float(previous.get('best_validation_bpt', best))
        previously_processed = int(previous.get('trained_tokens', 0))
        previous_training_seconds = float(previous.get('train_seconds', 0.0))
        if 'optimizer_state' in previous:
            opt.load_state_dict(previous['optimizer_state'])
        else:
            print('WARNING: optimizer state not saved in this checkpoint; resumed weights '
                  'with a FRESH optimizer (not an exact continuation).', flush=True)
        if 'numpy_rng_state' in previous:
            rng.bit_generator.state = previous['numpy_rng_state']
            choose.setstate(previous['python_rng_state'])
            torch.set_rng_state(previous['torch_rng_state'])
            if device.type == 'cuda' and 'cuda_rng_state_all' in previous:
                torch.cuda.set_rng_state_all(previous['cuda_rng_state_all'])
        else:
            print('WARNING: RNG state not stored; sampling will restart.', flush=True)
    out = Path(a.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics_file = out.with_suffix('.metrics.jsonl')
    if not a.resume:
        metrics_file.unlink(missing_ok=True)
    last_checkpoint = out.with_name(out.stem + '.last' + out.suffix)
    processed = previously_processed
    start_time = time.perf_counter()
    total_training = previous_training_seconds
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    router = None
    written = False
    last_step = start_step
    for step in range(start_step + 1, a.steps + 1):
        step_start = time.perf_counter()
        model.train()
        opt.zero_grad(set_to_none=True)
        lm_value = 0.0
        penalty_value = 0.0
        for _ in range(a.accum):
            source = 'code' if choose.random() < a.code_share else 'general'
            x, y = sample_batch(datasets[f'train_{source}'], a.batch, a.context, rng)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            lm = F.cross_entropy(logits.reshape(-1, cfg.vocab), y.reshape(-1))
            penalty, router = model.router_penalty()
            loss = (lm + penalty) / a.accum
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f'Nonfinite loss at step {step}')
            loss.backward()
            lm_value += float(lm.detach().item()) / a.accum
            penalty_value += float(penalty.detach().item()) / a.accum
            processed += y.numel()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        total_training += time.perf_counter() - step_start
        last_step = step
        if step == 1 or step % a.eval_every == 0 or step == a.steps or (
            a.max_seconds > 0 and time.perf_counter() - start_time > a.max_seconds):
            val = evaluate_both(model, datasets, a.context, a.eval_batch, a.eval_batches)
            current = val['val_mean_bpt']
            improved = current < best
            best = min(best, current)
            rec = {'step': step, 'train_language_bpt': round(lm_value/LN2, 5),
                   'router_penalty': round(penalty_value, 5),
                   'clip_grad_norm': round(float(grad_norm), 4),
                   **val, 'best_validation_bpt': round(best, 5),
                   'trained_tokens': processed,
                   'train_seconds': round(total_training, 3),
                   'wall_seconds': round(time.perf_counter() - start_time, 3),
                   'tokens_per_train_second': round(processed / max(total_training, 1e-9), 1),
                   'session_tokens_per_second': round((processed - previously_processed) /
                                                      max(total_training - previous_training_seconds, 1e-9), 1),
                   'session_train_seconds': round(total_training - previous_training_seconds, 3),
                   'device': str(device),
                   'peak_cuda_allocated_mib': round(torch.cuda.max_memory_allocated(device)/2**20, 1)
                                              if device.type == 'cuda' else None,
                   'peak_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 1),
                   'last_expert_routing': router['expert_selection'] if router else []}
            with metrics_file.open('a', encoding='utf-8') as f:
                f.write(json.dumps(rec) + '\n')
            print('RESULT '+json.dumps(rec), flush=True)
            payload = {'format': 'PhiStateMoE-v1', 'config': vars(cfg),
                       'state_dict': model.state_dict(), 'step': step,
                       'best_validation_bpt': best,
                       'data_fingerprint': manifest['data_fingerprint'],
                       'tokenizer': manifest['tokenizer'],
                       'train_args': vars(a), 'run_seed': a.seed,
                       'trained_tokens': processed,
                       'train_seconds': total_training,
                       'numpy_rng_state': rng.bit_generator.state,
                       'python_rng_state': choose.getstate(),
                       'torch_rng_state': torch.get_rng_state()}
            if device.type == 'cuda':
                payload['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
            if improved:
                # Best is inference-only and stays lightweight (no optimizer state).
                atomic_checkpoint(out, payload)
                written = True
                print('SAVED best', out, flush=True)
            if a.save_optimizer:
                payload['optimizer_state'] = opt.state_dict()
            # Last is a resumable state at the most recent validation point.
            atomic_checkpoint(last_checkpoint, payload)
            print('SAVED latest', last_checkpoint, flush=True)
        if a.max_seconds > 0 and time.perf_counter() - start_time >= a.max_seconds:
            print('STOP: reached --max-seconds; last and best checkpoints preserved.', flush=True)
            break
    print('FINISHED', json.dumps({'steps_done': last_step, 'best_val_bpt': best,
                                  'train_seconds': round(total_training, 2),
                                  'wall_seconds': round(time.perf_counter()-start_time, 2),
                                  'checkpoint': str(out)}), flush=True)


def evaluate(a):
    setup(a.threads, a.seed)
    model, data = ckpt_load(a.checkpoint)
    manifest, datasets, path = load_data(a.data_dir)
    if manifest['data_fingerprint'] != data['data_fingerprint']:
        raise ValueError('Dataset fingerprint mismatch; refusing to compare checkpoints on changed data')
    if a.split == 'test' and not a.final_test:
        raise ValueError('Sealed test split: pass --final-test only after model/hyperparameters are frozen')
    split = a.split
    out = {}
    for domain in ('code','general'):
        measure = evaluate_domain(model, datasets[f'{split}_{domain}'], a.context,
                                  a.eval_batch, a.eval_batches)
        out[domain] = {'bits_per_token': round(measure['bits_per_token'], 6),
                       'nats_per_token': round(measure['nats_per_token'], 6),
                       'evaluated_tokens': measure['tokens']}
    print(json.dumps({'split':split,'data_fingerprint':manifest['data_fingerprint'],
                      'model':vars(model.cfg), 'measure':out},indent=2),flush=True)


@torch.inference_mode()
def generate(a):
    setup(a.threads, a.seed)
    model, checkpoint = ckpt_load(a.checkpoint)
    corpus_root = Path(a.data_dir).expanduser().resolve()
    manifest = json.loads((corpus_root/'manifest.json').read_text())
    if checkpoint['data_fingerprint'] != manifest['data_fingerprint']:
        raise RuntimeError('Tokenizer/corpus mismatch')
    if manifest['tokenizer'] == 'bpe':
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=str(corpus_root / 'tokenizer.model'))
        prompt_ids = sp.encode(a.prompt, out_type=int)
    else:
        sp = None
        prompt_ids = list(a.prompt.encode('utf-8'))
    if not prompt_ids:
        raise ValueError('Prompt must not be empty')
    state = None
    for tid in prompt_ids:
        last, state = model.step(torch.tensor([[tid]],dtype=torch.long), state)
    generated = []
    rng = torch.Generator().manual_seed(a.seed)
    t0 = time.perf_counter()
    for _ in range(a.new_tokens):
        logit = last[0, -1] / max(a.temperature, 0.05)
        if a.greedy:
            next_id = int(torch.argmax(logit))
        else:
            values, choices = torch.topk(logit, min(a.top_k_sample, len(logit)))
            pr = torch.softmax(values,dim=-1)
            next_id = int(choices[torch.multinomial(pr,1,generator=rng)])
        generated.append(next_id)
        last, state = model.step(torch.tensor([[next_id]],dtype=torch.long), state)
    if sp is not None:
        text = sp.decode(prompt_ids + generated)
    else:
        text = bytes(prompt_ids+generated).decode('utf-8',errors='replace')
    print('GENERATED',len(generated),'tokens in',round(time.perf_counter()-t0,2),'seconds')
    print(text)


def smoke(a):
    setup(a.threads, a.seed)
    cfg = config_for('mini', vocab=256, mode='dft')
    model = PhiStateMoELM(cfg)
    a_tensor = torch.randint(0, cfg.vocab, (2, 11))
    logits = model(a_tensor)
    penalty, report = model.router_penalty()
    lm = F.cross_entropy(logits.reshape(-1,cfg.vocab),torch.randint(0,cfg.vocab,(22,)))
    total_loss=lm+penalty
    total_loss.backward()
    assert math.isfinite(float(total_loss.detach()))
    assert model.blocks[0].ff.router.weight.grad is not None, 'router gradient missing'
    assert torch.isfinite(model.blocks[0].ff.router.weight.grad).all(), 'router gradient nonfinite'
    model.eval()
    with torch.inference_mode():
        batch = a_tensor[:1]
        full = model(batch)
        cache = None
        stream = []
        for t in range(batch.shape[1]):
            single, cache = model.step(batch[:,t:t+1],cache)
            stream.append(single)
        err=(full-torch.cat(stream,dim=1)).abs().max().item()
        assert err < 3e-5, f'MoE full/step mismatch: {err}'
        altered=batch.clone()
        altered[:,6:]=torch.randint(0,cfg.vocab,(1,5))
        causal=(full[:,:6]-model(altered)[:,:6]).abs().max().item()
        assert causal < 3e-5, f'future leakage {causal}'
    print('SMOKE PASS',json.dumps({'total_params': count_parameters(model),
                                   'active_params': model.estimate_active_parameters(),
                                   'train_loss': round(float(lm.detach()),4),
                                   'expert_selection':report['expert_selection'],
                                   'stream_max_abs_err':err,'causal_max_abs_err':causal}),flush=True)


def add_model_args(p):
    p.add_argument('--profile', choices=['mini','35m','100m'],default='35m')
    p.add_argument('--mode',choices=['dft','phi','local'],default='dft')
    p.add_argument('--mix-basis',choices=['dft','phi','auto'],default=None)
    for name in ('width','layers','heads','window','experts','top-k','expert-ratio','shared-ratio'):
        p.add_argument('--'+name, type=int, default=None)
    p.add_argument('--router-aux-weight',type=float,default=None)
    p.add_argument('--router-z-weight',type=float,default=None)


def main():
    p = argparse.ArgumentParser(description='PhiState-MoE: 35M/100M CPU hybrid model, training and evaluation')
    sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('smoke');q.add_argument('--threads',type=int,default=4);q.add_argument('--seed',type=int,default=20261027)
    q=sub.add_parser('train');add_model_args(q)
    q.add_argument('--data-dir',required=True)
    q.add_argument('--out',required=True)
    q.add_argument('--steps',type=int,default=100)
    q.add_argument('--context',type=int,default=128)
    q.add_argument('--batch',type=int,default=1)
    q.add_argument('--accum',type=int,default=4)
    q.add_argument('--code-share',type=float,default=0.55)
    q.add_argument('--eval-batch',type=int,default=1)
    q.add_argument('--eval-batches',type=int,default=3)
    q.add_argument('--eval-every',type=int,default=20)
    q.add_argument('--lr',type=float,default=3e-4)
    q.add_argument('--clip',type=float,default=1.0)
    q.add_argument('--weight-decay',type=float,default=0.05)
    q.add_argument('--optimizer',choices=['adamw','adafactor'],default='adafactor')
    q.add_argument('--threads',type=int,default=4)
    q.add_argument('--device', choices=['cpu','cuda','auto'], default='cpu',
                   help='PyTorch compute device; CPU by default for backwards compatibility')
    q.add_argument('--seed',type=int,default=20261027)
    q.add_argument('--max-seconds',type=float,default=0.0)
    q.add_argument('--dry-run',action='store_true')
    q.add_argument('--save-optimizer',action='store_true')
    q.add_argument('--resume',default=None)
    q=sub.add_parser('evaluate');q.add_argument('--checkpoint',required=True);q.add_argument('--data-dir',required=True)
    q.add_argument('--split',choices=['val','test'],default='val');q.add_argument('--final-test',action='store_true')
    q.add_argument('--context',type=int,default=128);q.add_argument('--eval-batch',type=int,default=1)
    q.add_argument('--eval-batches',type=int,default=20);q.add_argument('--threads',type=int,default=4)
    q.add_argument('--seed',type=int,default=20261027)
    q=sub.add_parser('generate');q.add_argument('--checkpoint',required=True);q.add_argument('--data-dir',required=True)
    q.add_argument('--prompt',required=True);q.add_argument('--new-tokens',type=int,default=64)
    q.add_argument('--temperature',type=float,default=0.8);q.add_argument('--top-k-sample',type=int,default=32)
    q.add_argument('--greedy',action='store_true');q.add_argument('--threads',type=int,default=4)
    q.add_argument('--seed',type=int,default=20261027)
    a=p.parse_args()
    if a.action == 'smoke': smoke(a)
    elif a.action == 'train':
        if a.steps <= 0 or a.context < 8 or a.batch <= 0 or a.accum <= 0 or a.eval_every <= 0:
            p.error('invalid training steps/context/batch/accum/eval-every')
        train(a)
    elif a.action == 'evaluate':evaluate(a)
    elif a.action == 'generate':generate(a)


if __name__=='__main__':main()
