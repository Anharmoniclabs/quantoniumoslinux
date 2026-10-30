#!/usr/bin/env python3
"""Bounded matched-batch optimizer comparison; never overwrite the source run."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time
from types import SimpleNamespace

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F

import moe_train


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def whitespace_piece(piece):
    if re.fullmatch(r'<0x[0-9A-Fa-f]{2}>', piece):
        return int(piece[3:5], 16) in (9, 10, 11, 12, 13, 32)
    return piece.replace('▁', ' ').isspace()


def write_json(path, value):
    pending = path.with_suffix('.partial')
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    pending.replace(path)


@torch.inference_mode()
def evaluate(model, datasets, mask, context, windows):
    model.eval()
    device = next(model.parameters()).device
    report = {}
    for domain in ('code', 'general'):
        data = datasets['val_' + domain]
        sums = {key: [0, 0.] for key in ('all', 'whitespace', 'other')}
        starts = np.unique(np.linspace(0, len(data) - context - 1, windows, dtype=np.int64))
        for start in starts:
            slab = torch.tensor(np.array(data[start:start + context + 1], dtype=np.int64), device=device)
            y = slab[1:]
            logits = model(slab[:-1][None])[0]
            losses = F.cross_entropy(logits, y, reduction='none') / math.log(2)
            if not torch.isfinite(losses).all():
                raise RuntimeError('Nonfinite validation')
            ws = mask[y]
            for key, selected in [('all', torch.ones_like(ws)), ('whitespace', ws), ('other', ~ws)]:
                sums[key][0] += int(selected.sum())
                sums[key][1] += float(losses[selected].sum())
        report[domain] = {k: {'tokens': n, 'bpt': loss / n if n else None}
                          for k, (n, loss) in sums.items()}
    return report


@torch.inference_mode()
def generate(model, tokenizer, count):
    model.eval()
    device = next(model.parameters()).device
    rows = []
    prompts = [('code', 'def add(a, b):\n'), ('code', 'def fibonacci(n):\n'),
               ('general', 'Quantum physics is'),
               ('general', 'User: What is Python code?\nAssistant:')]
    for domain, prompt in prompts:
        prefix = f'<|{domain}|>\n'
        ids = tokenizer.encode(prefix + prompt)
        cache = None
        for token in ids:
            logits, cache = model.step(torch.tensor([[token]], device=device), cache)
        produced = []
        for _ in range(count):
            if not torch.isfinite(logits).all():
                raise RuntimeError('Nonfinite generation')
            token = int(logits[0, -1].argmax())
            if token == tokenizer.eos_id():
                break
            produced.append(token)
            logits, cache = model.step(torch.tensor([[token]], device=device), cache)
        text = tokenizer.decode(ids + produced)
        if text.startswith(prefix):
            text = text[len(prefix):]
        rows.append({'prompt': prompt, 'text': text, 'generated_tokens': len(produced),
                     'continuation_repr': repr(tokenizer.decode(produced))})
    return rows


def run(args):
    if not 1 <= args.steps <= 1000 or not 1 <= args.windows <= 512:
        raise ValueError('Use 1..1000 steps and 1..512 validation windows')
    device = moe_train.resolve_device(args.device)
    moe_train.setup(2, 20261030)
    source = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError('Output exists; choose a fresh experiment directory')
    source_hash = sha(source)
    state = torch.load(source, map_location='cpu', weights_only=True)
    if sha(source) != source_hash:
        raise RuntimeError('Source checkpoint changed during loading')
    manifest, datasets, data = moe_train.load_data(args.data_dir)
    if state.get('format') != 'PhiStateMoE-v1' or state.get('data_fingerprint') != manifest['data_fingerprint']:
        raise RuntimeError('Checkpoint/corpus mismatch')
    if state['step'] != 65000 or state['train_args']['optimizer'] != 'adafactor':
        raise RuntimeError('Expected the completed 65K Adafactor checkpoint')
    if 'optimizer_state' not in state:
        raise RuntimeError('Use latest.pt, which includes optimizer state')
    if sha(data / 'tokenizer.model') != manifest['whitespace_repair']['tokenizer_sha256']:
        raise RuntimeError('Tokenizer hash mismatch')
    for key, spec in manifest['bins'].items():
        if key.startswith(('train_', 'val_')) and sha(data / spec['file']) != spec['sha256']:
            raise RuntimeError('Corpus hash mismatch: ' + key)
    tokenizer = spm.SentencePieceProcessor(model_file=str(data / 'tokenizer.model'))
    if tokenizer.vocab_size() != state['config']['vocab']:
        raise RuntimeError('Vocabulary mismatch')
    model = moe_train.PhiStateMoELM(moe_train.MoEConfig(**state['config'])).to(device)
    model.load_state_dict(state['state_dict'], strict=True)
    mask = torch.tensor([whitespace_piece(tokenizer.id_to_piece(i))
                         for i in range(tokenizer.vocab_size())], device=device)
    training = dict(state['train_args'])
    context, batch, accum = (int(training[k]) for k in ('context', 'batch', 'accum'))
    output.mkdir(parents=True)
    report = {'status': 'running', 'backend': str(device), 'source_sha256': source_hash,
              'source_step': state['step'], 'steps_per_arm': args.steps,
              'tokens_per_arm': args.steps * context * batch * accum,
              'scope': 'PyTorch optimizer pilot; no C++ training; no automatic promotion; validation only',
              'whitespace_definition': 'Whitespace-only tokens including byte fallback; other includes special tokens',
              'arms': {}}
    report_path = output / 'report.json'
    write_json(report_path, report)
    try:
        report['baseline'] = evaluate(model, datasets, mask, context, args.windows)
        report['baseline_generation'] = generate(model, tokenizer, 64)
        print('BASELINE', json.dumps(report['baseline']), flush=True)
        write_json(report_path, report)
        for arm in ('adafactor_control', 'adamw_candidate'):
            model.load_state_dict(state['state_dict'], strict=True)
            torch.manual_seed(20261030)
            rng = np.random.default_rng(20261030)
            choose = random.Random(20261030)
            settings = dict(training)
            if arm == 'adafactor_control':
                optimizer = moe_train.optimizer_for(SimpleNamespace(**settings), model)
                optimizer.load_state_dict(copy.deepcopy(state['optimizer_state']))
            else:
                settings.update(optimizer='adamw', lr=1e-4)
                optimizer = moe_train.optimizer_for(SimpleNamespace(**settings), model)
            started = time.perf_counter()
            batch_hash = hashlib.sha256()
            history = []
            for local_step in range(1, args.steps + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                if arm == 'adamw_candidate':
                    for group in optimizer.param_groups:
                        group['lr'] = settings['lr'] * min(1., local_step / min(50, args.steps))
                lm_sum = 0.
                for _ in range(accum):
                    domain = 'code' if choose.random() < settings['code_share'] else 'general'
                    x, y = moe_train.sample_batch(datasets['train_' + domain], batch, context, rng)
                    batch_hash.update(domain.encode())
                    batch_hash.update(x.numpy().tobytes())
                    batch_hash.update(y.numpy().tobytes())
                    logits = model(x.to(device))
                    lm = F.cross_entropy(logits.reshape(-1, model.cfg.vocab), y.to(device).reshape(-1))
                    penalty, _ = model.router_penalty()
                    loss = (lm + penalty) / accum
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite training loss')
                    loss.backward()
                    lm_sum += float(lm.detach()) / accum
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['clip'], error_if_nonfinite=True)
                optimizer.step()
                if local_step % 25 == 0 or local_step == args.steps:
                    entry = {'step': local_step, 'batch_bpt': lm_sum / math.log(2), 'gradient_norm': float(grad)}
                    history.append(entry)
                    print(arm, json.dumps(entry), flush=True)
                if local_step % 100 == 0 or local_step == args.steps:
                    payload = {k: v for k, v in state.items() if k not in
                               ('state_dict', 'optimizer_state', 'last_validation', 'best_validation_bpt')}
                    payload.update(state_dict=model.state_dict(), optimizer_state=optimizer.state_dict(),
                                   step=state['step'] + local_step, train_args=settings,
                                   trained_tokens=state.get('trained_tokens', 0) + local_step * context * batch * accum,
                                   train_seconds=state.get('train_seconds', 0) + time.perf_counter() - started,
                                   numpy_rng_state=rng.bit_generator.state, python_rng_state=choose.getstate(),
                                   torch_rng_state=torch.get_rng_state(),
                                   optimizer_trial={'arm': arm, 'source_sha256': source_hash,
                                                    'fresh_optimizer': arm == 'adamw_candidate',
                                                    'warmup_steps': min(50, args.steps) if arm == 'adamw_candidate' else 0})
                    if device.type == 'cuda':
                        payload['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
                    moe_train.atomic_checkpoint(output / (arm + '.pt'), payload)
                    del payload
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            result = {'training_seconds_including_saves': elapsed, 'batches_sha256': batch_hash.hexdigest(),
                      'optimizer': settings['optimizer'], 'target_lr': settings['lr'],
                      'validation': evaluate(model, datasets, mask, context, args.windows),
                      'generation': generate(model, tokenizer, 64), 'training_log': history}
            report['arms'][arm] = result
            write_json(report_path, report)
            print('\nRESULT', arm, json.dumps(result['validation']), flush=True)
            for row in result['generation']:
                print(row['text'], '\nContinuation:', row['continuation_repr'], flush=True)
            del optimizer
        hashes = {r['batches_sha256'] for r in report['arms'].values()}
        if len(hashes) != 1 or sha(source) != source_hash:
            raise RuntimeError('Batch equality or source preservation check failed')
        report.update(status='complete', identical_batches=True, source_unchanged=True)
        write_json(report_path, report)
        print('\nCOMPLETE:', report_path, flush=True)
        print('Compare other-token BPT in BOTH domains and generated text. No candidate was promoted.', flush=True)
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        write_json(report_path, report)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'data-dir', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--windows', type=int, default=64)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    run(p.parse_args())
