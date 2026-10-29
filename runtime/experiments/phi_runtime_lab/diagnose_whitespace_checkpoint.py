"""Read-only checkpoint diagnostics; writes only a new JSON report."""
import argparse
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F

import moe_train
from continue_whitespace_repair import run_lock
from repair_whitespace_corpus import require, sha


def whitespace_piece(piece):
    if re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", piece):
        value = int(piece[3:5], 16)
        return value in (9, 10, 11, 12, 13, 32)
    return piece.replace("▁", " ").isspace()


@torch.inference_mode()
def parity(model, ids):
    device = next(model.parameters()).device
    x = torch.tensor([ids], device=device)
    full = model(x).float()
    cached, state = [], None
    for token in x.unbind(1):
        logits, state = model.step(token[:, None], state)
        cached.append(logits.float())
    cached = torch.cat(cached, dim=1)
    require(torch.isfinite(full).all() and torch.isfinite(cached).all(), 'Nonfinite logits')
    error = (full - cached).abs()
    return {'tokens': len(ids), 'max_abs_logit_error': error.max().item(),
            'mean_abs_logit_error': error.mean().item(),
            'top1_agreement': (full.argmax(-1) == cached.argmax(-1)).float().mean().item(),
            'allclose_atol_1e_3_rtol_1e_4': torch.allclose(full, cached, atol=1e-3, rtol=1e-4)}


@torch.inference_mode()
def evaluate(model, data, whitespace, context, windows):
    device = next(model.parameters()).device
    require(len(data) > context, 'Validation stream too short')
    starts = np.unique(np.linspace(0, len(data)-context-1, windows, dtype=np.int64))
    sums = {name: [0, 0.0, 0] for name in ('all', 'whitespace_only', 'other')}
    for i, start in enumerate(starts):
        slab = torch.tensor(np.array(data[start:start+context+1], dtype=np.int64), device=device)
        y = slab[1:]
        logits = model(slab[:-1][None])[0].float()
        loss = F.cross_entropy(logits, y, reduction='none') / math.log(2)
        require(torch.isfinite(loss).all(), 'Nonfinite validation loss')
        correct = logits.argmax(-1) == y
        ws = whitespace[y]
        for name, mask in [('all', torch.ones_like(ws)), ('whitespace_only', ws), ('other', ~ws)]:
            sums[name][0] += int(mask.sum())
            sums[name][1] += float(loss[mask].sum())
            sums[name][2] += int(correct[mask].sum())
        if (i+1) % 64 == 0:
            print(f'  validation windows {i+1}/{len(starts)}', flush=True)
    return {k: {'tokens': n, 'bpt': loss/n if n else None,
                'top1_accuracy': correct/n if n else None} for k, (n, loss, correct) in sums.items()}


@torch.inference_mode()
def probe(model, tokenizer, domain, prompt, new_tokens=32):
    device = next(model.parameters()).device
    ids = tokenizer.encode(f'<|{domain}|>\n' + prompt)
    initial = model(torch.tensor([ids], device=device))[0, -1].float().softmax(-1)
    values, choices = initial.topk(8)
    alternatives = [{'piece': tokenizer.id_to_piece(int(t)), 'probability': float(p)}
                    for p, t in zip(values, choices)]
    # Full forward on the entire growing prefix, bypassing the recurrent cache.
    produced = []
    for _ in range(new_tokens):
        token = int(model(torch.tensor([ids + produced], device=device))[0, -1].argmax())
        if token == tokenizer.eos_id():
            break
        produced.append(token)
    return {'prompt': prompt, 'next_token_top8': alternatives,
            'full_forward_greedy_text': tokenizer.decode(ids + produced),
            'generated_ids': produced, 'parity_on_generated_sequence': parity(model, ids + produced)}


def run(args):
    checkpoint, data = Path(args.checkpoint), Path(args.data_dir)
    with run_lock(checkpoint.parent):
        fingerprint = sha(checkpoint)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        require(state['step'] == args.expected_step, f"Expected step {args.expected_step}, found {state['step']}")
        manifest = json.loads((data / 'manifest.json').read_text())
        require(state['data_fingerprint'] == manifest['data_fingerprint'], 'Checkpoint/corpus mismatch')
        require(sha(data / 'tokenizer.model') == state['corpus_migration']['tokenizer_sha256'], 'Tokenizer mismatch')
        device = moe_train.resolve_device(args.device)
        moe_train.setup(2, 20261027)
        model = moe_train.PhiStateMoELM(moe_train.MoEConfig(**state['config']))
        model.load_state_dict(state['state_dict'], strict=True)
        model.to(device).eval()
        context = state['train_args']['context']
        del state
        tokenizer = spm.SentencePieceProcessor(model_file=str(data / 'tokenizer.model'))
        whitespace = torch.tensor([whitespace_piece(tokenizer.id_to_piece(i))
                                   for i in range(tokenizer.vocab_size())], device=device)
        report = {'step': args.expected_step, 'checkpoint_sha256': fingerprint,
                  'device': str(device), 'context': context,
                  'scope': 'Validation only; whitespace-only vocabulary pieces, not character-level loss',
                  'validation': {}, 'parity': {}, 'probes': []}
        for domain in ('code', 'general'):
            print(f'CHECKING {domain}: validation and cached/full inference', flush=True)
            spec = manifest['bins'][f'val_{domain}']
            file = data / spec['file']
            require(sha(file) == spec['sha256'], f'Validation hash mismatch: {domain}')
            stream = np.memmap(file, dtype=manifest['dtype'], mode='r')
            report['validation'][domain] = evaluate(model, stream, whitespace, context, args.windows)
            report['parity'][domain] = parity(model, stream[:context].astype(int).tolist())
            print(json.dumps({domain: report['validation'][domain], 'parity': report['parity'][domain]}), flush=True)
        for domain, prompt in [('code', 'def add(a, b):\n'), ('code', 'def add(a, b):\n    return'),
                               ('general', 'The purpose of an operating system is')]:
            result = probe(model, tokenizer, domain, prompt)
            report['probes'].append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        require(sha(checkpoint) == fingerprint, 'Checkpoint changed during inspection')
        report['checkpoint_unchanged'] = True
        output = checkpoint.parent / f'diagnostic_{args.expected_step}_{time.time_ns()}.json'
        with output.open('x') as out:
            json.dump(report, out, indent=2, ensure_ascii=False, allow_nan=False)
        print(f'DIAGNOSTIC COMPLETE. Checkpoint unchanged. Report: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--expected-step', type=int, default=61000)
    parser.add_argument('--windows', type=int, default=256)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto')
    args = parser.parse_args()
    if args.windows < 1:
        parser.error('--windows must be positive')
    run(args)
