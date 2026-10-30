"""Resumable answer-only conversational training with 20% original-corpus replay."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F

import moe_train
from chat_recovery_data import SOURCES, TEMPLATE, encode_example, prompt_text, read_jsonl, sha, verify_prepared

PROBES = ['Hello! What can you help me with?', 'What is quantum physics? Explain simply.',
          'What is the difference between RAM and storage?',
          'Write a Python function add(a, b) that returns their sum.']


def atomic_json(path, value):
    pending = path.with_suffix('.partial')
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    pending.replace(path)


@contextmanager
def run_lock(path):
    lock = Path(tempfile.gettempdir()) / ('phistate_chat_' + hashlib.sha256(str(path.resolve()).encode()).hexdigest() + '.lock')
    with lock.open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This chat run already has an active worker') from None
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load_rows(directory, tokenizer, context):
    splits = {}
    groups, prompts = {}, {}
    for split in ('train', 'val'):
        pools = defaultdict(list)
        for row in read_jsonl(directory / (split + '.jsonl')):
            if row['split'] != split or row['source'] not in SOURCES:
                raise RuntimeError('Bad prepared row')
            x, y = encode_example(row['messages'], tokenizer, context)
            if x != row['input_ids'] or y != row['labels']:
                raise RuntimeError('Prepared labels/tokenizer mismatch')
            pools[row['source']].append(row)
        splits[split] = dict(pools)
        groups[split] = {r['split_group'] for rs in pools.values() for r in rs}
        prompts[split] = {' '.join(r['messages'][-2]['content'].lower().split()) for rs in pools.values() for r in rs}
    if groups['train'] & groups['val'] or prompts['train'] & prompts['val']:
        raise RuntimeError('Train/validation leakage')
    for split, pools in splits.items():
        if set(pools) != set(SOURCES):
            raise RuntimeError('Missing source in ' + split)
    return splits


def tensors(row, device):
    return (torch.tensor([row['input_ids']], dtype=torch.long, device=device),
            torch.tensor([row['labels']], dtype=torch.long, device=device))


@torch.inference_mode()
def evaluate(model, pools, limit):
    model.eval()
    device = next(model.parameters()).device
    result = {}
    for source in SOURCES:
        nll, tokens = 0., 0
        rows = sorted(pools[source], key=lambda r: r['id'])[:limit]
        for row in rows:
            x, y = tensors(row, device)
            loss = F.cross_entropy(model(x).reshape(-1, model.cfg.vocab), y.flatten(),
                                   ignore_index=-100, reduction='sum')
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite held-out answer loss')
            nll += float(loss)
            tokens += int((y != -100).sum())
        result[source] = dict(answer_bpt=nll / tokens / math.log(2), answer_tokens=tokens, examples=len(rows))
    return {'val_answer_bpt': sum(SOURCES[s]['weight'] * result[s]['answer_bpt'] for s in SOURCES),
            'per_source': result}


@torch.inference_mode()
def generate(model, tokenizer, count=96):
    model.eval()
    device = next(model.parameters()).device
    result = []
    for prompt in PROBES:
        ids = tokenizer.encode(prompt_text([{'role': 'user', 'content': prompt}]))
        cache, produced = None, []
        for token in ids:
            logits, cache = model.step(torch.tensor([[token]], device=device), cache)
        ended = False
        for _ in range(count):
            if not torch.isfinite(logits).all():
                raise RuntimeError('Nonfinite generation logits')
            token = int(logits[0, -1].argmax())
            if token == tokenizer.eos_id():
                ended = True
                break
            produced.append(token)
            logits, cache = model.step(torch.tensor([[token]], device=device), cache)
        text = tokenizer.decode(produced)
        result.append(dict(prompt=prompt, answer=text, answer_repr=repr(text), eos=ended, tokens=len(produced)))
    return result


def run(a):
    run_dir = Path(a.run_dir).resolve()
    with run_lock(run_dir):
        return train(a, run_dir)


def train(a, run_dir):
    if a.steps < 1 or a.accum < 1 or a.context < 128 or a.eval_every < 1 or a.save_every < 1:
        raise ValueError('Invalid run settings')
    device = moe_train.resolve_device(a.device)
    moe_train.setup(2, 20261030)
    checkpoint, sft_dir, data_dir = map(Path, (a.checkpoint, a.sft_dir, a.data_dir))
    if run_dir == checkpoint.resolve().parent or run_dir in (sft_dir.resolve(), data_dir.resolve()):
        raise RuntimeError('Chat run needs a separate output directory')
    source_hash = sha(checkpoint)
    base = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if sha(checkpoint) != source_hash:
        raise RuntimeError('Source changed while loading')
    corpus, replay, _ = moe_train.load_data(str(data_dir))
    token_hash = sha(data_dir / 'tokenizer.model')
    prepared = verify_prepared(sft_dir, token_hash, a.context)
    if (base.get('format') != 'PhiStateMoE-v1' or
        base.get('data_fingerprint') != corpus['data_fingerprint'] or
        prepared['base_data_fingerprint'] != corpus['data_fingerprint'] or
        corpus.get('whitespace_repair', {}).get('tokenizer_sha256') != token_hash):
        raise RuntimeError('Checkpoint, repaired corpus and conversation data do not match')
    if base.get('training_stage') == 'chat_recovery_v1':
        raise RuntimeError('Keep --checkpoint pointing at the original pretraining checkpoint; resume is automatic')
    for key, spec in corpus['bins'].items():
        if key.startswith(('train_', 'val_')) and sha(data_dir / spec['file']) != spec['sha256']:
            raise RuntimeError('Replay corpus changed: ' + key)
    tokenizer = spm.SentencePieceProcessor(model_file=str(data_dir / 'tokenizer.model'))
    if base['config']['vocab'] != tokenizer.vocab_size():
        raise RuntimeError('Tokenizer vocabulary mismatch')
    pools = load_rows(sft_dir, tokenizer, a.context)
    model = moe_train.PhiStateMoELM(moe_train.MoEConfig(**base['config'])).to(device)
    model.load_state_dict(base['state_dict'], strict=True)
    parent_step, base_config = int(base['step']), base['config']
    base_processed = int(base.get('trained_tokens', 0))
    if base.get('train_args', {}).get('optimizer') != 'adafactor' or 'optimizer_state' not in base:
        raise RuntimeError('Use original latest.pt with its saved Adafactor state; the AdamW trial regressed')
    train_args = dict(base['train_args'])
    optimizer = moe_train.optimizer_for(SimpleNamespace(**train_args), model)
    optimizer.load_state_dict(base['optimizer_state'])
    source_lr = optimizer.param_groups[0]['lr']
    del base
    settings = dict(context=a.context, accum=a.accum, lr=source_lr, optimizer='adafactor_preserved',
                    replay_share=.20, replay_context=128, replay_code_share=.55,
                    retention_max_mean_bpt_increase=.5,
                    val_examples_per_source=a.val_examples, seed=20261030, template=TEMPLATE,
                    eval_every=a.eval_every, save_every=a.save_every, patience=a.patience)
    rng = random.Random(20261030)
    numpy_rng = np.random.default_rng(20261030)
    orders = {s: list(range(len(pools['train'][s]))) for s in SOURCES}
    positions = {s: len(orders[s]) for s in SOURCES}
    step, processed, answer_tokens, best, bad_checks = 0, 0, 0, float('inf'), 0
    latest = run_dir / 'latest.pt'
    best_path = run_dir / 'best.pt'
    if latest.exists():
        state = torch.load(latest, map_location='cpu', weights_only=True)
        if (state.get('training_stage') != 'chat_recovery_v1' or state['source_sha256'] != source_hash
            or state['sft_data_fingerprint'] != prepared['data_fingerprint'] or state['chat_settings'] != settings
            or state['config'] != base_config):
            raise RuntimeError('Saved chat run settings/source/data differ')
        model.load_state_dict(state['state_dict'], strict=True)
        optimizer.load_state_dict(state['optimizer_state'])
        step, processed, answer_tokens = state['sft_steps'], state['phase_tokens'], state['answer_tokens']
        best, bad_checks = state['best_validation_bpt'], state['bad_checks']
        rng.setstate(state['python_rng_state'])
        numpy_rng.bit_generator.state = state['numpy_rng_state']
        orders, positions = state['orders'], state['positions']
        torch.set_rng_state(state['torch_rng_state'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng_state_all']])
        last_validation = state['last_validation']
        replay_baseline = state['replay_baseline']
        del state
        print('RESUMED saved SFT step', step, flush=True)
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError('Partial directory without latest.pt; preserve it and choose a fresh run-dir')
        run_dir.mkdir(parents=True, exist_ok=True)
        last_validation = evaluate(model, pools['val'], a.val_examples)
        replay_baseline = moe_train.evaluate_both(model, replay, 128, 1, 16)
        best = last_validation['val_answer_bpt']
    stop_file = run_dir / 'STOP'
    if stop_file.exists():
        if a.clear_stop:
            stop_file.unlink()
        else:
            print('STOP file present. Rerun with --clear-stop to resume intentionally.', flush=True)
            return 'stopped'
    stop = [False]
    def signal_stop(signum, frame):
        stop[0] = True
    previous = {s: signal.signal(s, signal_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    started = time.perf_counter()

    def save(status, improved=False):
        record = dict(format='PhiStateMoE-v1', training_stage='chat_recovery_v1',
                      config=vars(model.cfg), state_dict=model.state_dict(),
                      step=parent_step + step, pretrain_step=parent_step, sft_steps=step,
                      trained_tokens=base_processed + processed, phase_tokens=processed, answer_tokens=answer_tokens,
                      data_fingerprint=corpus['data_fingerprint'], tokenizer='bpe',
                      tokenizer_sha256=token_hash, sft_data_fingerprint=prepared['data_fingerprint'],
                      source_sha256=source_hash, source_checkpoint=str(checkpoint.resolve()),
                      train_args={**train_args, 'context': a.context, 'batch': 1, 'accum': a.accum,
                                  'sft_dir': str(sft_dir.resolve()), 'run_dir': str(run_dir)},
                      chat_settings=settings, chat_template=TEMPLATE, best_validation_bpt=best,
                      best_validation_metric='weighted held-out assistant-token BPT', last_validation=last_validation,
                      replay_baseline=replay_baseline,
                      bad_checks=bad_checks, python_rng_state=rng.getstate(), numpy_rng_state=numpy_rng.bit_generator.state,
                      torch_rng_state=torch.get_rng_state(), orders=orders, positions=positions)
        if device.type == 'cuda':
            record['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
        record['optimizer_state'] = optimizer.state_dict()
        moe_train.atomic_checkpoint(latest, record)
        checked = torch.load(latest, map_location='cpu', weights_only=True)
        if checked['sft_steps'] != step or checked['sft_data_fingerprint'] != prepared['data_fingerprint']:
            raise RuntimeError('Checkpoint readback failed')
        del checked
        if improved:
            record.pop('optimizer_state')
            moe_train.atomic_checkpoint(best_path, record)
        atomic_json(run_dir / 'progress.json', dict(status=status, saved_sft_step=step,
                    source_step=parent_step, target_sft_steps=a.steps, latest=str(latest), best=str(best_path),
                    last_validation=last_validation, best_val_answer_bpt=best,
                    phase_tokens=processed, answer_tokens=answer_tokens,
                    last_validation_sft_step=last_validation.get('sft_step', 0),
                    elapsed_session_seconds=time.perf_counter() - started))
        print('SAVED', status, 'SFT', step, '/', a.steps, flush=True)

    try:
        if not latest.exists():
            save('prepared', True)
            baseline = generate(model, tokenizer)
            atomic_json(run_dir / 'generation_00000.json', baseline)
            print('CHAT BASELINE', json.dumps(last_validation), flush=True)
        if step >= a.steps:
            print('Already at target. Increase --steps to extend this same run.', flush=True)
            return 'complete'
        print('TRAINING: PyTorch', device, '| preserved Adafactor | assistant-only targets | 20% replay', flush=True)
        status = 'training'
        for index in range(step + 1, a.steps + 1):
            if stop[0] or stop_file.exists():
                status = 'stopped'
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            lm_total = 0.
            for _ in range(a.accum):
                if rng.random() < settings['replay_share']:
                    domain = 'code' if rng.random() < settings['replay_code_share'] else 'general'
                    x, y = moe_train.sample_batch(replay['train_' + domain], 1, settings['replay_context'], numpy_rng)
                    x, y = x.to(device), y.to(device)
                else:
                    source = rng.choices(list(SOURCES), weights=[s['weight'] for s in SOURCES.values()], k=1)[0]
                    if positions[source] == len(orders[source]):
                        rng.shuffle(orders[source])
                        positions[source] = 0
                    row = pools['train'][source][orders[source][positions[source]]]
                    positions[source] += 1
                    x, y = tensors(row, device)
                    answer_tokens += int((y != -100).sum())
                logits = model(x)
                lm = F.cross_entropy(logits.reshape(-1, model.cfg.vocab), y.flatten(), ignore_index=-100)
                penalty, _ = model.router_penalty()
                loss = (lm + penalty) / a.accum
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss; last saved checkpoint retained')
                loss.backward()
                lm_total += float(lm.detach()) / a.accum
                processed += x.numel()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            step = index
            if step % 25 == 0:
                print('TRAIN', step, 'mixed batch BPT', round(lm_total / math.log(2), 4), flush=True)
            improved = False
            if step % a.eval_every == 0 or step == a.steps:
                last_validation = evaluate(model, pools['val'], a.val_examples)
                last_validation['sft_step'] = step
                score = last_validation['val_answer_bpt']
                retention = moe_train.evaluate_both(model, replay, 128, 1, 16)
                retained = retention['val_mean_bpt'] <= replay_baseline['val_mean_bpt'] + settings['retention_max_mean_bpt_increase']
                improved = score < best and retained
                bad_checks = 0 if improved else bad_checks + 1
                if improved:
                    best = score
                record = dict(train_mixed_bpt=lm_total / math.log(2),
                              gradient_norm=float(grad), replay_validation=retention, **last_validation)
                with (run_dir / 'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(record) + '\n')
                print('VALIDATION', json.dumps(record), flush=True)
                samples = generate(model, tokenizer)
                atomic_json(run_dir / f'generation_{step:05d}.json', samples)
                for row in samples:
                    print('\nUSER:', row['prompt'], '\nASSISTANT:', row['answer_repr'], flush=True)
                if step >= 500 and a.patience > 0 and bad_checks >= a.patience:
                    status = 'early_stopped'
                if not retained:
                    status = 'retention_stopped'
            if stop[0] or stop_file.exists():
                status = 'stopped'
            elif step == a.steps and status == 'training':
                status = 'complete'
            if step % a.save_every == 0 or step % a.eval_every == 0 or status != 'training':
                save(status, improved)
            if status != 'training':
                break
        progress = json.loads((run_dir / 'progress.json').read_text())
        if progress['saved_sft_step'] != step or progress['status'] != status:
            save(status)
        if sha(checkpoint) != source_hash:
            raise RuntimeError('Original checkpoint changed externally during training')
        print('CHAT RUN', status, '| best:', best_path, flush=True)
        return status
    except BaseException as error:
        # Do not serialize possibly half-applied updates; preserve last atomic save.
        atomic_json(run_dir / 'error.json', dict(error=str(error), last_completed_sft_step=step))
        progress_path = run_dir / 'progress.json'
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            progress.update(status='failed', error=str(error))
            atomic_json(progress_path, progress)
        raise
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'data-dir', 'sft-dir', 'run-dir'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--context', type=int, default=512)
    p.add_argument('--accum', type=int, default=4)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--val-examples', type=int, default=32)
    p.add_argument('--patience', type=int, default=6)
    p.add_argument('--clear-stop', action='store_true')
    run(p.parse_args())
