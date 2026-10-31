"""Bounded, resumable policy comparison arm; no automatic native export.

Starts from an existing SFT checkpoint and keeps its data mixture, optimizer
moments and sampler positions. Candidate selection uses held-out generated-answer
quality plus replay retention. Loss-only best weights have a separate filename.
"""
import argparse
import copy
from contextlib import contextmanager
import json
import math
from pathlib import Path
import random
import signal
from types import SimpleNamespace

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F

import moe_train
from chat_recovery_data import SOURCES, TEMPLATE, sha, verify_prepared
from chat_recovery_train import atomic_json, load_rows, run_lock, tensors, evaluate
from coherence_quality import load_suite, evaluate_suite
from optimizer_policy import protect_structured_parameters

TERMINAL = {'completed_qualified', 'completed_failed_quality',
            'early_stopped_failed_quality', 'early_stopped_qualified', 'retention_stopped'}


@contextmanager
def evaluation_rng(device):
    """Generation must not change the continuation's training random stream."""
    py, np_state = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)


def empty_coverage():
    return {name: {'draws': 0, 'unique_indices': [], 'input_tokens': 0, 'answer_tokens': 0}
            for name in (*SOURCES, 'replay_code', 'replay_general')}


def _score(value):
    score = float(value['macro_score'])
    if not math.isfinite(score) or not 0 <= score <= 1 or not isinstance(value['qualified'], bool):
        raise RuntimeError('Invalid generated-answer quality result')
    return score


def run(a):
    if not 1 <= a.steps <= 2000 or a.eval_every < 1 or a.save_every < 1:
        raise ValueError('Use a fixed additional budget of 1..2000 and positive evaluation/save intervals')
    if a.policy not in ('original', 'protected'):
        raise ValueError('Unknown optimizer policy')
    source, run_dir = Path(a.checkpoint).resolve(), Path(a.run_dir).resolve()
    if run_dir in (source.parent, Path(a.data_dir).resolve(), Path(a.sft_dir).resolve()):
        raise ValueError('Use a separate arm output directory')
    with run_lock(source.parent), run_lock(run_dir):
        return train(a, source, run_dir)


def train(a, source, run_dir):
    moe_train.setup(2, 20261031)
    device = moe_train.resolve_device(a.device)
    source_hash = sha(source)
    parent = torch.load(source, map_location='cpu', weights_only=True)
    if sha(source) != source_hash:
        raise RuntimeError('Source changed while loading')
    if parent.get('training_stage') != 'chat_recovery_v1' or 'optimizer_state' not in parent:
        raise RuntimeError('Source must be a saved conversational SFT checkpoint with optimizer state')
    if parent.get('train_args', {}).get('optimizer') != 'adafactor':
        raise RuntimeError('This comparison preserves the source Adafactor optimizer')
    corpus, replay, data_dir = moe_train.load_data(a.data_dir)
    token_hash = sha(data_dir / 'tokenizer.model')
    chat_settings = parent['chat_settings']
    context, accum = int(chat_settings['context']), int(chat_settings['accum'])
    prepared = verify_prepared(Path(a.sft_dir), token_hash, context)
    if (parent.get('format') != 'PhiStateMoE-v1'
        or parent.get('data_fingerprint') != corpus['data_fingerprint']
        or parent.get('tokenizer_sha256') != token_hash
        or corpus.get('whitespace_repair', {}).get('tokenizer_sha256') != token_hash
        or parent.get('sft_data_fingerprint') != prepared['data_fingerprint']
        or prepared['base_data_fingerprint'] != corpus['data_fingerprint']
        or parent.get('chat_template') != TEMPLATE):
        raise RuntimeError('Source/corpus/tokenizer/SFT lineage mismatch')
    for key, spec in corpus['bins'].items():
        if key.startswith(('train_', 'val_')) and sha(data_dir / spec['file']) != spec['sha256']:
            raise RuntimeError('Replay hash mismatch: ' + key)
    tokenizer = spm.SentencePieceProcessor(model_file=str(data_dir / 'tokenizer.model'))
    if tokenizer.vocab_size() != parent['config']['vocab']:
        raise RuntimeError('Vocabulary mismatch')
    pools = load_rows(Path(a.sft_dir), tokenizer, context)
    suite = load_suite(Path(a.dev_suite))
    if suite.get('split') != 'dev':
        raise ValueError('Training selection requires the development suite; final suite is sealed')
    suite_hash = sha(Path(a.dev_suite))
    settings = dict(policy=a.policy, additional_steps=a.steps, eval_every=a.eval_every,
                    save_every=a.save_every, source_sha256=source_hash,
                    tokenizer_sha256=token_hash, sft_data_fingerprint=prepared['data_fingerprint'],
                    dev_suite_sha256=suite_hash, source_chat_settings=chat_settings,
                    source_weights={s: v['weight'] for s, v in SOURCES.items()},
                    no_gain_evaluations=3, device_type=device.type,
                    runner_sha256=sha(Path(__file__)))
    model = moe_train.PhiStateMoELM(moe_train.MoEConfig(**parent['config'])).to(device)
    optimizer = moe_train.optimizer_for(SimpleNamespace(**parent['train_args']), model)
    latest = run_dir / 'latest.pt'
    resume = latest.is_file()
    if resume:
        saved = torch.load(latest, map_location='cpu', weights_only=True)
        if saved.get('coherence_settings') != settings or saved.get('training_stage') != 'coherence_recovery_v1':
            raise RuntimeError('Saved arm budget/settings/source differ; automatic extension is forbidden')
        model.load_state_dict(saved['state_dict'], strict=True)
        # Converted checkpoints need the converted group layout before loading.
        if a.policy == 'protected':
            protect_structured_parameters(optimizer, model)
        optimizer.load_state_dict(saved['optimizer_state'])
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError('Arm directory is nonempty without a resumable checkpoint')
        model.load_state_dict(parent['state_dict'], strict=True)
        optimizer.load_state_dict(parent['optimizer_state'])
        if a.policy == 'protected':
            protect_structured_parameters(optimizer, model)
        saved = parent
    rng = random.Random()
    rng.setstate(saved['python_rng_state'])
    numpy_rng = np.random.default_rng()
    numpy_rng.bit_generator.state = saved['numpy_rng_state']
    orders, positions = copy.deepcopy(saved['orders']), copy.deepcopy(saved['positions'])
    for source_name in SOURCES:
        if sorted(orders[source_name]) != list(range(len(pools['train'][source_name]))):
            raise RuntimeError('Saved sampling order does not match prepared rows')
        if not 0 <= positions[source_name] <= len(orders[source_name]):
            raise RuntimeError('Invalid saved sampling position')
    torch.set_rng_state(saved['torch_rng_state'].cpu())
    if device.type == 'cuda' and 'cuda_rng_state_all' in saved:
        torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng_state_all']])
    phase = int(saved['phase_step']) if resume else 0
    processed = int(saved['phase_tokens']) if resume else 0
    answer_tokens = int(saved['answer_tokens']) if resume else 0
    coverage = copy.deepcopy(saved['coverage']) if resume else empty_coverage()
    unique = {s: set(v['unique_indices']) for s, v in coverage.items()}
    best_loss = float(saved['best_loss_bpt']) if resume else float('inf')
    best_score = float(saved['best_development_score']) if resume else -1.
    best_candidate = float(saved['best_candidate_score']) if resume else -1.
    bad_checks = int(saved['no_gain_evaluations']) if resume else 0
    evaluations = int(saved['quality_evaluations']) if resume else 0
    baseline = saved['baseline_evaluation'] if resume else None
    last = saved['last_evaluation'] if resume else None
    candidate_step = saved['candidate_phase_step'] if resume else None
    last_status = saved.get('arm_status', 'training') if resume else 'training'
    parent_step, parent_sft = int(parent['step']), int(parent['sft_steps'])
    parent_tokens = int(parent.get('trained_tokens', 0))
    train_args = dict(parent['train_args'])
    config = parent['config']
    run_dir.mkdir(parents=True, exist_ok=True)
    del saved, parent

    def assess():
        with evaluation_rng(device):
            model.eval()
            losses = evaluate(model, pools['val'], int(chat_settings['val_examples_per_source']))
            retention = moe_train.evaluate_both(model, replay, 128, 1, 16)
            quality = evaluate_suite(model, tokenizer, suite)
            _score(quality)
            if not math.isfinite(float(losses['val_answer_bpt'])):
                raise RuntimeError('Nonfinite answer validation')
            if not math.isfinite(float(retention['val_mean_bpt'])):
                raise RuntimeError('Nonfinite replay validation')
            return dict(phase_step=phase, answer_validation=losses, replay_validation=retention,
                        quality=quality)

    def progress(status):
        return dict(status=status, policy=a.policy, phase_step=phase, target_steps=a.steps,
                    source_step=parent_step, source_sft_steps=parent_sft,
                    saved_sft_step=parent_sft+phase, phase_tokens=processed,
                    answer_tokens=answer_tokens, best_development_score=best_score,
                    candidate_phase_step=candidate_step, qualified_candidate=candidate_step is not None,
                    latest=str(latest), best_loss=str(run_dir/'best_loss.pt'),
                    candidate=str(run_dir/'candidate.pt') if candidate_step is not None else None,
                    coverage={s: {**v, 'unique_indices': sorted(unique[s]),
                                  'unique_examples': len(unique[s])} for s, v in coverage.items()},
                    last_evaluation=last, source_unchanged=sha(source) == source_hash)

    def save(status, loss_improved=False, candidate_improved=False):
        for s in coverage:
            coverage[s]['unique_indices'] = sorted(unique[s])
        record = dict(format='PhiStateMoE-v1', training_stage='coherence_recovery_v1',
            config=config, state_dict=model.state_dict(), optimizer_state=optimizer.state_dict(),
            step=parent_step+phase, sft_steps=parent_sft+phase, source_step=parent_step,
            source_sft_steps=parent_sft, phase_step=phase, phase_tokens=processed,
            trained_tokens=parent_tokens+processed, answer_tokens=answer_tokens,
            data_fingerprint=corpus['data_fingerprint'], tokenizer='bpe', tokenizer_sha256=token_hash,
            sft_data_fingerprint=prepared['data_fingerprint'], source_sha256=source_hash,
            source_checkpoint=str(source), train_args=train_args, chat_template=TEMPLATE,
            chat_settings=chat_settings, coherence_settings=settings, arm_status=status,
            baseline_evaluation=baseline, last_evaluation=last,
            quality=last['quality'], last_validation=last['answer_validation'],
            best_loss_bpt=best_loss, best_development_score=best_score,
            best_candidate_score=best_candidate, candidate_phase_step=candidate_step,
            no_gain_evaluations=bad_checks, quality_evaluations=evaluations,
            coverage=coverage, python_rng_state=rng.getstate(),
            numpy_rng_state=numpy_rng.bit_generator.state, orders=orders, positions=positions,
            torch_rng_state=torch.get_rng_state())
        if device.type == 'cuda':
            record['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
        if loss_improved:
            moe_train.atomic_checkpoint(run_dir/'best_loss.pt', record)
        if candidate_improved:
            moe_train.atomic_checkpoint(run_dir/'candidate.pt', record)
        moe_train.atomic_checkpoint(latest, record)
        info = progress(status)
        if not info['source_unchanged']:
            raise RuntimeError('Source checkpoint changed externally')
        atomic_json(run_dir/'progress.json', info)
        print('SAVED', status, phase, '/', a.steps, flush=True)
        return info

    if not resume:
        last = baseline = assess()
        evaluations = 1
        best_loss = float(last['answer_validation']['val_answer_bpt'])
        best_score = _score(last['quality'])
        if last['quality']['qualified']:
            best_candidate, candidate_step = best_score, 0
        save('training', True, candidate_step is not None)
    elif last_status in TERMINAL or phase >= a.steps:
        # Reconcile progress if a previous session died after its final checkpoint.
        status = last_status if last_status in TERMINAL else (
            'completed_qualified' if candidate_step is not None else 'completed_failed_quality')
        info = progress(status)
        atomic_json(run_dir/'progress.json', info)
        return info
    stop_file = run_dir/'STOP'
    parent_stop = Path(a.parent_stop) if getattr(a, 'parent_stop', None) else None
    if stop_file.exists() and getattr(a, 'clear_stop', False):
        stop_file.unlink()
    if parent_stop is not None and parent_stop.exists() and getattr(a, 'clear_stop', False):
        parent_stop.unlink()
    def stop_requested():
        return stopped[0] or stop_file.exists() or (parent_stop is not None and parent_stop.exists())
    stopped = [False]
    old_handlers = {s: signal.signal(s, lambda *_: stopped.__setitem__(0, True))
                    for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        if stop_requested():
            return save('stopped')
        for index in range(phase+1, a.steps+1):
            if stop_requested():
                return save('stopped')
            model.train()
            optimizer.zero_grad(set_to_none=True)
            mixed_loss = 0.
            for _ in range(accum):
                if rng.random() < chat_settings['replay_share']:
                    domain = 'code' if rng.random() < chat_settings['replay_code_share'] else 'general'
                    x, y = moe_train.sample_batch(replay['train_'+domain], 1,
                        chat_settings['replay_context'], numpy_rng)
                    x, y = x.to(device), y.to(device)
                    name, supervised = 'replay_'+domain, 0
                else:
                    name = rng.choices(list(SOURCES), weights=[v['weight'] for v in SOURCES.values()], k=1)[0]
                    if positions[name] == len(orders[name]):
                        rng.shuffle(orders[name]); positions[name] = 0
                    row_index = orders[name][positions[name]]
                    positions[name] += 1
                    row = pools['train'][name][row_index]
                    unique[name].add(row_index)
                    x, y = tensors(row, device)
                    supervised = int((y != -100).sum())
                logits = model(x)
                lm = F.cross_entropy(logits.reshape(-1, model.cfg.vocab), y.flatten(), ignore_index=-100)
                penalty, _ = model.router_penalty()
                loss = (lm + penalty) / accum
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training objective; latest saved checkpoint retained')
                loss.backward()
                mixed_loss += float(lm.detach()) / accum
                coverage[name]['draws'] += 1
                coverage[name]['input_tokens'] += x.numel()
                coverage[name]['answer_tokens'] += supervised
                processed += x.numel(); answer_tokens += supervised
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            phase = index
            loss_improved = candidate_improved = False
            status = 'training'
            if phase % a.eval_every == 0 or phase == a.steps:
                last = assess(); evaluations += 1
                score = _score(last['quality'])
                improved = score > best_score + 1e-12
                bad_checks = 0 if improved else bad_checks + 1
                best_score = max(best_score, score)
                loss_value = float(last['answer_validation']['val_answer_bpt'])
                loss_improved = loss_value < best_loss
                best_loss = min(best_loss, loss_value)
                retained = last['replay_validation']['val_mean_bpt'] <= (
                    baseline['replay_validation']['val_mean_bpt'] + chat_settings.get('retention_max_mean_bpt_increase', .5))
                last['retained'] = retained
                if retained and last['quality']['qualified'] and score > best_candidate + 1e-12:
                    candidate_step, best_candidate, candidate_improved = phase, score, True
                with (run_dir/'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps({**last, 'mixed_train_bpt': mixed_loss/math.log(2),
                                              'preclip_gradient_norm': float(grad)})+'\n')
                print('QUALITY', json.dumps(last['quality']), flush=True)
                if not retained:
                    status = 'retention_stopped'
                elif bad_checks >= settings['no_gain_evaluations']:
                    status = 'early_stopped_qualified' if candidate_step is not None else 'early_stopped_failed_quality'
            if status == 'training' and phase == a.steps:
                status = 'completed_qualified' if candidate_step is not None else 'completed_failed_quality'
            if status == 'training' and stop_requested():
                status = 'stopped'
            if phase % a.save_every == 0 or phase % a.eval_every == 0 or status != 'training':
                info = save(status, loss_improved, candidate_improved)
            if status != 'training':
                return info
        raise RuntimeError('Arm ended without a terminal status')
    finally:
        for s, handler in old_handlers.items():
            signal.signal(s, handler)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'data-dir', 'sft-dir', 'run-dir', 'dev-suite'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--policy', choices=('original', 'protected'), required=True)
    p.add_argument('--steps', type=int, default=500, help='Fixed additional phase budget; maximum 2000')
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default='cuda')
    p.add_argument('--clear-stop', action='store_true')
    p.add_argument('--parent-stop', help='Optional coordinator-wide STOP file')
    print(json.dumps(run(p.parse_args()), indent=2))


if __name__ == '__main__':
    main()
