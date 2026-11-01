"""Train one fixed-budget data arm with the parent optimizer and exact answer-token accounting."""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import signal
import threading
from types import SimpleNamespace

import sentencepiece as spm
import torch
import torch.nn.functional as F

import moe_train
from chat_recovery_data import TEMPLATE, sha
from chat_recovery_train import atomic_json, run_lock
from coherence_recovery_train import evaluation_rng


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _inspect_schedule(pool, vocab, context, eos):
    """Bind even partial/noncontiguous target masks to the ordered causal inputs."""
    if not len(pool):
        raise ValueError('Empty training schedule')
    digest = hashlib.sha256()
    positions = []
    for row in pool:
        x, y = row['input_ids'], row['labels']
        if not 1 <= len(x) <= context or len(x) != len(y):
            raise ValueError('Invalid causal row length')
        if any(type(t) is not int or not 0 <= t < vocab for t in x):
            raise ValueError('Invalid input token')
        active = []
        for i, t in enumerate(y):
            if t == -100:
                continue
            expected = x[i+1] if i+1 < len(x) else eos
            if type(t) is not int or t != expected or not 0 <= t < vocab:
                raise ValueError('Invalid next-token target or EOS')
            active.append(i)
        if not active:
            raise ValueError('Training schedule contains a row without supervised targets')
        positions.append(active)
        digest.update(_json([row.get('id'), row.get('source', 'unspecified'), x, y]).encode())
        digest.update(b'\n')
    return positions, digest.hexdigest()


def _segments(positions, cursor, budget):
    """Consume exactly budget selected targets, carrying the unfinished row forward."""
    row, offset = cursor
    remaining = budget
    while remaining:
        row_index = row % len(positions)
        active = positions[row_index]
        if row < 0 or not 0 <= offset < len(active):
            raise ValueError('Invalid saved schedule cursor')
        count = min(remaining, len(active)-offset)
        target = active[offset:offset+count]
        offset += count
        finished = offset == len(active)
        if finished:
            row += 1
            offset = 0
        remaining -= count
        yield row_index, target, (row, offset), finished


@contextmanager
def _interrupt_flag():
    interrupted = [False]
    previous = {}
    def request_stop(*_):
        interrupted[0] = True
    if threading.current_thread() is threading.main_thread():
        previous = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield interrupted
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


def _check_weights(model):
    if not torch.stack([p.detach().isfinite().all() for p in model.parameters()]).all().item():
        raise FloatingPointError('Nonfinite model parameter; previous safe checkpoint retained')


def run_arm(*, parent, tokenizer, pool, output, policy_name, device, steps, tokens_per_step,
            seed, evaluate_callback, identity, stop_path=None, check_owner=None):
    """Pool order defines the data schedule. A partially consumed row resumes exactly.

    Each optimizer update sums cross-entropy over exactly tokens_per_step selected
    answer targets (including supervised EOS). The ratio of selected targets in
    each segment also weights its router penalty. Replay is disabled in both arms.
    """
    if type(steps) is not int or not 1 <= steps <= 2000:
        raise ValueError('steps must be a fixed integer budget in 1..2000')
    if type(tokens_per_step) is not int or not 1 <= tokens_per_step <= 32768:
        raise ValueError('tokens_per_step must be a fixed integer budget in 1..32768')
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if output == parent.parent or output == parent:
        raise ValueError('Use a separate output directory for each arm')
    device = moe_train.resolve_device(str(device)) if str(device) in ('cuda', 'cpu', 'auto') else torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Use cpu or cuda')
    if isinstance(tokenizer, (str, Path)):
        token_hash = sha(tokenizer)
        processor = spm.SentencePieceProcessor(model_file=str(tokenizer))
    else:
        processor = tokenizer
        token_hash = hashlib.sha256(processor.serialized_model_proto()).hexdigest()
    with run_lock(output), evaluation_rng(device), _interrupt_flag() as interrupted:
        return _train(parent, processor, token_hash, output, pool, str(policy_name), device,
                      steps, tokens_per_step, int(seed), evaluate_callback, identity,
                      Path(stop_path) if stop_path is not None else None, interrupted, check_owner)


def _train(parent, tokenizer, token_hash, output, pool, policy_name, device, steps, budget,
           seed, evaluate, identity, stop_path, interrupted, check_owner):
    parent_hash = sha(parent)
    source = torch.load(parent, map_location='cpu', weights_only=True)
    if sha(parent) != parent_hash:
        raise RuntimeError('Source changed while loading')
    if (source.get('format') != 'PhiStateMoE-v1' or source.get('training_stage') != 'chat_recovery_v1'
            or source.get('chat_template') != TEMPLATE or source.get('tokenizer_sha256') != token_hash
            or source['config']['vocab'] != tokenizer.vocab_size()
            or source.get('train_args', {}).get('optimizer') != 'adafactor'
            or 'optimizer_state' not in source):
        raise ValueError('Expected a matching conversational SFT checkpoint with its saved Adafactor state')
    context = int(source['chat_settings']['context'])
    positions, schedule_hash = _inspect_schedule(pool, tokenizer.vocab_size(), context, tokenizer.eos_id())
    settings = json.loads(_json(dict(version=1, runner_sha256=sha(Path(__file__)),
        source_sha256=parent_hash, tokenizer_sha256=token_hash, arm=policy_name,
        schedule_sha256=schedule_hash, scheduled_rows=len(pool), seed=seed, steps=steps,
        tokens_per_step=budget, device_type=device.type, caller_identity=identity)))
    latest = output/'latest.pt'
    if latest.is_file():
        saved = torch.load(latest, map_location='cpu', weights_only=True)
        if saved.get('data_comparison_settings') != settings or saved.get('training_stage') != 'data_comparison_v1':
            raise ValueError('Saved arm source/schedule/budget/settings differ; refusing restart or extension')
        resume = True
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Nonempty output has no resumable checkpoint')
        saved, resume = source, False
    cfg = source['config']
    model = moe_train.PhiStateMoELM(moe_train.MoEConfig(**cfg)).to(device)
    model.load_state_dict(saved['state_dict'], strict=True)
    optimizer = moe_train.optimizer_for(SimpleNamespace(**source['train_args']), model)
    optimizer.load_state_dict(saved['optimizer_state'])
    if (len(optimizer.param_groups) != 1 or optimizer.param_groups[0]['weight_decay'] !=
            source['optimizer_state']['param_groups'][0]['weight_decay']):
        raise ValueError('Expected original single-group optimizer policy')
    # The data schedule is supplied explicitly. Restore the source torch RNG on
    # both starts, and the saved RNG on continuation. Seed handles older fixtures.
    cpu_rng = saved.get('torch_rng_state', torch.Generator().manual_seed(seed).get_state())
    torch.set_rng_state(cpu_rng.cpu())
    if device.type == 'cuda':
        index = torch.cuda.current_device() if device.index is None else device.index
        states = saved.get('cuda_rng_state_all', [])
        gpu_rng = states[index].cpu() if index < len(states) else torch.Generator(device=device).manual_seed(seed).get_state()
        torch.cuda.set_rng_state(gpu_rng, device=device)
    phase = int(saved['phase_step']) if resume else 0
    cursor = tuple(saved['stream_cursor']) if resume else (0, 0)
    targets = int(saved['supervised_tokens']) if resume else 0
    consumed = int(saved['input_tokens']) if resume else 0
    per_source = saved['per_source'] if resume else {}
    curve = saved['development_curve'] if resume else []
    updates = saved['update_metrics'] if resume else []
    if targets != phase*budget or len(updates) != phase:
        raise ValueError('Invalid saved token/update counts')
    if resume:
        cycles, row_index = divmod(cursor[0], len(positions))
        consumed_by_cursor = (cycles*sum(map(len, positions))
                              + sum(len(p) for p in positions[:row_index]) + cursor[1])
        if (cycles < 0 or not 0 <= cursor[1] < len(positions[row_index])
                or consumed_by_cursor != targets
                or sum(v['supervised_tokens'] for v in per_source.values()) != targets
                or sum(v['input_tokens'] for v in per_source.values()) != consumed):
            raise ValueError('Saved cursor and exposure counters disagree')
    base_keys = ('format', 'pretrain_step', 'train_args', 'chat_template', 'chat_settings',
                 'tokenizer_sha256', 'tokenizer', 'sft_data_fingerprint', 'data_fingerprint')
    base = {key: source[key] for key in base_keys if key in source}
    parent_step = int(source['step'])
    parent_sft = int(source['sft_steps'])
    parent_tokens = int(source.get('trained_tokens', 0))
    del saved, source
    output.mkdir(parents=True, exist_ok=True)
    local_stop = output/'STOP'
    def stopping():
        return interrupted[0] or local_stop.exists() or (stop_path is not None and stop_path.exists())
    def measure():
        with evaluation_rng(device), torch.inference_mode():
            model.eval()
            result = evaluate(model)
        if not isinstance(result, dict):
            raise TypeError('Development callback must return a JSON metrics dictionary')
        curve.append({'step': phase, 'supervised_tokens': targets, 'metrics': json.loads(_json(result))})
    def record(status):
        return dict(status=status, arm=policy_name, phase_step=phase, target_steps=steps,
                    source_sft_steps=parent_sft, saved_sft_step=parent_sft+phase,
                    supervised_tokens=targets, tokens_per_step=budget, input_tokens=consumed,
                    per_source=per_source, stream_cursor=list(cursor), development_curve=curve,
                    latest=str(latest), source_sha256=parent_hash, schedule_sha256=schedule_hash)
    def save(status):
        _check_weights(model)
        if sha(parent) != parent_hash:
            raise RuntimeError('Source checkpoint changed during comparison')
        info = record(status)
        checkpoint = dict(**base, config=cfg, state_dict=model.state_dict(), optimizer_state=optimizer.state_dict(),
            training_stage='data_comparison_v1', data_comparison_settings=settings, source_checkpoint=str(parent),
            step=parent_step+phase, sft_steps=parent_sft+phase, phase_tokens=consumed, answer_tokens=targets,
            trained_tokens=parent_tokens+consumed, update_metrics=updates, torch_rng_state=torch.get_rng_state(), **info)
        if device.type == 'cuda':
            checkpoint['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
        if check_owner is not None:
            check_owner()
        moe_train.atomic_checkpoint(latest, checkpoint)
        atomic_json(output/'progress.json', info)
        return info
    if phase == steps:
        if not curve or curve[-1]['step'] != steps:
            raise RuntimeError('Completed checkpoint missing final development evaluation')
        info = record('complete')
        if check_owner is not None:
            check_owner()
        atomic_json(output/'progress.json', info)
        return info
    if stopping():
        return save('stopped')
    if not curve:
        measure()
        save('training')
    for next_step in range(phase+1, steps+1):
        if stopping():
            return save('stopped')
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.
        for row_index, chosen, next_cursor, finished in _segments(positions, cursor, budget):
            row = pool[row_index]
            # Every scored target sees its full original causal prefix. Already
            # scored positions and prompt positions contribute no cross-entropy.
            x = torch.tensor([row['input_ids'][:chosen[-1]+1]], dtype=torch.long, device=device)
            y = torch.tensor([row['labels'][i] for i in chosen], dtype=torch.long, device=device)
            logits = model(x)[0, chosen]
            nll = F.cross_entropy(logits, y, reduction='sum')
            penalty, _ = model.router_penalty()
            objective = (nll + penalty*len(chosen))/budget
            if not torch.isfinite(objective).item():
                raise FloatingPointError('Nonfinite objective; previous safe checkpoint retained')
            objective.backward()
            loss_sum += float(nll.detach())
            name = str(row.get('source', 'unspecified'))
            stat = per_source.setdefault(name, {'supervised_tokens': 0, 'input_tokens': 0,
                                               'segments': 0, 'completed_schedule_rows': 0})
            stat['supervised_tokens'] += len(chosen)
            stat['input_tokens'] += x.numel()
            stat['segments'] += 1
            stat['completed_schedule_rows'] += int(finished)
            consumed += x.numel()
            targets += len(chosen)
            cursor = next_cursor
        if targets != next_step*budget:
            raise RuntimeError('Supervised token budget mismatch')
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        phase = next_step
        updates.append({'step': phase, 'supervised_tokens': budget, 'answer_bpt': loss_sum/(budget*math.log(2)),
                        'preclip_gradient_norm': float(grad)})
        if phase in (max(1, steps//2), steps):
            measure()
        status = 'complete' if phase == steps else 'stopped' if stopping() else 'training'
        if status != 'training' or phase % 16 == 0 or curve[-1]['step'] == phase:
            info = save(status)
            print('DATA ARM', policy_name, status, phase, '/', steps, '| answer tokens', targets, flush=True)
            if status != 'training':
                return info
    raise RuntimeError('Training ended without a terminal state')
