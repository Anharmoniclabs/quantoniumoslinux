"""Bounded, resumable 16-example memorization and prompt-conditioning diagnostic.

This is not held-out validation or a replacement production checkpoint.
"""
import argparse
import json
import math
from pathlib import Path
import signal
from types import SimpleNamespace

import sentencepiece as spm
import torch
import torch.nn.functional as F

import moe_train
from chat_recovery_data import encode_example, prompt_text, sha
from chat_recovery_train import atomic_json, run_lock

# Paired questions have distinct answers so swapping a question is meaningful.
PAIRS = [
    ('What is two plus three?', 'Two plus three is five.'),
    ('What is two minus three?', 'Two minus three is negative one.'),
    ('What does RAM do?', 'RAM temporarily holds data used by running programs.'),
    ('What does storage do?', 'Storage keeps files when the power is off.'),
    ('What is Python?', 'Python is a programming language.'),
    ('What is Linux?', 'Linux is an operating system kernel.'),
    ('What is the capital of France?', 'The capital of France is Paris.'),
    ('What is the capital of Japan?', 'The capital of Japan is Tokyo.'),
    ('What do plants need for photosynthesis?', 'Plants need light, water, and carbon dioxide.'),
    ('What do humans breathe to stay alive?', 'Humans need oxygen to stay alive.'),
    ('Write a Python function add(a, b) that returns their sum.',
     '```python\ndef add(a, b):\n    return a + b\n```'),
    ('Write a Python function subtract(a, b) that returns their difference.',
     '```python\ndef subtract(a, b):\n    return a - b\n```'),
    ('What is quantum physics? Explain simply.',
     'Quantum physics studies matter and energy at very small scales.'),
    ('What is astronomy? Explain simply.',
     'Astronomy studies stars, planets, and other objects in space.'),
    ('Reply with exactly this word: apple', 'apple'),
    ('Reply with exactly this word: orange', 'orange'),
]
PARAPHRASES = [
    ('Calculate 2 + 3.', PAIRS[0][1]),
    ('Explain what computer RAM is used for.', PAIRS[2][1]),
    ('Name the capital city of Japan.', PAIRS[7][1]),
    ('Implement add(a, b) in Python to add the two inputs.', PAIRS[10][1]),
]


def row(tokenizer, question, answer):
    x, y = encode_example([{'role': 'user', 'content': question},
                           {'role': 'assistant', 'content': answer}], tokenizer, 256)
    prefix = tokenizer.encode(prompt_text([{'role': 'user', 'content': question}]))
    assert all(t == -100 for t in y[:len(prefix)-1])
    assert y[len(prefix)-1:] == x[len(prefix):] + [tokenizer.eos_id()]
    assert tokenizer.decode(x) == prompt_text([{'role': 'user', 'content': question}]) + answer
    return x, y


def tensors(encoded, device):
    return tuple(torch.tensor([a], dtype=torch.long, device=device) for a in encoded)


@torch.inference_mode()
def decode(model, tokenizer, question, max_tokens=96):
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt_text([{'role': 'user', 'content': question}]))
    cache = None
    for token in ids:
        logits, cache = model.step(torch.tensor([[token]], device=device), cache)
    produced, eos = [], False
    for _ in range(max_tokens):
        if not torch.isfinite(logits).all():
            raise RuntimeError('Nonfinite decoding logits')
        token = int(logits[0, -1].argmax())
        if token == tokenizer.eos_id():
            eos = True
            break
        produced.append(token)
        logits, cache = model.step(torch.tensor([[token]], device=device), cache)
    return tokenizer.decode(produced), eos


@torch.inference_mode()
def audit(model, encoded):
    model.eval()
    device = next(model.parameters()).device
    x, _ = tensors(encoded, device)
    full = model(x)
    altered = x.clone()
    cut = x.shape[1] // 2
    altered[:, cut:] = (altered[:, cut:] + 17) % model.cfg.vocab
    causal_error = float((full[:, :cut] - model(altered)[:, :cut]).abs().max())
    cache, parts = None, []
    for token in x[0]:
        logits, cache = model.step(token.reshape(1, 1), cache)
        parts.append(logits)
    incremental = torch.cat(parts, dim=1)
    step_error = float((full - incremental).abs().max())
    if not torch.isfinite(full).all() or causal_error > .002 or step_error > .005:
        raise RuntimeError(f'Forward/cached/causality audit failed: {causal_error=}, {step_error=}')
    return dict(causal_max_error=causal_error, full_vs_cached_max_error=step_error)


@torch.inference_mode()
def evaluate(model, tokenizer, rows):
    model.eval()
    device = next(model.parameters()).device
    samples, nll, count = [], 0., 0
    for i, ((question, answer), encoded) in enumerate(zip(PAIRS, rows)):
        def score(item):
            x, y = tensors(item, device)
            loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=-100, reduction='sum')
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite diagnostic loss')
            return float(loss), int((y != -100).sum())
        loss, tokens = score(encoded)
        wrong_loss, wrong_tokens = score(row(tokenizer, PAIRS[i ^ 1][0], answer))
        # Same answer, paired wrong question; positive gap favors the correct question.
        gap = wrong_loss / wrong_tokens / math.log(2) - loss / tokens / math.log(2)
        text, eos = decode(model, tokenizer, question)
        samples.append(dict(question=question, expected=answer, generated=text, eos=eos,
                            exact=text == answer and eos, correct_prompt_advantage_bpt=gap))
        nll += loss
        count += tokens
    return dict(training_answer_bpt=nll / count / math.log(2),
                exact_answers=sum(s['exact'] for s in samples), total=16,
                prompt_wins=sum(s['correct_prompt_advantage_bpt'] > .05 for s in samples),
                mean_prompt_advantage_bpt=sum(s['correct_prompt_advantage_bpt'] for s in samples)/16,
                samples=samples)


def run(a):
    output, source, data = Path(a.run_dir).resolve(), Path(a.checkpoint).resolve(), Path(a.data_dir).resolve()
    if output == source.parent or output == data:
        raise RuntimeError('Diagnostic requires its own output directory')
    with run_lock(output), run_lock(source.parent):
        return train(a, output, source, data)


def train(a, output, source, data):
    if a.steps < 1 or a.steps > 800 or a.eval_every < 1:
        raise ValueError('Use 1..800 diagnostic steps and positive eval interval')
    moe_train.setup(2, 20261030)
    device = moe_train.resolve_device(a.device)
    source_hash = sha(source)
    base = torch.load(source, map_location='cpu', weights_only=True)
    if sha(source) != source_hash:
        raise RuntimeError('Source changed while loading')
    token_hash = sha(data / 'tokenizer.model')
    manifest = json.loads((data / 'manifest.json').read_text())
    if (base.get('format') != 'PhiStateMoE-v1' or
        base.get('data_fingerprint') != manifest['data_fingerprint'] or
        token_hash != manifest.get('whitespace_repair', {}).get('tokenizer_sha256') or
        base.get('tokenizer_sha256') != token_hash):
        raise RuntimeError('Checkpoint and repaired tokenizer/corpus lineage differ')
    if base.get('train_args', {}).get('optimizer') != 'adafactor' or 'optimizer_state' not in base:
        raise RuntimeError('Use chat_recovery_v1/run/latest.pt with saved Adafactor state')
    tokenizer = spm.SentencePieceProcessor(model_file=str(data / 'tokenizer.model'))
    if tokenizer.vocab_size() != base['config']['vocab']:
        raise RuntimeError('Vocabulary mismatch')
    model = moe_train.PhiStateMoELM(moe_train.MoEConfig(**base['config'])).to(device)
    model.load_state_dict(base['state_dict'], strict=True)
    optimizer = moe_train.optimizer_for(SimpleNamespace(**base['train_args']), model)
    optimizer.load_state_dict(base['optimizer_state'])
    config, parent = base['config'], base['step']
    del base
    rows = [row(tokenizer, q, ans) for q, ans in PAIRS]
    settings = dict(pairs=PAIRS, source_sha256=source_hash, tokenizer_sha256=token_hash,
                    accum=4, optimizer='preserved_adafactor', seed=20261030)
    # JSON canonical form keeps tuple/list comparison stable after serialization.
    settings = json.loads(json.dumps(settings))
    step, last, best_exact, best_bpt, update_delta = 0, None, -1, float('inf'), None
    latest = output / 'latest.pt'
    if latest.exists():
        state = torch.load(latest, map_location='cpu', weights_only=True)
        if state.get('diagnostic_settings') != settings:
            raise RuntimeError('Existing diagnostic belongs to different source/settings')
        model.load_state_dict(state['state_dict'], strict=True)
        optimizer.load_state_dict(state['optimizer_state'])
        step, last = state['diagnostic_step'], state['last_evaluation']
        best_exact, best_bpt, update_delta = state['best_exact'], state['best_bpt'], state['first_update_max_delta']
        torch.set_rng_state(state['torch_rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
        del state
        print('RESUMED diagnostic step', step, flush=True)
    else:
        if output.exists() and any(output.iterdir()):
            raise RuntimeError('Output directory contains files without a resumable checkpoint')
        output.mkdir(parents=True, exist_ok=True)
    checks = audit(model, rows[10])
    print('SOURCE STEP:', parent, '| PyTorch', device, '| source unchanged; diagnostic only', flush=True)
    print('AUDIT', json.dumps(checks), flush=True)
    stop = [False]
    old = {s: signal.signal(s, lambda *_: stop.__setitem__(0, True)) for s in (signal.SIGINT, signal.SIGTERM)}

    def save(status):
        nonlocal best_exact, best_bpt
        improved = (last['exact_answers'], -last['training_answer_bpt']) > (best_exact, -best_bpt)
        if improved:
            best_exact, best_bpt = last['exact_answers'], last['training_answer_bpt']
        state = dict(format='PhiStateTinyDiagnostic-v1', config=config, source_step=parent,
                     diagnostic_step=step, state_dict=model.state_dict(), optimizer_state=optimizer.state_dict(),
                     diagnostic_settings=settings, last_evaluation=last, best_exact=best_exact, best_bpt=best_bpt,
                     first_update_max_delta=update_delta, torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
        moe_train.atomic_checkpoint(latest, state)
        if improved:
            state.pop('optimizer_state')
            moe_train.atomic_checkpoint(output / 'best_diagnostic.pt', state)
        atomic_json(output / 'progress.json', dict(status=status, saved_step=step, target_steps=a.steps,
                    exact_answers=last['exact_answers'], prompt_wins=last['prompt_wins'],
                    training_answer_bpt=last['training_answer_bpt'], last_evaluation_step=last['step']))

    def assess():
        result = evaluate(model, tokenizer, rows)
        result['step'] = step
        atomic_json(output / f'evaluation_{step:04d}.json', result)
        print('CHECK', json.dumps({k:v for k,v in result.items() if k != 'samples'}), flush=True)
        for i in (0, 2, 10, 14):
            s = result['samples'][i]
            print('PROMPT:', s['question'], '\nANSWER:', repr(s['generated']), flush=True)
        return result

    def passed():
        return last['exact_answers'] == 16 and last['prompt_wins'] >= 14

    status = 'running'
    try:
        if last is None:
            last = assess()
            save('prepared')
        for index in range(step + 1, a.steps + 1):
            if stop[0] or (output / 'STOP').exists() or passed():
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            before = model.embedding.weight.detach().clone() if step == 0 else None
            for micro in range(4):
                x, y = tensors(rows[((index-1)*4 + micro) % 16], device)
                logits = model(x)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100)
                penalty, _ = model.router_penalty()
                if not torch.isfinite(loss + penalty):
                    raise RuntimeError('Nonfinite loss; previous saved checkpoint retained')
                ((loss + penalty)/4).backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            step = index
            if before is not None:
                update_delta = float((model.embedding.weight.detach() - before).abs().max())
                if not math.isfinite(update_delta) or update_delta <= 0:
                    raise RuntimeError('Embedding weights did not receive a finite nonzero update')
                print('FIRST UPDATE MAX WEIGHT CHANGE:', update_delta, flush=True)
            if step % 25 == 0:
                print('TRAIN', step, '/', a.steps, '| grad norm', round(float(grad), 4), flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                last = assess()
                save('running')
            if passed():
                break
        if last['step'] != step:
            last = assess()
        status = 'stopped' if stop[0] or (output / 'STOP').exists() else ('tiny_task_passed' if passed() else 'bounded_test_not_passed')
        save(status)
        checks = audit(model, rows[10])
        # These four prompts never enter the diagnostic training loop.
        paraphrases = []
        model.eval()
        for q, expected in PARAPHRASES:
            text, eos = decode(model, tokenizer, q)
            paraphrases.append(dict(question=q, expected=expected, generated=text, eos=eos))
        if sha(source) != source_hash:
            raise RuntimeError('Source checkpoint changed externally')
        report = dict(status=status, steps=step, source_unchanged=True, audit=checks,
                      training_evaluation=last, unseen_wordings=paraphrases,
                      first_update_max_delta=update_delta,
                      scope='16-example memorization diagnostic; not broad language ability. '
                            'Failure within this budget does not by itself prove an architecture bug.')
        atomic_json(output / 'report.json', report)
        print('\nRESULT:', status, '| exact', last['exact_answers'], '/16 | prompt wins', last['prompt_wins'], '/16', flush=True)
        for p in paraphrases:
            print('UNSEEN WORDING:', p['question'], '\nANSWER:', repr(p['generated']), flush=True)
        print('REPORT:', output / 'report.json', flush=True)
        return report
    finally:
        for s, handler in old.items():
            signal.signal(s, handler)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'data-dir', 'run-dir'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--steps', type=int, default=800)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    run(p.parse_args())
