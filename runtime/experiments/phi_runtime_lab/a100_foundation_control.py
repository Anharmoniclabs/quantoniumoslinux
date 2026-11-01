"""Bounded, from-scratch contextual binding comparison after a failed warm-checkpoint screen.

Both architectures receive exactly the same counterfactual mini-batches. Development uses
new wording and key/value combinations absent from training. A key's answer changes when
two context values are swapped, so a fixed answer preference can score at most one of each
pair. This task measures a narrow kind of prompt sensitivity, not conversational quality.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import string

import sentencepiece as spm
import torch
from torch import nn
from torch.nn import functional as F

from chat_recovery_train import atomic_json, run_lock
from moe_model import PhiStateMoELM, config_for
from phistate import RMSNorm, SwiGLU, count_parameters

STEPS, BATCH_PAIRS, DEV_PAIRS, SAVE_EVERY = 256, 8, 96, 16
SEED, WIDTH, LAYERS, MAX_LENGTH = 42017, 32, 2, 128
ARMS = ('dense', 'phistate')
TRAIN_WORDING = ('{k}={v}; {j}={w}. Read {q}:', '{k}:{v} / {j}:{w}. Find {q}:')
DEV_WORDING = ('List {k}={v}, {j}={w}. Value for {q}:', 'Given {k}:{v}; {j}:{w}. Look up {q}:')


def _symbols(tok, alphabet):
    result = []
    for sym in alphabet:
        pieces = tok.encode(sym)
        if len(pieces) == 1 and tok.decode(pieces) == sym and pieces[0] not in result:
            result.append(pieces[0])
        if len(result) == 8:
            return result
    raise ValueError('Binding control requires eight distinct single-token symbols per alphabet')


def _bank(tok, development):
    keys = _symbols(tok, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    vals = _symbols(tok, '0123456789abcdefghijklmnopqrstuvwxyz')
    if set(keys) & set(vals):
        raise ValueError('Key and value symbols overlap')
    forms = DEV_WORDING if development else TRAIN_WORDING
    # Encoding each field explicitly keeps the isolated value token identical in context
    # and response. Literal markers are never encoded as input tokens.
    templates = [[(tok.encode(literal), field) for literal, field, _, _ in string.Formatter().parse(form)]
                 for form in forms]
    candidates = []
    for k, j in itertools.permutations(range(8), 2):
        if k % 2 != j % 2:
            continue
        group = [v for v in range(8) if (v % 2 != k % 2) == development]
        for v, w in itertools.combinations(group, 2):
            for q in (k, j):
                for form in range(len(forms)):
                    candidates.append((k, j, v, w, q, form))
    if development:
        random.Random(SEED + 1).shuffle(candidates)
        candidates = candidates[:DEV_PAIRS]
    result = []
    for k, j, v, w, q, form in candidates:
        pair = []
        for first, second in ((v, w), (w, v)):
            fields = dict(k=keys[k], j=keys[j], q=keys[q], v=vals[first], w=vals[second])
            tokens = []
            for literal, field in templates[form]:
                tokens.extend(literal)
                if field:
                    tokens.append(fields[field])
            if not 0 < len(tokens) <= MAX_LENGTH:
                raise ValueError('Binding context length exceeds the control specification')
            target, foil = (first, second) if k == q else (second, first)
            pair.append((tokens, vals[target], vals[foil]))
        result.append(pair)
    return result


class DenseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.n2 = RMSNorm(WIDTH), RMSNorm(WIDTH)
        self.attn = nn.MultiheadAttention(WIDTH, 4, dropout=0, bias=False, batch_first=True)
        self.ff = SwiGLU(WIDTH, 5)

    def forward(self, x, mask):
        z = self.n1(x)
        x = x + 0.7 * self.attn(z, z, z, attn_mask=mask, need_weights=False, is_causal=True)[0]
        return x + 0.5 * self.ff(self.n2(x))


class DenseReference(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tokens = nn.Embedding(vocab, WIDTH)
        self.positions = nn.Embedding(MAX_LENGTH, WIDTH)
        self.blocks = nn.ModuleList(DenseBlock() for _ in range(LAYERS))
        self.norm = RMSNorm(WIDTH)
        nn.init.normal_(self.tokens.weight, std=0.05)
        nn.init.normal_(self.positions.weight, std=0.02)

    def forward(self, ids):
        length = ids.size(1)
        x = self.tokens(ids) + self.positions(torch.arange(length, device=ids.device))[None]
        mask = torch.ones(length, length, dtype=torch.bool, device=ids.device).triu(1)
        for block in self.blocks:
            x = block(x, mask)
        return F.linear(self.norm(x), self.tokens.weight)


def _batch(model, pairs, device):
    rows = [row for pair in pairs for row in pair]
    # Right-padding lies after every scored position and causal attention cannot see it.
    bounds = torch.tensor([len(row[0]) - 1 for row in rows], device=device)
    ids = torch.zeros((len(rows), 1 + int(bounds.max().item())), dtype=torch.long, device=device)
    for i, (tokens, _, _) in enumerate(rows):
        ids[i, :len(tokens)] = torch.tensor(tokens, device=device)
    logits = model(ids)[torch.arange(len(rows), device=device), bounds]
    target, foil = (torch.tensor([row[field] for row in rows], device=device) for field in (1, 2))
    return logits, target, foil, sum(len(row[0]) for row in rows)


@torch.inference_mode()
def _score(model, pairs, device):
    model.eval()
    correct = winners = both = ties = 0
    nll = margin = 0.0
    for begin in range(0, len(pairs), BATCH_PAIRS):
        logits, target, foil, _ = _batch(model, pairs[begin:begin+BATCH_PAIRS], device)
        indices = torch.arange(len(target), device=device)
        diff = logits[indices, target] - logits[indices, foil]
        wins = diff > 0
        ties += int((diff == 0).sum().item())
        winners += int(wins.sum().item())
        both += int(wins.reshape(-1, 2).all(1).sum().item())
        correct += int((logits.argmax(-1) == target).sum().item())
        margin += float(diff.sum().item())
        nll += float(F.cross_entropy(logits, target, reduction='sum').item())
    total = 2 * len(pairs)
    return dict(pairs=len(pairs), full_vocab_accuracy=correct / total,
                two_choice_accuracy=(winners + 0.5 * ties) / total, both_contexts_correct=both,
                paired_rate=both / len(pairs), ties=ties, mean_target_logit_margin=margin / total,
                answer_bpt=nll / (total * math.log(2)))


def _arm(name, vocab, train, dev, output, device, stop_path, identity):
    # fork_rng isolates the caller's stream. Data order derives solely from experiment seed and step.
    torch.manual_seed(SEED)
    model = DenseReference(vocab) if name == 'dense' else PhiStateMoELM(
        config_for('mini', vocab, mode='dft', width=WIDTH, layers=LAYERS, window=8))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    path = output / (name + '.pt')
    # A fixed sample of training-bank pairs distinguishes fitting the mechanism from
    # transfer to the development wording/bindings. It is explicitly a train metric.
    train_probe = random.Random(SEED + 2).sample(train, len(dev))
    step = seen_inputs = 0
    if path.is_file():
        state = torch.load(path, weights_only=True, map_location='cpu')
        if state['identity'] != identity or state['name'] != name:
            raise RuntimeError('Binding control checkpoint identity changed')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['cpu_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(state['gpu_rng'], device)
        step, seen_inputs, baseline = state['step'], state['seen_inputs'], state['baseline']
        train_baseline = state['train_baseline']
        if not 0 <= step <= STEPS:
            raise RuntimeError('Invalid foundation control saved step')
        del state
    else:
        baseline = _score(model, dev, device)
        train_baseline = _score(model, train_probe, device)

    def save():
        pending = path.with_suffix('.pt.partial')
        torch.save(dict(identity=identity, name=name, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        cpu_rng=torch.get_rng_state(), gpu_rng=torch.cuda.get_rng_state(device)
                        if device.type == 'cuda' else None, step=step, seen_inputs=seen_inputs,
                        baseline=baseline, train_baseline=train_baseline), pending)
        pending.replace(path)

    while step < STEPS:
        if stop_path.exists():
            save()
            return dict(status='stopped', updates=step)
        rng = random.Random(SEED + 1000 + step)
        pairs = rng.sample(train, BATCH_PAIRS)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, target, _, n_input = _batch(model, pairs, device)
        loss = F.cross_entropy(logits, target)
        if name == 'phistate':
            loss = loss + model.router_penalty()[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        step += 1
        seen_inputs += n_input
        if step % SAVE_EVERY == 0 or step == STEPS:
            save()
            print(f'Binding control {name} {step}/{STEPS} | {2 * BATCH_PAIRS * step} answer tokens', flush=True)
    return dict(status='complete', updates=step, supervised_tokens=step * 2 * BATCH_PAIRS,
                context_tokens=seen_inputs, total_parameters=count_parameters(model),
                estimated_active_parameters=model.estimate_active_parameters()
                if name == 'phistate' else count_parameters(model), before=baseline, after=_score(model, dev, device),
                training_probe_before=train_baseline, training_probe_after=_score(model, train_probe, device))


def run(tokenizer, output, device='cuda', *, stop_path=None):
    """Persist two independent scratch arms. Rerunning resumes any incomplete save."""
    tok = tokenizer if hasattr(tokenizer, 'encode') and not isinstance(tokenizer, str) else spm.SentencePieceProcessor(
        model_file=str(tokenizer))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    stop_path = Path(stop_path) if stop_path is not None else output.parent / 'STOP'
    train, dev = _bank(tok, False), _bank(tok, True)
    identity = dict(version='binding-control-v1', file_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    tokenizer_sha256=hashlib.sha256(tok.serialized_model_proto()).hexdigest(),
                    device_type=device.type, seed=SEED, updates_per_arm=STEPS, pairs_per_batch=BATCH_PAIRS,
                    dev_pairs=len(dev), train_pairs=len(train),
                    train_bank_sha256=hashlib.sha256(json.dumps(train).encode()).hexdigest(),
                    dev_bank_sha256=hashlib.sha256(json.dumps(dev).encode()).hexdigest())
    manifest, report = output/'request.json', output/'report.json'
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with run_lock(output), torch.random.fork_rng(devices=devices):
        if manifest.exists():
            if json.loads(manifest.read_text()) != identity:
                raise RuntimeError('Binding control manifest changed; reserve a new output directory')
        elif any(x.name != 'request.partial' for x in output.iterdir()):
            raise RuntimeError('Binding control refuses to take ownership of a nonempty directory')
        else:
            atomic_json(manifest, identity)
        if report.exists():
            saved = json.loads(report.read_text())
            if saved['identity'] != identity:
                raise RuntimeError('Binding control result belongs to another request')
            if saved['status'] == 'complete':
                return saved
        arms = {}
        for name in ARMS:
            if stop_path.exists():
                break
            arms[name] = _arm(name, tok.vocab_size(), train, dev, output, device, stop_path, identity)
            if arms[name]['status'] == 'stopped':
                break
        complete = len(arms) == len(ARMS) and all(a['status'] == 'complete' for a in arms.values())
        passes = [name for name in arms if arms[name]['status'] == 'complete' and
                  arms[name]['after']['paired_rate'] >= 0.65 and arms[name]['after']['two_choice_accuracy'] >= 0.8]
        outcome = {'dense,phistate': 'both_show_binding_transfer', 'dense': 'only_dense_shows_binding_transfer',
                   'phistate': 'only_phistate_shows_binding_transfer', '': 'inconclusive'}[','.join(passes)]
        result = dict(status='complete' if complete else 'stopped', outcome=outcome if complete else 'unfinished',
                      scope='New small models on a synthetic retrieval task. Scores cannot qualify a chat checkpoint.',
                      criteria={'paired_rate_min': 0.65, 'two_choice_accuracy_min': 0.8,
                                'two_choice_fixed_preference_accuracy': 0.5},
                      identity=identity, arms=arms)
        atomic_json(report, result)
        return result
