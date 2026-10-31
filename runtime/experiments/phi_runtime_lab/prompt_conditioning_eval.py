"""Measure validation answer likelihood under its own, a wrong, and an empty prompt."""
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import statistics
import tempfile

import sentencepiece as spm
import torch
import torch.nn.functional as F

from chat_recovery_data import SOURCES, TEMPLATE, canonical, digest, encode_example, prompt_text, read_jsonl, sha
from moe_train import ckpt_load

FORMAT = 'PhiStatePromptConditioning-v1'
SEED = 20261031


def _targets(row, tokenizer, context):
    x, y = encode_example(row['messages'], tokenizer, context)
    if x != row['input_ids'] or y != row['labels']:
        raise ValueError('Prepared prompt/target mismatch in validation row ' + row['id'])
    prefix = prompt_text(row['messages'][:-1])
    prefix_ids = tokenizer.encode(prefix)
    targets = y[len(prefix_ids)-1:]
    if targets != x[len(prefix_ids):] + [tokenizer.eos_id()]:
        raise ValueError('Answer alignment mismatch in validation row ' + row['id'])
    question = ' '.join(row['messages'][-2]['content'].casefold().split())
    return dict(id=row['id'], question=question, prefix=prefix, ids=prefix_ids, targets=targets,
                turns=len(row['messages'])-1, answer=row['messages'][-1]['content'])


def _fits(prefix, ids, sample, tokenizer, context):
    # Reject an entire pairing if it would require cutting the prompt or answer.
    # Identical target token IDs include the natural final EOS in every condition.
    if len(ids) + len(sample['targets']) - 1 > context:
        return False
    combined = ids + sample['targets'][:-1]
    return combined == tokenizer.encode(prefix + sample['answer'])


def _pair(rows, tokenizer, context):
    """Maximum-cardinality one-to-one non-self assignment in the fixed sample."""
    empty = prompt_text([dict(role='user', content='')])
    empty_ids = tokenizer.encode(empty)
    allowed = []
    for item in rows:
        donors = []
        if _fits(empty, empty_ids, item, tokenizer, context):
            for j, other in enumerate(rows):
                if other['id'] != item['id'] and other['question'] != item['question'] and _fits(
                    other['prefix'], other['ids'], item, tokenizer, context
                ):
                    donors.append(j)
        donors.sort(key=lambda j: (abs(len(rows[j]['ids'])-len(item['ids'])), rows[j]['id']))
        allowed.append(donors)
    owner = {}

    def assign(i, visited):
        for j in allowed[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in owner or assign(owner[j], visited):
                owner[j] = i
                return True
        return False

    for i in sorted(range(len(rows)), key=lambda i: (len(allowed[i]), rows[i]['id'])):
        assign(i, set())
    return {i: j for j, i in owner.items()}, empty_ids


@torch.inference_mode()
def _score(model, prefix_ids, targets, device):
    ids = prefix_ids + targets[:-1]
    x = torch.tensor([ids], device=device, dtype=torch.long)
    y = torch.tensor(targets, device=device, dtype=torch.long)
    logits = model(x)[0, len(prefix_ids)-1:].float()
    losses = F.cross_entropy(logits, y, reduction='none') / math.log(2)
    if not torch.isfinite(losses).all():
        raise RuntimeError('Nonfinite held-out answer loss')
    return dict(answer_bpt=float(losses.mean()), first8_bpt=float(losses[:8].mean()))


def _comparison(values):
    if not values:
        return None
    deviation = statistics.stdev(values) if len(values) > 1 else None
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                wins=sum(v > 1e-7 for v in values), losses=sum(v < -1e-7 for v in values),
                ties=sum(abs(v) <= 1e-7 for v in values), examples=len(values),
                sample_sd=deviation, descriptive_se=deviation/math.sqrt(len(values)) if deviation is not None else None)


def _source(model, pool, tokenizer, context, device, limit):
    chosen = sorted(pool, key=lambda r: digest(f'{SEED}|{r["source"]}|{r["id"]}'))[:limit]
    samples = [_targets(r, tokenizer, context) for r in chosen]
    pairs, empty_ids = _pair(samples, tokenizer, context)
    examples, omitted = [], []
    for i, item in enumerate(samples):
        if i not in pairs:
            omitted.append(item['id'])
            continue
        other = samples[pairs[i]]
        targets = item['targets']
        scores = {key: _score(model, ids, targets, device) for key, ids in
                  [('correct', item['ids']), ('wrong', other['ids']), ('empty', empty_ids)]}
        differences = {alternate+'_minus_correct_'+metric: scores[alternate][metric]-scores['correct'][metric]
                       for alternate in ('wrong', 'empty') for metric in ('answer_bpt', 'first8_bpt')}
        examples.append(dict(id=item['id'], donor_id=other['id'],
                             answer_tokens_including_eos=len(targets), target_sha256=digest(canonical(targets)),
                             correct_prompt_tokens=len(item['ids']), wrong_prompt_tokens=len(other['ids']),
                             correct_turns=item['turns'], wrong_turns=other['turns'],
                             scores=scores, differences=differences, correct_prompt_excerpt=item['prefix'][-240:],
                             wrong_prompt_excerpt=other['prefix'][-240:], reference_excerpt=item['answer'][:180]))
    comparisons = {}
    for alternate in ('wrong', 'empty'):
        for metric in ('answer_bpt', 'first8_bpt'):
            name = alternate+'_minus_correct_'+metric
            comparisons[name] = _comparison([e['differences'][name] for e in examples])
    return dict(available=len(pool), requested=limit, selected=len(samples), scored=len(examples),
                answer_tokens=sum(e['answer_tokens_including_eos'] for e in examples),
                unpaired=len(omitted), unpaired_ids=omitted, comparisons=comparisons, examples=examples)


def run(checkpoint, sft_dir, output, device='cpu', limit=16):
    """Return a JSON-compatible report. `output` is a directory for keyed reports."""
    if type(limit) is not int or not 2 <= limit <= 128:
        raise ValueError('limit must be an integer from 2 through 128 examples per source')
    checkpoint, sft_dir, output = map(Path, (checkpoint, sft_dir, output))
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Choose an available CPU or CUDA device')
    manifest = json.loads((sft_dir/'manifest.json').read_text())
    fingerprint = manifest.pop('data_fingerprint')
    if digest(canonical(manifest)) != fingerprint or manifest.get('template') != TEMPLATE:
        raise ValueError('Prepared manifest fingerprint or template mismatch')
    val_path, tok_path = sft_dir/'val.jsonl', sft_dir/'tokenizer.model'
    val_hash, tok_hash = sha(val_path), sha(tok_path)
    if val_hash != manifest['files']['val.jsonl'] or tok_hash != manifest['tokenizer_sha256']:
        raise ValueError('Prepared validation rows or tokenizer changed')
    lab = Path(__file__).resolve().parent
    code = {name: sha(lab/name) for name in ('prompt_conditioning_eval.py','chat_recovery_data.py',
                                              'moe_train.py','moe_model.py','phistate.py')}
    identity = dict(format=FORMAT, checkpoint_sha256=sha(checkpoint), validation_sha256=val_hash,
                    tokenizer_sha256=tok_hash, prepared_fingerprint=fingerprint, source_sha256=code,
                    seed=SEED, limit_per_source=limit, device=str(device), torch_version=str(torch.__version__),
                    device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                    matmul_precision=torch.get_float32_matmul_precision())
    key = digest(canonical(identity))
    output.mkdir(parents=True, exist_ok=True)
    path = output/f'prompt_conditioning_{key}.json'
    if path.exists():
        cached = json.loads(path.read_text())
        saved_digest = cached.pop('report_sha256', None)
        if cached.get('identity') != identity or digest(canonical(cached)) != saved_digest:
            raise ValueError('Saved conditioning report failed identity/content verification')
        cached['report_sha256'] = saved_digest
        return cached
    pools = defaultdict(list)
    seen = set()
    for row in read_jsonl(val_path):
        if row['split'] != 'val' or row['source'] not in SOURCES or row['id'] in seen:
            raise ValueError('Invalid validation source, ID or split')
        seen.add(row['id'])
        pools[row['source']].append(row)
    if sha(val_path) != val_hash or set(pools) != set(SOURCES):
        raise ValueError('Prepared validation set changed or a source is missing')
    # Checkpoint construction initializes random weights before loading them.
    with torch.random.fork_rng(devices=[]):
        model, state = ckpt_load(str(checkpoint))
    if (sha(checkpoint) != identity['checkpoint_sha256'] or state.get('tokenizer_sha256') != tok_hash or
        state.get('sft_data_fingerprint') != fingerprint or state.get('data_fingerprint') != manifest['base_data_fingerprint'] or
        state.get('chat_template') != TEMPLATE):
        raise ValueError('Checkpoint, validation corpus, tokenizer, or chat template mismatch')
    step = state.get('sft_steps')
    del state
    tokenizer = spm.SentencePieceProcessor(model_file=str(tok_path))
    if tokenizer.vocab_size() != model.cfg.vocab or sha(tok_path) != tok_hash:
        raise ValueError('Checkpoint vocabulary or tokenizer mismatch')
    model.to(device).eval()
    previous_threads = torch.get_num_threads()
    try:
        if device.type == 'cpu':
            torch.set_num_threads(min(previous_threads, 2))
        per_source = {source: _source(model, pools[source], tokenizer, manifest['context'], device, limit)
                      for source in sorted(SOURCES)}
    finally:
        if device.type == 'cpu':
            torch.set_num_threads(previous_threads)
    complete = all(s['selected'] == s['scored'] for s in per_source.values())
    macro = {}
    for metric in next(iter(per_source.values()))['comparisons']:
        values = [s['comparisons'][metric] for s in per_source.values()]
        macro[metric] = statistics.mean(v['mean'] for v in values) if all(v is not None for v in values) else None
    report = dict(status='complete' if complete else 'partial_coverage', identity=identity,
                  sft_step=step, report_path=str(path.resolve()), per_source=per_source, macro=macro,
                  interpretation='Positive differences favor the true prompt. Each source uses a fixed seeded sample and complete, identical answer tokens including EOS. Wrong histories come from other same-source examples and may differ in length or incidental relevance. Infeasible pairings are omitted and reported without replacement. Means weight examples equally within each source; macro means weight sources equally. Sample SD and SE describe variability in this small sample. Shared donors and related tasks limit statistical inference. This teacher-forced diagnostic does not score generated answer correctness.')
    report['report_sha256'] = digest(canonical(report))
    # Separate temporary files allow concurrent readers to see only complete reports.
    with tempfile.NamedTemporaryFile(mode='w', dir=output, delete=False, encoding='utf-8') as stream:
        pending = Path(stream.name)
        try:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        except BaseException:
            pending.unlink(missing_ok=True)
            raise
    try:
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)
    return report
