"""Compare broad assistant targets against a fixed 50% substitution of verified paired targets."""
import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil

import sentencepiece as spm
import torch

import coherence_quality
import paired_instruction_data
import data_comparison_eval
import data_comparison_train
from chat_recovery_data import SOURCES, TEMPLATE, sha, verify_prepared
from chat_recovery_train import atomic_json, load_rows, run_lock
from coherence_recovery import copy_verified
from device_preflight import run_preflight
from moe_train import ckpt_load

SEED = 20261031
STEPS = 256
TOKENS = 128
OUTPUT = 'paired_data_comparison_v1'


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def broad_stream(pools, seed):
    rng = random.Random(seed)
    names = list(SOURCES)
    weights = [SOURCES[name]['weight'] for name in names]
    order = {name: [] for name in names}
    while True:
        name = rng.choices(names, weights=weights, k=1)[0]
        if not order[name]:
            order[name] = list(range(len(pools[name])))
            rng.shuffle(order[name])
        yield pools[name][order[name].pop()]


def paired_stream(rows, seed):
    rng = random.Random(seed)
    families = defaultdict(lambda: defaultdict(list))
    for row in rows:
        families[row['family']][row['pair_id']].append(row)
    if not families or any(len(pair) != 2 for groups in families.values() for pair in groups.values()):
        raise ValueError('Expected complete training pairs')
    while True:
        decks = {name: list(groups) for name, groups in families.items()}
        for deck in decks.values():
            rng.shuffle(deck)
        while any(decks.values()):
            for name in sorted(decks):
                if decks[name]:
                    members = list(families[name][decks[name].pop()])
                    rng.shuffle(members)
                    yield from members


class TargetStream:
    """Take exactly n supervised positions, preserving full causal inputs and the next offset."""
    def __init__(self, rows):
        self.rows = iter(rows)
        self.positions = []
        self.cursor = 0

    def take(self, count, slot):
        fragments = []
        while count:
            if self.cursor == len(self.positions):
                self.row = next(self.rows)
                self.positions = [i for i, label in enumerate(self.row['labels']) if label != -100]
                if not self.positions:
                    raise ValueError('Empty supervised example')
                self.cursor = 0
            end = min(self.cursor + count, len(self.positions))
            chosen = self.positions[self.cursor:end]
            # Keep the full row so the paired trainer can verify next-token alignment.
            length = len(self.row['input_ids'])
            labels = [-100] * length
            for i in chosen:
                labels[i] = self.row['labels'][i]
            fragments.append(dict(id=f"{slot}/{self.row['id']}/{self.cursor}-{end}",
                                  origin_id=self.row['id'], source=self.row['source'], slot=slot,
                                  input_ids=self.row['input_ids'][:length], labels=labels))
            count -= len(chosen)
            self.cursor = end
        return fragments


def schedules(pools, paired_rows, steps, tokens):
    if tokens % 2 or min(steps, tokens) < 1:
        raise ValueError('Use an even answer-token budget per update')
    common = TargetStream(broad_stream(pools, SEED))
    extra = TargetStream(broad_stream(pools, SEED + 1))
    pairs = TargetStream(paired_stream(paired_rows, SEED + 2))
    result = {'broad': [], 'paired': []}
    for _ in range(steps):
        shared = common.take(tokens // 2, 'common')
        result['broad'].extend(shared + extra.take(tokens // 2, 'broad_substitution'))
        result['paired'].extend(shared + pairs.take(tokens // 2, 'paired_substitution'))
    return result


def describe(rows):
    targets, unique = Counter(), defaultdict(set)
    for row in rows:
        targets[row['slot']] += sum(t != -100 for t in row['labels'])
        unique[row['source']].add(row['origin_id'])
    return dict(targets_per_slot=dict(targets), unique_rows_per_source={s:len(ids) for s,ids in unique.items()},
                input_token_visits=sum(max(i for i,t in enumerate(row['labels']) if t != -100)+1 for row in rows), schedule_sha256=digest(rows))


def _fixed_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f'Immutable experiment settings differ: {path}')
    else:
        atomic_json(path, value)


def _release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(root, *, steps=STEPS, tokens_per_step=TOKENS, device='cuda'):
    root = Path(root).resolve()
    out = root / OUTPUT
    with run_lock(out):
        return _run(root, out, steps, tokens_per_step, device)


def _run(root, out, steps, tokens, device):
    if steps < 1 or steps > 512 or tokens < 2 or tokens > 512 or tokens % 2:
        raise ValueError('Use 1..512 updates and an even 2..512 targets per update')
    torch.set_num_threads(2)
    prior = root / 'coherence_execution_v1'
    previous = json.loads((prior/'report.json').read_text())
    parent = prior/'source.pt'
    data = root/'chat_recovery_v1/data'
    tokenizer_path = data/'tokenizer.model'
    parent_hash = sha(parent)
    if previous.get('status') != 'failed_quality' or parent_hash != previous['request']['source_sha256']:
        raise ValueError('Expected the frozen SFT source from the completed failed-quality optimizer comparison')
    # Abort if the earlier final suite was already used for model selection/testing.
    if (prior/'final_result.json').exists():
        raise RuntimeError('Earlier final evaluation exists; the declared fresh final suite is unavailable')
    sources = [Path(__file__), Path(paired_instruction_data.__file__), Path(data_comparison_train.__file__),
               Path(data_comparison_eval.__file__), Path(coherence_quality.__file__)]
    lab = Path(coherence_quality.__file__).resolve().parent
    sources.extend(lab / name for name in ('chat_recovery_train.py', 'chat_recovery_data.py', 'moe_train.py',
                                           'moe_model.py', 'phistate.py', 'device_preflight.py'))
    manifest = json.loads((data/'manifest.json').read_text())
    request = dict(format='PhiStateDataMixture-v1', source_sha256=parent_hash, steps=steps, tokens_per_step=tokens,
                   common_broad_targets_per_step=tokens//2, changed_targets_per_step=tokens//2,
                   seed=SEED, optimizer='original saved Adafactor', replay=0,
                   prepared_manifest_sha256=sha(data/'manifest.json'), tokenizer_sha256=sha(tokenizer_path),
                   previous_report_sha256=sha(prior/'report.json'),
                   code={p.name:sha(p) for p in sources}, torch_version=str(torch.__version__),
                   final_min_pair_fraction=2/3, final_min_pairs_per_family=1)
    if out.exists() and any(out.iterdir()) and not (out/'request.json').is_file():
        raise RuntimeError('Experiment directory contains files without a valid request')
    out.mkdir(exist_ok=True)
    _fixed_json(out/'request.json', request)
    report_path = out/'report.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report['request'] != request:
            raise RuntimeError('Archived report belongs to another experiment')
        if report['status'] in ('screen_failed', 'final_pairs_failed', 'paired_transfer_only', 'qualified'):
            print_summary(report)
            return report

    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a Colab GPU runtime before starting the data comparison')
    state = torch.load(parent,map_location='cpu',weights_only=True)
    prepared = verify_prepared(data, request['tokenizer_sha256'], manifest['context'])
    if (state.get('training_stage') != 'chat_recovery_v1' or state.get('sft_steps') != 4000 or
        state.get('tokenizer_sha256') != request['tokenizer_sha256'] or state.get('chat_template') != TEMPLATE or
        state.get('sft_data_fingerprint') != prepared['data_fingerprint'] or
        state.get('data_fingerprint') != prepared['base_data_fingerprint']):
        raise RuntimeError('SFT step, source, tokenizer or prepared-data lineage mismatch')
    del state
    preflight_path = out/'device_preflight.json'
    if not preflight_path.exists() and (prior/'device_preflight.json').is_file():
        # Cache is checked against device, software, source code, and actual model hashes.
        shutil.copyfile(prior/'device_preflight.json', preflight_path)
    print('Checking device and prepared rows.', flush=True)
    if not run_preflight(parent, tokenizer_path, preflight_path, device=device)['passed']:
        raise RuntimeError('CPU/device preflight failed; see '+str(preflight_path))
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    pools = load_rows(data, tokenizer, manifest['context'])
    suites = coherence_quality.build_suites(data, out/'broad_suites')
    broad_dev = coherence_quality.load_suite(suites['dev'])
    broad_final = coherence_quality.load_suite(suites['final'])
    excluded = [row['messages'][-2]['content'] for split in pools.values() for pool in split.values() for row in pool]
    excluded += [q['prompt'] for suite in (broad_dev,broad_final) for q in suite['items']]
    paired = paired_instruction_data.build(tokenizer, manifest['context'], excluded_prompts=excluded)
    plan = schedules(pools['train'],paired['train'],steps,tokens)
    summary = {arm:describe(rows) for arm,rows in plan.items()}
    _fixed_json(out/'data_plan.json',dict(arms=summary, paired_train_rows=len(paired['train']),
        dev_pairs_sha256=digest(paired['dev']),final_pairs_sha256=digest(paired['final']),
        broad_dev=broad_dev['fingerprint'],broad_final=broad_final['fingerprint']))
    print(f'Each arm: {steps} updates × {tokens} answer targets = {steps*tokens:,}; common broad targets: {steps*tokens//2:,}.',flush=True)
    stop = out/'STOP'
    stop.unlink(missing_ok=True)  # Calling run again explicitly resumes this fixed request.
    def evaluate(model):
        return data_comparison_eval.evaluate_dev(model, tokenizer, paired['dev'], broad_dev, pools['val'])
    arms = {}
    for name in ('broad','paired'):
        if stop.exists():
            return {'status':'stopped','run_dir':str(out)}
        print('Starting/resuming',name,'arm',flush=True)
        info = data_comparison_train.run_arm(parent=parent, tokenizer=tokenizer, pool=plan[name], output=out/name,
            policy_name=name, device=device, steps=steps, tokens_per_step=tokens, seed=SEED,
            evaluate_callback=evaluate, identity=dict(request=request, schedule=summary[name]['schedule_sha256'],
            paired_dev=digest(paired['dev']),broad_dev=broad_dev['fingerprint']), stop_path=stop)
        if info['status'] == 'stopped':
            return {'status':'stopped','arm':name,'run_dir':str(out)}
        if info['status'] != 'complete' or info['phase_step']!=steps or info['supervised_tokens']!=steps*tokens:
            raise RuntimeError('Arm ended without completing the fixed token budget')
        arms[name]=info
        _release()
    if sha(parent) != parent_hash:
        raise RuntimeError('Source checkpoint changed externally during the run')
    get_metrics=lambda name, index: arms[name]['development_curve'][index]['metrics']
    baseline,control,candidate=get_metrics('broad',0),get_metrics('broad',-1),get_metrics('paired',-1)
    other_base=get_metrics('paired',0)
    if (baseline['paired']['rows']!=other_base['paired']['rows'] or
        baseline['broad']['rows']!=other_base['broad']['rows']):
        raise RuntimeError('Starting models produced different baseline answers; comparison is inconclusive')
    screen=data_comparison_eval.screen_candidate(baseline,control,candidate)
    report=dict(request=request, data_plan=summary, arms=arms, screening=screen,
                source_unchanged=True, status='screen_failed', native_export=None)
    if screen['eligible']:
        latest=Path(arms['paired']['latest'])
        identity=dict(checkpoint=sha(latest),pairs=digest(paired['final']), broad=broad_final['fingerprint'],
                      broad_dev_qualified=candidate['broad']['qualified'])
        final_path=out/'final_result.json'
        if final_path.exists():
            final=json.loads(final_path.read_text())
            if final['identity']!=identity:
                raise RuntimeError('Final results exist for another selected checkpoint')
        else:
            model,state=ckpt_load(str(latest))
            del state
            model.to(device).eval()
            pair_result=data_comparison_eval.evaluate_pairs(model,tokenizer,paired['final'])
            passed=(pair_result['pair_rate']>=2/3 and
                    all(b['pairs_passed']>=1 for b in pair_result['buckets'].values()))
            final=dict(identity=identity,paired=pair_result,paired_passed=passed,broad=None)
            if passed and candidate['broad']['qualified']:
                final['broad']=coherence_quality.evaluate_suite(model,tokenizer,broad_final)
            atomic_json(final_path,final)
            del model
            _release()
        report['final']=final
        report['status']='paired_transfer_only' if final['paired_passed'] else 'final_pairs_failed'
        if final['broad'] and final['broad']['qualified']:
            from chat_recovery_cpp import export
            native=out/'qualified_native'
            native.mkdir(exist_ok=True)
            copy_verified(latest,native/'best.pt',identity['checkpoint'])
            report['native_export']=str(export(native,data))
            report['status']='qualified'
    atomic_json(report_path,report)
    print_summary(report)
    return report


def print_summary(report):
    print('DATA COMPARISON', report['status'], flush=True)
    for arm,info in report['arms'].items():
        metrics=info['development_curve'][-1]['metrics']
        pair=metrics['paired']
        broad=metrics['broad']['buckets']
        print(f"{arm}: {info['supervised_tokens']:,} answer tokens | both halves {pair['pairs_passed']}/{pair['pairs_total']} "
              f"| broad tasks {sum(b['passed'] for b in broad.values())}/{sum(b['total'] for b in broad.values())} "
              f"| answer BPT {metrics['answer_validation']['val_answer_bpt']:.4f}", flush=True)
        for item in pair['rows'][0]['items']:
            print('  ',item['prompt'],'→',repr(item['text'])[:180],flush=True)
    if 'final' in report:
        final=report['final']['paired']
        print('Fresh final pairs:',final['pairs_passed'],'/',final['pairs_total'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    args=parser.parse_args()
    result=run(args.root)
    print('Saved experiment:',Path(args.root)/OUTPUT,'|',result['status'],flush=True)
