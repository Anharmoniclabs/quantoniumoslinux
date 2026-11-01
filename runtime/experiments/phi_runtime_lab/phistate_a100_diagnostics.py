"""Read-only checkpoint diagnostics on CPU or CUDA, before training a candidate."""
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import tempfile

import sentencepiece as spm
import torch
import torch.nn.functional as F

from chat_recovery_data import SOURCES, TEMPLATE, canonical, digest, read_jsonl, sha
from device_preflight import _identity as device_identity
from moe_train import ckpt_load
from phistate import GoldenState
import prompt_conditioning_eval as conditioning

FORMAT = 'phistate-training-readiness-v1'


def _class(name):
    leaf = name.rsplit('.', 1)[-1]
    if name == 'embedding.weight': return 'tied_embedding_output'
    if leaf in ('decay_raw', 'phase_delta', 'read_re', 'read_im'): return 'recurrent_'+leaf
    if '.router.' in name: return 'router'
    if '.experts.' in name: return 'routed_experts'
    if '.shared.' in name: return 'shared_expert'
    if '.attn.' in name: return 'attention_bias' if leaf == 'bias' else 'attention_projection'
    if '.mixer.' in name: return 'channel_mixer'
    if 'norm' in name: return 'normalization'
    if '.memory.' in name: return 'recurrent_projection'
    return 'other'


def _norm2(tensor):
    result = float(torch.linalg.vector_norm(tensor.detach().float())) ** 2
    if not math.isfinite(result):
        raise FloatingPointError('Nonfinite parameter, gradient or parent comparison')
    return result


def _parameter_stats(model, parent=None):
    sums = defaultdict(lambda: defaultdict(float))
    for name, param in model.named_parameters():
        group = sums[_class(name)]
        group['elements'] += param.numel()
        group['tensors'] += 1
        group['weight'] += _norm2(param)
        if param.grad is None:
            group['missing_gradient_tensors'] += 1
        else:
            squared = _norm2(param.grad)
            group['gradient'] += squared
            group['nonzero_gradient_tensors'] += int(squared > 0)
        if parent is not None:
            old = parent[name]
            group['parent'] += _norm2(old)
            group['displacement'] += _norm2(param.detach().cpu() - old)
    report = {}
    for name, values in sorted(sums.items()):
        n = values['elements']
        report[name] = dict(elements=int(n), tensors=int(values['tensors']),
                            weight_rms=math.sqrt(values['weight']/n),
                            gradient_rms=math.sqrt(values['gradient']/n),
                            grad_to_weight_rms=math.sqrt(values['gradient']/values['weight']) if values['weight'] else None,
                            missing_gradient_tensors=int(values['missing_gradient_tensors']),
                            nonzero_gradient_tensors=int(values['nonzero_gradient_tensors']))
        if parent is not None:
            report[name]['parent_to_current_net_delta_rms'] = math.sqrt(values['displacement']/n)
            report[name]['net_delta_to_parent_weight_rms'] = math.sqrt(values['displacement']/values['parent']) if values['parent'] else None
    return report, math.sqrt(sum(v['gradient'] for v in sums.values()))


def _halflives(model):
    result = {}
    for name, layer in model.named_modules():
        if isinstance(layer, GoldenState):
            decay = F.softplus(layer.decay_raw.detach().double().cpu()) + 1e-4
            half = math.log(2)/decay
            if not torch.isfinite(half).all(): raise FloatingPointError('Nonfinite recurrent decay')
            quantiles = torch.quantile(half, torch.tensor([0., .25, .5, .75, 1.], dtype=torch.float64)).tolist()
            result[name] = dict(zip(('min','p25','median','p75','max'), quantiles))
    return result


def _gradient_probe(model, pools, tokenizer, context, device, parent=None):
    model.eval()
    model.zero_grad(set_to_none=True)
    selected = {}
    routed = [torch.zeros(model.cfg.experts, dtype=torch.float64) for _ in model.blocks]
    for source, pool in sorted(pools.items()):
        row = min(pool, key=lambda r: digest(f'phistate-gradient-probe-v1|{source}|{r["id"]}'))
        item = conditioning._targets(row, tokenizer, context)
        x = torch.tensor([row['input_ids']], device=device)
        y = torch.tensor(row['labels'], device=device)
        logits = model(x)
        lm = F.cross_entropy(logits.flatten(0, 1), y, ignore_index=-100)
        penalty, _ = model.router_penalty()
        total = lm + penalty
        if not torch.isfinite(total): raise FloatingPointError('Nonfinite validation probe objective')
        (total/len(pools)).backward()
        selected[source] = dict(id=row['id'], input_tokens=x.numel(), answer_tokens=len(item['targets']),
                                answer_bpt=float(lm.detach())/math.log(2), router_penalty=float(penalty.detach()))
        for i, layer in enumerate(model.blocks):
            routed[i] += layer.ff.last_routing.detach().double().cpu() * x.numel()
    groups, grad_norm = _parameter_stats(model, parent)
    shares = []
    for counts in routed:
        p = counts/counts.sum(); positives = p[p>0]
        shares.append(dict(assignment_share=p.tolist(), zero_selected=int((p==0).sum()),
                           normalized_entropy=float(-(positives*positives.log()).sum()/math.log(len(p)))))
    model.zero_grad(set_to_none=True)
    return dict(rows=selected, global_gradient_l2=grad_norm,
                implied_clipping_multiplier_at_norm_one=min(1.,1./(grad_norm+1e-6)),
                by_parameter_class=groups, routing_by_layer=shares)


def _read_conditioning(folder, required):
    """Use only exact checkpoint, corpus, source version, seed and sample size matches."""
    if folder is None: return None
    for path in sorted(Path(folder).glob('prompt_conditioning_*.json')):
        try:
            report = json.loads(path.read_text())
            content_hash = report.pop('report_sha256')
            identity = report['identity']
            if (content_hash == digest(canonical(report)) and report.get('status') == 'complete'
                and all(identity.get(k) == v for k,v in required.items())):
                return dict(source='verified_existing_report', path=str(path), report_sha256=content_hash,
                            measured_device=identity.get('device'), status=report['status'], macro=report['macro'],
                            sources={s:dict(scored=v['scored'], comparisons=v['comparisons'])
                                     for s,v in report['per_source'].items()})
        except (ValueError, KeyError, TypeError):
            continue
    return None


def _measure_conditioning(model, pools, tokenizer, context, device, limit):
    sources = {}
    for s, pool in sorted(pools.items()):
        r = conditioning._source(model, pool, tokenizer, context, device, limit)
        sources[s] = {k:r[k] for k in ('selected','scored','unpaired','comparisons')}
    macro = {}
    for name in next(iter(sources.values()))['comparisons']:
        summaries = [v['comparisons'][name] for v in sources.values()]
        macro[name] = sum(x['mean'] for x in summaries)/len(sources) if all(summaries) else None
    complete = all(v['selected']==v['scored'] for v in sources.values())
    return dict(source='measured_on_loaded_weights', measured_device=str(device),
                status='complete' if complete else 'partial_coverage', macro=macro, sources=sources)


def _publish(report, path):
    report['report_sha256'] = digest(canonical(report))
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        pending = Path(f.name)
        try:
            json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False)
            f.write('\n')
        except BaseException:
            pending.unlink(missing_ok=True)
            raise
    try: os.replace(pending,path)
    finally: pending.unlink(missing_ok=True)


def run(checkpoint, sft_dir, output, device='cpu', limit=16, conditioning_dir=None, comparison_checkpoint=None):
    """Return a hash-bound report without changing parameters or advancing an optimizer.

    `output` and optional `conditioning_dir` are directories. `comparison_checkpoint`
    must be the exact direct parent recorded by this SFT checkpoint. Its net weight
    displacement combines all intervening updates and is not a per-step update.
    """
    if type(limit) is not int or not 2 <= limit <= 128: raise ValueError('limit must be in 2..128')
    checkpoint, sft_dir, output = map(Path,(checkpoint,sft_dir,output))
    device = torch.device(device)
    if device.type not in ('cpu','cuda') or (device.type=='cuda' and not torch.cuda.is_available()):
        raise ValueError('An available CPU or CUDA device is required')
    if device.type=='cuda' and device.index is None: device=torch.device('cuda',torch.cuda.current_device())
    val = sft_dir/'val.jsonl'; tok = sft_dir/'tokenizer.model'; lab = Path(__file__).resolve().parent
    manifest=json.loads((sft_dir/'manifest.json').read_text()); fingerprint=manifest.pop('data_fingerprint')
    val_hash, tok_hash=sha(val),sha(tok)
    if (digest(canonical(manifest))!=fingerprint or manifest['files']['val.jsonl']!=val_hash
        or manifest['tokenizer_sha256']!=tok_hash or manifest['template']!=TEMPLATE):
        raise ValueError('Prepared validation lineage mismatch')
    identity=device_identity(checkpoint,tok,device)
    source=identity['source_sha256']
    for name in ('phistate_a100_diagnostics.py','prompt_conditioning_eval.py','moe_train.py','chat_recovery_data.py'):
        source[name]=sha(lab/name)
    identity.update(format=FORMAT, limit_per_source=limit, prepared_fingerprint=fingerprint,
                    context=manifest['context'], validation_sha256=val_hash,
                    comparison_sha256=sha(comparison_checkpoint) if comparison_checkpoint is not None else None)
    required=dict(format=conditioning.FORMAT, checkpoint_sha256=identity['checkpoint_sha256'],
                  tokenizer_sha256=tok_hash, validation_sha256=val_hash, prepared_fingerprint=fingerprint,
                  source_sha256={n:source[n] for n in ('prompt_conditioning_eval.py','chat_recovery_data.py','moe_train.py','moe_model.py','phistate.py')},
                  seed=conditioning.SEED, limit_per_source=limit)
    previous=_read_conditioning(conditioning_dir,required)
    identity['reused_conditioning_sha256']=previous['report_sha256'] if previous else None
    output.mkdir(parents=True,exist_ok=True)
    path=output/f'training_readiness_{digest(canonical(identity))}.json'
    if path.is_file():
        cached=json.loads(path.read_text()); check=cached.pop('report_sha256',None)
        if cached.get('identity')!=identity or digest(canonical(cached))!=check:
            raise ValueError('Saved training readiness report failed content verification')
        return {**cached,'report_sha256':check}
    pools=defaultdict(list)
    for row in read_jsonl(val):
        if row['split']!='val' or row['source'] not in SOURCES: raise ValueError('Invalid validation row')
        pools[row['source']].append(row)
    if set(pools)!=set(SOURCES) or sha(val)!=val_hash: raise ValueError('Incomplete or changed validation set')
    tokenizer=spm.SentencePieceProcessor(model_file=str(tok))
    threads=torch.get_num_threads(); tf32=torch.backends.cuda.matmul.allow_tf32; cudnn=torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        if device.type=='cpu': torch.set_num_threads(min(2,threads))
        with torch.inference_mode(False), torch.enable_grad(), torch.random.fork_rng(devices=[device.index] if device.type=='cuda' else []):
            model,state=ckpt_load(str(checkpoint))
            if (state.get('sft_data_fingerprint')!=fingerprint or state.get('data_fingerprint')!=manifest['base_data_fingerprint']
                or state.get('tokenizer_sha256')!=tok_hash or state.get('chat_template')!=TEMPLATE
                or tokenizer.vocab_size()!=model.cfg.vocab or sha(tok)!=tok_hash):
                raise ValueError('Checkpoint does not match the prepared SFT tokenizer/corpus')
            parent=None
            if comparison_checkpoint is not None:
                if identity['comparison_sha256']!=state.get('source_sha256'):
                    raise ValueError('Comparison is not the recorded direct parent of this checkpoint')
                old=torch.load(comparison_checkpoint,map_location='cpu',weights_only=True)
                if (old['config']!=state['config'] or old.get('data_fingerprint')!=state.get('data_fingerprint')
                    or old.get('step',0)>=state.get('step',0) or sha(comparison_checkpoint)!=identity['comparison_sha256']):
                    raise ValueError('Direct parent config, step or corpus lineage differs')
                parent=old['state_dict']; del old
            optimizer_groups=[dict(lr=float(g['lr']),weight_decay=float(g.get('weight_decay',0)),parameter_tensors=len(g['params']))
                              for g in state.get('optimizer_state',{}).get('param_groups',[])]
            step=state.get('sft_steps'); del state
            model.to(device).eval()
            grad=_gradient_probe(model,pools,tokenizer,manifest['context'],device,parent)
            half=_halflives(model)
            probe=previous or _measure_conditioning(model,pools,tokenizer,manifest['context'],device,limit)
    finally:
        torch.backends.cuda.matmul.allow_tf32=tf32; torch.backends.cudnn.allow_tf32=cudnn
        if device.type=='cpu': torch.set_num_threads(threads)
    if sha(checkpoint)!=identity['checkpoint_sha256']: raise ValueError('Checkpoint changed during diagnostics')
    report=dict(status='complete',identity=identity,report_path=str(path.resolve()),sft_step=step,
                checkpoint_unchanged=True, optimizer_steps_taken=0, saved_optimizer_groups=optimizer_groups,
                validation_gradient_probe=grad, recurrent_half_life_tokens=half, prompt_conditioning=probe,
                scope='Gradient RMS and routing come from one fixed held-out row per source, weighted equally including the router penalty. Gradient/weight RMS measures sensitivity on this probe. It is not an optimizer update. Optional parent displacement is the measured net change across snapshots. Recurrent half-life describes free decay in each diagonal channel, not whole-model recall. Completion is not a quality pass or a recommendation to extend training.')
    _publish(report,path)
    return report
