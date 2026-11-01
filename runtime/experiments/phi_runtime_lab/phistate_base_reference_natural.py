"""Cheap, read-only-data natural next-token learning check for the conventional reference on CPU."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from phistate_base_reference import DenseCausalLM

TOKENIZER = '9f704ff69de6cb60fd7466a26b70cbd0472bb1a192f1825a7c52cfc30179ea04'
REPAIRED = '5a45ae09dfc42c6021202838439e77bd5cb05ff38d673d23e3219a1ecbc6c1d3'
CONTEXT = 48
STEPS = 160
BATCH = 4
VAL_COUNT = 48


def _filehash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _open(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest.get('data_fingerprint') != REPAIRED or manifest['dtype'] != 'uint16' or manifest['vocab'] != 8192:
        raise ValueError('Unexpected repaired production corpus')
    if _filehash(directory/'tokenizer.model') != TOKENIZER:
        raise ValueError('Unexpected tokenizer')
    arrays, hashes = {}, {}
    for split in ('train', 'val'):
        for domain in ('code', 'general'):
            name = f'{split}_{domain}'
            spec = manifest['bins'][name]
            p = directory/spec['file']
            hashes[name] = _filehash(p)
            if hashes[name] != spec['sha256'] or 2*spec['tokens'] != p.stat().st_size:
                raise ValueError(f'Changed input {name}')
            arrays[name] = np.memmap(p, '<u2', mode='r')
    return arrays, hashes


def _read_windows(array, offsets):
    return torch.from_numpy(np.stack([array[i:i+CONTEXT+1] for i in offsets]).astype('int64'))


@torch.inference_mode()
def evaluate(model, val):
    model.eval()
    summary = {}
    for name, windows in val.items():
        total = true = altered = 0.
        for rows in windows.split(8):
            x, y = rows[:, :-1], rows[:, 1:]
            logits = model(x)
            total += float(F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction='sum'))
            true += float(F.cross_entropy(logits[:, -1], y[:, -1], reduction='sum'))
            # Keep the last input ID and the target. Earlier positions come from
            # the next fixed validation window of this domain, cyclically in each batch.
            modified = torch.cat((x[:, :-1].roll(1, dims=0), x[:, -1:]), dim=-1)
            altered += float(F.cross_entropy(model(modified)[:, -1], y[:, -1], reduction='sum'))
        n = len(windows)
        summary[name] = {'target_count':n*CONTEXT,'bpt':total/(n*CONTEXT*math.log(2)),
                         'full_prefix_last_bpt':true/(n*math.log(2)),
                         'swapped_prefix_last_bpt':altered/(n*math.log(2)),
                         'prefix_gain_last_bpt':(altered-true)/(n*math.log(2))}
    return summary


def run(directory):
    directory=Path(directory)
    arrays, hashes = _open(directory)
    val_offsets, val = {}, {}
    for n, domain in enumerate(('code','general')):
        stream = arrays['val_'+domain]
        # One seeded window within each disjoint free interval, spread over the entire validation bin.
        edges = np.linspace(0, len(stream), VAL_COUNT+1, dtype=np.int64)
        rng = np.random.default_rng(91117+n)
        offsets = [int(rng.integers(a, b-CONTEXT)) for a,b in zip(edges, edges[1:])]
        val_offsets[domain] = offsets
        val[domain] = _read_windows(stream, offsets)
    runs=[]
    previous_threads=torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            for seed in (20261111,20261112):
                torch.manual_seed(seed)
                rng=np.random.default_rng(seed)
                model=DenseCausalLM(8192,width=24,layers=2,heads=3,context=CONTEXT,hidden=48)
                opt=torch.optim.AdamW(model.parameters(),lr=.0025,weight_decay=.01)
                before=evaluate(model,val)
                schedule=[]
                model.train()
                for _ in range(STEPS):
                    # Orthogonal, independent train/val corpus memberships from the repaired manifest.
                    domain = 'code' if rng.random()<.55 else 'general'
                    a=arrays['train_'+domain]
                    offsets=rng.integers(0,len(a)-CONTEXT,size=BATCH).tolist()
                    schedule.append([domain,offsets])
                    rows=_read_windows(a,offsets)
                    logits=model(rows[:,:-1])
                    loss=F.cross_entropy(logits.flatten(0,1),rows[:,1:].flatten())
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    opt.step()
                after=evaluate(model,val)
                gains={d:before[d]['bpt']-after[d]['bpt'] for d in val}
                runs.append({'seed':seed,'parameters':model.parameter_count(),'updates':STEPS,
                             'training_target_count':STEPS*BATCH*CONTEXT,'validation_target_count_per_domain':VAL_COUNT*CONTEXT,
                             'training_schedule_sha256':hashlib.sha256(json.dumps(schedule,separators=(',',':')).encode()).hexdigest(),
                             'before':before,'after':after,'domain_bpt_improvement':gains,'passed':min(gains.values())>=.25})
    finally:
        torch.set_num_threads(previous_threads)
    return {'format':'PhiStateDenseCPUNatural-v1','qualified':all(r['passed'] for r in runs),
            'runs':runs,'criterion':'At least 0.25 held-out BPT improvement in each domain for each of the two seeds.',
            'validation_offsets':val_offsets,'corpus_fingerprint':REPAIRED,'tokenizer_sha256':TOKENIZER,'file_sha256':hashes,
            'module_sha256':_filehash(Path(__file__)),
            'scope':'Tiny untrained dense models; train and validation windows use the original repaired split. '
                    'Prefix gain is a diagnostic on 48 final targets/domain, not a pass gate. No test bin, user checkpoint, '
                    'A100 allocation or general assistant score.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=run(a.corpus)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(result['qualified'],[(r['seed'],r['domain_bpt_improvement']) for r in result['runs']])
