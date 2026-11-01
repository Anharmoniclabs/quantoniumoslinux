"""Independent preregistered CPU delayed-value experiment with changed-position supervision."""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from phistate_base_reference import DenseCausalLM, _probe, _symbols

SEEDS = (20261113, 20261114)
SPLIT_SEED = 99131
STEPS = 768
EMPHASIS_STEPS = 640
EXTRA_CE_WEIGHT = 3.
BATCH = 32


def make_data(tok):
    keys = _symbols(tok, 'IJKLMNOPQRSTUVWXYZ')
    distractions = _symbols(tok, 'QRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
    values = _symbols(tok, '0123456789')
    if set(keys) & set(distractions) or (set(keys) | set(distractions)) & set(values):
        raise ValueError('The three symbol alphabets must be disjoint')
    # Both markers and both non-digit alphabets differ from the earlier '=' ... ':'
    # and A-H protocol. The complete verify windows never occur in earlier experiments.
    start, end = tok.encode('~'), tok.encode('?')
    if not start or not end or tok.eos_id() < 0 or start == tok.encode('=') or end == tok.encode(':'):
        raise ValueError('Distinct start/end markers and EOS are required.')
    groups = [(k, a, b) for k in keys for a in distractions for b in distractions]
    random.Random(SPLIT_SEED).shuffle(groups)
    splits = {'train':groups[:448], 'verify':groups[448:]}
    return {name:torch.tensor([[*start, key, value, a, b, *end, value, tok.eos_id()]
                               for key, a, b in triples for value in values])
            for name, triples in splits.items()}


def run(tok):
    pools = make_data(tok)
    old_threads = torch.get_num_threads()
    runs = []
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            for seed in SEEDS:
                torch.manual_seed(seed)
                rng = torch.Generator().manual_seed(seed+1)
                model = DenseCausalLM(tok.vocab_size(),width=48,layers=2,heads=4,context=32,hidden=96)
                optimizer = torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.01)
                before = _probe(model,pools['verify'])
                model.train()
                for step in range(1, STEPS+1):
                    rows = pools['train'][torch.randint(len(pools['train']), (BATCH,), generator=rng)]
                    x,y = rows[:, :-1], rows[:, 1:]
                    logits = model(x)
                    objective = F.cross_entropy(logits.flatten(0,1),y.flatten())
                    if step <= EMPHASIS_STEPS:
                        # Same token and same full-vocabulary logits as ordinary causal LM CE.
                        # The model sees only x. No parsed key, value or query index is an input.
                        objective = objective + EXTRA_CE_WEIGHT*F.cross_entropy(logits[:, -2],y[:, -2])
                    optimizer.zero_grad(set_to_none=True)
                    objective.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step()
                after = _probe(model,pools['verify'])
                seen = _probe(model,pools['train'])
                passed = (after['value_accuracy'] >= .9 and after['accuracy_after_changing_only_earlier_value'] <= .2
                          and after['bpt'] < before['bpt'] and seen['value_accuracy'] >= .9)
                runs.append(dict(seed=seed,steps=STEPS,emphasis_updates=EMPHASIS_STEPS,
                          parameters=model.parameter_count(), before=before, after=after, training=seen, passed=passed,
                          all_token_targets=STEPS*BATCH*(pools['train'].shape[1]-1),
                          extra_weighted_positions=EMPHASIS_STEPS*BATCH))
                print(f"CPU seed {seed}: held-out {after['correct_generated_value']}/{after['windows']}, "
                      f"swapped {after['accuracy_after_changing_only_earlier_value']:.3f}, train {seen['value_accuracy']:.3f}",flush=True)
    finally:
        torch.set_num_threads(old_threads)
    return dict(protocol='changed-position-CE-then-full-CE-v1',qualified=all(r['passed'] for r in runs),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                dense_module_sha256=hashlib.sha256(Path(__file__).with_name('phistate_base_reference.py').read_bytes()).hexdigest(),
                tokenizer_sha256=hashlib.sha256(tok.serialized_model_proto()).hexdigest(),
                split_seed=SPLIT_SEED,splits={name:{'windows':len(t),'sha256':hashlib.sha256(t.numpy().tobytes()).hexdigest()}
                                               for name,t in pools.items()},
                budget=dict(seeds=list(SEEDS),updates=STEPS,extra_ce=EXTRA_CE_WEIGHT,emphasis_updates=EMPHASIS_STEPS,
                            batch=BATCH,optimizer='AdamW',lr=.0015,weight_decay=.01,clip=1.),
                thresholds={'verify_value_accuracy':.90,'maximum_wrong_prefix_accuracy':.20,
                            'train_value_accuracy':.90, 'all_token_bpt_must_drop':True},
                scope='Narrow synthetic prerequisite with all vocabulary logits and raw repaired token IDs. '
                      'These latest two runs train a changed objective and use new templates, distractors, seeds and '
                      'held-out examples. They do not reverse the earlier failed ordinary-CE results, establish natural '
                      'instruction-following, or establish a PhiState architecture effect.',runs=runs)


if __name__ == '__main__':
    import sentencepiece as spm
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run(spm.SentencePieceProcessor(model_file=str(args.tokenizer)))
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
