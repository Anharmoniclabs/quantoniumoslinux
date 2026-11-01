"""Post hoc, same-seed, same-window ordinary-CE control for the changed-position CPU study."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def _load_control(path):
    # Isolated module instance; importing/running this script never changes the
    # production model or source, or the default weight in a normal import.
    spec = importlib.util.spec_from_file_location('_matched_ce_control', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load reference implementation')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.EXTRA_CE_WEIGHT != 3.:
        raise ValueError('The published treatment weight changed.')
    module.EXTRA_CE_WEIGHT = 0.
    return module


def run(tokenizer, treatment):
    source = Path(__file__).with_name('phistate_base_reference_curriculum.py')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if (treatment['source_sha256'] != source_hash or treatment['budget']['extra_ce'] != 3.
            or treatment['protocol'] != 'changed-position-CE-then-full-CE-v1'):
        raise ValueError('Treatment must match the published fixed procedure.')
    module = _load_control(source)
    result = module.run(tokenizer)
    if result['budget'] != {**treatment['budget'], 'extra_ce':0.}:
        raise ValueError('Unexpected difference in training settings.')
    for key in ('source_sha256','dense_module_sha256','tokenizer_sha256','split_seed','splits','thresholds'):
        if result[key] != treatment[key]:
            raise ValueError(f'Matched protocol invariant changed: {key}')
    changes=[]
    for standard,weighted in zip(result['runs'],treatment['runs'],strict=True):
        for key in ('seed','parameters','before','steps','emphasis_updates','all_token_targets'):
            if standard[key] != weighted[key]:
                raise ValueError(f'Matched run invariant changed: {key}')
        # These positions had a nonzero bonus in the treatment only.
        standard['bonus_eligible_positions'] = standard['extra_weighted_positions']
        standard['extra_weighted_positions'] = 0
        changes.append({'seed':standard['seed'],'control_accuracy':standard['after']['value_accuracy'],
                        'treatment_accuracy':weighted['after']['value_accuracy'],
                        'difference':weighted['after']['value_accuracy']-standard['after']['value_accuracy']})
    result.update(protocol='matched-ordinary-next-token-ce-v1',
                  ablation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  treatment_run_sha256=hashlib.sha256(json.dumps(treatment,sort_keys=True,separators=(',',':'),
                                                                  allow_nan=False).encode()).hexdigest(),
                  changes=changes,
                  scope='Post hoc ablation, identical code/weights at initialization, model, examples, ordering, '
                        'seeds, budget, optimizer and verification groups. Only the coefficient of the extra '
                        'value-position cross entropy differs. This cannot establish generalization beyond this '
                        'synthetic task or causal conclusions about the warm PhiState model.')
    return result


if __name__=='__main__':
    import sentencepiece as spm
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--treatment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.resolve() in {args.treatment.resolve(),args.tokenizer.resolve()}:
        p.error('Use a new output for the ablation.')
    rec=run(spm.SentencePieceProcessor(model_file=str(args.tokenizer)),json.loads(args.treatment.read_text()))
    args.output.write_text(json.dumps(rec,indent=2,allow_nan=False))
    print(json.dumps(rec['changes']))
