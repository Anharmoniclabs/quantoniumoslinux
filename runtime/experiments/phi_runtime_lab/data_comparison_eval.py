"""Matched development measurements for the bounded paired-data experiment."""
from collections import Counter
import hashlib
import json
import math

from chat_recovery_data import SOURCES
from chat_recovery_train import evaluate as evaluate_answers
from coherence_quality import evaluate_suite, generate_answer, grade_answer

RETENTION_PER_SOURCE = 8
MIN_EXTRA_PAIRS = 2
MAX_BPT_REGRESSION = 0.5


def _fingerprint(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def evaluate_pairs(model, tokenizer, probes):
    """Generate from a fresh prompt for each item. A pair passes if both items pass."""
    probes = list(probes)
    if not probes or any(len(pair['items']) != 2 for pair in probes):
        raise ValueError('Expected one or more pairs of exactly two items.')
    pair_ids = [pair['pair_id'] for pair in probes]
    item_ids = [item['id'] for pair in probes for item in pair['items']]
    if len(set(pair_ids)) != len(pair_ids) or len(set(item_ids)) != len(item_ids):
        raise ValueError('Duplicate pair or item identifiers.')
    rows = []
    pair_total, pair_wins, item_total, item_wins = (Counter() for _ in range(4))
    for pair in probes:
        answers = []
        for item in pair['items']:
            answer = generate_answer(model, tokenizer, item['prompt'], max_tokens=item['max_tokens'])
            grade = grade_answer(item, answer['text'])
            bucket = item['bucket']
            item_total[bucket] += 1
            item_wins[bucket] += int(grade['passed'])
            answers.append(dict(id=item['id'], prompt=item['prompt'], bucket=bucket, **answer, **grade))
        bucket = answers[0]['bucket'] if answers[0]['bucket'] == answers[1]['bucket'] else 'mixed'
        passed = all(answer['passed'] for answer in answers)
        pair_total[bucket] += 1
        pair_wins[bucket] += int(passed)
        rows.append(dict(pair_id=pair['pair_id'], bucket=bucket, passed=passed, items=answers))
    buckets = {bucket: dict(pairs_passed=pair_wins[bucket], pairs_total=pair_total[bucket],
                            items_passed=item_wins[bucket], items_total=item_total[bucket])
               for bucket in sorted(set(item_total) | set(pair_total))}
    return dict(pairs_passed=sum(pair_wins.values()), pairs_total=len(rows),
                pair_rate=sum(pair_wins.values())/len(rows), items_passed=sum(item_wins.values()),
                items_total=len(item_ids), item_rate=sum(item_wins.values())/len(item_ids),
                buckets=buckets, rows=rows, suite_fingerprint=_fingerprint(probes),
                decoding=dict(temperature=0., repetition_penalty=1.))


def evaluate_dev(model, tokenizer, probes, broad_suite, heldout_rows=None):
    """Score distinct paired, broad-transfer and answer-retention measurements.

    heldout_rows is the prepared validation pool keyed by source, as received by
    chat_recovery_train.evaluate. Its first eight sorted rows per source are fixed.
    The caller controls the split and calls this on final probes only after selection.
    """
    was_training = model.training
    try:
        paired = evaluate_pairs(model, tokenizer, probes)
        broad = evaluate_suite(model, tokenizer, broad_suite)
        retention = None
        if heldout_rows is not None:
            selected = {name: sorted(heldout_rows[name], key=lambda row: row['id'])[:RETENTION_PER_SOURCE]
                        for name in SOURCES}
            if any(len(rows) != RETENTION_PER_SOURCE for rows in selected.values()):
                raise ValueError('Need eight complete held-out examples per source.')
            retention = evaluate_answers(model, selected, RETENTION_PER_SOURCE)
            # Bind the same selected answer tokens and masks across all models.
            signatures = {name: [(r['id'], r['input_ids'], r['labels']) for r in rows]
                          for name, rows in selected.items()}
            retention['rows_fingerprint'] = _fingerprint(signatures)
        return dict(paired=paired, broad=broad, answer_validation=retention)
    finally:
        model.train(was_training)


def screen_candidate(baseline, control, paired):
    """Apply the fixed development screen once to the paired arm's endpoint."""
    reports = [baseline, control, paired]
    for name in ('paired', 'broad'):
        fingerprints = [r[name]['suite_fingerprint'] for r in reports]
        if not fingerprints[0] or len(set(fingerprints)) != 1:
            raise ValueError(f'Cannot compare different {name} probe sets.')
    if any(r['broad']['split'] != 'dev' for r in reports):
        raise ValueError('Candidate selection requires the development split.')
    pair_scores = [r['paired']['pairs_passed'] for r in reports]
    broad_scores = [sum(b['passed'] for b in r['broad']['buckets'].values()) for r in reports]
    retention = [r.get('answer_validation') for r in reports]
    bpts = []
    if all(retention):
        identifiers = [entry['rows_fingerprint'] for entry in retention]
        if not identifiers[0] or len(set(identifiers)) != 1:
            raise ValueError('Cannot compare different held-out answer rows.')
        bpts = [float(entry['val_answer_bpt']) for entry in retention]
    checks = dict(paired_improves_over_baseline_and_control=pair_scores[2] >= max(pair_scores[:2]) + MIN_EXTRA_PAIRS,
                  broad_pass_count_retained=broad_scores[2] >= max(broad_scores[:2]),
                  heldout_answer_bpt_retained=bool(bpts) and all(math.isfinite(n) for n in bpts)
                  and bpts[2] <= min(bpts[:2]) + MAX_BPT_REGRESSION)
    return dict(eligible=all(checks.values()), checks=checks, reasons=[k for k, v in checks.items() if not v],
                thresholds={'min_additional_complete_pairs_over_each_reference': MIN_EXTRA_PAIRS,
                            'max_broad_pass_count_decline': 0, 'max_answer_bpt_regression': MAX_BPT_REGRESSION},
                scope='Fixed development screening criterion for this experiment.')
