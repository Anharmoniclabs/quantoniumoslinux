"""Offline fixtures, independent patch grading, and strict paired comparisons.

No model calls, dependencies, downloads, or repository mutations.
Generated candidate code executes: use an OS sandbox for untrusted submissions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from fixtures import TASKS

VERSION = 'nessa-smoke-v1'
FAIR_FIELDS = ('suite_sha256', 'model_weights_sha256', 'tokenizer_sha256',
               'quantization', 'inference_engine', 'inference_version', 'hardware',
               'temperature', 'seed', 'max_steps', 'max_output_tokens_per_call',
               'max_total_tokens', 'wall_seconds', 'workers', 'attempts_per_task')


def suite_hash():
    return hashlib.sha256(json.dumps(TASKS, sort_keys=True).encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prepare(dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)  # never silently replace a run
    rows = []
    for task_id, task in TASKS.items():
        snap = dest / 'snapshots' / task_id
        snap.mkdir(parents=True)
        (snap / 'solution.py').write_text(task['source'])
        (snap / 'README.md').write_text(task['prompt'] + '\n')
        rows.append({'instance_id': task_id, 'problem_statement': task['prompt'],
                     'snapshot': f'snapshots/{task_id}'})
    (dest / 'tasks.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (dest / 'suite.json').write_text(json.dumps({'version': VERSION, 'sha256': suite_hash()}, indent=2))


def command(argv, cwd, timeout=10, input=None):
    return subprocess.run(argv, cwd=cwd, input=input, capture_output=True,
                          timeout=timeout, text=True)


def grade_patch(task_id, patch, timeout=5):
    """Apply only a regular solution.py patch to a clean snapshot, then verify.

    Agent-reported checks/status never determine success. This is process
    isolation, NOT a security sandbox or tamper-proof adversarial evaluator.
    """
    start = time.monotonic()
    result = {'task_id': task_id, 'verified': False}
    if not isinstance(patch, str) or not patch.strip():
        return {**result, 'failure': 'empty_patch', 'grade_seconds': 0.0}
    if len(patch.encode()) > 100_000:
        return {**result, 'failure': 'patch_too_large', 'grade_seconds': 0.0}
    try:
        with tempfile.TemporaryDirectory(prefix='nessa-grade-') as tmp:
            root = Path(tmp)
            (root / 'solution.py').write_text(TASKS[task_id]['source'])
            # -z avoids git pathname quoting ambiguities. Only this file may change.
            stat = command(['git', 'apply', '--numstat', '-z', '-'], root, input=patch)
            entries = [entry for entry in stat.stdout.split('\0') if entry]
            valid = stat.returncode == 0 and len(entries) == 1
            if valid:
                fields = entries[0].split('\t', 2)
                valid = len(fields) == 3 and fields[2] == 'solution.py' and all(x.isdigit() for x in fields[:2])
            # Reject non-regular modes, including symlinks, before applying.
            modes = [line.split()[-1] for line in patch.splitlines()
                     if line.startswith(('new file mode ', 'old mode ', 'new mode ', 'deleted file mode '))]
            if not valid or any(mode not in ('100644', '100755') for mode in modes):
                return {**result, 'failure': 'patch_scope', 'grade_seconds': time.monotonic()-start}
            applied = command(['git', 'apply', '--whitespace=nowarn', '-'], root, input=patch)
            if applied.returncode:
                return {**result, 'failure': 'patch_rejected', 'grade_seconds': time.monotonic()-start}
            solution = root / 'solution.py'
            if not solution.is_file() or solution.is_symlink():
                return {**result, 'failure': 'missing_solution', 'grade_seconds': time.monotonic()-start}
            # -I excludes user site packages and environment-supplied PYTHONPATH.
            proof = root / 'completed-proof'
            script = ('import sys\nsys.path.insert(0, ' + repr(str(root)) + ')\n'
                      + TASKS[task_id]['tests']
                      + '\nfrom pathlib import Path as _ProofPath\n_ProofPath(' + repr(str(proof)) + ').write_text(\"completed\")\n')
            # Write logs to disk so runaway stdout cannot exhaust parent RAM.
            with (root / 'grade.log').open('w+') as log:
                proc = subprocess.run([sys.executable, '-I', '-c', script], cwd=root,
                                      stdout=log, stderr=subprocess.STDOUT, timeout=timeout,
                                      env={'PATH': os.environ.get('PATH', ''), 'PYTHONDONTWRITEBYTECODE': '1'})
                log.seek(0)
                output = log.read(4000)
            passed = proc.returncode == 0 and proof.is_file() and proof.read_text() == 'completed'
            result.update(verified=passed,
                          failure=None if passed else 'tests_failed',
                          test_exit_code=proc.returncode, diagnostic=output)
    except subprocess.TimeoutExpired:
        result['failure'] = 'grader_timeout'
    except (OSError, ValueError) as exc:
        result.update(failure='grader_error', diagnostic=str(exc)[:1000])
    result['grade_seconds'] = round(time.monotonic() - start, 4)
    return result


def metric(row, key):
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None
    return value


def evidence_metrics(events):
    """Successful-response token usage only; never estimate missing server usage.

    Schema 1 marks complete protocol-error instrumentation. An interrupted trace
    cannot supply complete totals, even when some model events are present.
    """
    absent = dict.fromkeys(('prompt_tokens', 'completion_tokens', 'tool_calls', 'invalid_tool_calls'))
    if not events or events[0].get('event') != 'start' or events[-1].get('event') != 'end':
        return absent
    model = [event for event in events if event.get('event') == 'model']
    if not model:
        return absent
    out = {}
    for key in ('prompt_tokens', 'completion_tokens'):
        values = [metric(event, key) for event in model]
        out[key] = sum(values) if all(value is not None for value in values) else None
    schema = events[0].get('telemetry_schema', 0)
    out['tool_calls'] = sum(len(event['calls']) for event in model) if all(isinstance(event.get('calls'), list) for event in model) else None
    out['invalid_tool_calls'] = (sum(sum(bool(error) for error in event['errors'])
                                      for event in events if event.get('event') == 'protocol_error')
                                 if schema == 1 else None)
    # Old prompt usage defaults omitted usage to zero, so it is not measurable.
    if schema != 1:
        out['prompt_tokens'] = None
    return out


def enrich_predictions(rows, runs):
    """Read only known fixed fixture IDs, never a path supplied in predictions."""
    enriched = []
    for original in rows:
        row = dict(original)
        tid = row.get('instance_id', row.get('task_id'))
        if tid not in TASKS:
            raise ValueError(f'Unknown task: {tid!r}')
        events = Path(runs) / tid / 'evidence' / 'events.jsonl'
        if events.is_file():
            row.update(evidence_metrics(read_jsonl(events)))
        enriched.append(row)
    return enriched


def grade_predictions(rows, manifest, timeout=5):
    if manifest.get('suite_sha256') != suite_hash():
        raise ValueError('Manifest suite hash differs from installed fixtures')
    indexed = {}
    for row in rows:
        tid = row.get('instance_id', row.get('task_id'))
        if tid not in TASKS:
            raise ValueError(f'Unknown task: {tid!r}')
        if tid in indexed:
            raise ValueError(f'Duplicate task {tid!r}; select one predeclared attempt, never best-of retries')
        indexed[tid] = row
    results = []
    for tid in TASKS:
        row = indexed.get(tid)
        if row is None:
            results.append({'task_id': tid, 'verified': False, 'failure': 'missing_prediction'})
            continue
        grade = grade_patch(tid, row.get('model_patch', row.get('patch')), timeout)
        grade['harness_status'] = row.get('status')
        grade['metrics'] = {key: metric(row, key) for key in
                            ('seconds', 'steps', 'prompt_tokens', 'completion_tokens', 'tool_calls', 'invalid_tool_calls')}
        results.append(grade)
    sums = {}
    coverage = {}
    for key in ('seconds', 'steps', 'prompt_tokens', 'completion_tokens', 'tool_calls', 'invalid_tool_calls'):
        values = [r.get('metrics', {}).get(key) for r in results]
        coverage[key] = sum(v is not None for v in values)
        sums[key] = sum(values) if all(v is not None for v in values) else None
    tool_calls, invalid = sums['tool_calls'], sums['invalid_tool_calls']
    valid_rate_data = tool_calls is not None and invalid is not None and 0 <= invalid <= tool_calls
    summary = {'tasks': len(TASKS), 'verified_solves': sum(r['verified'] for r in results),
               'solve_rate': sum(r['verified'] for r in results) / len(TASKS),
               'failure_counts': dict(Counter(r['failure'] for r in results if r['failure'])),
               'totals': sums, 'metric_coverage_tasks': coverage,
               'invalid_tool_rate': invalid / tool_calls if valid_rate_data and tool_calls > 0 else None}
    return {'suite_version': VERSION, 'manifest': manifest, 'summary': summary, 'tasks': results}


def compare(baseline, candidate):
    differences = []
    for key in FAIR_FIELDS:
        a, b = baseline['manifest'].get(key), candidate['manifest'].get(key)
        if a is None or b is None or a != b:
            differences.append(key)
    if differences:
        raise ValueError('Unpaired/missing comparison controls: ' + ', '.join(differences))
    if baseline['manifest']['attempts_per_task'] != 1:
        raise ValueError('This v1 grader supports exactly one predeclared attempt per task')
    if not baseline['manifest'].get('harness_revision') or not candidate['manifest'].get('harness_revision'):
        raise ValueError('Both harness revisions must be recorded')
    a = {r['task_id']: r['verified'] for r in baseline['tasks']}
    b = {r['task_id']: r['verified'] for r in candidate['tasks']}
    if set(a) != set(TASKS) or set(b) != set(TASKS):
        raise ValueError('Comparison must cover every task')
    return {'baseline_verified': sum(a.values()), 'candidate_verified': sum(b.values()),
            'gained': [tid for tid in TASKS if b[tid] and not a[tid]],
            'regressed': [tid for tid in TASKS if a[tid] and not b[tid]],
            'warning': 'Four-task smoke suite only. No statistical or broad capability claim justified.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('dest')
    sub.add_parser('hash')
    p = sub.add_parser('grade'); p.add_argument('predictions'); p.add_argument('manifest'); p.add_argument('out'); p.add_argument('--runs')
    p = sub.add_parser('compare'); p.add_argument('baseline'); p.add_argument('candidate')
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args.dest)
    elif args.command == 'hash': print(suite_hash())
    elif args.command == 'grade':
        rows = read_jsonl(args.predictions)
        if args.runs:
            rows = enrich_predictions(rows, args.runs)
        report = grade_predictions(rows, json.loads(Path(args.manifest).read_text()))
        with Path(args.out).open('x') as out: out.write(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report['summary'], indent=2))
    else:
        print(json.dumps(compare(json.loads(Path(args.baseline).read_text()), json.loads(Path(args.candidate).read_text())), indent=2))

if __name__ == '__main__': main()
