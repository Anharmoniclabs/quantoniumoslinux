"""Foreground Colab journal for the fixed coherence comparison. Linux/Colab only."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile


EXPERIMENT = 'coherence_execution_v1'
POLICIES = ('original', 'protected')
TERMINAL = {'qualified', 'failed_quality', 'failed_final'}


def _json(path):
    if path.is_file():
        return json.loads(path.read_text(encoding='utf-8'))
    return {}


def _settings(out, steps):
    request = _json(out / 'request.json')
    if request:
        return {key: request[key] for key in ('steps_per_arm', 'eval_every', 'save_every')}
    return {'steps_per_arm': steps, 'eval_every': 250, 'save_every': 50}


def _active(out):
    from chat_recovery_train import run_lock
    try:
        with run_lock(out / 'execution'):
            return False
    except RuntimeError as exc:
        if 'active worker' not in str(exc):
            raise
        return True


def _quality(result):
    if not result:
        return {}
    samples = {}
    for row in result.get('rows', []):
        samples.setdefault(row['bucket'], {key: row.get(key) for key in ('prompt', 'text', 'passed')})
    return {'macro_score': result.get('macro_score'), 'qualified': result.get('qualified'),
            'buckets': result.get('buckets', {}), 'examples': samples}


def snapshot(root):
    """Read saved progress from each arm. Probe the same-runtime coordinator lock."""
    root = Path(root).resolve()
    out = root / EXPERIMENT
    request, report = _json(out / 'request.json'), _json(out / 'report.json')
    active = _active(out)
    arms = {}
    for policy in POLICIES:
        progress = _json(out / policy / 'progress.json')
        quality = (progress.get('last_evaluation') or {}).get('quality', {})
        coverage = {name: {k: v for k, v in counts.items() if k != 'unique_indices'}
                    for name, counts in progress.get('coverage', {}).items()}
        arms[policy] = {'status': progress.get('status', 'pending'),
                        'phase_step': progress.get('phase_step', 0),
                        'target_steps': progress.get('target_steps', request.get('steps_per_arm')),
                        'coverage': coverage, 'quality': _quality(quality)}
    status = report.get('status')
    if status not in TERMINAL:
        if (out / 'STOP').exists() or (not active and any(a['status'] == 'stopped' for a in arms.values())):
            status = 'stopping' if active else 'stopped'
        elif active:
            status = 'comparing'
        elif status != 'native_export_failed':
            status = 'interrupted' if request else 'not_started'
    return {'status': status, 'worker_active': active, 'request': request, 'arms': arms,
            'final': _quality(report.get('final', {}).get('result', {})),
            'reason': report.get('reason'), 'native_run': report.get('native_run'),
            'log': str(root / 'coherence_continuation_journal.log')}


def _print_quality(value):
    if not value:
        return
    print('  macro', value['macro_score'], '| qualified', value['qualified'])
    for name, counts in value['buckets'].items():
        print(' ', name, f"{counts['passed']}/{counts['total']}")
        sample = value['examples'].get(name)
        if sample:
            print('   ', sample['prompt'], '→', repr(sample['text'])[:230], '| pass', sample['passed'])


def print_summary(root):
    """Display the last atomic saves, sampled-source counts and actual generated text."""
    state = snapshot(root)
    print('Experiment', state['status'], '| worker active on this runtime', state['worker_active'])
    for name, arm in state['arms'].items():
        if arm['target_steps'] is None:
            print(name, 'pending')
        else:
            print(name, arm['status'], '| saved', arm['phase_step'], '/', arm['target_steps'])
        for source, counts in arm['coverage'].items():
            print(' ', source, '| draws', counts['draws'], '| unique rows',
                  counts['unique_examples'] if not source.startswith('replay_') else 'stream windows',
                  '| input tokens', counts['input_tokens'], '| assistant targets', counts['answer_tokens'])
        _print_quality(arm['quality'])
    if state['final']:
        print('Final evaluation')
        _print_quality(state['final'])
    for key in ('reason', 'native_run', 'log'):
        if state[key]:
            print(key, state[key])


class _StoppedBeforeRequest(Exception):
    pass


def _worker(root, ready, steps):
    """Run in an isolated process. Only the owner of the coordinator lock may pause it."""
    import coherence_recovery as coordinator
    out = root / EXPERIMENT
    original_lock = coordinator.run_lock
    owned = False

    def pause(*_):
        if owned and (out / 'request.json').exists():
            (out / 'STOP').touch()
        else:
            raise _StoppedBeforeRequest()

    @contextmanager
    def observed_lock(path):
        nonlocal owned
        with original_lock(path):
            if path != out / 'execution':
                yield
                return
            # Make initial resume-marker cleanup and ownership publication one
            # operation with respect to the parent's targeted pause signal.
            old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})
            try:
                if (out / 'request.json').is_file():
                    for file in [out / 'STOP', *(out / arm / 'STOP' for arm in POLICIES)]:
                        file.unlink(missing_ok=True)
                owned = True
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
            try:
                yield
            finally:
                owned = False

    # Signal masking protects request creation from leaving a half-written
    # first request if the user interrupts startup, before STOP may exist.
    write_json = coordinator.atomic_json
    def safe_write(path, value):
        if path != out / 'request.json':
            return write_json(path, value)
        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})
        try:
            return write_json(path, value)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)

    original_handler = signal.signal(signal.SIGUSR1, pause)
    coordinator.run_lock, coordinator.atomic_json = observed_lock, safe_write
    try:
        ready.touch()
        settings = _settings(out, steps)
        result = coordinator.run(root, steps=settings['steps_per_arm'], device='cuda',
                                 eval_every=settings['eval_every'], save_every=settings['save_every'],
                                 clear_stop=False)
        return result['status']
    except _StoppedBeforeRequest:
        return 'stopped'
    finally:
        coordinator.run_lock, coordinator.atomic_json = original_lock, write_json
        signal.signal(signal.SIGUSR1, original_handler)


def _display_line(line):
    if line.startswith('QUALITY '):
        try:
            q = json.loads(line[8:])
            return 'QUALITY macro {} | qualified {} | {}'.format(q['macro_score'], q['qualified'],
                ', '.join(f"{k} {v['passed']}/{v['total']}" for k, v in q['buckets'].items()))
        except (ValueError, KeyError):
            pass
    return line.rstrip('\n')


def _verified_terminal(root, out, report):
    import coherence_recovery as coordinator
    from chat_recovery_data import sha
    request = _json(out / 'request.json')
    if not request or request != report.get('request') or not (out / 'source.pt').is_file():
        raise RuntimeError('Saved terminal report does not match its immutable request/source.')
    # Reuse the coordinator's input, code, environment and frozen-copy checks.
    # A terminal report returns before device preflight or training.
    checked = coordinator.run(root, steps=request['steps_per_arm'], device='cpu',
                              eval_every=request['eval_every'], save_every=request['save_every'])
    if checked != report:
        raise RuntimeError('Terminal report changed during verification.')
    if report['status'] in {'qualified', 'failed_final'}:
        policy, final = report.get('selected'), _json(out / 'final_result.json')
        if policy not in POLICIES or not final or final != report.get('final'):
            raise RuntimeError('Saved final result and selection disagree with the terminal report.')
        candidate_hash = sha(out / policy / 'candidate.pt')
        if final['identity'] != {'candidate_sha256': candidate_hash,
                                 'suite_sha256': sha(out / 'qualification/final.json')}:
            raise RuntimeError('Final candidate or suite changed after evaluation.')
        if report['status'] == 'qualified' and sha(out / 'qualified_native/best.pt') != candidate_hash:
            raise RuntimeError('Native checkpoint differs from the final candidate.')
    return checked


def run_or_resume(root, lab, steps=500):
    """Run to completion in the foreground. Rerunning resumes the recorded budget."""
    from chat_recovery_train import run_lock
    root, lab = Path(root).resolve(), Path(lab).resolve()
    if not root.is_dir():
        raise FileNotFoundError('Missing Drive project folder: ' + str(root))
    out = root / EXPERIMENT
    with run_lock(root / 'coherence_continuation_journal_launcher'):
        if _active(out):
            raise RuntimeError('A coherence comparison is already running on this runtime.')
        report = _json(out / 'report.json')
        if report.get('status') in TERMINAL:
            checked = _verified_terminal(root, out, report)
            print('Verified saved result:', checked['status'], flush=True)
            return checked
        with tempfile.TemporaryDirectory(prefix='phistate_journal_') as tmp:
            ready = Path(tmp) / 'ready'
            result_file = Path(tmp) / 'status'
            cmd = [sys.executable, '-u', str(lab / 'continuation_journal.py'),
                   str(root), str(ready), str(result_file), str(steps)]
            with (root / 'coherence_continuation_journal.log').open('a', encoding='utf-8', buffering=1) as log:
                print(datetime.now(timezone.utc).isoformat(), '| launch', _settings(out, steps), file=log)
                print('Starting fixed comparison. Interrupt this cell to save and pause.', flush=True)
                proc = subprocess.Popen(cmd, cwd=lab, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, errors='replace', start_new_session=True)
                paused = False
                try:
                    while True:
                        try:
                            for line in proc.stdout:
                                log.write(line)
                                print(_display_line(line), flush=True)
                            code = proc.wait()
                            break
                        except KeyboardInterrupt:
                            if not paused and proc.poll() is None:
                                paused = True
                                if ready.exists():
                                    proc.send_signal(signal.SIGUSR1)
                                else:
                                    proc.terminate()  # The coordinator has not started yet.
                                print('Pause requested. Waiting for the current save.', flush=True)
                finally:
                    proc.stdout.close()
                    if proc.poll() is None:
                        if ready.exists():
                            proc.send_signal(signal.SIGUSR1)
                        else:
                            proc.terminate()
                        proc.wait()
                if code and not (paused and code == -signal.SIGTERM):
                    raise RuntimeError(f'Journal worker exited with code {code}. See {log.name}')
            status = result_file.read_text() if result_file.exists() else 'stopped'
            report = _json(out / 'report.json')
            return report if report.get('status') == status else {'status': status}


if __name__ == '__main__':
    root, ready, result_file, steps = sys.argv[1:]
    status = _worker(Path(root).resolve(), Path(ready), int(steps))
    Path(result_file).write_text(status)
    print('JOURNAL', status, flush=True)
