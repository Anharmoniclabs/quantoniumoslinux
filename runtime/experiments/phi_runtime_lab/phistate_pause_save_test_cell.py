# PhiState-MoE: one Colab cell with Resume and Stop / Save / Test controls.
# FIRST: click the stop square on the OLD training cell and wait for it to end.
# The first switch uses the last verified save; an unfinished chunk is not saved.
# Thereafter, use this panel's Stop button: it finishes and saves the current chunk.

if globals().get('PHISTATE_PANEL') is not None and PHISTATE_PANEL.busy:
    raise RuntimeError('This panel is already working. Use its Stop / Save / Test button.')
if globals().get('PHISTATE_PANEL') is not None:
    for old_button in (PHISTATE_PANEL.start, PHISTATE_PANEL.stop, PHISTATE_PANEL.test):
        old_button.disabled = True

import gc
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch

LAB_PATH = Path('/content/phistate_l4/quantoniumoslinux/runtime/experiments/phi_runtime_lab')
DRIVE_PATH = Path('/content/drive/MyDrive/Quantonium/PhiState-L4')
RUN_PATH = DRIVE_PATH / '50k'
WORK_PATH = Path('/content/phistate_50k_work')
EXPECTED_CORPUS = '29e4997527c221f48180d30f5d5d3245d0b5476e69a3b735d5b0895204982474'
NEW_TOKENS = 80
EVAL_BATCHES = 16


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    pending = path.with_name(path.name + '.pending')
    with pending.open('w', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(pending, path)


def checked_copy(source, destination):
    source, destination = Path(source), Path(destination)
    pending = destination.with_name(destination.name + '.pending')
    try:
        with source.open('rb') as src, pending.open('wb') as dst:
            shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        source_hash = digest(source)
        require(digest(pending) == source_hash, f'Copy verification failed: {destination}')
        os.replace(pending, destination)
        return source_hash
    finally:
        pending.unlink(missing_ok=True)


def read_progress(run):
    state = json.loads((Path(run) / 'progress.json').read_text())
    require(state.get('format') == 'PhiStateMoE-L4-50K-progress-v1', 'Wrong progress format.')
    require(state.get('active_slot') in ('a', 'b'), 'Invalid active checkpoint slot.')
    require(state.get('data_fingerprint') == EXPECTED_CORPUS, 'Corpus fingerprint mismatch.')
    require(type(state.get('step')) is int and state['step'] >= 10250, 'Invalid saved step.')
    require(math.isfinite(float(state['best_validation_bpt'])), 'Nonfinite validation score.')
    require(math.isfinite(float(state['last_val_mean_bpt'])), 'Nonfinite latest validation score.')
    return state


def verify_saved_pair(latest_path, best_path, state):
    latest = torch.load(latest_path, map_location='cpu', weights_only=True)
    best = torch.load(best_path, map_location='cpu', weights_only=True)
    for payload in (latest, best):
        require(payload.get('format') == 'PhiStateMoE-v1', 'Wrong checkpoint format.')
        require(payload.get('data_fingerprint') == EXPECTED_CORPUS, 'Wrong checkpoint corpus.')
        require(isinstance(payload.get('state_dict'), dict), 'Checkpoint lacks model weights.')
        require(type(payload.get('step')) is int, 'Checkpoint lacks a valid step.')
    require(latest['step'] == state['step'], 'Progress and latest checkpoint disagree.')
    require('optimizer_state' in latest, 'Latest checkpoint lacks resumable optimizer state.')
    require(isinstance(latest.get('train_args'), dict), 'Latest checkpoint lacks training settings.')
    require(best['config'] == latest['config'], 'Best/latest architecture mismatch.')
    require(best['step'] <= latest['step'], 'Best checkpoint is newer than latest.')
    for payload in (latest, best):
        score = float(payload['best_validation_bpt'])
        require(math.isfinite(score) and abs(score - state['best_validation_bpt']) < 1e-5,
                'Checkpoint score does not match progress.')
    return {'latest_step': latest['step'], 'best_step': best['step'],
            'train_args': latest['train_args'], 'config': latest['config']}


def take_snapshot(run, data, destination_root):
    state = read_progress(run)
    slot = state['active_slot']
    sources = [Path(run) / f'slot_{slot}.last.pt', Path(run) / f'slot_{slot}.pt']
    verify_saved_pair(*sources, state)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination = Path(destination_root) / f'step_{state["step"]}_{stamp}_{uuid.uuid4().hex[:8]}'
    destination.mkdir(parents=True, exist_ok=False)
    latest = destination / f'phi_{state["step"]}.last.pt'
    best = destination / f'phi_{state["step"]}.pt'
    checksums = {}
    for src, dst in zip(sources, (latest, best)):
        checksums[dst.name] = checked_copy(src, dst)
    metadata = verify_saved_pair(latest, best, state)
    require(read_progress(run) == state,
            'Training changed the active save while copying. Stop training and retry; this snapshot is incomplete.')
    for name in ('tokenizer.model', 'manifest.json'):
        checksums[name] = checked_copy(Path(data) / name, destination / name)
    # Written last: a folder without snapshot.json is an incomplete snapshot.
    record = {'format': 'PhiState-manual-snapshot-v1', 'created_utc': stamp,
              'progress': state, 'checkpoints': metadata, 'sha256': checksums,
              'latest_file': latest.name, 'best_file': best.name,
              'data_directory': str(data),
              'note': 'The tokenized corpus remains in its original data directory; it is not bundled.'}
    write_json(destination / 'snapshot.json', record)
    return destination, latest, record


def arg_value(command, flag):
    for i, token in enumerate(command):
        if token == flag and i + 1 < len(command):
            return command[i + 1]
        if token.startswith(flag + '='):
            return token.split('=', 1)[1]
    return None


def old_training_processes(lab, run, work):
    import psutil
    found = {}
    runner = str((lab / 'colab_50k_runner.py').resolve())
    trainer = str((lab / 'pretrain_continue_10k.py').resolve())
    for proc in psutil.process_iter(['pid', 'cmdline']):
        if proc.pid == os.getpid():
            continue
        command = proc.info['cmdline'] or []
        drive_arg, output_arg = arg_value(command, '--drive-dir'), arg_value(command, '--out')
        is_runner = runner in command and drive_arg and Path(drive_arg).resolve() == run.resolve()
        is_trainer = (trainer in command and output_arg and
                      Path(output_arg).resolve().is_relative_to(work.resolve()))
        if is_runner or is_trainer:
            found[proc.pid] = proc
            try:
                for child in proc.children(recursive=True):
                    found[child.pid] = child
            except psutil.NoSuchProcess:
                pass
    return list(found.values())


def stop_leftover_training(lab, run, work):
    # Only this run's exact runner and its work-directory trainer are selected.
    # This is initial adoption after interrupting the old foreground Colab cell.
    import psutil
    processes = old_training_processes(lab, run, work)
    for proc in processes:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=5)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=5)
    require(not alive, 'Previous training processes have not stopped.')
    return len(processes)


class PhiStatePanel:
    def __init__(self, lab, drive, run, work, data):
        import ipywidgets as widgets
        from IPython.display import display
        self.lab, self.drive, self.run, self.work, self.data = lab, drive, run, work, data
        self.busy = False
        self.stop_requested = threading.Event()
        self.last_snapshot = None
        self.start = widgets.Button(description='Resume training', button_style='success')
        self.stop = widgets.Button(description='Stop / Save / Test', button_style='warning', disabled=True)
        self.test = widgets.Button(description='Save / Test now', button_style='info')
        for button in (self.start, self.stop, self.test):
            button.layout.width = '190px'
        self.status = widgets.Label(value='Ready: training is paused.')
        self.output = widgets.Output(layout={'border': '1px solid #ddd', 'max_height': '560px', 'overflow': 'auto'})
        self.start.on_click(lambda _: self.launch(train=True))
        self.test.on_click(lambda _: self.launch(train=False))
        self.stop.on_click(self.request_stop)
        display(widgets.VBox([widgets.HBox([self.start, self.stop, self.test]), self.status, self.output]))

    def say(self, message):
        self.output.append_stdout(str(message) + '\n')

    def request_stop(self, _):
        self.stop_requested.set()
        self.stop.disabled = True
        self.status.value = 'Stop requested: finishing and saving the current chunk, then testing.'
        self.say('Stop requested. Keep the runtime connected; do not press the cell stop square.')

    def launch(self, train):
        if self.busy:
            return
        self.busy = True
        self.stop_requested.clear()
        self.start.disabled = self.test.disabled = True
        self.stop.disabled = not train
        self.status.value = 'Training one chunk at a time...' if train else 'Saving and testing...'
        threading.Thread(target=self.worker, args=(train,), daemon=True).start()

    def worker(self, train):
        try:
            require(not old_training_processes(self.lab, self.run, self.work),
                    'Another trainer is running. Stop it before using this panel.')
            if train:
                self.train_until_paused()
            self.stop.disabled = True
            self.status.value = 'Archiving the latest verified save and testing it...'
            self.snapshot_and_test()
            self.status.value = 'Paused. Checkpoint and test report saved. Resume training when ready.'
        except Exception as error:
            self.status.value = 'Stopped with an error; inspect the message below.'
            self.say(f'{type(error).__name__}: {error}')
        finally:
            self.busy = False
            self.start.disabled = self.test.disabled = False
            self.stop.disabled = True
            gc.collect()

    def train_until_paused(self):
        require(torch.cuda.is_available(), 'Select an L4/CUDA runtime before resuming training.')
        while not self.stop_requested.is_set():
            state = read_progress(self.run)
            target = int(state['target_steps'])
            if state['step'] >= target:
                return
            command = [sys.executable, '-u', str(self.lab / 'colab_50k_runner.py'),
                       '--lab', str(self.lab), '--data-dir', str(self.data),
                       '--initial-last', str(self.drive / 'phi_10250.last.pt'),
                       '--initial-best', str(self.drive / 'phi_10250.pt'),
                       '--drive-dir', str(self.run), '--work-dir', str(self.work),
                       '--target-steps', str(target), '--chunk-steps', str(state['chunk_steps']),
                       '--eval-batches', str(EVAL_BATCHES), '--threads', '2', '--max-chunks', '1']
            self.say(f'Training from step {state["step"]:,}; next save within {state["chunk_steps"]} steps.')
            with tempfile.TemporaryFile(mode='w+', encoding='utf8') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
                if result.returncode:
                    log.seek(0)
                    raise RuntimeError('Training failed:\n' + log.read()[-6000:])
            saved = read_progress(self.run)
            require(saved['step'] > state['step'], 'Runner exited without advancing the verified save.')
            self.say(f'SAVED {saved["step"]:,}/{target:,} | latest validation {saved["last_val_mean_bpt"]:.5f} BPT')

    def snapshot_and_test(self):
        from moe_train import ckpt_load, evaluate_both, load_data, setup
        from compare_phistate_generations import PROMPTS, generate, python_syntax
        folder, latest, record = take_snapshot(self.run, self.data, self.drive / 'manual_snapshots')
        self.last_snapshot = str(folder)
        self.say(f'VERIFIED SNAPSHOT: {folder}')
        self.say(f'Testing LATEST weights at step {record["checkpoints"]["latest_step"]:,}; '
                 f'best weights are separately stored at step {record["checkpoints"]["best_step"]:,}.')
        setup(threads=2, seed=20261027)
        model, payload = ckpt_load(str(latest))  # strict weight loading; exact latest, not best
        args = payload['train_args']
        del payload  # release the optimizer copy during inference
        manifest, datasets, _ = load_data(str(self.data))
        require(manifest['data_fingerprint'] == EXPECTED_CORPUS, 'Wrong evaluation corpus.')
        import sentencepiece as spm
        tokenizer = spm.SentencePieceProcessor(model_file=str(folder / 'tokenizer.model'))
        require(tokenizer.vocab_size() == model.cfg.vocab, 'Tokenizer vocabulary does not match model.')
        self.say('Running validation on CPU using 16 batches per domain; sealed test splits are unused.')
        validation = evaluate_both(model, datasets, int(args['context']), int(args.get('eval_batch', 1)), EVAL_BATCHES)
        report = {'checkpoint_step': record['checkpoints']['latest_step'], 'checkpoint': str(latest),
                  'validation': validation, 'context': int(args['context']),
                  'eval_batch': int(args.get('eval_batch', 1)), 'eval_batches': EVAL_BATCHES,
                  'seed': 20261027, 'new_tokens': NEW_TOKENS, 'temperature': 0.8, 'top_k': 32,
                  'note': 'Small generation probe. Python parsing checks syntax only. Generated code is not executed.',
                  'results': []}
        self.say(json.dumps(validation, indent=2))
        write_json(folder / 'test_report.json', report)
        lines = [f'PhiState latest checkpoint: step {report["checkpoint_step"]}',
                 json.dumps(validation, indent=2), report['note'], '']
        for case in PROMPTS:
            for mode in ('greedy', 'sample'):
                text, diversity, repetition = generate(model, tokenizer, case['prompt'], NEW_TOKENS,
                                                      mode, seed=20261027)
                syntax_ok, syntax_error = python_syntax(text) if case['kind'] == 'python' else (None, None)
                row = {'case': case['id'], 'prompt': case['prompt'], 'decoding': mode,
                       'full_generated_text': text, 'python_source_parses': syntax_ok,
                       'syntax_error': syntax_error, 'unique_token_fraction': diversity,
                       'longest_repeated_token_run': repetition}
                report['results'].append(row)
                section = f'--- {case["id"]} / {mode} / Python parses: {syntax_ok} ---\n{text}\n'
                lines.append(section)
                self.say(section)
                write_json(folder / 'test_report.json', report)
        report['complete'] = True
        write_json(folder / 'test_report.json', report)
        (folder / 'test_report.txt').write_text('\n'.join(lines), encoding='utf8')
        self.say(f'FINISHED: {folder / "test_report.txt"}')


def launch_panel():
    for name in ('psutil', 'ipywidgets', 'sentencepiece'):
        require(importlib.util.find_spec(name) is not None,
                f'Missing {name}. Run %pip install {name}, then rerun this cell.')
    require((RUN_PATH / 'progress.json').is_file(),
            'Mount your original Google Drive and run the original notebook setup first.')
    for name in ('colab_50k_runner.py', 'moe_train.py', 'compare_phistate_generations.py'):
        require((LAB_PATH / name).is_file(), f'Missing source: {LAB_PATH / name}')
    candidates = [Path('/content/phistate_l4/data'),
                  Path('/content/phistate_l4/dataset_transfer/hf_coding_knowledge_v2'),
                  DRIVE_PATH / 'data/hf_coding_knowledge_v2']
    data = None
    for candidate in candidates:
        manifest_path = candidate / 'manifest.json'
        if manifest_path.is_file() and (candidate / 'tokenizer.model').is_file():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('data_fingerprint') == EXPECTED_CORPUS:
                data = candidate
                break
    require(data is not None, 'Original corpus/tokenizer not found. Run the original setup cell first.')
    state = read_progress(RUN_PATH)
    # Make sure a verified recovery point exists before stopping leftover processes.
    slot = state['active_slot']
    verify_saved_pair(RUN_PATH / f'slot_{slot}.last.pt', RUN_PATH / f'slot_{slot}.pt', state)
    count = stop_leftover_training(LAB_PATH, RUN_PATH, WORK_PATH)
    print(f'Closed {count} leftover training processes. Using the last verified save; unfinished work was not saved.')
    sys.path.insert(0, str(LAB_PATH))
    from google.colab import output
    output.enable_custom_widget_manager()
    panel = PhiStatePanel(LAB_PATH, DRIVE_PATH, RUN_PATH, WORK_PATH, data)
    panel.launch(train=False)  # immediately archive and test the latest saved checkpoint
    return panel


if __name__ == '__main__':
    PHISTATE_PANEL = launch_panel()
