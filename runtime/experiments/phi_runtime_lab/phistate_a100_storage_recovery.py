"""Resume the unchanged A100 experiment using local disk and verified Drive API backups."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time
import uuid

import torch
from chat_recovery_train import atomic_json, run_lock
import phistate_a100_base_journal as journal
import phistate_base_cuda_train as training
from phistate_checkpoint_io import atomic_checkpoint, file_receipt
from phistate_drive_transport import DriveStore, DriveTransferError

FORMAT = 'PhiState-A100-local-Drive-recovery-v1'
BACKUP_INTERVAL = 128
GIB = 1024**3
# The source receipt below is generated from byte-verified Drive artifacts.
RECOVERY = json.loads('{"immutable_sources":[{"bytes":154128843,"drive_id":"1JRWHPBABRiezpTNereKOt3T9KzEQbdES","relative_path":"run_60k_to_61k/latest.pt","sha256":"c09d5d63207bc69812d8e70608ed5be017619f53995164c642185c2413a41127"},{"bytes":154128779,"drive_id":"1DD0RZKGYImpBKvamoHdNvRrEYaDYwpHy","relative_path":"continue_60k/latest.pt","sha256":"50f9bfe9b63082041b201bd6420531cbe6384cfc3ef5cce097588d8ec2396f46"},{"bytes":2561,"drive_id":"150R-zXJCUT_YWYMogBAQS4dnJeHYhivu","relative_path":"corpus_repaired_v1/manifest.json","sha256":"489997dacf681fc58b58133f447dce57bc3366b829969933d4ead8091d7c7341"},{"bytes":109778,"drive_id":"1sx9xDb2RRbfooydEF1lWjWvdk3N7Ox5a","relative_path":"corpus_repaired_v1/tokenizer.model","sha256":"9f704ff69de6cb60fd7466a26b70cbd0472bb1a192f1825a7c52cfc30179ea04"},{"bytes":51248512,"drive_id":"1e9g8DgTiXF64X6AKA4NmrTfZxTA-A8IQ","relative_path":"corpus_repaired_v1/train_code.bin","sha256":"167b7ccda85163ebc10b9ca5e7b0b4f82ecab1ae9a390f3d33c3cd615f902989"},{"bytes":32425054,"drive_id":"19O-6Qwoe2gnWNQINo8w7ZOTxbySLz3qu","relative_path":"corpus_repaired_v1/train_general.bin","sha256":"80a33592f4ec52fa37a66d00ccdfad70cefed39e5cea2ab699be7b4dbe447ac8"},{"bytes":6590152,"drive_id":"1TYTRFgwp_S8BdmeeY11P665mX7QVM8gF","relative_path":"corpus_repaired_v1/val_code.bin","sha256":"956a1c79dc3adffe00c434684b580a28d0d1e46057658452c9198a21532f2ddf"},{"bytes":4231100,"drive_id":"1WQs0BP72NqE139RR27xbU-5VcUv1IKpy","relative_path":"corpus_repaired_v1/val_general.bin","sha256":"a43592c6519ae70b2b2d64d367d46e7688f1cae5efd0c60e537ecdaa388dcfab"}],"old_arm_request":{"arm":"dense","base_sha256":"50f9bfe9b63082041b201bd6420531cbe6384cfc3ef5cce097588d8ec2396f46","batch":8,"code":{"phistate_base_cuda_train.py":"e3c4b813e66c31d81432ce28fda22e23e3d4b9d54dec89df54421ae786a87c2e","phistate_base_reference.py":"4e06ce9379cf53691619bd58b8839c507d6735fe70c1d653df16a0868be38c6b","phistate_base_windows.py":"c145eaa5f82fc7f893f305598acc94879ca4b733ebaa413dcc2c8a7537d27eb4"},"context":128,"cuda":"13.0","data_fingerprint":"5a45ae09dfc42c6021202838439e77bd5cb05ff38d673d23e3219a1ecbc6c1d3","gpu":"NVIDIA A100-SXM4-40GB","model_config":{"context":128,"heads":8,"hidden":1920,"layers":6,"vocab":8192,"width":320},"optimizer":"fresh AdamW lr=.0003 betas=(.9,.95) wd=.01","schedule_sha256":"9688e1474a1d401a045b6a443a9f63509e1c93054ce2bc7489cf80097e55b046","seed":20261101,"source_step":0,"steps":512,"torch":"2.11.0+cu130","validation_sha256":"b6711e8172961837984bd02a823e1abbee0d3fea53386f035ff6dfbf0c4f1e55","version":"PhiState-base-A100-arm-v1"},"old_experiment_request":{"bases":{"65K":["run_60k_to_61k/latest.pt","c09d5d63207bc69812d8e70608ed5be017619f53995164c642185c2413a41127",65000],"66K":["continue_60k/latest.pt","50f9bfe9b63082041b201bd6420531cbe6384cfc3ef5cce097588d8ec2396f46",66000]},"batch":8,"confirm":"3e5be7305cc9849141dced8e73669f8fd134ec1a6718be65efff05942db18c8a","confirm_seed":729411,"context":128,"corpus":"5a45ae09dfc42c6021202838439e77bd5cb05ff38d673d23e3219a1ecbc6c1d3","cuda":"13.0","dev":"b6711e8172961837984bd02a823e1abbee0d3fea53386f035ff6dfbf0c4f1e55","experiment":"PhiState-A100-base-v1","fp32":true,"modules":{"device_preflight.py":"f1a6c7e3027f69a508658c315fb65bcb4cec623150df8a3923bb12a8d3fa2cd3","phistate_a100_base_journal.py":"f10b6b3c8cc33b42a474f80b42b9ee90f2b647bde5eba8ed9efc9ea0fdc3ab6b","phistate_base_cuda_train.py":"e3c4b813e66c31d81432ce28fda22e23e3d4b9d54dec89df54421ae786a87c2e","phistate_base_prerequisite.py":"cdba04e0c1e752dd60d7ebf90295e73dd598ebc3b63da685cd97a4fd213cc2e9","phistate_base_reference.py":"4e06ce9379cf53691619bd58b8839c507d6735fe70c1d653df16a0868be38c6b","phistate_base_windows.py":"c145eaa5f82fc7f893f305598acc94879ca4b733ebaa413dcc2c8a7537d27eb4"},"plans":{"20261101":"9688e1474a1d401a045b6a443a9f63509e1c93054ce2bc7489cf80097e55b046","20261102":"088c04e808d4708c86b68f4d34177bb98ed28e157e8f6ed4f0d1d1e3953db53b"},"reference_evidence":"ceb31915dd12517afdd366c3375581b40fffe135f8bbdf52f55185342deb49f7","seeds":[20261101,20261102],"source_pin":"73edfa1a704e8d198269351ebe597e5ff8fed8ed","steps":512,"thresholds":{"history_domain_tolerance":0.02,"maximum_domain_other_bpt_regression":0.05,"minimum_longer_context_gain":0.02,"minimum_other_bpt_drop":0.15,"minimum_warm_bpt_gain":0.01,"minimum_wrong_history_mean":0.01},"torch":"2.11.0+cu130"},"old_experiment_request_key":"9c77437f669f2bc37e262f51b0c7b214a45b0f5163c3dac5bf7fcfdb0dac905c","old_module_sha256":{"device_preflight.py":"f1a6c7e3027f69a508658c315fb65bcb4cec623150df8a3923bb12a8d3fa2cd3","phistate_a100_base_journal.py":"f10b6b3c8cc33b42a474f80b42b9ee90f2b647bde5eba8ed9efc9ea0fdc3ab6b","phistate_base_cuda_train.py":"e3c4b813e66c31d81432ce28fda22e23e3d4b9d54dec89df54421ae786a87c2e","phistate_base_prerequisite.py":"cdba04e0c1e752dd60d7ebf90295e73dd598ebc3b63da685cd97a4fd213cc2e9","phistate_base_reference.py":"4e06ce9379cf53691619bd58b8839c507d6735fe70c1d653df16a0868be38c6b","phistate_base_windows.py":"c145eaa5f82fc7f893f305598acc94879ca4b733ebaa413dcc2c8a7537d27eb4"},"root_folder_id":"1gB2SHXGRuYcuANkbHgi0d0PKYf2Aqtr1","run_files":[{"bytes":142,"drive_id":"1giHrgumVUbwQilOMlkSKqt27G6icmuES","relative_path":"phistate_a100_base_v1/report.json","sha256":"6047e22ba9b9a5a60e4d2526ed254d444adc0f92421e876514b4665278c4dd42"},{"bytes":1984,"drive_id":"1n2-LBcrWekRYDNpRNxIXs1nLsOulkNzc","relative_path":"phistate_a100_base_v1/request.json","sha256":"0d8b6401b22b674d0aa37015275bbe7f7b469417231f5fd031c86b5e5a586af3"},{"bytes":2766,"drive_id":"14mY7FQ6zRulcrXRonv-z1szBlMdMrMiN","relative_path":"phistate_a100_base_v1/baseline_65K.json","sha256":"a93202e234f8c5e922ff1b072c774fae4d0b0b5a4ec6f57ed4a7ed154b4782bc"},{"bytes":2809,"drive_id":"11J0QQnnnW7nyTQcenvQq9KTfdXUyW9sm","relative_path":"phistate_a100_base_v1/baseline_66K.json","sha256":"6587591dfc6677873fd14c3e68053941e71f0576bcad83d508da2f9c5cc3154f"},{"bytes":1521,"drive_id":"15EL_vIgv0RRoBW6tb_CkdMFHm8CZyIcP","relative_path":"phistate_a100_base_v1/device_preflight.json","sha256":"e71d285bf91ed0dbdcd731318fb27c77e2b47bc25f276ec753962f7cacaf2d54"},{"bytes":1101,"drive_id":"1f07JCxuuNGDhJjQN5uhiCGRSdCPFnfTB","relative_path":"phistate_a100_base_v1/dense_20261101/request.json","sha256":"c21212b9bd00c931065cf4c7765d630dcf32a9fc3f6ef26b363e43d69a222949"},{"bytes":4331,"drive_id":"1b4oKgItaBxQqGCCVNLSG0hfcfLQOdScZ","relative_path":"phistate_a100_base_v1/dense_20261101/progress.json","sha256":"4f7b8f6544faf7330617257b689f60f86038f675894b8ff805739409e0024850"},{"bytes":193766987,"drive_id":"16Y-F9Go2tiP8bU787SZ4ui5siOgVEAY-","relative_path":"phistate_a100_base_v1/dense_20261101/latest.pt","sha256":"6c75900d6b1a9fe94ca692c6d7432c3a78790ffda11c2bd8e104380bd9033774"}]}')


def safe_relative(value):
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0] != journal.OUTPUT:
        raise ValueError('Unexpected recovery file path')
    arms = {f'{arm}_{seed}' for arm in ('dense', 'phistate') for seed in journal.SEEDS} | {'continued_66k'}
    if len(p.parts) not in (2, 3) or (len(p.parts) == 3 and p.parts[1] not in arms):
        raise ValueError('Unexpected recovery directory')
    if p.suffix != '.json' and not (len(p.parts) == 3 and p.name == 'latest.pt'):
        raise ValueError('Unexpected recovery file type')
    return p


def check_compute():
    for name, expected in RECOVERY['old_module_sha256'].items():
        actual = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Research implementation changed: {name}; refusing checkpoint migration.')
    request = RECOVERY['old_experiment_request']
    if str(torch.__version__) != request['torch'] or torch.version.cuda != request['cuda']:
        raise RuntimeError('Resume requires the recorded PyTorch/CUDA versions; no updates were started.')


def ensure_local(work):
    work = Path(work).resolve()
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, right = line.split(' - ', 1)
        mount = Path(left.split()[4].replace('\\040', ' '))
        if right.split()[0].startswith('fuse') and (work == mount or mount in work.parents):
            raise RuntimeError('Training workspace must be local, outside mounted Drive.')
    work.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(work).free < 4 * GIB:
        raise RuntimeError('Need at least 4 GiB free local disk for verified checkpoints.')
    return work


def _index_files(value):
    if value.get('format') != FORMAT or value.get('experiment') != RECOVERY['old_experiment_request_key']:
        raise ValueError('Recovery index belongs to a different experiment')
    files = value.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Recovery index has no files')
    for relative, entry in files.items():
        safe_relative(relative)
        if not isinstance(entry.get('id'), str) or len(entry.get('sha256', '')) != 64:
            raise ValueError('Invalid recovery file receipt')
    return files


def newest_index(store):
    """Read only this experiment's small API-created indexes. A new runtime forks its own destination."""
    query = (f"'{RECOVERY['root_folder_id']}' in parents and trashed = false "
             "and appProperties has { key='phistate_role' and value='recovery_index' }")
    indexes, page = [], None
    while True:
        result = store.service.files().list(q=query, spaces='drive', pageSize=100, pageToken=page,
            fields='nextPageToken,files(id,name,size,md5Checksum,modifiedTime)', orderBy='modifiedTime desc').execute(num_retries=3)
        indexes.extend(result.get('files', []))
        page = result.get('nextPageToken')
        if not page:
            break
    candidates = []
    for meta in indexes:
        if not meta.get('name', '').startswith('phistate_a100_recovery_'):
            continue
        if int(meta.get('size', 0)) > 2 * 1024**2:
            raise RuntimeError('Unexpected oversized recovery index')
        body = store.service.files().get_media(fileId=meta['id']).execute(num_retries=3)
        if hashlib.md5(body).hexdigest() != meta.get('md5Checksum'):
            raise RuntimeError('Recovery index checksum mismatch')
        value = json.loads(body)
        if value.get('format') == FORMAT and value.get('experiment') == RECOVERY['old_experiment_request_key']:
            _index_files(value)
            candidates.append((sum(value.get('steps', {}).values()), meta['modifiedTime'], value))
    if not candidates:
        return None
    for _, _, value in sorted(candidates, key=lambda x: (x[0], x[1]), reverse=True):
        valid = True
        for entry in value['files'].values():
            try:
                remote = store.metadata(entry['id'])
            except Exception as exc:
                if getattr(getattr(exc, 'resp', None), 'status', None) in (404, 410):
                    valid = False
                    break
                raise
            if (remote.get('trashed') or int(remote.get('size', -1)) != entry['size']
                    or ('md5' in entry and entry['md5'] != remote.get('md5Checksum'))):
                valid = False
                break
        if valid:
            return value
        print('A recovery index references an unavailable object; checking an earlier committed index.', flush=True)
    raise RuntimeError('Recovery indexes exist but none has all files available. Refusing to restart from step 32.')


class Backups:
    def __init__(self, store, work, state):
        self.store, self.work, self.state = store, Path(work), state
        self.project = self.work / 'project'
        self.state_file = self.work / 'transport.json'

    def persist(self):
        atomic_json(self.state_file, self.state)

    def folder(self, relative):
        key = str(relative)
        if key not in self.state['folders']:
            self.state['folders'][key] = self.store.ensure_folder(
                self.state['folder_id'], key, self.state['token'])
            self.persist()
        return self.state['folders'][key]

    def weights(self, path, step):
        relative = str(Path(path).relative_to(self.project))
        safe_relative(relative)
        arm = Path(path).parent.name
        old = self.state['arms'].get(arm)
        new = self.store.sync_checkpoint(path, self.folder(arm), self.state['token'], step, old,
                                         defer_cleanup=True)
        self.state['arms'][arm] = new
        self.state['files'][relative] = new['snapshots'][-1]
        self.state['steps'][arm] = step
        self.persist()
        print(f'DRIVE VERIFIED {arm} step {step} | local checkpoint retained', flush=True)

    def metadata(self):
        out = self.project / journal.OUTPUT
        for path in sorted(out.rglob('*.json')):
            relative = str(path.relative_to(self.project))
            safe_relative(relative)
            if path.stat().st_size > 2 * 1024**2:
                raise RuntimeError('Unexpected oversized experiment metadata')
            checksum = file_receipt(path)['sha256']
            prior = self.state['files'].get(relative)
            if prior and prior['sha256'] == checksum:
                continue
            parent = self.state['folder_id'] if path.parent == out else self.folder(path.parent.name)
            # Versioned metadata keeps the previous committed index valid during upload.
            name = f'{path.stem}_{checksum[:16]}.json'
            receipt = self.store.upload(path, parent, name, self.state['token'], role='experiment_metadata')
            self.state['files'][relative] = receipt
        self.persist()

    def commit_index(self, status):
        self.metadata()
        value = {'format': FORMAT, 'experiment': RECOVERY['old_experiment_request_key'],
                 'files': self.state['files'], 'steps': self.state['steps'], 'status': status,
                 'output_folder_id': self.state['folder_id'], 'token': self.state['token'],
                 'storage_code': {p.name: file_receipt(p)['sha256'] for p in
                    (Path(__file__), Path(__file__).with_name('phistate_checkpoint_io.py'),
                     Path(__file__).with_name('phistate_drive_transport.py'),
                     Path(__file__).with_name('a100_run_guard.py'))}}
        receipt = self.store.upload_json(value, RECOVERY['root_folder_id'],
            f"phistate_a100_recovery_{self.state['token']}.json", self.state['token'],
            file_id=self.state.get('index_id'), role='recovery_index')
        self.state['index_id'] = receipt['id']
        self.state['status'] = status
        self.persist()
        # Retire older owned snapshots only after the global recovery index commits.
        protected = {v['id'] for v in self.state['files'].values()}
        for arm, saved in list(self.state['arms'].items()):
            self.state['arms'][arm] = self.store.cleanup_snapshots(
                self.folder(arm), self.state['token'], saved, protected)
        self.persist()

    def checkpoint(self, path, payload):
        # This replacement changes storage only; the existing optimizer loop is byte-identical.
        return atomic_checkpoint(path, payload)

    def progress(self, path, info):
        atomic_json(path, info)
        if Path(path).name != 'progress.json':
            return
        step = info['completed_steps']
        if step and (step % BACKUP_INTERVAL == 0 or info['status'] in ('complete', 'stopped')):
            self.weights(Path(path).with_name('latest.pt'), step)
            self.commit_index('training' if info['status'] == 'complete' else info['status'])


@contextmanager
def storage_hooks(backups):
    old_ck, old_json = training.atomic_checkpoint, training.atomic_json
    training.atomic_checkpoint, training.atomic_json = backups.checkpoint, backups.progress
    try:
        yield
    finally:
        training.atomic_checkpoint, training.atomic_json = old_ck, old_json


def validate_local_checkpoint(project):
    for path in (project / journal.OUTPUT).glob('*/latest.pt'):
        s = torch.load(path, map_location='cpu', weights_only=True)
        req = s['experiment_request']
        if req['base_sha256'] != training.BASE_SHA256 or req['data_fingerprint'] != journal.CORPUS:
            raise ValueError('Recovery checkpoint source mismatch')
        expected_code = RECOVERY['old_arm_request']['code']
        if req['code'] != expected_code or req['torch'] != str(torch.__version__) or req['cuda'] != torch.version.cuda:
            raise ValueError('Recovery checkpoint implementation mismatch')
        if s['phase_step'] != len(s['history']) or s['phase_trained_tokens'] != s['phase_step'] * journal.BATCH * journal.CONTEXT:
            raise ValueError('Recovery checkpoint update/token count mismatch')
        if not all(torch.isfinite(v).all().item() for v in s['state_dict'].values()):
            raise ValueError('Recovery checkpoint contains nonfinite weights')
        print(f"RESUME VERIFIED {path.parent.name} | {s['phase_step']}/512 saved updates", flush=True)
        del s


def run(work, service=None):
    training.require_a100()
    check_compute()
    work = ensure_local(work)
    with run_lock(work):
        if service is None:
            import google.auth
            from googleapiclient.discovery import build
            credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/drive'])
            service = build('drive', 'v3', credentials=credentials, cache_discovery=False)
        store = DriveStore(service)
        state_path = work / 'transport.json'
        saved_state = json.loads(state_path.read_text()) if state_path.exists() else None
        occupied = sum(s['size'] for a in (saved_state or {}).get('arms', {}).values()
                       for s in a.get('snapshots', []))
        quota = store.quota_guard(max(64 * 1024**2, 4 * GIB - occupied))
        print('Drive API storage check |', json.dumps(quota), flush=True)
        project = work / 'project'
        project.mkdir(exist_ok=True)
        for spec in RECOVERY['immutable_sources']:
            path = project / spec['relative_path']
            store.download(spec['drive_id'], path, spec['sha256'])
            print('Verified source |', spec['relative_path'], flush=True)
        if saved_state is not None:
            state = saved_state
            _index_files(state)
        else:
            prior = newest_index(store)
            files = prior['files'] if prior else {s['relative_path']:
                {'id': s['drive_id'], 'sha256': s['sha256'], 'size': s['bytes']} for s in RECOVERY['run_files']}
            token = uuid.uuid4().hex
            state = {'format': FORMAT, 'experiment': RECOVERY['old_experiment_request_key'],
                     'token': token, 'files': files, 'steps': prior['steps'] if prior else {'dense_20261101': 32},
                     'arms': {}, 'folders': {}, 'status': 'recovering'}
            state['folder_id'] = store.ensure_folder(RECOVERY['root_folder_id'],
                f'phistate_a100_base_recovered_{token[:12]}', token)
            atomic_json(state_path, state)
        for relative, spec in state['files'].items():
            safe_relative(relative)
            path = project / relative
            # Local optimizer progress can be newer than the last successful cloud backup.
            if not path.exists():
                store.download(spec['id'], path, spec['sha256'])
        validate_local_checkpoint(project)
        backups = Backups(store, work, state)
        print('Drive backups | https://drive.google.com/drive/folders/' + state['folder_id'], flush=True)
        print('Local recovery |', project / journal.OUTPUT, flush=True)
        # Upload any newer local recovery point before doing further optimizer updates.
        for path in (project / journal.OUTPUT).glob('*/latest.pt'):
            relative = str(path.relative_to(project))
            receipt = file_receipt(path)
            if receipt['sha256'] != state['files'].get(relative, {}).get('sha256'):
                s = torch.load(path, map_location='cpu', weights_only=True)
                step = int(s['phase_step']); del s
                backups.weights(path, step)
        # Prove that the API can write and read back an owned object before more updates.
        backups.commit_index('ready')
        try:
            with storage_hooks(backups):
                result = journal.run(project, recover=True)
            for path in (project / journal.OUTPUT).glob('*/latest.pt'):
                s = torch.load(path, map_location='cpu', weights_only=True)
                step = int(s['phase_step']); del s
                backups.weights(path, step)
            backups.commit_index(result['status'])
        except BaseException as exc:
            # Local-only reporting cannot mask a Drive exception or lose the recovery path.
            error = {'status': 'backup_or_training_failed', 'error': f'{type(exc).__name__}: {exc}',
                     'local_project': str(project), 'keep_runtime_connected': True}
            try:
                atomic_json(work / 'recovery_error.json', error)
            except Exception as secondary:
                exc.add_note(f'Local error reporting also failed: {secondary}')
            print('STOPPED. Local checkpoints retained at', project / journal.OUTPUT,
                  '| Keep this runtime connected and rerun this cell to retry.', flush=True)
            raise
        print('COMPLETE | verified Drive backups | https://drive.google.com/drive/folders/' + state['folder_id'], flush=True)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', default='/content/phistate_a100_recovery')
    args = parser.parse_args()
    run(args.work)
