"""Verified immutable checkpoint snapshots for local or mounted Drive storage.

Incomplete writes have no valid COMPLETE.json and are never restored. Original
snapshots are retained; this helper never deletes an older checkpoint.
"""
import json
from pathlib import Path
import shutil
import time
import uuid
import torch

from open_flow import sha256

FILES = ('latest.pt','best.pt','metrics_latest.json','run_manifest.json')


def verify_snapshot(folder):
    folder = Path(folder)
    receipt = json.loads((folder/'COMPLETE.json').read_text())
    if set(receipt['files']) != set(FILES):
        raise ValueError('Snapshot file list differs')
    for name,digest in receipt['files'].items():
        if sha256(folder/name) != digest:
            raise ValueError('Snapshot checksum differs: '+name)
    state = torch.load(folder/'latest.pt',map_location='cpu',weights_only=True)
    best = torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    metrics = json.loads((folder/'metrics_latest.json').read_text())
    manifest = json.loads((folder/'run_manifest.json').read_text())
    if (int(state['step']) != receipt['step'] or metrics['phase_step'] != receipt['step']
        or best['step'] > state['step'] or best['config'] != state['config']
        or manifest['config'] != state['config']):
        raise ValueError('Snapshot step/config mismatch')
    for value in (state,best):
        if not all(torch.isfinite(x).all() for x in value['model'].values()):
            raise ValueError('Nonfinite snapshot weights')
    return receipt,state['config']


def save_snapshot(run, destination):
    run,destination = Path(run),Path(destination)
    state = torch.load(run/'latest.pt',map_location='cpu',weights_only=True)
    step = int(state['step'])
    folder = destination/'snapshots'/f'step_{step:09d}_{uuid.uuid4().hex[:12]}'
    folder.mkdir(parents=True,exist_ok=False)
    hashes = {}
    for name in FILES:
        hashes[name] = sha256(run/name)
        shutil.copyfile(run/name,folder/name)
        if sha256(folder/name) != hashes[name]:
            raise IOError('Backup copy did not verify: '+name)
    receipt = dict(step=step,created_ns=time.time_ns(),files=hashes)
    # Written last. Readers verify its full content and all file hashes.
    (folder/'COMPLETE.json').write_text(json.dumps(receipt,indent=2)+'\n')
    verify_snapshot(folder)
    return folder


def restore_latest(destination, run, expected_config):
    """Restore highest valid step for this exact lineage; report rejected receipts."""
    candidates, rejected = [],[]
    for receipt_path in (Path(destination)/'snapshots').glob('*/COMPLETE.json'):
        try:
            receipt,config = verify_snapshot(receipt_path.parent)
            if config != expected_config:
                raise ValueError('Different checkpoint lineage/config')
            candidates.append((receipt['step'],receipt['created_ns'],receipt_path.parent))
        except (ValueError,KeyError,OSError,RuntimeError,EOFError) as exc:
            rejected.append(str(receipt_path.parent)+': '+str(exc))
    if rejected:
        print('Rejected snapshots:',*rejected,sep='\n')
    if not candidates:
        if rejected:
            raise ValueError('No valid snapshot; refusing a silent reset')
        return None
    _,_,folder = max(candidates)
    run = Path(run)
    run.mkdir(parents=True,exist_ok=True)
    if (run/'latest.pt').exists():
        local = torch.load(run/'latest.pt',map_location='cpu',weights_only=True)
        if local['config'] != expected_config:
            raise ValueError('Local run has another config')
        if local['step'] > max(candidates)[0]:
            raise ValueError('Local checkpoint is newer; save it before restoring')
    for name in FILES:
        temp = run/(name+'.restoring')
        shutil.copyfile(folder/name,temp)
        if sha256(temp) != sha256(folder/name):
            raise IOError('Restore copy failed verification')
        temp.replace(run/name)
    return folder
