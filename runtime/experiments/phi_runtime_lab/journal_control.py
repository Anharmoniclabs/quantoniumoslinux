"""Colab journal audit and isolated continuation helpers. Never promote checkpoints."""
from collections import defaultdict
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from chat_recovery_data import sha, read_jsonl, SOURCES, verify_prepared
from chat_recovery_train import atomic_json, run_lock


def inventory(root):
    import torch
    root=Path(root)
    entries=[]
    for label, relative in [
        ('repaired pretraining','run_60k_to_61k/latest.pt'),
        ('conversation','chat_recovery_v1/run/latest.pt'),
        ('tiny diagnostic','tiny_learning_check_v1/latest.pt'),
        ('generalization','generalization_v1/latest.pt'),
        ('broad continuation','broad_continuation_v1/latest.pt')]:
        path=root/relative
        if not path.exists():
            entries.append(dict(run=label,exists=False,path=str(path)))
            continue
        with run_lock(path.parent):
            state=torch.load(path,map_location='cpu',weights_only=True)
            cfg=state.get('config',{})
            entries.append(dict(run=label,exists=True,step=state.get('step'),
                phase_step=state.get('phase_step',state.get('sft_steps',state.get('diagnostic_step'))),
                token_exposures=state.get('trained_tokens'),phase_token_exposures=state.get('phase_tokens'),
                supervised_answer_exposures=state.get('answer_tokens'),
                optimizer_saved='optimizer_state' in state,vocab=cfg.get('vocab'),
                state_tensor_elements=sum(v.numel() for v in state['state_dict'].values()),path=str(path)))
            del state
        gc.collect()
    return entries


def audit_data(root):
    root=Path(root)
    corpus=root/'corpus_repaired_v1'
    sft=root/'chat_recovery_v1/data'
    manifest=json.loads((corpus/'manifest.json').read_text())
    token_hash=sha(corpus/'tokenizer.model')
    prepared=verify_prepared(sft,token_hash,512)
    if prepared['base_data_fingerprint']!=manifest['data_fingerprint']:
        raise RuntimeError('Prepared data and repaired corpus lineage mismatch')
    if manifest.get('whitespace_repair',{}).get('tokenizer_sha256')!=token_hash:
        raise RuntimeError('Repaired tokenizer hash mismatch')
    counts=[]
    for split in ('train','val'):
        sources=defaultdict(lambda:dict(examples=0,input_tokens=0,answer_tokens=0,multi_turn=0))
        for row in read_jsonl(sft/(split+'.jsonl')):
            s=sources[row['source']]
            s['examples']+=1
            s['input_tokens']+=len(row['input_ids'])
            s['answer_tokens']+=sum(v!=-100 for v in row['labels'])
            s['multi_turn']+=int(len(row['messages'])>2)
        for name,stats in sources.items():
            counts.append(dict(split=split,source=name,**stats,
                configured_sft_example_share=SOURCES[name]['weight'],
                mean_input_tokens=stats['input_tokens']/stats['examples'],
                mean_answer_tokens=stats['answer_tokens']/stats['examples']))
    train=[r for r in counts if r['split']=='train']
    total_weighted_length=sum(r['configured_sft_example_share']*r['mean_input_tokens'] for r in train)
    for r in train:
        r['expected_sft_input_token_share']=r['configured_sft_example_share']*r['mean_input_tokens']/total_weighted_length
    return dict(tokenizer_sha256=token_hash,data_fingerprint=manifest['data_fingerprint'],
        bins={k:v['tokens'] for k,v in manifest['bins'].items()},sft=counts,
        notes=['Example sampling shares differ from token shares because examples have different lengths.',
               'Input tokens include prompts; answer tokens include only supervised assistant targets and EOS.',
               'Checkpoint exposure counters are not counts of unique tokens or complete corpus passes.',
               'Do not sum ancestor and descendant checkpoints: their exposure counters overlap.',
               'Old-tokenizer pretraining exposures cannot be divided by repaired corpus size to infer epochs.'])


def fork_conversation(root):
    """Copy resumable conversational state, not the overfit curriculum state."""
    import torch
    root=Path(root)
    source=root/'chat_recovery_v1/run'
    target=root/'broad_continuation_v1'
    with run_lock(target),run_lock(source):
        if target.exists():
            metadata=json.loads((target/'fork.json').read_text())
            if metadata.get('source_run')!=str(source.resolve()) or not (target/'latest.pt').exists():
                raise RuntimeError('Existing continuation is not a valid journal fork')
            return target
        source_hashes={name:sha(source/name) for name in ('latest.pt','best.pt')}
        latest=torch.load(source/'latest.pt',map_location='cpu',weights_only=True)
        if latest.get('training_stage')!='chat_recovery_v1' or 'optimizer_state' not in latest:
            raise RuntimeError('Expected resumable conversation checkpoint')
        original=root/'run_60k_to_61k/latest.pt'
        if latest['source_sha256']!=sha(original):
            raise RuntimeError('Original pretraining checkpoint differs from chat lineage')
        settings=latest['chat_settings']
        metadata=dict(source_run=str(source.resolve()),source_hashes=source_hashes,
            initial_sft_steps=latest['sft_steps'],settings=settings,
            copied_at_unix=time.time(),scope='Independent continuation on existing broad data; no curriculum examples.')
        del latest
        stage=Path(tempfile.mkdtemp(prefix='broad_continuation.pending_',dir=root))
        try:
            for name,expected in source_hashes.items():
                shutil.copyfile(source/name,stage/name)
                if sha(stage/name)!=expected or sha(source/name)!=expected:
                    raise RuntimeError('Checkpoint copy or source hash changed')
            atomic_json(stage/'fork.json',metadata)
            atomic_json(stage/'progress.json',dict(status='forked',saved_sft_step=metadata['initial_sft_steps'],
                source_step=None,target_sft_steps=None,latest=str(target/'latest.pt'),best=str(target/'best.pt')))
            stage.rename(target)
        except BaseException:
            shutil.rmtree(stage,ignore_errors=True)
            raise
    return target


def worker_active(run):
    path=Path(run)/'worker.json'
    if not path.exists(): return False
    info=json.loads(path.read_text())
    proc=Path('/proc')/str(info['pid'])
    try:
        cmdline=(proc/'cmdline').read_bytes().split(b'\0')
        start=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]
        return start==info['process_start'] and str(Path(run).resolve()).encode() in cmdline
    except (OSError,IndexError):
        return False


def launch(root,lab,additional_steps=2000):
    """Start/resume a bounded independent continuation; repeat calls don't add a new budget."""
    root,lab=Path(root).resolve(),Path(lab).resolve()
    if not 1<=additional_steps<=4000: raise ValueError('Choose 1..4000 additional SFT steps')
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('Select a GPU runtime in Colab')
    target=root/'broad_continuation_v1'
    if worker_active(target):
        print('Worker already running. Use the monitor cell.')
        return target
    target=fork_conversation(root)
    with run_lock(target):
        metadata=json.loads((target/'fork.json').read_text())
        destination=metadata['initial_sft_steps']+additional_steps
        request=target/'request.json'
        if request.exists() and json.loads(request.read_text())['target_sft_steps']!=destination:
            raise RuntimeError('Existing run has a different fixed budget; preserve it')
        status=json.loads((target/'progress.json').read_text()).get('status')
        if status in ('complete','early_stopped','retention_stopped'):
            print('Run already ended:',status,'— inspect generations and metrics; no automatic extension.')
            return target
        s=metadata['settings']
        command=[sys.executable,'-u',str(lab/'chat_recovery_train.py'),
            '--checkpoint',str(root/'run_60k_to_61k/latest.pt'),
            '--data-dir',str(root/'corpus_repaired_v1'),'--sft-dir',str(root/'chat_recovery_v1/data'),
            '--run-dir',str(target),'--steps',str(destination),'--context',str(s['context']),
            '--accum',str(s['accum']),'--device','cuda','--eval-every',str(s['eval_every']),
            '--save-every',str(s['save_every']),'--val-examples',str(s['val_examples_per_source']),
            '--patience',str(s['patience']),'--clear-stop']
        atomic_json(request,dict(target_sft_steps=destination,additional_steps=additional_steps,command=command))
    # The trainer takes the run lock. Launch lock serializes concurrent notebook requests.
    with run_lock(target/'launcher'):
        if worker_active(target): return target
        with (target/'training.log').open('a',buffering=1) as log:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            start=(Path('/proc')/str(proc.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
        except OSError:
            raise RuntimeError('Worker exited immediately; inspect training.log') from None
        atomic_json(target/'worker.json',dict(pid=proc.pid,process_start=start,command=command))
    print('STARTED/RESUMED PID',proc.pid,'| target SFT step',destination,'|',target/'training.log')
    return target


def monitor(root,lines=35):
    target=Path(root)/'broad_continuation_v1'
    print('Worker active:',worker_active(target))
    if (target/'progress.json').exists():
        print((target/'progress.json').read_text())
    if (target/'training.log').exists():
        # Read a bounded tail even for a long-running log.
        with (target/'training.log').open('rb') as f:
            f.seek(0,2); size=f.tell(); f.seek(max(0,size-30000))
            tail=f.read().decode('utf-8',errors='replace').splitlines()
        print('\n'.join(tail[-lines:]))
    else: print('No continuation has been started.')


def request_stop(root):
    target=Path(root)/'broad_continuation_v1'
    if not target.exists():
        print('No continuation exists.')
        return
    (target/'STOP').touch()
    print('Stop requested. The worker saves after its current complete update. Run monitor to confirm stopped.')
