"""Run the CPU binding reference, then an isolated SFT-4000 learning check."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from a100_run_guard import guarded_run
from chat_recovery_data import sha
from chat_recovery_train import atomic_json

import phistate_binding_bridge as bridge
import phistate_binding_fit as fit
import phistate_binding_reference as reference
import phistate_copy_calibration as calibration

PARENT = 'dcc79a5a3b05411725fefcefa04fd0aafd4393f4efb21e018c60da1e9a80d8ea'
TOKENIZER = '9f704ff69de6cb60fd7466a26b70cbd0472bb1a192f1825a7c52cfc30179ea04'
PIN = '73edfa1a704e8d198269351ebe597e5ff8fed8ed'
OUTPUT = 'phistate_cpu_binding_v1'
SEED = 20261101
STEPS = 96
THREADS = 2
MODULES = ('phistate_binding_data.py','phistate_binding_fit.py',
           'phistate_binding_reference.py','phistate_binding_journal.py','phistate_copy_calibration.py',
           'phistate_binding_bridge.py')
GATES = {'seen_quartet_rate':.875, 'dev_familiar_wording_quartet_rate':.75,
         'dev_new_wording_quartet_rate':.75}


def decision(result):
    if result['seen_quartet_rate'] < GATES['seen_quartet_rate']:
        return 'training_examples_not_yet_fit'
    if result['dev_familiar_wording_quartet_rate'] < GATES['dev_familiar_wording_quartet_rate']:
        return 'seen_fit_only'
    if result['dev_new_wording_quartet_rate'] < GATES['dev_new_wording_quartet_rate']:
        return 'transfer_with_familiar_wording'
    return 'transfer_on_this_binding_task'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def run(root, recover=False):
    root = Path(root).resolve()
    out = root / OUTPUT
    parent = root / 'coherence_execution_v1' / 'source.pt'
    tokenizer = root / 'chat_recovery_v1' / 'data' / 'tokenizer.model'
    if torch.cuda.is_available():
        raise RuntimeError('Select CPU under Runtime → Change runtime type, reconnect, and run this cell.')
    if sha(parent) != PARENT or sha(tokenizer) != TOKENIZER:
        raise ValueError('The SFT-4000 parent or repaired tokenizer did not match the verified Drive files.')
    request = {'format':'PhiState-CPU-binding-v1','parent':PARENT,'tokenizer':TOKENIZER,'basis_commit':PIN,
               'modules':{name:sha(Path(__file__).parent/name) for name in MODULES},
               'seed':SEED,'warm_steps':STEPS,'warm_threads':THREADS,'copy_max_steps':160,'bridge_steps':STEPS,'gates':GATES,'torch':str(torch.__version__)}
    with guarded_run(out,recover=recover) as owner:
        out.mkdir(exist_ok=True)
        req, report = out/'request.json', out/'report.json'
        if req.exists():
            if json.loads(req.read_text()) != request:
                raise RuntimeError('Saved run identity differs. Use the same notebook and PyTorch version to resume.')
        else:
            if any(p.name not in ('STOP','request.partial') for p in out.iterdir()):
                raise RuntimeError('Unexpected files in the experiment output before first verified start.')
            atomic_json(req,request)
        if report.is_file():
            saved = json.loads(report.read_text())
            if saved.get('status') == 'complete':
                if saved['request'] != fingerprint(request):
                    raise RuntimeError('Completed report does not match this experiment.')
                print('Already complete:',saved['outcome'],'|',report,flush=True)
                return saved
        def publish(status,phase,**fields):
            owner.check()
            result = {'status':status,'phase':phase,'request':fingerprint(request),**fields}
            atomic_json(report,result)
            print(status.upper(), phase, '|', report, flush=True)
            return result
        def stopped():
            return publish('stopped','resume',note='Remove STOP if you created it, then rerun this cell.')
        if (out/'STOP').exists():
            return stopped()
        phase='learned reference'
        try:
            publish('running',phase)
            ref_path = out/'reference.json'
            if ref_path.is_file():
                rec = json.loads(ref_path.read_text())
                if rec['request'] != fingerprint(request) or rec['hash'] != fingerprint(rec['result']):
                    raise RuntimeError('Saved reference result changed.')
                ref = rec['result']
            else:
                ref = reference.run(seed=SEED)
                owner.check()
                atomic_json(ref_path,{'request':fingerprint(request),'hash':fingerprint(ref),'result':ref})
            print('Learned reference passed:',ref['qualified'],
                  '| generic dense passed:',ref['generic_dense_qualified'],flush=True)
            if not ref['qualified']:
                return publish('complete','finished',outcome='positive_reference_failed',source_unchanged=sha(parent)==PARENT,
                               reference=str(ref_path),narrow_task_only=True)
            if (out/'STOP').exists():
                return stopped()
            phase='SFT-4000 binding fit'
            publish('running',phase,reference=str(ref_path))
            warm_dir=out/'warm'
            warm = fit.run(parent,tokenizer,warm_dir,steps=STEPS,seed=SEED,threads=THREADS)
            if warm['status']=='stopped':
                return stopped()
            if warm['status']!='complete' or warm['steps_completed'] != STEPS:
                raise RuntimeError('The warm comparison ended without its expected terminal state.')
            if sha(parent)!=PARENT:
                raise RuntimeError('The frozen source changed during the run.')
            rates = {name:warm[name] for name in GATES}
            if any(type(value) not in (int,float) or not 0 <= value <= 1 for value in rates.values()):
                raise RuntimeError('Invalid quartet-rate report.')
            outcome=decision(rates)
            copy_result = None
            bridge_result = None
            final_outcome = outcome
            if outcome != 'transfer_on_this_binding_task':
                phase = 'single-value copying calibration'
                publish('running',phase,warm_outcome=outcome,measured_rates=rates)
                if (out/'STOP').exists():
                    return stopped()
                calibrated = calibration.run(parent,tokenizer,out/'single_copy',max_steps=160,seed=SEED,threads=THREADS)
                if calibrated['status']=='stopped':
                    return stopped()
                if calibrated['status']!='complete':
                    raise RuntimeError('Copy calibration did not finish.')
                copy_result = {key:calibrated[key] for key in ('outcome','steps_completed','seen_correct',
                               'seen_total','dev_correct','dev_total','new_wording_correct','new_wording_total')}
                if copy_result['outcome'] == 'single_copy_passed':
                    phase='two-value retraining from the copy checkpoint'
                    publish('running',phase,warm_outcome=outcome,copy_calibration=copy_result)
                    if (out/'STOP').exists():
                        return stopped()
                    resumed=bridge.run(parent,tokenizer,out/'single_copy',out/'from_copy',
                                       steps=STEPS,seed=SEED,threads=THREADS)
                    if resumed['status']=='stopped':
                        return stopped()
                    if resumed['status']!='complete' or resumed['steps_completed']!=STEPS:
                        raise RuntimeError('Copy-to-binding stage ended without its expected terminal state.')
                    bridge_rates={name:resumed[name] for name in GATES}
                    if not all(type(v) in (float,int) and 0<=v<=1 for v in bridge_rates.values()):
                        raise RuntimeError('Invalid copy-to-binding report.')
                    bridge_result={'outcome':decision(bridge_rates),'rates':bridge_rates,'directory':str(out/'from_copy')}
                    final_outcome='after_copy_'+bridge_result['outcome']
                elif copy_result['outcome']=='single_copy_failed':
                    final_outcome='single_value_copy_failed'
                else:
                    raise RuntimeError('Unrecognized copy calibration outcome.')
            if sha(parent)!=PARENT:
                raise RuntimeError('Frozen checkpoint changed before publication.')
            result=publish('complete','finished',outcome=final_outcome,direct_outcome=outcome,measured_rates=rates,
                           copy_calibration=copy_result,from_copy=bridge_result,source_unchanged=True,
                           learned_reference_passed=ref['qualified'], generic_dense_reference_passed=ref['generic_dense_qualified'],
                           reference=str(ref_path),warm_output=str(warm_dir),narrow_task_only=True)
            print(json.dumps({'outcome':final_outcome,'direct_rates':rates,'copy_calibration':copy_result,
                              'from_copy':bridge_result},indent=2),flush=True)
            return result
        except Exception as exc:
            publish('error',phase,error=f'{type(exc).__name__}: {exc}')
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--recover-disconnected',action='store_true')
    args=parser.parse_args()
    run(args.root,args.recover_disconnected)
