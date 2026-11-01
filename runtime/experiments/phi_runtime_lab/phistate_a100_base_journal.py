"""Audit stage 4. Compare clean next-token learning, then conditionally continue the archived base."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from a100_run_guard import guarded_run
from chat_recovery_data import sha
from chat_recovery_train import atomic_json
from coherence_quality import code_passes
from device_preflight import run_preflight
from moe_train import ckpt_load
from phistate_base_reference import DenseCausalLM

import phistate_base_cuda_train as training
import phistate_base_windows as windows
from phistate_base_prerequisite import EVIDENCE

OUTPUT = 'phistate_a100_base_v1'
BASES = {
    '65K': ('run_60k_to_61k/latest.pt','c09d5d63207bc69812d8e70608ed5be017619f53995164c642185c2413a41127',65000),
    '66K': ('continue_60k/latest.pt','50f9bfe9b63082041b201bd6420531cbe6384cfc3ef5cce097588d8ec2396f46',66000),
}
CORPUS = '5a45ae09dfc42c6021202838439e77bd5cb05ff38d673d23e3219a1ecbc6c1d3'
STEPS, BATCH, CONTEXT = 512, 8, 128
SEEDS = (20261101,20261102)
DEV_SEED, CONFIRM_SEED = 152301, 729411
MIN_LEARNING = .15
MIN_CONTEXT_GAIN = .02
MIN_WRONG_HISTORY_MEAN = .01
HISTORY_DOMAIN_TOLERANCE = .02
MAX_DOMAIN_REGRESSION = .05
MIN_WARM_GAIN = .01
MODULES = ('phistate_a100_base_journal.py','phistate_base_windows.py','phistate_base_reference.py',
           'phistate_base_cuda_train.py','phistate_base_prerequisite.py','device_preflight.py')
PIN = '73edfa1a704e8d198269351ebe597e5ff8fed8ed'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def memo(path, identity, measure):
    if path.is_file():
        rec=read(path)
        if rec['identity']!=identity or rec['hash']!=digest(rec['result']):
            raise RuntimeError('Measurement from another experiment or damaged: '+str(path))
        return rec['result']
    result=measure()
    atomic_json(path,{'identity':identity,'hash':digest(result),'result':result})
    return result


def disjoint(left,right):
    for domain in ('code','general'):
        for a in left[domain]:
            for b in right[domain]:
                if a['offset']<=b['offset']+b['context'] and b['offset']<=a['offset']+a['context']:
                    return False
    return True


def validation_plans(corpus):
    dev=windows.validation_plan(corpus,context=CONTEXT,count=32,seed=DEV_SEED)
    for seed in range(CONFIRM_SEED,CONFIRM_SEED+1000):
        confirm=windows.validation_plan(corpus,context=CONTEXT,count=32,seed=seed)
        if disjoint(dev,confirm):
            return dev,confirm,seed
    raise RuntimeError('Could not construct a disjoint confirmation plan.')


def learns(start,end):
    history = [end[d]['wrong_history_gain_bpt'] for d in ('code','general')]
    return (all(start[d]['other_bpt']-end[d]['other_bpt'] >= MIN_LEARNING and
                end[d]['context_gain_bpt'] >= MIN_CONTEXT_GAIN for d in ('code','general'))
            and min(history) >= -HISTORY_DOMAIN_TOLERANCE and sum(history)/2 >= MIN_WRONG_HISTORY_MEAN)


def retains(start,end):
    return (start['mean_bpt']-end['mean_bpt']>=MIN_WARM_GAIN and
            all(end[d]['other_bpt']<=start[d]['other_bpt']+MAX_DOMAIN_REGRESSION for d in ('code','general')))


@torch.inference_mode()
def completions(model,sp,device):
    """Descriptive base-prompt readout. Generated tokens are the only tokens scored."""
    model.eval()
    cases = [('add','code','def add(a, b):\n'),('square','code','def square(x):\n'),
             ('storage','general','Computer storage is used to'),('os','general','The purpose of an operating system is')]
    out=[]
    for name,domain,prompt in cases:
        ids=sp.encode(f'<|{domain}|>\n'+prompt)
        emitted=[]; ended=False
        for _ in range(48):
            scores=model(torch.tensor([ids[-CONTEXT:]],device=device))[:, -1, :]
            if not torch.isfinite(scores).all():
                raise FloatingPointError('Nonfinite generation logits')
            token=int(scores.argmax(-1).item())
            if token==sp.eos_id():
                ended=True
                break
            emitted.append(token);ids.append(token)
        # Decode the whole prefix to preserve leading whitespace/byte fallback at the boundary.
        prefix=sp.decode(ids[:len(ids)-len(emitted)]) if emitted else sp.decode(ids)
        text=sp.decode(ids)[len(prefix):]
        row={'name':name,'prompt':prompt,'continuation':text,'ended':ended,'tokens':len(emitted)}
        if domain=='code':
            args=['a','b'] if name=='add' else ['x']
            inputs=([(2,-3),(-4,7),(0,0)] if name=='add' else [(0,),(-3,),(4,)])
            rubric={'function':name,'args':args,'cases':[{'args':list(vals),'expected':sum(vals) if name=='add' else vals[0]**2} for vals in inputs]}
            row['passes_implementation']=code_passes(prompt+text,rubric)
        out.append(row)
    return out


def endpoint_samples(path, arm, corpus, device):
    if arm == 'phistate':
        model,state=ckpt_load(str(path))
    else:
        state=torch.load(path,map_location='cpu',weights_only=True)
        model=DenseCausalLM(**state['config'])
        model.load_state_dict(state['state_dict'],strict=True)
    del state
    try:
        return completions(model.to(device),corpus.tokenizer,device)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()


def model_measure(path,corpus,val,device,probe=True):
    model,state=ckpt_load(str(path));del state
    try:
        model.to(device).eval()
        metrics=windows.evaluate(model,corpus,val,device)
        return {'metrics':metrics,'samples':completions(model,corpus.tokenizer,device) if probe else []}
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def prerequisite():
    payload=EVIDENCE['payload']
    if digest(payload)!=EVIDENCE['sha256']:
        raise RuntimeError('The recorded local reference evidence has changed.')
    if payload['reference_module_sha256']!=sha(Path(__file__).with_name('phistate_base_reference.py')):
        raise RuntimeError('Reference implementation differs from the one tested.')
    substantive=payload.get('changed_position_confirmation',{}).get('qualified') and payload.get('natural_cpu',{}).get('qualified')
    if bool(substantive)!=payload['qualified']:
        raise RuntimeError('Reference receipt qualification disagrees with its measurements.')
    return payload


def run(root,recover=False):
    proof=prerequisite()
    if not proof['qualified']:
        raise RuntimeError('Research prerequisite did not pass. Keep the A100 released.')
    training.require_a100()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    root=Path(root).resolve()
    if not root.is_dir(): raise FileNotFoundError(root)
    out=root/OUTPUT
    with guarded_run(out,recover=recover) as owner:
        out.mkdir(exist_ok=True)
        corpus=windows.open_corpus(root/'corpus_repaired_v1')
        if corpus.fingerprint!=CORPUS:
            raise RuntimeError('Repaired corpus fingerprint mismatch.')
        paths={}
        for name,(relative,expected,step) in BASES.items():
            path=root/relative
            if sha(path)!=expected: raise RuntimeError('Wrong '+name+' archived base.')
            state=torch.load(path,map_location='cpu',weights_only=True)
            if state.get('step')!=step or state.get('data_fingerprint')!=CORPUS or state.get('sft_steps',0):
                raise RuntimeError('Base lineage mismatch: '+name)
            paths[name]=path
            del state
        dev,confirm,conf_seed=validation_plans(corpus)
        plans={s:windows.schedule(corpus,steps=STEPS,batch_size=BATCH,context=CONTEXT,seed=s,code_share=.55) for s in SEEDS}
        request={'experiment':'PhiState-A100-base-v1','source_pin':PIN,'bases':BASES,'seeds':list(SEEDS),
                 'steps':STEPS,'batch':BATCH,'context':CONTEXT,'corpus':CORPUS,'torch':str(torch.__version__),
                 'cuda':torch.version.cuda,'fp32':True,'dev':digest(dev),'confirm':digest(confirm),'confirm_seed':conf_seed,
                 'plans':{str(s):p['fingerprint'] for s,p in plans.items()},'reference_evidence':EVIDENCE['sha256'],
                 'thresholds':{'minimum_other_bpt_drop':MIN_LEARNING,'minimum_longer_context_gain':MIN_CONTEXT_GAIN,
                              'minimum_wrong_history_mean':MIN_WRONG_HISTORY_MEAN,'history_domain_tolerance':HISTORY_DOMAIN_TOLERANCE,
                              'minimum_warm_bpt_gain':MIN_WARM_GAIN,'maximum_domain_other_bpt_regression':MAX_DOMAIN_REGRESSION},
                 'modules':{n:sha(Path(__file__).with_name(n)) for n in MODULES}}
        # Canonicalize tuples for immutable JSON comparison on later notebook runs.
        request=json.loads(json.dumps(request)); key=digest(request)
        req_path=out/'request.json'; report_path=out/'report.json'
        if req_path.exists():
            if read(req_path)!=request: raise RuntimeError('Configuration differs from this experiment. Use its original software to resume.')
        else:
            if any(p.name not in ('STOP','request.partial') for p in out.iterdir()):
                raise RuntimeError('Unexpected existing experiment output.')
            atomic_json(req_path,request)
        if report_path.exists():
            existing=read(report_path)
            if existing.get('status')=='complete':
                if existing.get('request')!=key: raise RuntimeError('Final report belongs to another run.')
                print('Already complete:',existing['outcome'],'|',report_path,flush=True)
                return existing
        def publish(status,phase,**fields):
            owner.check()
            record={'status':status,'phase':phase,'request':key,**fields}
            atomic_json(report_path,record);print(status.upper(),phase,'|',report_path,flush=True)
            return record
        stop=out/'STOP'
        phase='device check'
        try:
            if stop.exists():return publish('stopped',phase)
            publish('running',phase)
            pre=run_preflight(paths['66K'],root/'corpus_repaired_v1/tokenizer.model',out/'device_preflight.json',device='cuda')
            if not pre['passed']:raise RuntimeError('CPU/CUDA agreement check failed; see device_preflight.json.')
            archived={}
            for name,path in paths.items():
                phase='measure archived '+name
                publish('running',phase)
                archived[name]=memo(out/('baseline_'+name+'.json'),[key,BASES[name][1],'dev'],
                                    lambda path=path:model_measure(path,corpus,dev,'cuda'))
            passes={}; details={}
            for seed in SEEDS:
                passes[str(seed)]={}
                for arm in ('dense','phistate'):
                    phase=f'{arm}, seed {seed}'; arm_out=out/f'{arm}_{seed}'
                    if stop.exists():return publish('stopped',phase)
                    publish('running',phase)
                    info=training.run_arm(paths['66K'],corpus,arm_out,arm=arm,seed=seed,plan=plans[seed],validation=dev,
                        steps=STEPS,batch=BATCH,context=CONTEXT,stop_path=stop,ownership_check=owner.check)
                    if info['status']=='stopped':return publish('stopped',phase)
                    if (info['status'],info['trained_tokens'])!=('complete',STEPS*BATCH*CONTEXT):
                        raise RuntimeError('Unexpected controlled-arm budget.')
                    before,after=info['curve'][0]['metrics'],info['curve'][-1]['metrics']
                    passes[str(seed)][arm]=learns(before,after)
                    details[f'{arm}_{seed}']={'initial':before,'endpoint':after,'device':info['device_receipt'],
                                                 'checkpoint':info['latest'],'supervised_tokens':info['trained_tokens']}
                    endpoint=Path(info['latest'])
                    details[f'{arm}_{seed}']['samples']=memo(out/f'samples_{arm}_{seed}.json',
                        [key,sha(endpoint)],lambda:endpoint_samples(endpoint,arm,corpus,'cuda'))
            dense_ok=all(v['dense'] for v in passes.values());phi_ok=all(v['phistate'] for v in passes.values())
            if not (dense_ok and phi_ok):
                result=('phistate_context_gate_failed' if dense_ok else 'dense_context_gate_failed' if phi_ok
                        else 'scratch_control_inconclusive')
                if any(sha(paths[n])!=BASES[n][1] for n in paths):
                    raise RuntimeError('Archive changed during controlled comparison.')
                return publish('complete','scratch comparison finished',outcome=result,scratch_passes=passes,
                       baseline_files={n:str(out/f'baseline_{n}.json') for n in paths},source_unchanged=True,
                       comparison=details,scope='Base next-token learning; no native export or assistant qualification.',warm_started=False)
            phase='continue archived 66K on the first seed schedule'
            if stop.exists():return publish('stopped',phase)
            publish('running',phase)
            warm=training.run_arm(paths['66K'],corpus,out/'continued_66k',arm='warm',seed=SEEDS[0],
                      plan=plans[SEEDS[0]],validation=dev,steps=STEPS,batch=BATCH,context=CONTEXT,
                      stop_path=stop,ownership_check=owner.check)
            if warm['status']=='stopped':return publish('stopped',phase)
            if (warm['status'],warm['trained_tokens'])!=('complete',STEPS*BATCH*CONTEXT):
                raise RuntimeError('Incomplete warm training.')
            candidate=Path(warm['latest']); candidate_sha=sha(candidate)
            start,end=warm['curve'][0]['metrics'],warm['curve'][-1]['metrics']
            better=retains(start,end)
            conf=None
            if better:
                phase='confirm warm endpoint on disjoint validation windows'
                publish('running',phase)
                def compare():
                    return {'parent':model_measure(paths['66K'],corpus,confirm,'cuda',False),
                            'candidate':model_measure(candidate,corpus,confirm,'cuda')}
                conf=memo(out/'confirmation.json',[key,BASES['66K'][1],candidate_sha,'confirm'],compare)
                better=retains(conf['parent']['metrics'],conf['candidate']['metrics'])
            for name,path in paths.items():
                if sha(path)!=BASES[name][1]:raise RuntimeError('Archived parent changed externally.')
            if sha(candidate)!=candidate_sha:raise RuntimeError('Candidate changed before reporting.')
            return publish('complete','finished',outcome='base_loss_improved' if better else 'base_improvement_unconfirmed',
                scratch_passes=passes,comparison=details,warm_started=True,warm_checkpoint=str(candidate),
                warm_checkpoint_sha256=candidate_sha,warm_dev={'parent':start,'candidate':end},confirmation=conf,
                baseline_files={name:str(out/f'baseline_{name}.json') for name in paths},
                scope='Base next-token validation. Instruction following and native serving are separate qualifications.')
        except Exception as exc:
            publish('error',phase,error=f'{type(exc).__name__}: {exc}')
            raise


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',required=True)
    ap.add_argument('--recover-disconnected',action='store_true');a=ap.parse_args()
    run(a.root,a.recover_disconnected)
