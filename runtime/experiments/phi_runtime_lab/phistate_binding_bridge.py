"""Continue the verified single-copy checkpoint on the same two-record crossover cases."""
import json
from pathlib import Path

import sentencepiece as spm
import torch

import phistate_binding_data as data
import phistate_binding_fit as warm
import phistate_copy_calibration as copy
from chat_recovery_data import TEMPLATE, sha
from chat_recovery_train import atomic_json, run_lock
from moe_train import ckpt_load


def source_report(copy_dir):
    report=json.loads((copy_dir/'report.json').read_text())
    if (report.get('status')!='complete' or report.get('outcome')!='single_copy_passed'
            or report.get('seen_total')!=32 or report.get('dev_total')!=32
            or report.get('seen_correct',0)<29 or report.get('dev_correct',0)<24
            or report.get('request',{}).get('protocol')!='phistate-copy-calibration-v1'
            or report['request'].get('source')!=warm.SOURCE or report['request'].get('tokenizer')!=warm.TOKENIZER):
        raise ValueError('The bridge requires a passing verified copy calibration')
    return report


def measure_copy(model,pools):
    return {k:warm.evaluate(model,pools[k],1) for k in ('train_familiar','dev_familiar','dev_new')}


def run(parent,tokenizer_path,copy_dir,output,*,steps=96,seed=20261101,threads=2):
    parent,tokenizer_path,copy_dir,out=map(lambda p:Path(p).resolve(),(parent,tokenizer_path,copy_dir,output))
    if (out in (copy_dir,parent,tokenizer_path.parent,parent.parent) or not 1<=steps<=192
            or not 1<=threads<=8):
        raise ValueError('Use an independent bridge output, 1..192 updates and 1..8 CPU threads')
    if sha(parent)!=warm.SOURCE or sha(tokenizer_path)!=warm.TOKENIZER:
        raise ValueError('Wrong frozen SFT-4000 source or tokenizer')
    base=source_report(copy_dir)
    src=copy_dir/'latest.pt'
    req=dict(protocol='phistate-copy-to-binding-v1',source=warm.SOURCE,tokenizer=warm.TOKENIZER,
             copy_weights=sha(src),copy_report=sha(copy_dir/'report.json'),copy_completed=base['steps_completed'],
             steps=steps,threads=threads,seed=seed,lr=warm.LR,quartets_per_update=4,torch=str(torch.__version__),
             code={p.name:sha(p) for p in (Path(__file__),Path(warm.__file__),Path(copy.__file__),Path(data.__file__))},
             optimizer='resume the passing single-copy AdamW weights and moments')
    out.mkdir(parents=True,exist_ok=True)
    with run_lock(out):
        manifest=out/'request.json'
        if manifest.exists():
            if json.loads(manifest.read_text())!=req: raise RuntimeError('Bridge request mismatch')
        elif any(p.name not in ('STOP','request.partial') for p in out.iterdir()):
            raise RuntimeError('Unidentified bridge directory')
        else:atomic_json(manifest,req)
        old_threads=torch.get_num_threads()
        try:
            torch.set_num_threads(threads)
            with torch.random.fork_rng(devices=[]):
                result=_fit(parent,tokenizer_path,src,base,out,req)
            if sha(parent)!=warm.SOURCE or sha(src)!=req['copy_weights']:
                raise RuntimeError('Source or copy checkpoint changed during bridge')
            return result
        finally:
            torch.set_num_threads(old_threads)


def _fit(parent,tok_path,src,base,out,req):
    report=out/'report.json';latest=out/'latest.pt'
    if report.exists():
        record=json.loads(report.read_text())
        if record['request']!=req:raise RuntimeError('Bridge report mismatch')
        if record['status']=='complete':return record
    m,s=ckpt_load(str(parent))
    if s.get('sft_steps')!=4000 or s.get('chat_template')!=TEMPLATE: raise ValueError('Wrong parent format')
    del s
    tok=spm.SentencePieceProcessor(model_file=str(tok_path))
    pools=warm.banks(tok)
    copy_bank=copy.datasets(tok)
    state=torch.load(src,map_location='cpu',weights_only=True)
    if state.get('request')!=base['request'] or state.get('step')!=base['steps_completed']:
        raise ValueError('Passing copy report does not identify these weights')
    m.load_state_dict(state['weights'])
    opt=torch.optim.AdamW(m.parameters(),lr=warm.LR,weight_decay=0.,foreach=False)
    opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['rng']);del state
    copy_start=measure_copy(m,copy_bank)
    if (copy_start['train_familiar']['exact_digit_then_eos']!=base['seen_correct'] or
            copy_start['dev_familiar']['exact_digit_then_eos']!=base['dev_correct'] or
            copy_start['dev_new']['exact_digit_then_eos']!=base['new_wording_correct']):
        raise ValueError('Could not reproduce the recorded passing copy checkpoint')
    step=0;curve={}
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=True)
        if state['request']!=req: raise RuntimeError('Bridge continuation checkpoint mismatch')
        m.load_state_dict(state['weights']);opt.load_state_dict(state['optimizer'])
        step=state['step'];curve=state['curve'];torch.set_rng_state(state['rng']);del state
    def checkpoint():
        warm.save(latest,dict(request=req,weights=m.state_dict(),optimizer=opt.state_dict(),
                              step=step,curve=curve,rng=torch.get_rng_state()))
    def record(status):
        result=dict(request=req,status=status,steps_completed=step,curve=curve,
                    copy_baseline=copy_start,checkpoint=str(latest) if latest.exists() else None)
        if str(step) in curve:
            r=curve[str(step)]
            result.update(seen_quartet_rate=r['binding']['train_familiar']['group_rate'],
                          dev_familiar_wording_quartet_rate=r['binding']['dev_familiar']['group_rate'],
                          dev_new_wording_quartet_rate=r['binding']['dev_new']['group_rate'],
                          copy_retention=r['copy'])
        atomic_json(report,result);return result
    def stop():return (out/'STOP').exists() or (out.parent/'STOP').exists()
    if stop():return record('stopped')
    if '0' not in curve:
        curve['0']=dict(binding=warm.measure(m,pools),copy=copy_start);record('running')
    train=pools['train_familiar']
    for n in range(step+1,req['steps']+1):
        if stop():
            if step: checkpoint()
            return record('stopped')
        ids=warm.pick(len(train),4,(n-1)*4,req['seed']+1000)
        rows=[r for i in ids for r in train[i]]
        m.train();opt.zero_grad(set_to_none=True);loss=warm.objective(m,rows)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite bridge objective')
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
        opt.step();step=n
        if n in (req['steps']//2,req['steps']):
            curve[str(n)]=dict(binding=warm.measure(m,pools),copy=measure_copy(m,copy_bank))
        if n%16==0 or n in (req['steps']//2,req['steps']):
            checkpoint();record('running')
            print(f'Copy → two-record binding {step}/{req["steps"]}, loss {float(loss.detach()):.4f}',flush=True)
    return record('complete')
