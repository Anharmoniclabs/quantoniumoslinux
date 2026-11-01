"""A CPU positive-control ladder on the warm checkpoint's simplest prerequisite: copy one digit."""
import json
from pathlib import Path

import sentencepiece as spm
import torch

import phistate_binding_data as data
import phistate_binding_fit as warm
from chat_recovery_data import TEMPLATE, sha
from chat_recovery_train import atomic_json, run_lock
from moe_train import ckpt_load


def datasets(tok):
    bank=data.make_bank()
    output={}
    bindings={}
    for split,groups in bank.items():
        edges=sorted({(row['query'],row['answer']) for group in groups for row in data.cases(group)})
        bindings[split]=set(edges)
        for regime in ('familiar','new'):
            rows=[]
            for k,v in edges:
                if regime=='familiar':
                    p=f'Record: {k}={v}. Return the value of {k}. Reply with one digit.'
                else:
                    p=f'The digit stored for {k} is {v}. What digit does {k} hold? Reply with one digit.'
                other=sorted(val for key,val in edges if key==k and val!=v)
                rows.append(warm.encode(tok,p,v,other[int(v)%len(other)]))
            output[split+'_'+regime]=rows
    if bindings['train'] & bindings['dev'] or any(len(edges)!=32 for edges in bindings.values()):
        raise ValueError('Single-copy bindings overlap or the fixed bank is incomplete')
    for rows in output.values():
        if len({r[0][:-1] for r in rows})!=32:
            raise ValueError('Repeated copy prompt')
    return output


def run(parent,tokenizer_path,output,*,max_steps=160,seed=20261101,threads=2):
    parent,tokenizer_path,out=map(lambda p:Path(p).resolve(),(parent,tokenizer_path,output))
    if out in (parent.parent,tokenizer_path.parent) or out==parent or not 16<=max_steps<=256 or not 1<=threads<=8:
        raise ValueError('Use a separate output, 16..256 updates and 1..8 CPU threads')
    if sha(parent)!=warm.SOURCE or sha(tokenizer_path)!=warm.TOKENIZER:
        raise ValueError('Expected the archived SFT-4000 source and repaired tokenizer')
    req=dict(protocol='phistate-copy-calibration-v1',source=warm.SOURCE,tokenizer=warm.TOKENIZER,
             max_steps=max_steps,seed=seed,threads=threads,torch=str(torch.__version__),lr=warm.LR,
             state='fresh AdamW, all PhiState parameters',seen_threshold=.9,dev_threshold=.75,
             code={p.name:sha(p) for p in (Path(__file__),Path(data.__file__),Path(warm.__file__))})
    out.mkdir(parents=True,exist_ok=True)
    with run_lock(out):
        manifest=out/'request.json'
        if manifest.exists():
            if json.loads(manifest.read_text())!=req:
                raise RuntimeError('Copy calibration request differs')
        elif any(p.name not in ('STOP','request.partial') for p in out.iterdir()):
            raise RuntimeError('Unidentified copy calibration directory')
        else:
            atomic_json(manifest,req)
        old=torch.get_num_threads()
        try:
            torch.set_num_threads(threads)
            with torch.random.fork_rng(devices=[]):
                result=_run(parent,tokenizer_path,out,req)
            if sha(parent)!=warm.SOURCE:
                raise RuntimeError('Frozen source changed during single-copy training')
            return result
        finally:
            torch.set_num_threads(old)


def _run(parent,tokenizer_path,out,req):
    report_path=out/'report.json';ck=out/'latest.pt'
    if report_path.exists():
        saved=json.loads(report_path.read_text())
        if saved['request']!=req:
            raise RuntimeError('Copy outcome belongs to another run')
        if saved['status']=='complete':
            return saved
    tok=spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    pools=datasets(tok)
    model,state=ckpt_load(str(parent))
    if state.get('sft_steps')!=4000 or state.get('chat_template')!=TEMPLATE:
        raise ValueError('Wrong chat lineage')
    del state
    torch.manual_seed(req['seed'])
    opt=torch.optim.AdamW(model.parameters(),lr=warm.LR,weight_decay=0.,foreach=False)
    step=0;curve={}
    if ck.exists():
        state=torch.load(ck,weights_only=True,map_location='cpu')
        if state['request']!=req:raise RuntimeError('Checkpoint identity mismatch')
        model.load_state_dict(state['weights']);opt.load_state_dict(state['optimizer'])
        curve=state['curve'];step=state['step'];torch.set_rng_state(state['rng']);del state
    def publish(status,outcome,metrics=None,novel=None):
        m=metrics or {};seen=m.get('train_familiar',{});dev=m.get('dev_familiar',{})
        result=dict(request=req,status=status,outcome=outcome,steps_completed=step,curve=curve,
                    seen_correct=seen.get('exact_digit_then_eos'),seen_total=seen.get('items'),
                    dev_correct=dev.get('exact_digit_then_eos'),dev_total=dev.get('items'),
                    new_wording_correct=(novel or {}).get('exact_digit_then_eos'),
                    new_wording_total=(novel or {}).get('items'))
        atomic_json(report_path,result)
        return result
    def save():
        warm.save(ck,dict(request=req,weights=model.state_dict(),optimizer=opt.state_dict(),
                          step=step,curve=curve,rng=torch.get_rng_state()))
    def stop():
        return (out/'STOP').exists() or (out.parent/'STOP').exists()
    if stop():return publish('stopped','stopped')
    def measure():
        return {k:warm.evaluate(model,pools[k],group_size=1) for k in ('train_familiar','dev_familiar')}
    if '0' not in curve:
        curve['0']=measure()
    while True:
        if str(step) in curve:
            m=curve[str(step)]
            passed=m['train_familiar']['group_rate']>=.9 and m['dev_familiar']['group_rate']>=.75
            if passed or step==req['max_steps']:
                # New wording is examined only after the predetermined stopping rule.
                novel=warm.evaluate(model,pools['dev_new'],group_size=1)
                return publish('complete','single_copy_passed' if passed else 'single_copy_failed',m,novel)
        if stop():
            if step:save()
            return publish('stopped','stopped')
        rows=pools['train_familiar']
        group=[rows[i] for i in warm.pick(len(rows),8,step*8,req['seed'])]
        model.train();opt.zero_grad(set_to_none=True)
        loss=warm.objective(model,group)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite copy objective')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step();step+=1
        if step%16==0 or step==req['max_steps']:
            m=measure();curve[str(step)]=m;save()
            publish('running','evaluating',m)
            print(f"Single copy {step}/{req['max_steps']}: seen {m['train_familiar']['exact_digit_then_eos']}/32,"
                  f" heldout {m['dev_familiar']['exact_digit_then_eos']}/32",flush=True)
