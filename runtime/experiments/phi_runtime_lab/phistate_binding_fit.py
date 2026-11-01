"""Bounded CPU-only fit test for the frozen broad SFT-4000 PhiState checkpoint."""
import hashlib
import json
import random
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

from chat_recovery_data import TEMPLATE, encode_example, sha
from chat_recovery_train import atomic_json, run_lock
from moe_train import ckpt_load
from phistate_binding_data import cases, make_bank, wording

SOURCE = 'dcc79a5a3b05411725fefcefa04fd0aafd4393f4efb21e018c60da1e9a80d8ea'
TOKENIZER = '9f704ff69de6cb60fd7466a26b70cbd0472bb1a192f1825a7c52cfc30179ea04'
STEPS = 96
COPY_STEPS = 16
GROUPS_PER_UPDATE = 4
SEED = 20261101
LR = 0.0004


def encode(tokenizer, prompt, answer, foil):
    def make(target):
        return encode_example([{'role':'user','content':prompt},{'role':'assistant','content':target}],
                              tokenizer, 512)
    x,y = make(answer)
    fx,fy = make(foil)
    active = [j for j,t in enumerate(y) if t != -100]
    if (len(active) != 2 or active != [len(x)-2,len(x)-1] or y[-1] != tokenizer.eos_id()
            or x[:-1] != fx[:-1] or fy[-1] != tokenizer.eos_id() or x[-1] != y[-2]
            or fx[-1] != fy[-2] or x[-1] == fx[-1]):
        raise ValueError('Expected identical prefix and one distinct answer digit followed by EOS.')
    return (tuple(x), x[-1], fx[-1], y[-1])


def banks(tok):
    groups = make_bank()
    banks = {}
    train_edges = set()
    for split in groups:
        for regime in ('familiar','new'):
            collection = []
            for group in groups[split]:
                examples=[]
                for row in cases(group):
                    if split == 'train':
                        train_edges.add((row['query'],row['answer']))
                    examples.append(encode(tok, wording(row, regime), row['answer'],row['foil']))
                collection.append(examples)
            banks[split+'_'+regime]=collection
    # Single-copy priming only uses allowed training associations.
    copy=[]
    for key,answer in sorted(train_edges):
        prompt=f'Record: {key}={answer}. Return the value of {key}. Reply with one digit.'
        # Keep priming contrast labels within the training bindings for this key.
        choices=sorted(value for k,value in train_edges if k==key and value!=answer)
        foil=choices[int(answer)%len(choices)]
        copy.append(encode(tok,prompt,answer,foil))
    banks['copy']=copy
    train={r[0][:-1] for group in banks['train_familiar'] for r in group}
    for collection in ('dev_familiar','dev_new','train_new'):
        if any(r[0][:-1] in train for g in banks[collection] for r in g):
            raise ValueError('Warm fit/transfer prefix collision')
    return banks


def predict(model,batch):
    length=max(len(x[0]) for x in batch)
    ids=torch.tensor([list(r[0])+[0]*(length-len(r[0])) for r in batch])
    pos=torch.tensor([len(r[0])-2 for r in batch])
    target,foil,eos=torch.tensor([r[1:] for r in batch]).T
    logits=model(ids)
    samples=torch.arange(len(batch))
    return logits[samples,pos],logits[samples,pos+1],target,foil,eos


def objective(model,batch):
    numeric,stop,target,foil,eos=predict(model,batch)
    rows=torch.arange(len(batch))
    # Full-vocabulary answer CE plus a counterfactual term at the digit itself.
    # Supervise EOS separately; boilerplate and prompt tokens receive no targets.
    return (F.cross_entropy(numeric,target) + .5*F.cross_entropy(stop,eos)
            + F.softplus(1. + numeric[rows,foil] - numeric[rows,target]).mean())


@torch.inference_mode()
def evaluate(model,rows,group_size=4):
    model.eval()
    flags=[]; digit_hits=choice_hits=0
    for i in range(0,len(rows),16):
        batch=rows[i:i+16]
        first,second,target,foil,eos=predict(model,batch)
        idx=torch.arange(len(batch))
        digit=first.argmax(1)==target
        # When the first token is right this is exactly the second greedy step.
        flags.extend((digit & (second.argmax(1)==eos)).tolist())
        digit_hits+=int(digit.sum())
        choice_hits+=int((first[idx,target]>first[idx,foil]).sum())
    total=len(rows)
    successes=sum(all(flags[i:i+group_size]) for i in range(0,total,group_size))
    return dict(items=total,exact_digit_then_eos=sum(flags),top_digit_hits=digit_hits,
                digit_over_foil=choice_hits,groups=total//group_size,all_correct_groups=successes,
                group_rate=successes/(total//group_size))


def measure(model,bank):
    metrics={name:evaluate(model,[row for group in bank[name] for row in group])
             for name in ('train_familiar','train_new','dev_familiar','dev_new')}
    metrics['single_copy_seen']=evaluate(model,bank['copy'],1)
    return metrics


def pick(length, n, offset, seed):
    """Balanced repeating decks; no group starves and retries are reproducible."""
    result=[]
    prev=None
    for i in range(offset,offset+n):
        block,pos=divmod(i,length)
        if block!=prev:
            deck=list(range(length))
            random.Random(seed+block).shuffle(deck)
            prev=block
        result.append(deck[pos])
    return result


def save(path, value):
    tmp=path.with_suffix('.partial.pt')
    torch.save(value,tmp)
    tmp.replace(path)


def fit(model,bank,out,request,steps,seed):
    state_file=out/'latest.pt'
    report_file=out/'report.json'
    if report_file.exists():
        result=json.loads(report_file.read_text())
        if result['request'] != request:
            raise RuntimeError('Warm fit output belongs to another request')
        if result['status']=='complete':
            return result
    torch.manual_seed(seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.,foreach=False)
    phase=0; curve={}
    if state_file.exists():
        state=torch.load(state_file,map_location='cpu',weights_only=True)
        if state['request']!=request:
            raise RuntimeError('Warm fit checkpoint belongs to another request')
        model.load_state_dict(state['weights']);optimizer.load_state_dict(state['optimizer'])
        phase=state['step'];curve=state['curve'];torch.set_rng_state(state['rng'])
        del state
    copy_steps=min(COPY_STEPS,max(1,steps//4))
    def record(status):
        result=dict(status=status,steps_completed=phase,request=request,curve=curve,
                    checkpoint=str(state_file) if state_file.exists() else None)
        if str(phase) in curve:
            metrics=curve[str(phase)]
            result.update(seen_quartet_rate=metrics['train_familiar']['group_rate'],
                          dev_familiar_wording_quartet_rate=metrics['dev_familiar']['group_rate'],
                          dev_new_wording_quartet_rate=metrics['dev_new']['group_rate'])
        atomic_json(report_file,result)
        return result
    def stopping():
        return (out/'STOP').exists() or (out.parent/'STOP').exists()
    def snapshot():
        save(state_file,dict(request=request,step=phase,curve=curve,weights=model.state_dict(),
                             optimizer=optimizer.state_dict(),rng=torch.get_rng_state()))
    if stopping():
        return record('stopped')
    if '0' not in curve:
        curve['0']=measure(model,bank);record('running')
    for step in range(phase+1,steps+1):
        if stopping():
            if phase: snapshot()
            return record('stopped')
        if step<=copy_steps:
            rows=[bank['copy'][i] for i in pick(len(bank['copy']),8,(step-1)*8,seed)]
        else:
            train=bank['train_familiar']
            ids=pick(len(train),GROUPS_PER_UPDATE,(step-copy_steps-1)*GROUPS_PER_UPDATE,seed+1000)
            rows=[r for i in ids for r in train[i]]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss=objective(model,rows)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite binding objective')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();phase=step
        if step in (copy_steps,steps):
            curve[str(step)]=measure(model,bank)
        if step%16==0 or step in (copy_steps,steps):
            snapshot()
            print(f'Warm PhiState binding {step}/{steps}, loss {float(loss.detach()):.4f}',flush=True)
            record('running')
    return record('complete')


def run(parent,tokenizer_path,output,*,steps=STEPS,seed=SEED,threads=2):
    parent,tokenizer_path,out=map(lambda p:Path(p).resolve(),(parent,tokenizer_path,output))
    if not 2<=steps<=192 or not 1<=threads<=8:
        raise ValueError('Use 2..192 CPU updates and 1..8 threads')
    if out in (parent.parent,tokenizer_path.parent) or out==parent:
        raise ValueError('Use a separate diagnostic output directory')
    if sha(parent)!=SOURCE or sha(tokenizer_path)!=TOKENIZER:
        raise ValueError('Expected the frozen SFT-4000 source and repaired tokenizer')
    import phistate_binding_data
    req=dict(protocol='phistate-binding-fit-1',source=SOURCE,tokenizer=TOKENIZER,steps=steps,seed=seed,
             torch=str(torch.__version__),threads=threads,optimizer='fresh AdamW, no decay',lr=LR,
             source_code_sha256=sha(Path(__file__)),data_code_sha256=sha(Path(phistate_binding_data.__file__)),
             device='cpu',candidate_scope='narrow disposable diagnostic; no native promotion')
    out.mkdir(parents=True,exist_ok=True)
    with run_lock(out):
        manifest=out/'request.json'
        if manifest.is_file():
            if json.loads(manifest.read_text())!=req:
                raise ValueError('Warm fit settings changed; use a new output directory')
        elif any(p.name not in ('STOP','request.partial') for p in out.iterdir()):
            raise ValueError('Unidentified files in warm fit output directory')
        else:
            atomic_json(manifest,req)
        old_threads=torch.get_num_threads()
        try:
            torch.set_num_threads(threads)
            model,state=ckpt_load(str(parent))
            if (state.get('sft_steps')!=4000 or state.get('chat_template')!=TEMPLATE
                    or state.get('tokenizer_sha256')!=TOKENIZER or state.get('training_stage')!='chat_recovery_v1'):
                raise ValueError('Wrong source lineage')
            del state
            tok=spm.SentencePieceProcessor(model_file=str(tokenizer_path))
            with torch.random.fork_rng(devices=[]):
                result=fit(model,banks(tok),out,req,steps,seed)
            if sha(parent)!=SOURCE:
                raise RuntimeError('Source changed during diagnostic')
            return result
        finally:
            torch.set_num_threads(old_threads)
