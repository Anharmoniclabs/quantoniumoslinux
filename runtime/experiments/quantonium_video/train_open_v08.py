"""Warm-start V0.7 on open RGB movie triples with frozen RAFT and JC rehearsal.

This is a separate fine-tuning lineage. It cannot resume into a V0.7 output
directory and does not replace the original character checkpoint or automation.
"""
import argparse
import fcntl
import json
from pathlib import Path
import random
import signal
import os

import torch
from torch.nn import functional as F

from open_flow import guides_from_flows, sha256, RAFT_SHA256
from train_v07 import (PoseInterpolator,PoseData,DEFAULT_ASSETS,DATA_ID,
                       atomic_json,batch,supervised_loss,evaluate)

VERSION = 'qvideo-open-v08-raft-mixed-v1'


class OpenData:
    def __init__(self, root, size):
        root = Path(root).expanduser().resolve()
        self.fingerprint = sha256(root/'manifest.json')
        self.manifest = m = json.loads((root/'manifest.json').read_text())
        if m['schema'] != 1 or m['size'] != size or m['teacher_sha256'] != RAFT_SHA256:
            raise ValueError('Unsupported dataset/size/teacher')
        self.items = {'train':[],'validation':[]}
        films = {'train':set(),'validation':set()}
        for row in m['records']:
            path = (root/row['file']).resolve()
            if not path.is_relative_to(root) or sha256(path) != row['sha256']:
                raise ValueError('Dataset path or checksum differs')
            split = row['split']
            if split not in films or m['sources'][row['source']]['split'] != split:
                raise ValueError('Invalid source split')
            value = torch.load(path,map_location='cpu',weights_only=True)
            if value['frames'].shape != (3,3,size,size) or value['frames'].dtype != torch.uint8:
                raise ValueError('Expected uint8 RGB triplets')
            for key in ('ab','ba'):
                if value[key].shape != (2,size,size) or not torch.isfinite(value[key]).all():
                    raise ValueError('Invalid cached flow')
            if row['t'] != .5:
                raise ValueError('Only evenly spaced midpoint triples are supported')
            self.items[split].append(value)
            films[split].add(row['source'])
        if not all(self.items.values()) or films['train'] & films['validation']:
            raise ValueError('Need nonempty film-disjoint splits')

    def tensors(self, values):
        rgb = torch.stack([v['frames'] for v in values]).float()/255
        alpha = torch.ones_like(rgb[:,:,:1])
        a,y,b = torch.cat((rgb,alpha),2).unbind(1)
        guides = guides_from_flows(torch.stack([v['ab'] for v in values]),
                                  torch.stack([v['ba'] for v in values]),.5)
        return a,y,b,guides


@torch.no_grad()
def external_eval(model, data):
    model.eval()
    device = next(model.parameters()).device
    totals = dict(model_l1=0.,crossfade_l1=0.,raft_warp_l1=0.)
    for value in data.items['validation']:
        a,y,b,g = (x.to(device) for x in data.tensors([value]))
        pred,*_ = model(a,b,.5,0.,g)
        warp = .5*(model.warp(a,g[:,9:11])+model.warp(b,g[:,11:13]))
        for key,output in [('model_l1',pred),('crossfade_l1',(a+b)/2),('raft_warp_l1',warp)]:
            totals[key] += F.l1_loss(output[:,:3],y[:,:3]).item()
    model.train()
    return {**{k:v/len(data.items['validation']) for k,v in totals.items()},
            'samples':len(data.items['validation']),'scope':'held-out Sintel trailer; not unseen-animation proof'}


def evaluate_jc(model, data):
    # The source-preserving Blender evaluator/renderer uses CPU image geometry.
    # Evaluate a small CPU copy; do not move the live optimizer's parameters.
    if next(model.parameters()).device.type == 'cpu':
        return evaluate(model,data)
    import copy
    return evaluate(copy.deepcopy(model).cpu(),data)


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(808)
    rng = random.Random(808)
    parent = Path(args.init).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if out == parent.parent:
        raise ValueError('Use a separate output directory; original checkpoint is read-only')
    requested = getattr(args,'device','cpu')
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; select a GPU runtime')
    device = torch.device(requested)
    out.mkdir(parents=True,exist_ok=True)
    with (out/'training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state = torch.load(parent,map_location='cpu',weights_only=True)
        if state['config']['version'] != DATA_ID:
            raise ValueError('Warm-start requires a V0.7 checkpoint')
        size,width = state['config']['size'],state['config']['width']
        jc = PoseData(args.assets,size,args.motion_checkpoint)
        jc.attach_shot(args.shot)
        if state['config']['data'] != jc.fingerprint or state['config']['shot'] != jc.shot_hash:
            raise ValueError('JC rehearsal must match the parent checkpoint data and shot')
        data = OpenData(args.data,size)
        model = PoseInterpolator(size,width).to(device)
        model.load_state_dict(state['model'],strict=True)
        optim = torch.optim.AdamW(model.parameters(),lr=args.lr)
        config = dict(version=VERSION,size=size,width=width,parent_sha256=sha256(parent),
            parent_step=int(state['step']),data=data.fingerprint,jc=jc.fingerprint,shot=jc.shot_hash,
            batch=args.batch,lr=args.lr,jc_weight=args.jc_weight)
        step,best = 0,float('inf')
        latest = out/'latest.pt'
        if getattr(args,'require_resume',False) and not latest.exists():
            raise FileNotFoundError('Resume required: latest.pt is missing; restore the saved run')
        if latest.exists():
            saved = torch.load(latest,map_location='cpu',weights_only=True)
            if saved['config'] != config:
                raise ValueError('Resume config/data/parent changed; use original inputs')
            model.load_state_dict(saved['model'],strict=True)
            optim.load_state_dict(saved['optimizer'])
            rng.setstate(saved['rng'])
            torch.set_rng_state(saved['torch_rng'])
            if device.type == 'cuda' and saved.get('cuda_rng'):
                if len(saved['cuda_rng']) != torch.cuda.device_count():
                    raise ValueError('CUDA device count changed; cannot restore all RNG states')
                torch.cuda.set_rng_state_all(saved['cuda_rng'])
            step,best = saved['step'],saved['best']
            print('RESUMED V08',step,flush=True)
        else:
            atomic_json(out/'metrics_initial.json',dict(external=external_eval(model,data),jc=evaluate_jc(model,jc)))
            print('WARM START',config['parent_step'],'new optimizer; phase step 0',flush=True)
        atomic_json(out/'run_manifest.json',dict(config=config,device=str(device),torch_version=str(torch.__version__),
            training='one external minibatch plus one JC rehearsal minibatch per update',
            teacher='frozen RAFT C_T_V2 endpoint-only; no target-frame leakage',
            sources=data.manifest['sources'],auto_promote=False))
        stopping = False
        def stop(*_):
            nonlocal stopping
            stopping = True
        signal.signal(signal.SIGTERM,stop)
        signal.signal(signal.SIGINT,stop)

        def save():
            nonlocal best
            metrics = dict(phase_step=step,parent_step=config['parent_step'],
                external=external_eval(model,data),jc=evaluate_jc(model,jc))
            improved = metrics['external']['model_l1'] < best
            best = min(best,metrics['external']['model_l1'])
            tmp = out/'latest.tmp.pt'
            torch.save(dict(config=config,step=step,best=best,model=model.state_dict(),
                optimizer=optim.state_dict(),rng=rng.getstate(),torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                saved_device=str(device)),tmp)
            os.replace(tmp,latest)
            if improved:
                import shutil
                shutil.copyfile(latest,out/'best.tmp.pt')
                os.replace(out/'best.tmp.pt',out/'best.pt')
            atomic_json(out/'metrics_latest.json',metrics)
            if getattr(args,'backup_dir',''):
                from colab_video_runtime import save_snapshot
                saved_path = save_snapshot(out,Path(args.backup_dir).expanduser())
                print('DRIVE SNAPSHOT VERIFIED',step,saved_path,flush=True)
            print(json.dumps(metrics),flush=True)

        while (args.steps == 0 or step < args.steps) and not stopping:
            if getattr(args,'stop_file','') and Path(args.stop_file).exists():
                stopping = True
                break
            examples = [rng.choice(data.items['train']) for _ in range(args.batch)]
            a,y,b,g = (x.to(device) for x in data.tensors(examples))
            p,*_ = model(a,b,.5,0.,g)
            # Opaque movies carry no silhouette/contact annotations. RGB + alpha
            # coverage only; never invent JC floor labels for arbitrary films.
            external = F.l1_loss(p[:,:3],y[:,:3])+.1*F.l1_loss(p[:,3:],y[:,3:])
            ja,jy,jb,t,c,bg,jg = (x.to(device) for x in batch(jc,rng,args.batch))
            jp,*_ = model(ja,jb,t,c,jg)
            rehearsal,_ = supervised_loss(jp,jy,bg,c)
            loss = external+args.jc_weight*rehearsal
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss; last checkpoint retained')
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optim.step()
            step += 1
            if step%args.save_every == 0:
                save()
        if step%args.save_every or not latest.exists():
            save()
        print('STOPPED phase' if stopping else 'FINISHED phase',step,flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('init','out','data','shot'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--assets',default=str(DEFAULT_ASSETS))
    p.add_argument('--motion-checkpoint',default='')
    p.add_argument('--steps',type=int,default=100,help='TOTAL phase steps; 0 keeps training until a stop request')
    p.add_argument('--require-resume',action='store_true',help='Refuse to start without latest.pt')
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--save-every',type=int,default=25)
    p.add_argument('--lr',type=float,default=.00003)
    p.add_argument('--jc-weight',type=float,default=1.)
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='cpu')
    p.add_argument('--backup-dir',default='',help='Write verified immutable snapshots here after every save')
    p.add_argument('--stop-file',default='',help='Finish current update and save when this file exists')
    args = p.parse_args()
    if args.steps < 0 or min(args.batch,args.threads,args.save_every) < 1 or not 0 < args.lr <= .001 or not .1 <= args.jc_weight <= 10:
        p.error('Nonnegative steps; positive batch/threads/save interval; lr <= .001; jc-weight .1..10')
    train(args)
