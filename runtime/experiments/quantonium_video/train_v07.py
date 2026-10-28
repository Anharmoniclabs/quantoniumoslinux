#!/usr/bin/env python3
"""JC V0.7: learn RGBA in-betweens from the eight V0.6 cels on CPU.

Sparse-pose supervision plus controlled affine examples, temporal regularization,
sole/contact losses and layered-scene composition. Experimental interpolation,
not a pretrained generative video model or evidence of animation generalization.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from train_v03 import TemporalWarpNet, edges
from v07_scene import (FLOOR, SUPPORT_X, KEY_TIMES, LIFT, contact_point,
                       grounded, over, conditioning, shift)

DATA_ID = 'jc-v07-blender-guided-rgba-v2'
DEFAULT_ASSETS = Path(__file__).parent / 'assets/jc_v06_lossless'


class PoseInterpolator(TemporalWarpNet):
    """V0.3 U-Net backbone, new RGBA/contact inputs and independently trained head."""
    def __init__(self, size=128, width=16):
        super().__init__(size, width)
        self.enc1[0] = nn.Conv2d(23, width, 3, padding=1)
        self.head = nn.Conv2d(width, 9, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def warp(self, img, pixels):
        h, w = img.shape[-2:]
        delta = pixels.permute(0, 2, 3, 1) * pixels.new_tensor([2/(w-1), 2/(h-1)])
        return F.grid_sample(img, self.grid+delta, align_corners=True,
                             padding_mode='zeros')

    def forward(self, a, b, t, contact, guides, return_fields=False):
        n, _, h, w = a.shape
        t = torch.as_tensor(t, device=a.device, dtype=a.dtype).reshape(-1,1,1,1)
        t = t.expand(n,1,1,1)
        c = torch.as_tensor(contact, device=a.device, dtype=a.dtype).reshape(-1,1,1,1)
        c = c.expand(n,1,h,w)
        if guides.shape != (n,13,h,w):
            raise ValueError('Expected Blender guides (9) and backward rig flow (4)')
        inputs = torch.cat((guides[:,:9],guides[:,9:]/self.size),1)
        e1 = self.enc1(torch.cat((a, b, t.expand(n,1,h,w), c, inputs), 1))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d2 = self.dec2(torch.cat((e2, F.interpolate(e3, size=e2.shape[-2:],
                               mode='bilinear', align_corners=False)), 1))
        z = self.head(self.dec1(torch.cat((e1, F.interpolate(d2, size=e1.shape[-2:],
                                   mode='bilinear', align_corners=False)), 1)))
        bound = 4*t*(1-t)
        fa = guides[:,9:11]+torch.tanh(z[:,:2])*self.size*.12*bound
        fb = guides[:,11:13]+torch.tanh(z[:,2:4])*self.size*.12*bound
        # Weighted sigmoid keeps the interpolation endpoints exactly specified.
        wa = (1-t)*torch.sigmoid(z[:,4:5])
        wb = t*torch.sigmoid(-z[:,4:5])
        mix = wa/(wa+wb).clamp_min(1e-6)
        aw, bw = self.warp(a, fa), self.warp(b, fb)
        support = (aw[:,3:4]+bw[:,3:4]).clamp(0,1)
        residual = .20*torch.tanh(z[:,5:9])*(4*t*(1-t))
        result = mix*aw+(1-mix)*bw + residual*support
        alpha = result[:,3:4].clamp(0,1)
        result = torch.cat((torch.minimum(result[:,:3].clamp(0,1), alpha), alpha), 1)
        result = torch.where(t == 0, a, torch.where(t == 1, b, result))
        if return_fields:
            return result,fa,fb,mix,residual
        return result, fa, fb


class PoseData:
    def __init__(self, assets, size, motion_checkpoint=''):
        self.size = size
        self.motion = None
        self.motion_step = None
        self.hashes = {}
        self.motion_hash = None
        if motion_checkpoint:
            from train_v04 import MotionController, DATA_ID as V04_ID
            path = Path(motion_checkpoint).expanduser().resolve()
            state = torch.load(path, map_location='cpu', weights_only=True)
            if len(state.get('config', [])) != 3 or state['config'][2] != V04_ID:
                raise ValueError('Expected a V0.4 motion checkpoint')
            self.motion = MotionController(int(state['config'][0])).eval()
            self.motion.load_state_dict(state['model'], strict=True)
            self.motion.requires_grad_(False)
            self.motion_step = int(state['step'])
            self.motion_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        self.frames = []
        self.registration = []
        self.asset_root = root = Path(assets).expanduser().resolve()
        checksum_file = root/'SHA256SUMS'
        if not checksum_file.is_file():
            raise ValueError('Missing tracked pose SHA256SUMS: '+str(root))
        expected = dict((line.split()[1].lstrip('*'), line.split()[0])
                        for line in checksum_file.read_text().splitlines() if line.strip())
        for i in range(8):
            name = f'pose_{i+1:02d}.webp'
            path = root/name
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if expected.get(name) != digest:
                raise ValueError('Pose checksum mismatch: '+str(path))
            self.hashes[name] = digest
            with Image.open(path) as source:
                rgba = np.asarray(source.convert('RGBA')).copy()/255.
            if not 0 < rgba[:,:,3].mean() < 1:
                raise ValueError('Expected a transparent nonempty cel: '+name)
            # Same pixel scale for every cel: do not normalize each bounding box.
            raw = torch.from_numpy(rgba).permute(2,0,1).float()[None]
            raw[:,:3] *= raw[:,3:4]
            h, w = raw.shape[-2:]
            scale = size*.0015
            raw = F.interpolate(raw, size=(round(h*scale), round(w*scale)),
                                mode='bilinear', align_corners=False)
            foot_x, foot_y = contact_point(raw[0,3].numpy())
            canvas = torch.zeros(1,4,size,size)
            rh, rw = raw.shape[-2:]
            canvas[:,:,:rh,:rw] = raw
            x = SUPPORT_X*size + self.root_x(KEY_TIMES[i])
            y = round(FLOOR*size)-LIFT[i]*size/540
            self.frames.append(shift(canvas, x-foot_x, y-foot_y)[0])
            self.registration.append(dict(scale_x=rw/w,scale_y=rh/h,
                dx=x-foot_x,dy=y-foot_y,source=str(path),floor=round(FLOOR*size)))
        self.frames = torch.stack(self.frames)
        self.fingerprint = hashlib.sha256(json.dumps({
            'poses':self.hashes, 'motion':self.motion_hash, 'version':DATA_ID,
            'times':KEY_TIMES, 'lift':LIFT, 'size':size,
        }, sort_keys=True).encode()).hexdigest()

    def highres_frames(self, size):
        """Rebuild source cels at output scale, avoiding low-resolution face upscaling."""
        frames = []
        ratio = size/self.size
        for reg in self.registration:
            with Image.open(reg['source']) as image:
                raw = torch.from_numpy(np.asarray(image.convert('RGBA')).copy()).permute(2,0,1).float()[None]/255
            raw[:,:3] *= raw[:,3:4]
            h,w = raw.shape[-2:]
            # Use the SAME registered transform as the train image, including pixel centers.
            rh,rw = round(h*reg['scale_y']*ratio),round(w*reg['scale_x']*ratio)
            raw = F.interpolate(raw,size=(rh,rw),mode='bilinear',align_corners=False)
            canvas = torch.zeros(1,4,size,size)
            canvas[:,:,:rh,:rw] = raw
            frames.append(shift(canvas,reg['dx']*ratio,reg['dy']*ratio)[0])
        return torch.stack(frames)

    def attach_shot(self, folder):
        root = Path(folder).expanduser().resolve()
        manifest = root/'shot_manifest.json'
        self.shot = json.loads(manifest.read_text())
        if self.shot.get('data_fingerprint') != self.fingerprint:
            raise ValueError('Blender shot does not match image/checkpoint/size; rerun prepare stage')
        if self.shot.get('max_projection_error_px',float('inf')) > .01:
            raise ValueError('Blender registration failed its projection gate')
        for name,digest in self.shot['artifacts'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
                raise ValueError('Blender artifact changed: '+name)
        self.shot_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
        self.shot_dir = root
        self.guide_times = np.array([f['time'] for f in self.shot['frames']])
        self.guide_points = np.array([f['points'] for f in self.shot['frames']])
        with Image.open(root/'background.png') as bg:
            self.background = bg.convert('RGB').copy()
        h = self.background.height
        x = (self.background.width-h)//2
        crop = self.background.crop((x,0,x+h,h)).resize((self.size,self.size),Image.Resampling.LANCZOS)
        self.bg_tensor = torch.from_numpy(np.asarray(crop).copy()).permute(2,0,1).float()/255

    def points(self, t):
        return np.stack([np.interp(t,self.guide_times,self.guide_points[:,j,k])
                           for j in range(14) for k in range(2)]).reshape(14,2)

    def controls(self,start,t,end):
        return conditioning(self.points(start),self.points(t),self.points(end),self.size)

    def root_x(self, t):
        if self.motion is None:
            return 0.
        from train_v04 import features
        # Freeze world-space translation during both planted intervals. Smooth
        # reparameterization starts/stops learned V0.4 travel in the air only.
        u = min(1., max(0., (t-KEY_TIMES[1])/(KEY_TIMES[6]-KEY_TIMES[1])))
        u = u*u*(3-2*u)
        control = (.22,.78,.74,.78,.05,0.,1.,3)
        with torch.inference_mode():
            x = self.motion(features(control,u)[None])[0,0].item()
        return (x-control[0])*.10*self.size

    def interval(self, t):
        i = min(6, max(0, int(np.searchsorted(KEY_TIMES, t, side='right')-1)))
        return i, (t-KEY_TIMES[i])/(KEY_TIMES[i+1]-KEY_TIMES[i])

    def sample(self, rng):
        if rng.random() < .65:
            # Actual middle artwork supervises pose change, NOT a crossfade target.
            i = rng.randrange(6)
            t = (KEY_TIMES[i+1]-KEY_TIMES[i])/(KEY_TIMES[i+2]-KEY_TIMES[i])
            a, y, b = self.frames[[i,i+1,i+2]]
            c = float(grounded(KEY_TIMES[i+1]))
            stamp = KEY_TIMES[i+1]
            points = [self.points(KEY_TIMES[j]) for j in (i,i+1,i+2)]
        else:
            i = rng.randrange(8)
            t = rng.uniform(.15,.85)
            c = float(grounded(KEY_TIMES[i]))
            stamp = KEY_TIMES[i]
            dx = 0. if c else rng.uniform(-.05,.05)*self.size
            dy = 0. if c else rng.uniform(-.035,.035)*self.size
            a = self.frames[i]
            y = shift(a[None],dx*t,dy*t)[0]
            b = shift(a[None],dx,dy)[0]
            points = [self.points(stamp)+[dx*v,dy*v] for v in (0,t,1)]
        # Shared translation preserves sole height and supplies held-out augmentations.
        jitter = rng.uniform(-.015,.015)*self.size
        a,y,b = shift(torch.stack((a,y,b)),jitter,0.)
        guides = conditioning(*(p+[jitter,0] for p in points),self.size)
        # This is the actual Blender plate from the staged shot, not a preview-only override.
        bg = self.bg_tensor
        return a,y,b,t,c,bg,guides


def batch(data, rng, count):
    samples = [data.sample(rng) for _ in range(count)]
    return tuple(torch.stack(items) if isinstance(items[0],torch.Tensor)
                 else torch.tensor(items,dtype=torch.float32) for items in zip(*samples))


def contact_loss(pred, target, contact):
    """Penetration and support-band mass/moments, only for planted examples."""
    size = pred.shape[-1]
    floor = round(FLOOR*size)
    p, y = pred[:,3:4], target[:,3:4]
    # Targets include bilinear antialiasing; compare excess below the sole plane.
    penetration = F.relu(p[:,:,floor+2:]-y[:,:,floor+2:]).mean((1,2,3))
    p = p[:,:,floor-4:floor+2]
    y = y[:,:,floor-4:floor+2]
    xs = torch.linspace(0,1,size,device=p.device)[None,None,None,:]
    pm, ym = p.sum((1,2,3)), y.sum((1,2,3))
    px = (p*xs).sum((1,2,3))/pm.clamp_min(1e-5)
    yx = (y*xs).sum((1,2,3))/ym.clamp_min(1e-5)
    mass = (pm-ym).abs()/size
    return ((penetration+mass+(px-yx).abs())*contact).sum()/contact.sum().clamp_min(1)


def supervised_loss(pred, y, bg, contact):
    weight = 1+6*y[:,3:4]
    pixel = ((over(bg,pred)-over(bg,y)).abs()*weight).mean()
    alpha = F.l1_loss(pred[:,3:4],y[:,3:4])
    pe, ye = edges(pred), edges(y)
    edge = sum(F.l1_loss(p,q) for p,q in zip(pe,ye))/2
    foot = contact_loss(pred,y,contact)
    return pixel+2*alpha+.4*edge+.25*foot, dict(image=pixel,alpha=alpha,edge=edge,contact=foot)


def temporal_loss(model, data, rng):
    i = rng.randrange(7)
    u = rng.uniform(.1,.9)
    ts = torch.tensor([u-.05,u,u+.05])
    a = data.frames[i:i+1].expand(3,-1,-1,-1)
    b = data.frames[i+1:i+2].expand(3,-1,-1,-1)
    stamps = [KEY_TIMES[i]+float(v)*(KEY_TIMES[i+1]-KEY_TIMES[i]) for v in ts]
    c = torch.tensor([float(grounded(t)) for t in stamps])
    guides = torch.stack([data.controls(KEY_TIMES[i],t,KEY_TIMES[i+1]) for t in stamps])
    p,fa,fb = model(a,b,ts,c,guides)
    # Second differences penalize jitter, while first-order movement is allowed.
    acceleration = (p[0]-2*p[1]+p[2]).abs().mean()
    flow_accel = ((fa[0]-2*fa[1]+fa[2]).abs().mean()+
                  (fb[0]-2*fb[1]+fb[2]).abs().mean())/data.size
    reference = (1-ts[:,None,None,None])*a+ts[:,None,None,None]*b
    foot = contact_loss(p, reference, c)
    return acceleration+.2*flow_accel+.25*foot


@torch.no_grad()
def evaluate(model, data, count=12):
    model.eval()
    rng = random.Random(90707)
    totals = dict(model_l1=0.,crossfade_l1=0.,rig_warp_l1=0.,mask_l1=0.,contact_loss=0.)
    for _ in range(count):
        a,y,b,t,c,bg,guides = batch(data,rng,1)
        p,_,_ = model(a,b,t,c,guides)
        mix = t[:,None,None,None]
        totals['model_l1'] += (over(bg,p)-over(bg,y)).abs().mean().item()
        totals['crossfade_l1'] += (over(bg,(1-mix)*a+mix*b)-over(bg,y)).abs().mean().item()
        rig = (1-mix)*model.warp(a,guides[:,9:11])+mix*model.warp(b,guides[:,11:13])
        totals['rig_warp_l1'] += (over(bg,rig)-over(bg,y)).abs().mean().item()
        totals['mask_l1'] += (p[:,3:4]-y[:,3:4]).abs().mean().item()
        totals['contact_loss'] += contact_loss(p,y,c).item()
    model.train()
    return {**{k:round(v/count,6) for k,v in totals.items()},
            'samples':count,'scope':'same eight poses; held-out augmentation seed, not held-out motion'}


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(tmp,path)


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(707)
    rng = random.Random(707)
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    args.out = str(out)
    # A direct invocation and the shell launcher share this lock.
    import fcntl
    with (out/'training.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another V0.7 process is using this output directory')
        data = PoseData(args.assets,args.size,args.motion_checkpoint)
        data.attach_shot(args.shot)
        model = PoseInterpolator(args.size,args.width)
        optim = torch.optim.AdamW(model.parameters(),lr=args.lr)
        config = dict(version=DATA_ID,size=args.size,width=args.width,data=data.fingerprint,
                      shot=data.shot_hash,batch=args.batch,lr=args.lr,temporal_weight=args.temporal_weight)
        ckpt = out/'latest.pt'
        step, best = 0, float('inf')
        if ckpt.exists():
            state = torch.load(ckpt,map_location='cpu',weights_only=True)
            if state['config'] != config:
                raise ValueError('V0.7 checkpoint/data/config differs; use a new --out directory')
            model.load_state_dict(state['model'],strict=True)
            optim.load_state_dict(state['optimizer'])
            step, best = int(state['step']), float(state['best'])
            rng.setstate(state['rng'])
            torch.set_rng_state(state['torch_rng'])
            print('RESUMED',step,flush=True)
        elif args.preview_only:
            raise SystemExit('No V0.7 checkpoint; train first')
        else:
            print('NEW V0.7',sum(p.numel() for p in model.parameters()),'parameters',flush=True)
        atomic_json(out/'run_manifest.json',dict(config=config,poses=data.hashes,
            v04_checkpoint_sha256=data.motion_hash,v04_step=data.motion_step,
            supervision='six real middle-pose triples + controlled affine examples',
            background='Blender courtyard plate; composited during training',
            limitations='Eight poses cannot establish general animation quality. No pretrained diffusion, rig, or text-to-video.'))

        def save(improved=False):
            tmp = out/'latest.tmp.pt'
            torch.save(dict(config=config,step=step,best=best,model=model.state_dict(),
                optimizer=optim.state_dict(),rng=rng.getstate(),torch_rng=torch.get_rng_state()),tmp)
            os.replace(tmp,ckpt)
            if improved:
                shutil.copyfile(ckpt,out/'best.tmp.pt')
                os.replace(out/'best.tmp.pt',out/'best.pt')
            print('SAVED',step,ckpt,flush=True)

        def report():
            nonlocal best
            stats = evaluate(model,data,args.val_samples)
            stats['step'] = step
            print('EVAL',json.dumps(stats),flush=True)
            atomic_json(out/'metrics_latest.json',stats)
            improved = stats['model_l1'] < best
            best = min(best,stats['model_l1'])
            save(improved)
            if not args.no_preview:
                from v07_preview import export_preview
                export_preview(model,data,args,step)

        if args.preview_only:
            from v07_preview import export_preview
            export_preview(model,data,args,step)
            return
        stopping = False
        def stop_request(signum, frame):
            nonlocal stopping
            stopping = True
        signal.signal(signal.SIGTERM,stop_request)
        signal.signal(signal.SIGINT,stop_request)
        start = time.monotonic()
        last_report = -1
        try:
            while not stopping and (args.steps == 0 or step < args.steps):
                a,y,b,t,c,bg,guides = batch(data,rng,args.batch)
                p,_,_ = model(a,b,t,c,guides)
                loss,parts = supervised_loss(p,y,bg,c)
                temporal = temporal_loss(model,data,rng)
                loss = loss+args.temporal_weight*temporal
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss; previous saved checkpoint is retained')
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optim.step()
                step += 1
                if step == 1 or step%25 == 0:
                    print(json.dumps(dict(step=step,loss=round(loss.item(),6),
                        temporal=round(temporal.item(),6),**{k:round(v.item(),6) for k,v in parts.items()},
                        elapsed_s=round(time.monotonic()-start,1))),flush=True)
                if step%args.save_every == 0:
                    report()
                    last_report = step
        except Exception:
            # Do not replace a good checkpoint with a potentially invalid update.
            print('FAILED: retained last completed checkpoint',flush=True)
            raise
        else:
            if last_report != step:
                report()
            print('STOPPED' if stopping else 'FINISHED',step,flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',default=str(DEFAULT_ASSETS))
    p.add_argument('--out',default='~/quantonium-video-runs/v07_codespace')
    p.add_argument('--motion-checkpoint',default='',help='Read-only V0.4 checkpoint; recommended for existing runs')
    p.add_argument('--shot',required=True,help='Verified Blender prepass directory from prepare_v07.py')
    p.add_argument('--size',type=int,default=128)
    p.add_argument('--width',type=int,default=16)
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--steps',type=int,default=5000,help='TOTAL V0.7 optimizer steps; 0=continuous')
    p.add_argument('--save-every',type=int,default=250)
    p.add_argument('--val-samples',type=int,default=12)
    p.add_argument('--lr',type=float,default=.0003)
    p.add_argument('--temporal-weight',type=float,default=.15)
    p.add_argument('--frames',type=int,default=72)
    p.add_argument('--fps',type=int,default=24)
    p.add_argument('--render-width',type=int,default=960)
    p.add_argument('--render-height',type=int,default=540)
    p.add_argument('--no-preview',action='store_true')
    p.add_argument('--preview-only',action='store_true')
    a = p.parse_args()
    if (a.size < 64 or a.size > 512 or a.size%4 or a.width < 4 or a.batch < 1 or a.threads < 1
        or a.steps < 0 or a.save_every < 1 or a.val_samples < 1 or a.lr <= 0
        or not 0 <= a.temporal_weight <= 10 or a.frames < 16 or a.fps < 1
        or a.render_height < 64 or a.render_width < a.render_height
        or a.render_width%2 or a.render_height%2):
        p.error('Invalid training or render dimensions/rates (size: 64..512, multiple of 4)')
    train(a)


if __name__ == '__main__':
    main()
