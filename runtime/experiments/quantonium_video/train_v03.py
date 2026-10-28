#!/usr/bin/env python3
"""Quantonium Anime Video V0.3: conditional multi-time frame generation on CPU.

Research prototype: given two anchor frames, predicts intermediate frames at arbitrary
timestamps using learned backward warping, an occlusion mask, and a small pixel residual.
This is not text-to-video or a Seedance model. Synthetic training scenes have camera
pans, parallax, and simplified cel-style characters. Optional owned animation shots.
"""
import argparse
import hashlib
import json
import math
import os
import random
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F


def synthetic_frame(seed, t, size):
    """Produce one frame in a deterministic 2D scene with camera parallax."""
    r = random.Random(seed)
    w = size
    horizon = int(size * 0.76)
    palette = [(236, 183, 206), (191, 204, 229), (216, 192, 214),
               (245, 196, 189)][seed % 4]
    im = Image.new("RGB", (w, size), palette)
    d = ImageDraw.Draw(im)
    camera = (0.11 + (seed % 5) * 0.018) * w * t
    d.ellipse((int(.74*w-camera*.11), int(.12*w),
               int(.89*w-camera*.11), int(.27*w)),
              fill=(248, 220, 218))
    for i in range(13):
        bx = i * size / 8 - 30
        building = random.Random(seed*191 + i)
        bh = building.randint(size//8, size//2)
        far_x = int(bx - camera * .34)
        d.rectangle((far_x, horizon-bh, far_x+size//7, horizon),
                    fill=(113 + seed%12, 115, 146))
    for i in range(10):
        building = random.Random(seed*991 + i)
        bx = i * size / 6 - 30
        bh = building.randint(size//7, size//3)
        near_x = int(bx - camera * .77)
        d.rectangle((near_x, horizon-bh, near_x+size//8, horizon),
                    fill=(59 + seed%13, 69, 96))
        d.line((near_x+3, horizon-bh+4, near_x+3, horizon-3),
               fill=(84, 93, 121), width=1)
    d.rectangle((0, horizon, size, size), fill=(65, 70, 87))
    d.line((0, horizon, size, horizon), fill=(113, 103, 118), width=2)
    speed = 0.42 + (seed % 4)*.055
    x = int(w * (.22 + speed*t) - camera*.23)
    y = int(size*.50 + size*.013*math.sin(2*math.pi*t*2))
    radius = max(4, size//13)
    skin = [(156, 99, 75), (117, 76, 60), (178, 113, 82)][seed%3]
    d.ellipse((x-radius, y-2*radius, x+radius, y),
              fill=skin, outline=(33, 31, 44), width=max(1,size//64))
    d.pieslice((x-radius, y-2*radius, x+radius, y-radius),
               180, 360, fill=(29, 29, 40))
    d.ellipse((x+1,y-radius+1,x+radius//2,y-radius//2+1),
              fill=(245, 242, 237))
    d.ellipse((x+radius//3,y-radius+1,x+radius//2,y-radius//2),
              fill=(35, 32, 40))
    d.polygon([(x-radius//2,y),(x+radius//2,y),
               (x+radius,y+2*radius),(x-radius,y+2*radius)],
              fill=(38, 47, 70))
    leg = int(radius*.75*math.sin(4*math.pi*t))
    d.line((x-radius//3,y+2*radius,x-radius//3+leg,y+3*radius),
           fill=(32,33,43), width=max(2,size//42))
    d.line((x+radius//3,y+2*radius,x+radius//3-leg,y+3*radius),
           fill=(32,33,43), width=max(2,size//42))
    return im


def tensor(im):
    return torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2,0,1).float()/255.0


def load_shots(root):
    """Split entire shots (never adjacent frames) into train/val partitions."""
    if not root:
        return None, None, "synthetic-v03"
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise ValueError("Missing --frames-dir: "+str(base))
    shots = []
    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue
        paths = sorted(p for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        if len(paths) >= 4:
            shots.append(paths)
    if len(shots) < 2:
        raise ValueError("Need at least TWO separate shot folders with 4+ frames each: one for training, one for validation")
    shots.sort(key=lambda p: hashlib.sha256(str(p[0].parent).encode()).hexdigest())
    n_val = max(1, len(shots)//5)
    validation, training = shots[:n_val], shots[n_val:]
    fingerprint = hashlib.sha256()
    for shot in shots:
        for path in shot:
            fingerprint.update(str(path).encode())
            fingerprint.update(str(path.stat().st_size).encode())
    tag = "frames-v03:" + fingerprint.hexdigest()
    print("SHOTS",json.dumps({"train":len(training),"val":len(validation)}),flush=True)
    return training, validation, tag


def example(seed, t_index, frames, size, shots):
    """Fetch anchor/target frames; each target uses a distinct time index."""
    t = t_index / (frames-1)
    if shots is None:
        return (tensor(synthetic_frame(seed, 0., size)),
                tensor(synthetic_frame(seed, t, size)),
                tensor(synthetic_frame(seed, 1., size)), t)
    shot = shots[seed % len(shots)]
    span = min(frames,len(shot))
    start = (seed//len(shots)) % (len(shot)-span+1)
    idx = round(t * (span-1))
    idx = max(1, min(span-2, idx))
    selection = [shot[start], shot[start+idx], shot[start+span-1]]
    values = []
    for p in selection:
        with Image.open(p) as im:
            values.append(tensor(im.convert("RGB").resize((size,size),Image.Resampling.BILINEAR)))
    return *values, idx/(span-1)


class TemporalWarpNet(nn.Module):
    def __init__(self, size, width=24):
        super().__init__()
        self.size = size
        self.max_flow = size*.35
        def conv(ci,co,stride=1):
            return nn.Conv2d(ci,co,3,stride=stride,padding=1)
        self.enc1 = nn.Sequential(conv(7,width),nn.SiLU(),
                                  conv(width,width),nn.SiLU())
        self.enc2 = nn.Sequential(conv(width,width*2,2),nn.SiLU(),
                                  conv(width*2,width*2),nn.SiLU())
        self.enc3 = nn.Sequential(conv(width*2,width*3,2),nn.SiLU(),
                                  conv(width*3,width*3),nn.SiLU())
        self.dec2 = nn.Sequential(conv(width*5,width*2),nn.SiLU())
        self.dec1 = nn.Sequential(conv(width*3,width),nn.SiLU(),
                                  conv(width,width),nn.SiLU())
        self.head = nn.Conv2d(width,8,3,padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        ys,xs=torch.meshgrid(torch.linspace(-1,1,size),
                             torch.linspace(-1,1,size),indexing="ij")
        self.register_buffer("grid",torch.stack((xs,ys),-1)[None],persistent=False)

    def warp(self, img, pixels):
        h,w=img.shape[-2:]
        delta=pixels.permute(0,2,3,1)
        delta=delta*delta.new_tensor([2/(w-1),2/(h-1)])
        return F.grid_sample(img,self.grid+delta,mode="bilinear",
                             padding_mode="border",align_corners=True)

    def forward(self, a, b, t):
        if isinstance(t,(float,int)):
            t=torch.full((a.shape[0],),float(t),device=a.device,dtype=a.dtype)
        t=t.reshape(-1,1,1,1).to(a)
        t_map=t.expand(-1,1,a.shape[2],a.shape[3])
        x=torch.cat((a,b,t_map),dim=1)
        e1=self.enc1(x)
        e2=self.enc2(e1)
        e3=self.enc3(e2)
        u2=F.interpolate(e3,size=e2.shape[-2:],mode="bilinear",align_corners=False)
        d2=self.dec2(torch.cat((e2,u2),dim=1))
        u1=F.interpolate(d2,size=e1.shape[-2:],mode="bilinear",align_corners=False)
        z=self.head(self.dec1(torch.cat((e1,u1),dim=1)))
        fa=torch.tanh(z[:,0:2])*self.max_flow*t
        fb=torch.tanh(z[:,2:4])*self.max_flow*(1-t)
        prior=torch.log((1-t).clamp(.001)/t.clamp(.001))
        alpha=torch.sigmoid(prior+z[:,4:5])
        residual=.15*torch.tanh(z[:,5:8])*(4*t*(1-t))
        result=(alpha*self.warp(a,fa)+(1-alpha)*self.warp(b,fb)+residual).clamp(0,1)
        return result,fa,fb


def edges(x):
    return (x[:,:,:,1:]-x[:,:,:,:-1],
            x[:,:,1:,:]-x[:,:,:-1,:])


def evaluate(model,args,val_shots):
    model.eval()
    error_model=error_cross=error_edges=0.0
    count=0
    with torch.no_grad():
        for i in range(args.val_scenes):
            seed=950000+i*97
            for frac in (.25,.5,.75):
                idx=max(1,min(args.frames-2,round(frac*(args.frames-1))))
                a,y,b,t=example(seed,idx,args.frames,args.size,val_shots)
                a=a[None];y=y[None];b=b[None]
                output,*_=model(a,b,torch.tensor([t]))
                cross=(1-t)*a+t*b
                error_model+=F.l1_loss(output,y).item()
                error_cross+=F.l1_loss(cross,y).item()
                p1,p2=edges(output);y1,y2=edges(y)
                error_edges+=(F.l1_loss(p1,y1)+F.l1_loss(p2,y2)).item()/2
                count+=1
    model.train()
    return {"model_l1":round(error_model/count,6),
            "crossfade_l1":round(error_cross/count,6),
            "edge_l1":round(error_edges/count,6),
            "samples":count}


def preview(model,args,val_shots):
    """Write bounded MP4 (or GIF if ffmpeg absent) from arbitrary-time predictions."""
    model.eval()
    a,_,b,_=example(978321,(args.frames-1)//2,args.frames,args.size,val_shots)
    a=a[None];b=b[None]
    frames=[]
    with torch.no_grad():
        for i in range(args.frames):
            if i==0: out=a[0]
            elif i==args.frames-1: out=b[0]
            else: out=model(a,b,torch.tensor([i/(args.frames-1)]))[0]
            pixels=(out.clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)
            frames.append(Image.fromarray(pixels).resize((args.size*3,args.size*3),
                                                        Image.Resampling.NEAREST))
    output=Path(args.out).expanduser()
    output.mkdir(parents=True,exist_ok=True)
    if shutil.which("ffmpeg"):
        with tempfile.TemporaryDirectory(dir=output) as tmp:
            temp=Path(tmp)
            for i,im in enumerate(frames):
                im.save(temp/f"frame_{i:03}.png")
            target=output/"preview_next.mp4"
            subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y",
                "-framerate",str(args.fps),"-i",str(temp/"frame_%03d.png"),
                "-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart",
                str(target)],check=True,timeout=120)
            os.replace(target,output/"preview_latest.mp4")
        print("PREVIEW",output/"preview_latest.mp4",flush=True)
    else:
        target=output/"preview_next.gif"
        frames[0].save(target,save_all=True,append_images=frames[1:],
                       duration=max(25,round(1000/args.fps)),loop=0)
        os.replace(target,output/"preview_latest.gif")
        print("PREVIEW",output/"preview_latest.gif","(install ffmpeg for MP4)",flush=True)
    model.train()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="~/quantonium-video-runs/v03")
    p.add_argument("--frames-dir",default="",help="Optional OWNED frame sequence folders, at least two shots")
    p.add_argument("--size",type=int,default=128)
    p.add_argument("--frames",type=int,default=12)
    p.add_argument("--width",type=int,default=24)
    p.add_argument("--batch",type=int,default=2)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--steps",type=int,default=1000,help="Total steps; 0=unbounded")
    p.add_argument("--save-every",type=int,default=200)
    p.add_argument("--val-scenes",type=int,default=12)
    p.add_argument("--fps",type=int,default=12)
    p.add_argument("--lr",type=float,default=.0004)
    args=p.parse_args()
    if args.size<32 or args.size%4 or args.frames<5 or args.batch<1 or args.save_every<1:
        p.error("size must be >=32 and divisible by 4; frames>=5; batch/save-every>=1")
    args.out=str(Path(args.out).expanduser().resolve())
    torch.set_num_threads(args.threads)
    torch.manual_seed(4813)
    random.seed(4813)
    train_shots,val_shots,dataset_id=load_shots(args.frames_dir)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    model=TemporalWarpNet(args.size,args.width)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    step=0
    best=float("inf")
    checkpoint=out/"latest.pt"
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location="cpu",weights_only=True)
        expected=(args.size,args.width,args.frames,dataset_id)
        if tuple(state["config"])!=expected:
            raise ValueError("Incompatible checkpoint size/width/frames/dataset. Use a NEW --out path")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        step=int(state["step"])
        best=float(state.get("best_val",float("inf")))
        if "python_rng" in state: random.setstate(state["python_rng"])
        if "torch_rng" in state: torch.set_rng_state(state["torch_rng"])
        print("RESUMED",step,"BEST",best,flush=True)
    else:
        print("NEW RUN",sum(q.numel() for q in model.parameters()),
              "parameters","DATASET",dataset_id,flush=True)
    def save(is_best=False):
        path=out/"checkpoint_tmp.pt"
        torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),
                    "step":step,"best_val":best,
                    "config":(args.size,args.width,args.frames,dataset_id),
                    "python_rng":random.getstate(),"torch_rng":torch.get_rng_state()},path)
        os.replace(path,checkpoint)
        if is_best:
            shutil.copyfile(checkpoint,out/"best.pt")
        print("SAVED",step,str(checkpoint),flush=True)

    stopping=False
    def request_stop(signum, frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    started=time.monotonic()
    try:
        while not stopping and (args.steps==0 or step<args.steps):
            batch=[]
            for i in range(args.batch):
                idx=random.randrange(1,args.frames-1)
                seed=random.randrange(0,750000)
                batch.append(example(seed,idx,args.frames,args.size,train_shots))
            a=torch.stack([x[0] for x in batch])
            y=torch.stack([x[1] for x in batch])
            b=torch.stack([x[2] for x in batch])
            t=torch.tensor([x[3] for x in batch],dtype=torch.float32)
            pred,fa,fb=model(a,b,t)
            l1=F.l1_loss(pred,y)
            x1,x2=edges(pred);y1,y2=edges(y)
            edge_loss=(F.l1_loss(x1,y1)+F.l1_loss(x2,y2))/2
            smooth=(fa[:,:,:,1:]-fa[:,:,:,:-1]).abs().mean()
            smooth+=(fb[:,:,:,1:]-fb[:,:,:,:-1]).abs().mean()
            loss=l1+0.15*edge_loss+1e-4*smooth
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step+=1
            if step==1 or step%25==0:
                print(json.dumps({"step":step,"train_l1":round(l1.item(),6),
                    "train_edge_l1":round(edge_loss.item(),6),
                    "elapsed_s":round(time.monotonic()-started,1)}),flush=True)
            if step%args.save_every==0:
                metrics=evaluate(model,args,val_shots)
                print("EVAL",json.dumps({"step":step,**metrics}),flush=True)
                improved=metrics["model_l1"]<best
                if improved: best=metrics["model_l1"]
                save(improved)
                preview(model,args,val_shots)
    finally:
        save()
        print("FINISHED",step,"BEST",best,flush=True)


if __name__=="__main__":
    main()
