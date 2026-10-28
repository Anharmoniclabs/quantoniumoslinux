#!/usr/bin/env python3
"""Quantonium Video V0.4: supervised layered motion for stable 2D animation.

Synthetic ground-truth layers teach a small learned motion controller to position a
foreground cutout and parallax camera. Explicit alpha composition prevents vanishing
characters. This is NOT free-form video generation, learned human pose synthesis,
or Seedance. The synthetic sprites/pose cycles are drawn procedurally.

CPU: pip install torch numpy pillow; optional ffmpeg for MP4.
"""
import argparse
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


STYLES = ("linear", "ease", "jump", "dash")
DATA_ID = "quantonium-v04-layered-synthetic-v1"


def ctrl(seed):
    r = random.Random(seed)
    kind = seed % len(STYLES)
    x0 = r.uniform(.17, .31)
    x1 = r.uniform(.63, .81)
    y0 = r.uniform(.75, .80)
    y1 = y0 + r.uniform(-.024, .024)
    camera_delta = r.uniform(-.12, .14)
    jump_height = r.uniform(.10, .19) if kind == 2 else 0.
    camera_ease = float((seed // 4) % 2)
    return (x0, y0, x1, y1, camera_delta, jump_height,
            camera_ease, kind)


def features(c, t):
    x0, y0, x1, y1, camera, jump, ease, kind = c
    hot = [float(kind == i) for i in range(4)]
    return torch.tensor([x0, y0, x1, y1, camera, jump,
                         ease, t, *hot], dtype=torch.float32)


def analytic_position(c, t):
    x0, y0, x1, y1, camera, jump, ease, kind = c
    u = t
    if kind == 1:
        u = t*t*(3-2*t)
    elif kind == 3:
        u = 1-(1-t)**3
    elif kind == 2:
        u = t*t*(3-2*t)
    y = y0 + (y1-y0)*u
    if kind == 2:
        y -= jump * math.sin(math.pi*t)
    cam = camera * (t*t*(3-2*t) if ease else t)
    return torch.tensor([x0+(x1-x0)*u, y, cam], dtype=torch.float32)


def linear_position(c, t):
    x0, y0, x1, y1, camera, _, _, _ = c
    return torch.tensor([x0+(x1-x0)*t,
                         y0+(y1-y0)*t, camera*t], dtype=torch.float32)


class MotionController(nn.Module):
    """Learn deviations from linear interpolation. Endpoints remain pinned."""
    def __init__(self, width=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12,width), nn.SiLU(),
            nn.Linear(width,width), nn.SiLU(),
            nn.Linear(width,width), nn.SiLU(),
            nn.Linear(width,3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, f):
        t = f[:,7:8]
        baseline = torch.cat((
            f[:,0:1] + (f[:,2:3]-f[:,0:1])*t,
            f[:,1:2] + (f[:,3:4]-f[:,1:2])*t,
            f[:,4:5]*t),dim=1)
        bound = (4*t*(1-t)).clamp(0,1)
        max_delta = f.new_tensor([.27,.25,.16])
        return baseline + bound*max_delta*torch.tanh(self.net(f))


def to_tensor(im):
    return torch.from_numpy(np.asarray(im,dtype=np.uint8).copy()).permute(2,0,1).float()/255.


def synthetic_layers(seed, size):
    r = random.Random(seed+177)
    sky = Image.new("RGB",(size,size),[(236,186,207),(206,215,236),
          (243,202,197),(196,192,225)][seed%4])
    s = ImageDraw.Draw(sky)
    sun_x=int(size*(.72+ r.random()*.11))
    sun_y=int(size*(.12+r.random()*.08))
    s.ellipse((sun_x-9,sun_y-9,sun_x+9,sun_y+9),fill=(252,224,219))

    far=Image.new("RGBA",(size,size),(0,0,0,0))
    near=Image.new("RGBA",(size,size),(0,0,0,0))
    fd=ImageDraw.Draw(far); nd=ImageDraw.Draw(near)
    horizon=int(size*.83)
    for i in range(12):
        rr=random.Random(seed*77+i*127)
        x=i*size//9-size//8
        h=rr.randrange(size//7,size//2)
        fd.rectangle((x,horizon-h,x+size//7,horizon),
                     fill=(114+i%3*5,123+i%3*3,151,255))
        if i%3==0:
            fd.rectangle((x+3,horizon-h+6,x+5,horizon-h+12),
                         fill=(245,218,189,255))
    for i in range(9):
        rr=random.Random(seed*99+i*337)
        x=i*size//7-size//8
        h=rr.randrange(size//8,size//3)
        nd.rectangle((x,horizon-h,x+size//8,horizon),
                     fill=(58+i%3*3,66+i%3*2,92,255))
        if i%2:
            nd.line((x+3,horizon-h+4,x+3,horizon-5),
                    fill=(91,94,122,255),width=1)
    nd.rectangle((0,horizon,size,size),fill=(63,69,84,255))
    nd.line((0,horizon,size,horizon),fill=(151,126,143,255),width=2)
    return (to_tensor(sky),to_tensor(far),to_tensor(near))


def character(seed,size,x0,y0,t=0.0):
    """Deterministic 2D cel figure; limb poses are SCRIPTED, not neural."""
    im=Image.new("RGBA",(size,size),(0,0,0,0))
    d=ImageDraw.Draw(im)
    x=int(x0*size)
    foot=int(y0*size)
    unit=max(3,size//22)
    skin=[(157,101,73,255),(115,76,60,255),(180,116,82,255)][seed%3]
    shirt=[(34,47,76,255),(79,43,69,255),(39,64,70,255)][seed%3]
    ink=(28,32,45,255)
    knee=foot-2*unit
    hip=foot-4*unit
    shoulder=foot-7*unit
    neck=foot-8*unit
    head_top=foot-13*unit
    swing=round(1.15*unit*math.sin(4*math.pi*t))
    d.line((x-unit//2,hip,x-unit//2+swing,knee,x-unit//2-swing,foot),
           fill=ink,width=max(2,unit//2))
    d.line((x+unit//2,hip,x+unit//2-swing,knee,x+unit//2+swing,foot),
           fill=ink,width=max(2,unit//2))
    d.polygon(((x-unit,hip),(x+unit,hip),(x+unit,shoulder),(x-unit,shoulder)),
              fill=shirt)
    d.line((x-unit,shoulder,x-2*unit,hip+swing//2),fill=skin,width=max(2,unit//2))
    d.line((x+unit,shoulder,x+2*unit,hip-swing//2),fill=skin,width=max(2,unit//2))
    d.ellipse((x-2*unit,head_top,x+2*unit,neck),
              fill=skin,outline=ink,width=1)
    d.pieslice((x-2*unit,head_top,x+2*unit,neck-unit),
               180,360,fill=(27,30,42,255))
    eye_y=foot-10*unit
    d.ellipse((x+unit//2,eye_y,x+unit+unit//2,eye_y+unit//2),
              fill=(248,245,243,255))
    d.ellipse((x+unit,eye_y,x+unit+1,eye_y+unit//2),
              fill=ink)
    return to_tensor(im)


def warp(image,dx,dy,grid=None):
    """Differentiable shift by +dx,+dy in image-normalized coordinates."""
    n,c,h,w=image.shape
    if grid is None:
        yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=image.device),
                             torch.linspace(-1,1,w,device=image.device),
                             indexing="ij")
        grid=torch.stack((xx,yy),-1)[None].expand(n,-1,-1,-1)
    delta=torch.stack((-dx*2*w/(w-1),-dy*2*h/(h-1)),dim=-1)
    grid=grid+delta[:,None,None,:]
    return F.grid_sample(image,grid,mode="bilinear",
                         padding_mode="zeros",align_corners=True)


def over(base,rgba):
    alpha=rgba[:,3:4].clamp(0,1)
    return base*(1-alpha)+rgba[:,:3]*alpha


def render(layers,actor,control,positions,return_parts=False):
    """Composite camera-shifted sky/buildings/character with nontransparent alpha."""
    sky,far,near=layers
    x0=control[:,0]
    y0=control[:,1]
    x,y,cam=positions.unbind(1)
    bg=over(sky,warp(far,-.33*cam,torch.zeros_like(cam)))
    bg=over(bg,warp(near,-.70*cam,torch.zeros_like(cam)))
    fg=warp(actor,x-x0-cam,y-y0)
    fg_alpha=fg[:,3:4].clamp(0,1)
    image=over(bg,fg)
    if return_parts:
        return image,fg_alpha,bg
    return image


def case(seed,size,t):
    c=ctrl(seed)
    f=features(c,t)
    target=analytic_position(c,t)
    linear=linear_position(c,t)
    sky,far,near=synthetic_layers(seed,size)
    act=character(seed,size,c[0],c[1],t)
    return f,target,linear,(sky,far,near),act


def batch_cases(seeds,times,size):
    entries=[case(seed,size,t) for seed,t in zip(seeds,times)]
    f=torch.stack([e[0] for e in entries])
    target=torch.stack([e[1] for e in entries])
    linear=torch.stack([e[2] for e in entries])
    layers=tuple(torch.stack([e[3][i] for e in entries]) for i in range(3))
    actor=torch.stack([e[4] for e in entries])
    return f,target,linear,layers,actor


def evaluate(model,args):
    model.eval()
    totals={"model_l1":0.,"layered_linear_l1":0.,"crossfade_l1":0.,
            "character_l1":0.,"mask_iou":0.,"position_mae_px":0.}
    count=0
    with torch.no_grad():
        for i in range(args.val_scenes):
            seed=900000+i*193
            for t in (.25,.5,.75):
                f,gt,linear,layers,actor=batch_cases([seed],[t],args.size)
                pred=model(f)
                img,mask,bg=render(layers,actor,f,pred,True)
                true,true_mask,_=render(layers,actor,f,gt,True)
                lin=render(layers,actor,f,linear)
                # Approximate crossfade of the SAME synthetic scene at both endpoints.
                c=ctrl(seed)
                f0,g0,_,l0,a0=batch_cases([seed],[0.],args.size)
                f1,g1,_,l1,a1=batch_cases([seed],[1.],args.size)
                start=render(l0,a0,f0,g0)
                end=render(l1,a1,f1,g1)
                cross=(1-t)*start+t*end
                weight=1+8*true_mask
                totals["model_l1"]+=F.l1_loss(img,true).item()
                totals["layered_linear_l1"]+=F.l1_loss(lin,true).item()
                totals["crossfade_l1"]+=F.l1_loss(cross,true).item()
                totals["character_l1"]+=(torch.abs(img-true)*weight).mean().item()
                union=((mask>.4)|(true_mask>.4)).float().sum().item()
                intersection=((mask>.4)&(true_mask>.4)).float().sum().item()
                totals["mask_iou"]+=intersection/max(1.,union)
                totals["position_mae_px"]+=torch.abs(pred[:,:2]-gt[:,:2]).mean().item()*args.size
                count+=1
    model.train()
    return {"samples":count,**{k:round(v/count,6) for k,v in totals.items()}}


def override_preview_assets(args, f, layers, actor):
    """Optional OWNED PNG asset preview. Training and validation remain synthetic.

    Background PNG is a flat plate; camera parallax requires separate layer passes.
    Character PNG must have transparency, and stays a rigid cutout in V0.4.
    """
    if args.background_png:
        with Image.open(args.background_png) as source:
            bg=source.convert("RGB").resize((args.size,args.size),
                                             Image.Resampling.LANCZOS)
        sky=to_tensor(bg)[None]
        blank=torch.zeros((1,4,args.size,args.size))
        layers=(sky,blank,blank)
    if args.character_png:
        with Image.open(args.character_png) as source:
            sprite=source.convert("RGBA")
        if sprite.getextrema()[3] == (255,255):
            raise ValueError("Character PNG must contain real transparency (alpha channel)")
        max_w=int(args.size*.32)
        max_h=int(args.size*.40)
        sprite.thumbnail((max_w,max_h),Image.Resampling.LANCZOS)
        canvas=Image.new("RGBA",(args.size,args.size),(0,0,0,0))
        x0=float(f[0,0]);y0=float(f[0,1])
        at=(round(x0*args.size-sprite.width/2),
            round(y0*args.size-sprite.height))
        canvas.alpha_composite(sprite,at)
        actor=to_tensor(canvas)[None]
    return layers,actor


def export_preview(model,args):
    model.eval()
    seed=900777
    frames=[]
    comparison=[]
    with torch.no_grad():
        for i in range(args.frames):
            t=i/(args.frames-1)
            f,gt,linear,layers,actor=batch_cases([seed],[t],args.size)
            layers,actor=override_preview_assets(args,f,layers,actor)
            output=render(layers,actor,f,model(f))[0]
            reference=render(layers,actor,f,gt)[0]
            baseline=render(layers,actor,f,linear)[0]
            rgb=(output.clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)
            frame=Image.fromarray(rgb).resize((args.size*3,args.size*3),
                                              Image.Resampling.NEAREST)
            frames.append(frame)
            if i in (0,args.frames//4,args.frames//2,3*args.frames//4,args.frames-1):
                trio=[output,baseline,reference]
                pieces=[]
                for x in trio:
                    a=(x.clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)
                    pieces.append(Image.fromarray(a).resize((args.size*2,args.size*2),
                                                             Image.Resampling.NEAREST))
                comparison.append(pieces)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    panel=Image.new("RGB",(args.size*2*3,args.size*2*len(comparison)))
    for row,images in enumerate(comparison):
        for col,image in enumerate(images):
            panel.paste(image,(col*args.size*2,row*args.size*2))
    tmp=out/"comparison.tmp.png"
    panel.save(tmp)
    os.replace(tmp,out/"comparison_latest.png")
    if shutil.which("ffmpeg"):
        with tempfile.TemporaryDirectory(dir=out) as tmpdir:
            path=Path(tmpdir)
            for i,im in enumerate(frames): im.save(path/f"frame_{i:03}.png")
            tmpmp4=out/"preview_tmp.mp4"
            subprocess.run(["ffmpeg","-y","-hide_banner","-loglevel","error",
                            "-framerate",str(args.fps),"-i",str(path/"frame_%03d.png"),
                            "-c:v","libx264","-pix_fmt","yuv420p",
                            str(tmpmp4)],check=True,timeout=120)
            os.replace(tmpmp4,out/"preview_latest.mp4")
        print("PREVIEW",out/"preview_latest.mp4",flush=True)
    else:
        tmpgif=out/"preview_tmp.gif"
        frames[0].save(tmpgif,save_all=True,append_images=frames[1:],
                       duration=max(25,int(1000/args.fps)),loop=0)
        os.replace(tmpgif,out/"preview_latest.gif")
        print("PREVIEW",out/"preview_latest.gif",flush=True)
    print("COMPARISON",out/"comparison_latest.png",flush=True)
    model.train()


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(224)
    rng=random.Random(224)
    out=Path(args.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    args.out=str(out)
    model=MotionController(args.width)
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr)
    checkpoint=out/"latest.pt"
    step=0
    best=float("inf")
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location="cpu",weights_only=True)
        config=[args.width,args.size,DATA_ID]
        if state["config"]!=config:
            raise ValueError("Checkpoint config differs; use a fresh --out directory")
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optimizer"])
        step=int(state["step"])
        best=float(state.get("best",float("inf")))
        if "rng" in state: rng.setstate(state["rng"])
        if "torch_rng" in state: torch.set_rng_state(state["torch_rng"])
        print("RESUMED",step,flush=True)
    else:
        print("NEW RUN",sum(p.numel() for p in model.parameters()),
              "parameters",DATA_ID,flush=True)

    def save(improved=False):
        tmp=out/"latest_tmp.pt"
        torch.save(dict(step=step,best=best,model=model.state_dict(),
                        optimizer=optim.state_dict(),rng=rng.getstate(),
                        torch_rng=torch.get_rng_state(),
                        config=[args.width,args.size,DATA_ID]),tmp)
        os.replace(tmp,checkpoint)
        if improved: shutil.copyfile(checkpoint,out/"best.pt")
        print("SAVED",step,checkpoint,flush=True)

    stopping=False
    def stop_request(sig,frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop_request)
    signal.signal(signal.SIGINT,stop_request)
    started=time.monotonic()
    try:
        while not stopping and (args.steps==0 or step<args.steps):
            seeds=[rng.randrange(800000) for _ in range(args.batch)]
            times=[rng.randrange(1,args.frames-1)/(args.frames-1)
                   for _ in range(args.batch)]
            f,target,linear,layers,actor=batch_cases(seeds,times,args.size)
            pred=model(f)
            predicted,pred_mask,_=render(layers,actor,f,pred,True)
            true,true_mask,_=render(layers,actor,f,target,True)
            # Explicit character-weighted supervision; background cannot
            # dominate the gradient purely by its pixel count.
            weight=(1+8*true_mask.detach())
            image_loss=(torch.abs(predicted-true)*weight).mean()
            segmentation_loss=F.l1_loss(pred_mask,true_mask)
            coord_loss=F.smooth_l1_loss(pred,target,beta=.04)
            loss=8*coord_loss+image_loss+2*segmentation_loss
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optim.step()
            step+=1
            if step==1 or step%25==0:
                print(json.dumps({"step":step,
                                  "weighted_image_l1":round(image_loss.item(),6),
                                  "mask_l1":round(segmentation_loss.item(),6),
                                  "trajectory_loss":round(coord_loss.item(),6),
                                  "elapsed_s":round(time.monotonic()-started,1)}),
                      flush=True)
            if step%args.save_every==0:
                stats=evaluate(model,args)
                print("EVAL",json.dumps({"step":step,**stats}),flush=True)
                improved=stats["character_l1"]<best
                if improved: best=stats["character_l1"]
                save(improved)
                export_preview(model,args)
    finally:
        save()
        print("FINISHED",step,"BEST",best,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="~/quantonium-video-runs/v04")
    p.add_argument("--size",type=int,default=128)
    p.add_argument("--frames",type=int,default=12)
    p.add_argument("--fps",type=int,default=12)
    p.add_argument("--width",type=int,default=96)
    p.add_argument("--batch",type=int,default=4)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--steps",type=int,default=1000,help="total optimizer steps; 0=continuous")
    p.add_argument("--save-every",type=int,default=200)
    p.add_argument("--val-scenes",type=int,default=12)
    p.add_argument("--character-png",default="",help="Optional original transparent RGBA character cutout for preview only")
    p.add_argument("--background-png",default="",help="Optional background plate for preview only")
    p.add_argument("--lr",type=float,default=.0008)
    args=p.parse_args()
    if args.size<96 or args.frames<5 or args.save_every<1 or args.batch<1:
        p.error("size >=96, frames >=5, save-every/batch >=1 required")
    train(args)


if __name__=="__main__":
    main()
