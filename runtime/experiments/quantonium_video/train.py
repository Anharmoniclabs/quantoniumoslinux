#!/usr/bin/env python3
"""Quantonium Anime Video v0.1 — CPU-first learned inbetweening.
Original tiny convolutional model; NOT Seedance or a text-to-video foundation model.
Synthetic cel-style scene motion, optional external image sequences.
Dependencies: pip install torch pillow numpy
"""
import argparse, json, math, os, random, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F

def frame(seed, t, size):
    rng = random.Random(seed)
    sky = [(205,214,235),(243,184,202),(173,193,222),(237,217,200)][seed % 4]
    img = Image.new("RGB",(size,size),sky); d = ImageDraw.Draw(img)
    horizon=int(size*.67)
    d.rectangle((0,horizon,size,size),fill=(71,77,96))
    for i in range(6):
        x=(i*size//5 + seed*13)%(size+25)-15
        h=11+rng.randrange(size//3)
        d.rectangle((x,horizon-h,x+size//9,horizon),fill=(64+i*3,72+i*4,103))
    # Animated character, simple 2-D cel with consistent outline/shape
    phase=2*math.pi*t
    x=int(size*(.23+.48*t)+5*math.sin(phase+seed))
    y=int(size*.60+4*math.sin(2*phase))
    radius=max(3,size//12)
    skin=[(147,93,65),(173,111,77),(114,72,60)][seed%3]
    d.ellipse((x-radius,y-2*radius,x+radius,y),fill=skin,outline=(32,29,44),width=2)
    d.pieslice((x-radius,y-2*radius,x+radius,y-radius),180,360,fill=(29,30,39))
    d.ellipse((x+radius//3,y-radius,x+radius//2,y-radius+radius//3),fill="white")
    d.ellipse((x+radius//2-1,y-radius+1,x+radius//2,y-radius+radius//3),fill=(27,27,31))
    d.polygon([(x-radius//2,y),(x+radius//2,y),(x+radius,y+2*radius),(x-radius,y+2*radius)],fill=(32,43,64))
    swing=int(radius*.9*math.sin(phase*2))
    d.line((x-radius//3,y+2*radius,x-radius//3+swing,y+3*radius),fill=(33,34,45),width=3)
    d.line((x+radius//3,y+2*radius,x+radius//3-swing,y+3*radius),fill=(33,34,45),width=3)
    return img

def tensor(im):
    return torch.from_numpy(np.asarray(im, dtype=np.float32).copy()).permute(2,0,1)/127.5-1

class VideoInbetweener(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(7,width,5,padding=2),nn.SiLU(),
          nn.Conv2d(width,width,3,padding=1),nn.SiLU(),
          nn.Conv2d(width,width,3,padding=1),nn.SiLU(),
          nn.Conv2d(width,3,3,padding=1))
    def forward(self,a,b,t):
        tt=torch.ones_like(a[:,:1])*t
        baseline=a*(1-t)+b*t
        return (baseline+0.4*torch.tanh(self.net(torch.cat((a,b,tt),dim=1)))).clamp(-1,1)

def samples(args, seed):
    folder=Path(args.frames_dir) if args.frames_dir else None
    if folder and folder.exists():
        sequences=[]
        for sub in sorted(folder.iterdir()):
            if sub.is_dir():
                paths=[p for p in sorted(sub.iterdir()) if p.suffix.lower() in (".png",".jpg",".jpeg")]
                if len(paths)>=3: sequences.append(paths)
        if sequences:
            paths=random.choice(sequences)
            start=random.randrange(len(paths)-2)
            picks=paths[start:start+3]
            imgs=[Image.open(p).convert("RGB").resize((args.size,args.size)) for p in picks]
            return [tensor(x) for x in imgs]
    return [tensor(frame(seed,t,args.size)) for t in (0,0.5,1)]

def save_preview(model,args,step,device):
    model.eval()
    a,_,b=samples(args,101)
    a=a.unsqueeze(0).to(device);b=b.unsqueeze(0).to(device)
    pictures=[]
    with torch.no_grad():
        for i in range(9):
            t=i/8
            x=model(a,b,t)[0].cpu()
            arr=((x.clamp(-1,1)+1)*127.5).byte().permute(1,2,0).numpy()
            pictures.append(Image.fromarray(arr,"RGB").resize((args.size*3,args.size*3)))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    target=out/"preview_latest.gif"
    tmp=out/"preview_tmp.gif"
    pictures[0].save(tmp,save_all=True,append_images=pictures[1:],duration=100,loop=0)
    os.replace(tmp,target)
    model.train()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="runtime/experiments/quantonium_video/runs/optiplex")
    p.add_argument("--size",type=int,default=64)
    p.add_argument("--width",type=int,default=32)
    p.add_argument("--batch",type=int,default=4)
    p.add_argument("--steps",type=int,default=1000,help="0 means continuous")
    p.add_argument("--save-every",type=int,default=200)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--frames-dir",default="",help="Optional folders of ordered frames from Blender/OpenToonz")
    p.add_argument("--lr",type=float,default=0.0005)
    args=p.parse_args()
    torch.set_num_threads(args.threads)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    # One fixed checkpoint + one fixed preview; bounded output storage.
    torch.manual_seed(42);random.seed(42)
    device=torch.device("cpu")
    model=VideoInbetweener(args.width).to(device)
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr)
    ckpt=out/"latest.pt"; step=0
    if ckpt.exists():
        saved=torch.load(ckpt,map_location="cpu",weights_only=True)
        if saved["width"]!=args.width or saved["size"]!=args.size:
            raise SystemExit("Checkpoint dimensions differ; use another --out directory")
        model.load_state_dict(saved["model"]);optim.load_state_dict(saved["optimizer"])
        step=saved["step"]; print("RESUMED",step,flush=True)
    else:
        print("NEW RUN",sum(p.numel() for p in model.parameters()),"parameters",flush=True)
    started=time.monotonic()
    try:
        while args.steps==0 or step<args.steps:
            xs=[samples(args,random.randrange(100000)) for _ in range(args.batch)]
            a=torch.stack([x[0] for x in xs]); target=torch.stack([x[1] for x in xs]);b=torch.stack([x[2] for x in xs])
            pred=model(a,b,.5)
            loss=F.l1_loss(pred,target)
            optim.zero_grad(set_to_none=True);loss.backward();optim.step();step+=1
            if step==1 or step%25==0:
                print(json.dumps({"step":step,"l1":round(loss.item(),6),"elapsed_s":round(time.monotonic()-started,1)}),flush=True)
            if step%args.save_every==0:
                tmp=out/"checkpoint_tmp.pt"
                torch.save({"step":step,"model":model.state_dict(),"optimizer":optim.state_dict(),"width":args.width,"size":args.size},tmp)
                os.replace(tmp,ckpt);save_preview(model,args,step,device)
                print("SAVED",ckpt,"PREVIEW",out/"preview_latest.gif",flush=True)
    except KeyboardInterrupt:
        print("Interrupted: saving",flush=True)
    finally:
        tmp=out/"checkpoint_tmp.pt"
        torch.save({"step":step,"model":model.state_dict(),"optimizer":optim.state_dict(),"width":args.width,"size":args.size},tmp)
        os.replace(tmp,ckpt)
        print("FINISHED",step,ckpt,flush=True)
if __name__=="__main__":main()
