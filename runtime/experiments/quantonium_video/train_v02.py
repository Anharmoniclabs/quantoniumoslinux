#!/usr/bin/env python3
"""V0.2: CPU frame warping from two keyframes, with crossfade baseline."""
import argparse, json, math, os, random, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F

def scene(seed,t,size):
    r=random.Random(seed)
    im=Image.new("RGB",(size,size),[(240,188,204),(200,211,231)][seed%2])
    d=ImageDraw.Draw(im);ground=int(size*.7)
    for k in range(7):
        x=k*size//6;h=r.randrange(size//6,size//2)
        d.rectangle((x,ground-h,x+size//9,ground),fill=(48+k*2,55+k*2,80+k*2))
    d.rectangle((0,ground,size,size),fill=(67,71,88))
    x=int(size*(.2+(.36+seed%5*.04)*t)); y=int(size*.53);rad=max(3,size//11)
    skin=[(151,100,76),(113,72,58),(180,112,77)][seed%3]
    d.ellipse((x-rad,y-2*rad,x+rad,y),fill=skin,outline=(28,30,42),width=2)
    d.pieslice((x-rad,y-2*rad,x+rad,y-rad),180,360,fill=(27,30,40))
    d.ellipse((x+1,y-rad,x+rad//2,y-rad//2),fill="white")
    d.polygon([(x-rad//2,y),(x+rad//2,y),(x+rad,y+2*rad),(x-rad,y+2*rad)],fill=(37,46,67))
    swing=int(rad*.6*math.sin(4*math.pi*t))
    d.line((x-rad//3,y+2*rad,x-rad//3+swing,y+3*rad),fill=(30,31,42),width=3)
    d.line((x+rad//3,y+2*rad,x+rad//3-swing,y+3*rad),fill=(30,31,42),width=3)
    return im

def tensor(im):
    return torch.from_numpy(np.asarray(im,dtype=np.uint8).copy()).permute(2,0,1).float()/255

def triplet(seed,size,frames_dir):
    if frames_dir:
        root=Path(frames_dir)
        shots=[]
        for folder in sorted(root.iterdir()):
            if folder.is_dir():
                items=sorted(p for p in folder.iterdir() if p.suffix.lower() in (".png",".jpg",".jpeg"))
                if len(items)>=3: shots.append(items)
        if not shots: raise ValueError("Expected subfolders with 3+ contiguous frames")
        paths=shots[seed%len(shots)]
        start=(seed//len(shots))%(len(paths)-2)
        return tuple(tensor(Image.open(p).convert("RGB").resize((size,size))) for p in paths[start:start+3])
    return tuple(tensor(scene(seed,t,size)) for t in (0,.5,1))

class MotionNet(nn.Module):
    def __init__(self,width=32,max_flow=32):
        super().__init__();self.max_flow=max_flow
        self.net=nn.Sequential(nn.Conv2d(6,width,5,padding=2),nn.SiLU(),
            nn.Conv2d(width,width,3,padding=1),nn.SiLU(),
            nn.Conv2d(width,width,3,padding=1),nn.SiLU(),
            nn.Conv2d(width,5,3,padding=1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def warp(self,img,flow):
        n,c,h,w=img.shape
        ys,xs=torch.meshgrid(torch.linspace(-1,1,h),torch.linspace(-1,1,w),indexing="ij")
        grid=torch.stack((xs,ys),dim=-1)[None].expand(n,-1,-1,-1)
        displacement=flow.permute(0,2,3,1)*flow.new_tensor((2/(w-1),2/(h-1)))
        return F.grid_sample(img,grid+displacement,align_corners=True,padding_mode="border")
    def forward(self,a,b):
        x=self.net(torch.cat((a,b),dim=1))
        fa=torch.tanh(x[:,0:2])*self.max_flow
        fb=torch.tanh(x[:,2:4])*self.max_flow
        alpha=torch.sigmoid(x[:,4:5])
        return alpha*self.warp(a,fa)+(1-alpha)*self.warp(b,fb),fa,fb

def evaluate(model,size,frames_dir):
    model.eval();ml=bl=0
    with torch.no_grad():
        for i in range(24):
            a,m,b=[v[None] for v in triplet(800000+i*173,size,frames_dir)]
            pred,*_=model(a,b)
            ml+=F.l1_loss(pred,m).item()
            bl+=F.l1_loss((a+b)/2,m).item()
    model.train()
    return dict(model_l1=round(ml/24,6),crossfade_l1=round(bl/24,6))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="runtime/experiments/quantonium_video/runs/v02")
    p.add_argument("--frames-dir",default="")
    p.add_argument("--size",type=int,default=64)
    p.add_argument("--width",type=int,default=32)
    p.add_argument("--batch",type=int,default=4)
    p.add_argument("--steps",type=int,default=1000,help="0=continuous")
    p.add_argument("--save-every",type=int,default=200)
    p.add_argument("--threads",type=int,default=4)
    args=p.parse_args()
    torch.set_num_threads(args.threads);torch.manual_seed(17);random.seed(17)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    model=MotionNet(args.width,args.size/2)
    opt=torch.optim.AdamW(model.parameters(),lr=.0005)
    step=0;ckpt=out/"latest.pt"
    if ckpt.exists():
        state=torch.load(ckpt,map_location="cpu",weights_only=True)
        if (state["size"],state["width"])!=(args.size,args.width): raise ValueError("Checkpoint dimensions changed")
        model.load_state_dict(state["model"]);opt.load_state_dict(state["optimizer"]);step=state["step"]
        print("RESUMED",step,flush=True)
    else: print("NEW RUN",sum(p.numel() for p in model.parameters()),"parameters",flush=True)
    def save():
        tmp=out/"latest.tmp.pt"
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,size=args.size,width=args.width),tmp)
        os.replace(tmp,ckpt)
        model.eval()
        a,m,b=triplet(801337,args.size,args.frames_dir)
        with torch.no_grad(): pred,*_=model(a[None],b[None])
        frames=[a,(a+b)/2,pred[0],m,b]
        images=[Image.fromarray((v.clamp(0,1).permute(1,2,0).numpy()*255).astype("uint8")).resize((args.size*3,args.size*3)) for v in frames]
        temp=out/"preview.tmp.gif"
        images[0].save(temp,save_all=True,append_images=images[1:],duration=500,loop=0)
        os.replace(temp,out/"preview_latest.gif")
        model.train();print("SAVED",step,ckpt,flush=True)
    try:
        started=time.monotonic()
        while args.steps==0 or step<args.steps:
            records=[triplet(random.randrange(800000),args.size,args.frames_dir) for _ in range(args.batch)]
            a=torch.stack([v[0] for v in records]);m=torch.stack([v[1] for v in records]);b=torch.stack([v[2] for v in records])
            pred,fa,fb=model(a,b)
            reconstruction=F.l1_loss(pred,m)
            smooth=(fa[:,:,:,1:]-fa[:,:,:,:-1]).abs().mean()+(fb[:,:,:,1:]-fb[:,:,:,:-1]).abs().mean()
            loss=reconstruction+1e-4*smooth
            opt.zero_grad(set_to_none=True);loss.backward();opt.step();step+=1
            if step==1 or step%25==0:
                print(json.dumps(dict(step=step,train_l1=round(reconstruction.item(),6),elapsed_s=round(time.monotonic()-started,1))),flush=True)
            if step%args.save_every==0:
                print("EVAL",json.dumps(evaluate(model,args.size,args.frames_dir)),flush=True)
                save()
    except KeyboardInterrupt: print("INTERRUPTED: saving",flush=True)
    finally: save();print("FINISHED",step,flush=True)
if __name__=="__main__": main()
