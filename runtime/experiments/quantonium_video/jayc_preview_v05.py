#!/usr/bin/env python3
"""V0.5 Jayc identity-locked video: trained V0.4 motion, actual approved artwork.

Moves only locally supplied transparent PNG cels; optional pose pack changes
drawn poses using stepped cel timing. No new faces/poses are synthesized.
"""
import argparse
import hashlib
import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw

STYLE_LOCK={
    'eyes':'large elongated eyes; preserve source pixels',
    'hair':'dark Caesar waves and taper',
    'streetwear':'black hoodie/cargo trousers/headphones',
    'jewelry':'gold pendant distinct from story coin',
    'rendering':'clean fine 1990s anime linework; no redesign'
}

def load_rgba(path):
    with Image.open(path) as source: im=source.convert('RGBA')
    if im.getchannel('A').getextrema()==(255,255):
        raise SystemExit(f'Art has no transparency: {path}. Extract cutout using jayc_assets_v05.py.')
    return im

def synthetic_background(w,h,camera,t,seed=19):
    im=Image.new('RGBA',(w,h),(229,184,207,255))
    d=ImageDraw.Draw(im)
    sunx=int(w*.82-camera*.1*w)
    d.ellipse((sunx-32,h*.14-32,sunx+32,h*.14+32),fill=(252,224,222,255))
    horizon=int(h*.83)
    for layer,parallax,color,num in [(0,.25,(117,121,149,255),12),
                                     (1,.80,(65,72,102,255),12)]:
        for i in range(num):
            rr=random.Random(seed*121+i*359+layer)
            bx=round(w*(i/9-.1)-camera*w*parallax)
            top=horizon-rr.randint(h//8,h//2 if layer==0 else h//3)
            width=rr.randint(w//10,w//7)
            d.rectangle((bx,top,bx+width,horizon),fill=color)
    d.rectangle((0,horizon,w,h),fill=(70,76,92,255))
    return im

def supplied_background(path,w,h,camera):
    with Image.open(path) as source: bg=source.convert('RGBA')
    scale=max(1.10*w/bg.width,1.10*h/bg.height)
    bg=bg.resize((int(bg.width*scale)+2,int(bg.height*scale)+2),Image.Resampling.LANCZOS)
    slack_x=bg.width-w
    off=max(0,min(slack_x,int(slack_x*.5+camera*w*.35)))
    y=max(0,(bg.height-h)//2)
    return bg.crop((off,y,off+w,y+h))

def approved_poses(directory,single):
    if not directory: return [load_rgba(single)]
    root=Path(directory).expanduser()
    if not root.is_dir(): raise SystemExit('No pose directory: '+str(root))
    files=sorted(x for x in root.iterdir() if x.suffix.lower()=='.png')
    if len(files)<2:
        raise SystemExit('--poses-dir needs >=2 approved transparent JC pose PNGs.')
    return [load_rgba(x) for x in files]

def motion_positions(args):
    import torch
    from train_v04 import MotionController,features
    defaults={
        'walk':(.18,.78,.74,.78,.055,0.,0.,0),
        'ease':(.18,.78,.74,.78,.055,0.,1.,1),
        'jump':(.18,.78,.74,.78,.09,.16,1.,2),
        'dash':(.18,.78,.78,.78,.07,0.,1.,3)
    }
    motion=defaults[args.action]
    ckpt=Path(args.motion_checkpoint).expanduser()
    if not ckpt.is_file():
        raise SystemExit(f'Missing V0.4 checkpoint: {ckpt}. Train V0.4 first.')
    data=torch.load(ckpt,map_location='cpu',weights_only=True)
    config=data.get('config')
    if not isinstance(config,(tuple,list)) or len(config)<1:
        raise SystemExit('Checkpoint missing V0.4 model config.')
    model=MotionController(int(config[0]))
    model.load_state_dict(data['model'])
    model.eval()
    points=[]
    with torch.no_grad():
        for index in range(args.frames):
            t=index/(args.frames-1)
            x,y,cam=model(features(motion,t).unsqueeze(0))[0].tolist()
            points.append((x,y,cam))
    return points,int(data.get('step',-1))

def render(args):
    import numpy as np
    out=Path(args.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    sprite=Path(args.jayc).expanduser()
    if not sprite.is_file(): raise SystemExit('No Jayc cutout: '+str(sprite))
    poses=approved_poses(args.poses_dir,str(sprite))
    positions,trained_step=motion_positions(args)
    frames=[]
    for i,(x,y,camera) in enumerate(positions):
        t=i/(args.frames-1)
        plate=(supplied_background(args.background,args.width,args.height,camera)
               if args.background else synthetic_background(args.width,args.height,camera,t))
        # Never generate, blend or substitute Jayc's face.
        pose=poses[(i//args.cel_hold)%len(poses)]
        scale=args.height*args.character_height/pose.height
        fig=pose.resize((max(1,int(pose.width*scale)),
                         max(1,int(pose.height*scale))),Image.Resampling.LANCZOS)
        px=round(x*args.width-fig.width/2)
        py=round(y*args.height-fig.height)
        plate.alpha_composite(fig,(px,py))
        frames.append(plate.convert('RGB'))

    indices=[0,args.frames//4,args.frames//2,3*args.frames//4,args.frames-1]
    thumb=192
    contact=Image.new('RGB',(thumb*len(indices),thumb))
    for j,index in enumerate(indices):
        im=frames[index].copy()
        im.thumbnail((thumb,thumb),Image.Resampling.LANCZOS)
        contact.paste(im,(j*thumb+(thumb-im.width)//2,(thumb-im.height)//2))
    tmp=out/'contact_tmp.png'
    contact.save(tmp)
    tmp.replace(out/'jayc_contact_latest.png')

    if shutil.which('ffmpeg'):
        with tempfile.TemporaryDirectory(dir=out) as temp:
            folder=Path(temp)
            for i,frame in enumerate(frames):
                frame.save(folder/f'frame_{i:04}.png')
            staged=out/'jayc_preview_tmp.mp4'
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y',
                '-framerate',str(args.fps),'-i',str(folder/'frame_%04d.png'),
                '-c:v','libx264','-pix_fmt','yuv420p',str(staged)],
                check=True,timeout=90)
            staged.replace(out/'jayc_preview_latest.mp4')
        clip=out/'jayc_preview_latest.mp4'
    else:
        clip=out/'jayc_preview_latest.gif'
        staged=out/'jayc_preview_tmp.gif'
        frames[0].save(staged,save_all=True,append_images=frames[1:],
            duration=max(33,int(1000/args.fps)),loop=0)
        staged.replace(clip)

    meta={
        'mode':'V0.5 pixel-preserving Jayc composition from approved character assets',
        'model_checkpoint':str(Path(args.motion_checkpoint).expanduser()),
        'motion_trained_steps':trained_step,
        'source_asset':sprite.name,
        'source_sha256':hashlib.sha256(sprite.read_bytes()).hexdigest(),
        'style_lock':STYLE_LOCK,'frames':args.frames,'fps':args.fps,
        'pose_sources':len(poses),'action':args.action,
        'limitation':'No new pose synthesis: a single source is a rigid translated cutout. Articulated motion requires approved pose frames.'
    }
    (out/'render_manifest.json').write_text(json.dumps(meta,indent=2))
    print('JAYC POSES',len(poses),'V0.4 MOTION CHECKPOINT STEP',trained_step)
    if len(poses)==1:
        print('NOTICE: one approved image => rigid cutout motion. Add pose frames for true limb changes.')
    print('SAVED VIDEO',clip)
    print('SAVED QC',out/'jayc_contact_latest.png')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jayc',required=True,help='Locally extracted JC RGBA cutout')
    p.add_argument('--motion-checkpoint',default='~/quantonium-video-runs/v04/latest.pt')
    p.add_argument('--poses-dir',default='',help='Folder of approved transparent Jayc pose PNGs')
    p.add_argument('--background',default='',help='Approved cinematic background plate, optional')
    p.add_argument('--out',default='~/quantonium-video-runs/v05')
    p.add_argument('--frames',type=int,default=24)
    p.add_argument('--fps',type=int,default=24)
    p.add_argument('--width',type=int,default=640)
    p.add_argument('--height',type=int,default=360)
    p.add_argument('--character-height',type=float,default=.62)
    p.add_argument('--cel-hold',type=int,default=2)
    p.add_argument('--action',choices=('walk','ease','jump','dash'),default='dash')
    args=p.parse_args()
    if args.frames<5 or args.fps<1 or args.cel_hold<1:
        p.error('frames>=5, fps>=1, cel-hold>=1')
    render(args)

if __name__=='__main__':
    main()
