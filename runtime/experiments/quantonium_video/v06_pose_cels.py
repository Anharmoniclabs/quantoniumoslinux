#!/usr/bin/env python3
"""V0.6: 8 distinct creator-drawn JC cels + optional trained V0.4 root motion.

This is pose-aware 2D cel animation, NOT trained pose synthesis or Blender.
No private artwork, PDF or model weight is uploaded by this script.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

# Beat holds: anticipation -> crouch -> takeoff -> airborne kick -> recovery.
BEATS = [(1,8),(2,4),(3,4),(4,3),(5,3),(6,4),(7,6),(8,10),(1,6)]
LIFT = [0,0,55,92,100,65,15,0]
SHIFT = [-12,-9,3,16,28,14,2,0]


def restore_cels(folder):
    folder=Path(folder).expanduser().resolve()
    files=[folder/f'pose_{index:02d}.png' for index in range(1,9)]
    if any(not path.is_file() for path in files):
        raise SystemExit('Expected eight pose_01.png ... pose_08.png in '+str(folder))
    cels=[]
    for file in files:
        with Image.open(file) as image:
            cel=image.convert('RGBA')
        amin,amax=cel.getchannel('A').getextrema()
        if amin==255 or amax==0 or cel.width<120 or cel.height<150:
            raise SystemExit('Rejected incomplete or nontransparent cel: '+str(file))
        cels.append(cel)
    return files,cels


def motion_model(checkpoint, times):
    if not checkpoint:
        return [0.0]*len(times),None
    import torch
    from train_v04 import MotionController,features
    ckpt=Path(checkpoint).expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError('Missing trained V0.4 checkpoint: '+str(ckpt))
    state=torch.load(ckpt,map_location='cpu',weights_only=True)
    if 'model' not in state or 'config' not in state:
        raise ValueError('This is not a V0.4 motion checkpoint')
    model=MotionController(int(state['config'][0]))
    model.load_state_dict(state['model'],strict=True)
    model.eval()
    control=(.22,.78,.74,.78,.05,0.,1.,3)
    with torch.inference_mode():
        xs=[model(features(control,t).unsqueeze(0))[0,0].item() for t in times]
    # Maximum 10% of learned movement: prevent whole cutout sliding across scene.
    return [(x-xs[0])*.10 for x in xs],int(state.get('step',-1))


def background_plate(width,height,background):
    if background:
        with Image.open(background) as source:
            im=source.convert('RGB')
        scale=max(width/im.width,height/im.height)
        im=im.resize((math.ceil(im.width*scale),math.ceil(im.height*scale)),Image.Resampling.LANCZOS)
        x=(im.width-width)//2; y=(im.height-height)//2
        return im.crop((x,y,x+width,y+height)).convert('RGBA')
    # Neutral proof background: no game-like pixel skyline or fake perspective.
    im=Image.new('RGBA',(width,height))
    line=ImageDraw.Draw(im)
    for y in range(height):
        mix=y/max(height-1,1)
        line.line((0,y,width,y),fill=(int(231+8*mix),int(191+27*mix),int(209+21*mix),255))
    return im


def run(args):
    files,cels=restore_cels(args.poses)
    output=Path(args.out).expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    sequence=[index for index,hold in BEATS for _ in range(hold)]
    indices=[sequence[min(47,int(i*48/args.frames))] for i in range(args.frames)]
    assert sorted(set(indices))==list(range(1,9)), 'Every source pose must appear'
    times=[i/(args.frames-1) for i in range(args.frames)]
    deltas,trained_step=motion_model(args.motion_checkpoint,times)
    base=background_plate(args.width,args.height,args.background)
    shots=[]
    for frame,pose_index in enumerate(indices):
        index=pose_index-1
        cel=cels[index]
        # ONE global pixel scale maintains JC face/head/body proportions across poses.
        scaled=cel.resize((int(cel.width*args.scale),int(cel.height*args.scale)),Image.Resampling.LANCZOS)
        scene=base.copy()
        ground=round(args.height*.89)
        center_x=round(args.width*.52)
        painter=ImageDraw.Draw(scene,'RGBA')
        shadow_width=round(args.width*.11*(.55 if LIFT[index]>50 else 1))
        painter.ellipse((center_x-shadow_width,ground,center_x+shadow_width,ground+12),
                        fill=(51,43,69,24))
        x=round(center_x+deltas[frame]*args.width+SHIFT[index]*args.height/540)
        y=round(ground-scaled.height-LIFT[index]*args.height/540)
        scene.alpha_composite(scaled,(x-scaled.width//2,y))
        shots.append(scene.convert('RGB'))
    # QC contact contains the rendered animation frames, not the PDF crop.
    picks=[0,9,15,18,21,25,30,40]
    qc=Image.new('RGB',(1280,420),'#f1d6df')
    d=ImageDraw.Draw(qc)
    for i,k in enumerate(picks):
        k=min(k,args.frames-1)
        tile=shots[k].resize((320,180),Image.Resampling.LANCZOS)
        x=(i%4)*320; y=(i//4)*210
        qc.paste(tile,(x,y))
        d.text((x+6,y+185),f'FRAME {k+1:02d} | POSE {indices[k]:02d}',fill=(55,39,55))
    qc.save(output/'jc_pose_frame_review.png')
    mp4=output/'jc_action_cels_v06.mp4'
    with tempfile.TemporaryDirectory(dir=output) as work:
        directory=Path(work)
        for i,im in enumerate(shots):
            im.save(directory/f'frame_{i:04d}.png')
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y',
                        '-framerate',str(args.fps),'-i',str(directory/'frame_%04d.png'),
                        '-c:v','libx264','-crf','19','-pix_fmt','yuv420p',
                        '-movflags','+faststart',str(mp4)],check=True)
    meta={
        'mode':'8 creator-drawn JC action cels, not pose synthesis',
        'v04_checkpoint':args.motion_checkpoint or None,
        'v04_trained_step':trained_step,
        'pose_count':8,'frames':args.frames,'fps':args.fps,
        'pose_shas':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'limitations':'No learned in-between pose synthesis, no Blender rig, no new limbs drawn',
        'approval':'JC pose extraction and animation timing require creator QC'
    }
    (output/'v06_proof_manifest.json').write_text(json.dumps(meta,indent=2))
    print('V0.6: distinct JC poses:',meta['pose_count'],'V0.4 checkpoint step:',trained_step,flush=True)
    print('SAVED MP4',mp4,flush=True)
    print('SAVED QC',output/'jc_pose_frame_review.png',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--poses',required=True)
    p.add_argument('--out',default='~/quantonium-video-runs/v06_codespace')
    p.add_argument('--motion-checkpoint',default='')
    p.add_argument('--background',default='',help='Optional creator-approved background plate')
    p.add_argument('--frames',type=int,default=48)
    p.add_argument('--fps',type=int,default=24)
    p.add_argument('--width',type=int,default=960)
    p.add_argument('--height',type=int,default=540)
    p.add_argument('--scale',type=float,default=.86)
    a=p.parse_args()
    if a.frames<16 or a.width%2 or a.height%2 or a.fps<1 or not .2<=a.scale<=2:
        p.error('Invalid frames/size/fps/scale')
    run(a)


if __name__=='__main__':
    main()
