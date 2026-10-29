#!/usr/bin/env python3
"""Mandatory image-to-rig and Blender shot prepass before V0.7 training/render."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw
import torch

from train_v07 import PoseData, DEFAULT_ASSETS
from v07_image_rig import fit_image
from v07_scene import BONES, JOINT_NAMES, KEY_TIMES, frame_times, grounded


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',default=str(DEFAULT_ASSETS))
    p.add_argument('--motion-checkpoint',default='')
    p.add_argument('--out',required=True)
    p.add_argument('--size',type=int,default=128)
    p.add_argument('--frames',type=int,default=72)
    p.add_argument('--fps',type=int,default=24)
    p.add_argument('--render-width',type=int,default=960)
    p.add_argument('--render-height',type=int,default=540)
    p.add_argument('--blender',default=os.environ.get('QVIDEO_BLENDER','blender'))
    p.add_argument('--image',default='',help='Fit a standalone new full-body image into a static Blender diagnostic rig (not V0.7 training data)')
    p.add_argument('--force-detector',action='store_true',help='Use the local pretrained pose detector even for a known cel')
    a = p.parse_args()
    if a.size < 64 or a.frames < 16 or a.fps < 1 or a.render_width<a.render_height or min(a.render_height,a.render_width)<64 or a.render_width%2 or a.render_height%2:
        p.error('Invalid prepass dimensions or timing')
    executable = shutil.which(a.blender)
    if not executable:
        raise SystemExit('Blender is required BEFORE training. Install Blender or set QVIDEO_BLENDER to its executable.')
    torch.set_num_threads(4)
    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    if a.image:
        fitted = fit_image(a.image,DEFAULT_ASSETS,a.force_detector)
        points = np.array(fitted['points'])
        scale = a.size*.65/max(fitted['dimensions'])
        points = (points-points[-1])*scale + np.array([a.size*.56,round(a.size*.87)])
        anchors = [points.tolist()]*8
        sources = [fitted]
        fingerprint = 'standalone-image-rig:'+fitted['pixel_sha256']
    else:
        data = PoseData(a.assets,a.size,a.motion_checkpoint)
        sources,anchors = [],[]
        for i,reg in enumerate(data.registration):
            fitted = fit_image(reg['source'],a.assets,a.force_detector)
            sources.append(fitted)
            points = np.array(fitted['points'])
            points[:,0] = (points[:,0]+.5)*reg['scale_x']-.5+reg['dx']
            points[:,1] = (points[:,1]+.5)*reg['scale_y']-.5+reg['dy']
            # The measured sole, not crop height, sets contact; airborne support
            # remains a trajectory reference and is never treated as grounded.
            from v07_scene import SUPPORT_X, FLOOR, LIFT
            points[-1] = [SUPPORT_X*a.size+data.root_x(KEY_TIMES[i]),round(FLOOR*a.size)-LIFT[i]*a.size/540]
            anchors.append(points.tolist())
        fingerprint = data.fingerprint
    frames = []
    for index,t in enumerate(frame_times(a.frames)):
        i = min(6,max(0,int(np.searchsorted(KEY_TIMES,t,side='right')-1)))
        u = (t-KEY_TIMES[i])/(KEY_TIMES[i+1]-KEY_TIMES[i])
        u = u*u*(3-2*u)
        points = (1-u)*np.array(anchors[i])+u*np.array(anchors[i+1])
        frames.append(dict(frame=index+1,time=t,grounded=bool(grounded(t)),points=points.tolist()))
    script = Path(__file__).with_name('v07_blender_shot.py')
    # Pack original pixels and readable overlays with the motion shot for review.
    for i,source in enumerate(sources):
        with Image.open(source['source']) as image:
            original = image.convert('RGBA')
        name = f'reference_{i+1:02d}.png'
        original.save(out/name)
        source['packed_reference'] = str(out/name)
        source['packed_reference_sha256'] = hashlib.sha256((out/name).read_bytes()).hexdigest()
        review = Image.new('RGBA',original.size,(210,210,220,255))
        review.alpha_composite(original)
        painter = ImageDraw.Draw(review)
        for start,end in BONES:
            painter.line((*source['points'][start],*source['points'][end]),fill=(0,240,180,255),width=2)
        review.convert('RGB').save(out/f'joint_review_{i+1:02d}.png')
    spec = dict(version=1,data_fingerprint=fingerprint,size=a.size,fps=a.fps,
                width=a.render_width,height=a.render_height,sources=sources,anchors=anchors,
                joint_names=JOINT_NAMES,bones=BONES,frames=frames,
                builder_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                source_status='Existing eight artworks reused; joint centers estimated, depth inferred')
    payload = json.dumps(spec,sort_keys=True,indent=2)+'\n'
    input_sha = hashlib.sha256(payload.encode()).hexdigest()
    manifest = out/'shot_manifest.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        valid = previous.get('input_sha256') == input_sha
        valid = valid and all((out/n).is_file() and hashlib.sha256((out/n).read_bytes()).hexdigest()==h
                              for n,h in previous.get('artifacts',{}).items())
        if valid:
            print('VERIFIED EXISTING BLENDER SHOT',out,flush=True)
            return
    (out/'shot_input.json').write_text(payload)
    print('IMAGE-TO-RIG -> BLENDER SHOT',out,flush=True)
    subprocess.run([executable,'--background','--factory-startup','--threads','4','--python-exit-code','1',
                    '--python',str(script),'--',str(out/'shot_input.json'),str(out)],check=True)
    # Blender can return 0 despite a Python exception without --python-exit-code.
    if not manifest.is_file() or json.loads(manifest.read_text()).get('input_sha256') != input_sha:
        raise RuntimeError('Blender did not produce a matching verified shot')


if __name__ == '__main__':
    main()
