"""Render source-preserving keys and Blender-conditioned learned in-between fields."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from v07_scene import FLOOR,SUPPORT_X,KEY_TIMES,grounded,over,pin_contact,contact_point


def warp_highres(image,flow,training_size):
    h,w = image.shape[-2:]
    flow = F.interpolate(flow,size=(h,w),mode='bilinear',align_corners=False)
    # Source transforms scale pixel distances by output_size/training_size.
    # Convert those scaled distances to the high-resolution grid coordinates.
    flow = flow.permute(0,2,3,1)*flow.new_tensor([
        2*w/training_size/(w-1),2*h/training_size/(h-1)])
    yy,xx = torch.meshgrid(torch.linspace(-1,1,h),torch.linspace(-1,1,w),indexing='ij')
    grid = torch.stack((xx,yy),-1)[None]+flow
    return F.grid_sample(image,grid,align_corners=True,padding_mode='zeros')


@torch.no_grad()
def export_preview(model,data,args,step):
    if len(data.shot['frames']) != args.frames or data.shot['fps'] != args.fps:
        raise ValueError('Preview frame count/fps differs from staged Blender shot')
    if data.background.size != (args.render_width,args.render_height):
        raise ValueError('Preview size differs from staged Blender camera')
    model.eval()
    out = Path(args.out)
    width,height = args.render_width,args.render_height
    sources = data.highres_frames(height)
    bg = torch.from_numpy(np.asarray(data.background).copy()).permute(2,0,1).float()[None]/255
    left = (width-height)//2
    selected = set(np.rint(np.linspace(0,args.frames-1,8)).astype(int).tolist())
    tiles,records = [],[]
    key_errors = []
    with tempfile.TemporaryDirectory(dir=out) as temp:
        temp = Path(temp)
        for index,spec in enumerate(data.shot['frames']):
            t = spec['time']
            i,u = data.interval(t)
            a,b = data.frames[i:i+1],data.frames[i+1:i+2]
            guides = data.controls(KEY_TIMES[i],t,KEY_TIMES[i+1])[None]
            _,fa,fb,mix,residual = model(a,b,u,float(grounded(t)),guides,True)
            is_key = any(abs(t-k)<1e-8 for k in KEY_TIMES)
            if is_key:
                k = min(range(8),key=lambda j:abs(KEY_TIMES[j]-t))
                frame = sources[k:k+1].clone()
                key_errors.append(float((frame-sources[k:k+1]).abs().max()))
            else:
                aw = warp_highres(sources[i:i+1],fa,data.size)
                bw = warp_highres(sources[i+1:i+2],fb,data.size)
                mix = F.interpolate(mix,size=(height,height),mode='bilinear',align_corners=False)
                residual = F.interpolate(residual,size=(height,height),mode='bilinear',align_corners=False)
                support = (aw[:,3:4]+bw[:,3:4]).clamp(0,1)
                frame = mix*aw+(1-mix)*bw+residual*support
                alpha = frame[:,3:4].clamp(0,1)
                frame = torch.cat((torch.minimum(frame[:,:3].clamp(0,1),alpha),alpha),1)
            correction = (0.,0.)
            raw_contact = contact_point(frame[0,3].numpy())
            if grounded(t) and not is_key:
                # Correct only new frames. The eight registered source keyframes
                # bypass all learned synthesis and remain pixel-identical to sources.
                target_x = (SUPPORT_X*data.size+data.root_x(t))*height/data.size
                target_y = round(FLOOR*data.size)*height/data.size
                frame,correction = pin_contact(frame,target_x,target_y)
            corrected_contact = contact_point(frame[0,3].numpy())
            plate = bg.clone()
            # Ground shadow is rendered beneath the cel, following its root and lift.
            shadow = Image.new('RGBA',(width,height))
            d = ImageDraw.Draw(shadow)
            ground = round(FLOOR*data.size)*height/data.size
            x = left+(SUPPORT_X*data.size+data.root_x(t))*height/data.size
            lift = max(0,ground-raw_contact[1])
            radius = height*.085/(1+lift/height*2)
            d.ellipse((x-radius,ground-3,x+radius,ground+7),fill=(14,18,28,65))
            shadow = torch.from_numpy(np.asarray(shadow).copy()).permute(2,0,1).float()[None]/255
            shadow[:,:3] *= shadow[:,3:4]
            plate = over(plate,shadow)
            plate[:,:,:,left:left+height] = over(plate[:,:,:,left:left+height],frame)
            rgb = (plate[0].clamp(0,1).permute(1,2,0).numpy()*255).round().astype('uint8')
            image = Image.fromarray(rgb)
            image.save(temp/f'frame_{index:04d}.png')
            if index in selected:
                tiles.append((index,is_key,image.resize((320,180),Image.Resampling.LANCZOS)))
            records.append(dict(frame=index+1,time=t,keyframe=is_key,grounded=grounded(t),
                                raw_contact_px=raw_contact,contact_px=corrected_contact,
                                correction_px=correction))
        if shutil.which('ffmpeg'):
            temporary = out/'preview_tmp.mp4'
            subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-framerate',str(args.fps),
                            '-i',str(temp/'frame_%04d.png'),'-c:v','libx264','-crf','18',
                            '-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)],check=True)
            os.replace(temporary,out/'preview_latest.mp4')
        else:
            frames = [Image.open(temp/f'frame_{i:04d}.png').convert('RGB') for i in range(args.frames)]
            frames[0].save(out/'preview_tmp.gif',save_all=True,append_images=frames[1:],
                           duration=round(1000/args.fps),loop=0)
            os.replace(out/'preview_tmp.gif',out/'preview_latest.gif')
    panel = Image.new('RGB',(1280,420),(27,33,48))
    d = ImageDraw.Draw(panel)
    for k,(index,is_key,tile) in enumerate(tiles):
        x,y = k%4*320,k//4*210
        panel.paste(tile,(x,y))
        d.text((x+6,y+185),f'FRAME {index+1:02d} | '+('SOURCE KEY' if is_key else 'LEARNED IN-BETWEEN'),fill='white')
    panel.save(out/'comparison_tmp.png')
    os.replace(out/'comparison_tmp.png',out/'comparison_latest.png')
    report = dict(step=step,frames=args.frames,fps=args.fps,resolution=[width,height],
                  blender_shot_sha256=data.shot_hash,max_registered_keyframe_error=max(key_errors),
                  keyframe_count=len(key_errors),frames_detail=records,
                  limits='Source keys bypass synthesis before video encoding. Rescaling and lossy MP4 are not original-file pixel identity. In-between anatomy and likeness require visual review.')
    (out/'preview_manifest.tmp.json').write_text(json.dumps(report,indent=2)+'\n')
    os.replace(out/'preview_manifest.tmp.json',out/'preview_manifest.json')
    print('PREVIEW',out/'preview_latest.mp4','SOURCE KEYS',len(key_errors),flush=True)
    model.train()
