"""Render between RGB endpoints with a V0.8 checkpoint and frozen RAFT guidance."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageOps
import torch

from open_flow import FlowGuide, guides_from_flows, sha256
from train_open_v08 import VERSION
from train_v07 import PoseInterpolator


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','first','last','out'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--frames',type=int,default=12)
    p.add_argument('--fps',type=int,default=12)
    args = p.parse_args()
    if not 2 <= args.frames <= 240 or not 1 <= args.fps <= 60:
        p.error('frames 2..240; fps 1..60')
    torch.set_num_threads(4)
    state = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    cfg = state['config']
    if cfg['version'] != VERSION:
        p.error('Expected a V0.8 open-data checkpoint')
    model = PoseInterpolator(cfg['size'],cfg['width']).eval()
    model.load_state_dict(state['model'],strict=True)
    images = []
    for path in (args.first,args.last):
        with Image.open(path) as im:
            im = ImageOps.fit(im.convert('RGB'),(cfg['size'],cfg['size']),method=Image.Resampling.BICUBIC)
            images.append(torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float()[None]/255)
    a,b = images
    teacher = FlowGuide()
    ab,ba = teacher.endpoints(a,b)
    alpha = torch.ones_like(a[:,:1])
    a,b = torch.cat((a,alpha),1),torch.cat((b,alpha),1)
    root = Path(args.out).expanduser().resolve()
    root.mkdir(parents=True,exist_ok=False)
    with torch.inference_mode():
        for i,t in enumerate(torch.linspace(0,1,args.frames)):
            pred,*_ = model(a,b,t,0.,guides_from_flows(ab,ba,t))
            array = (pred[0,:3].permute(1,2,0).clamp(0,1)*255).round().byte().numpy()
            Image.fromarray(array).save(root/f'{i:04d}.png')
    subprocess.run(['ffmpeg','-v','error','-framerate',str(args.fps),'-i',str(root/'%04d.png'),
        '-c:v','libx264','-pix_fmt','yuv420p',str(root/'preview.mp4')],check=True)
    (root/'provenance.json').write_text(json.dumps(dict(checkpoint_sha256=sha256(args.checkpoint),
        phase_step=state['step'],frames=args.frames,fps=args.fps,
        endpoint_sha256=[sha256(args.first),sha256(args.last)],
        limitation='square RGB interpolation, constant-velocity flow approximation; no JC rig or contact pin'),indent=2)+'\n')


if __name__ == '__main__':
    main()
