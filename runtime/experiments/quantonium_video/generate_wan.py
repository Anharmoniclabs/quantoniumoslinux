"""Optional open-weight GPU text-to-video backend; not a V0.7 checkpoint loader."""
import argparse
import json
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompt',required=True)
    p.add_argument('--out',required=True,help='New directory for frames, MP4 and provenance')
    p.add_argument('--seed',type=int,default=808)
    p.add_argument('--frames',type=int,default=49)
    p.add_argument('--steps',type=int,default=30)
    args = p.parse_args()
    if args.frames < 5 or args.frames > 81 or (args.frames-1)%4 or args.steps < 1:
        p.error('frames must be 4n+1 in 5..81, steps positive')
    if not torch.cuda.is_available():
        p.error('Wan generation requires a CUDA GPU. No weights downloaded or remote job started.')
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.utils import export_to_video
    source = json.loads(Path(__file__).with_name('open_sources.json').read_text())['weights']['wan']
    root = Path(args.out).expanduser().resolve()
    root.mkdir(parents=True,exist_ok=False)
    # All components use the same immutable upstream revision, safetensors only.
    kwargs = dict(revision=source['revision'],use_safetensors=True)
    vae = AutoencoderKLWan.from_pretrained(source['repo'],subfolder='vae',torch_dtype=torch.float32,**kwargs)
    pipe = WanPipeline.from_pretrained(source['repo'],vae=vae,torch_dtype=torch.bfloat16,**kwargs)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    frames = pipe(prompt=args.prompt,height=480,width=832,num_frames=args.frames,
        num_inference_steps=args.steps,guidance_scale=5.,
        generator=torch.Generator(device='cpu').manual_seed(args.seed),output_type='pil').frames[0]
    shot = root/'shot001'
    shot.mkdir()
    for i,frame in enumerate(frames):
        frame.save(shot/f'{i:05d}.png')
    export_to_video(frames,str(root/'generated.mp4'),fps=16)
    (root/'provenance.json').write_text(json.dumps(dict(source=source,prompt=args.prompt,
        seed=args.seed,frames=args.frames,fps=16,steps=args.steps,
        kind='synthetic generated video; requires review before training',
        compatible_frame_importer='train_v03.py --frames-dir; need multiple reviewed shots for split'),indent=2)+'\n')


if __name__ == '__main__':
    main()
