"""Optional open-weight GPU text-to-video backend; not a V0.7 checkpoint loader."""
import argparse
import gc
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
    p.add_argument('--height',type=int,default=480)
    p.add_argument('--width',type=int,default=832)
    p.add_argument('--lora-run',default='',help='Verified Wan LoRA run folder; omit for base model')
    p.add_argument('--checkpoint',choices=['best','latest'],default='best')
    args = p.parse_args()
    if (args.frames < 5 or args.frames > 81 or (args.frames-1)%4 or args.steps < 1
        or min(args.height,args.width)<128 or args.height%16 or args.width%16):
        p.error('frames must be 4n+1 in 5..81; steps positive; dimensions >=128 and multiples of 16')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        p.error('Wan generation requires a BF16 CUDA GPU, such as L4/A100.')
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.utils import export_to_video
    source = json.loads(Path(__file__).with_name('open_sources.json').read_text())['weights']['wan']
    adapter, adapter_info = None, None
    if args.lora_run:
        from wan_training import find_checkpoint, verify_checkpoint, digest
        adapter = find_checkpoint(args.lora_run,best=args.checkpoint=='best')
        if adapter is None:
            p.error('No trained Wan checkpoint exists yet')
        state,_ = verify_checkpoint(adapter)
        if state['config']['source'] != source:
            p.error('Adapter belongs to a different base model')
        adapter_info = dict(step=state['step'],folder=str(adapter),
            sha256=digest(adapter/'pytorch_lora_weights.safetensors'))
    root = Path(args.out).expanduser().resolve()
    root.mkdir(parents=True,exist_ok=False)
    # All components use the same immutable upstream revision, safetensors only.
    kwargs = dict(revision=source['revision'],use_safetensors=True)
    # Encode and unload the large T5 before loading the transformer and VAE.
    encoder_pipe = WanPipeline.from_pretrained(source['repo'],transformer=None,vae=None,
        torch_dtype=torch.bfloat16,**kwargs)
    encoder_pipe.text_encoder.to('cuda')
    with torch.no_grad():
        prompt_embeds,negative_embeds = encoder_pipe.encode_prompt(prompt=args.prompt,
            negative_prompt='',device=torch.device('cuda'),max_sequence_length=512)
    del encoder_pipe
    gc.collect()
    torch.cuda.empty_cache()
    vae = AutoencoderKLWan.from_pretrained(source['repo'],subfolder='vae',torch_dtype=torch.float32,**kwargs)
    pipe = WanPipeline.from_pretrained(source['repo'],vae=vae,text_encoder=None,tokenizer=None,
        torch_dtype=torch.bfloat16,**kwargs)
    if adapter:
        pipe.load_lora_weights(str(adapter))
        print('WAN GENERATION USING TRAINED ADAPTER STEP',adapter_info['step'],flush=True)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    frames = pipe(prompt_embeds=prompt_embeds,negative_prompt_embeds=negative_embeds,
        height=args.height,width=args.width,num_frames=args.frames,
        num_inference_steps=args.steps,guidance_scale=5.,
        generator=torch.Generator(device='cpu').manual_seed(args.seed),output_type='pil').frames[0]
    shot = root/'shot001'
    shot.mkdir()
    for i,frame in enumerate(frames):
        frame.save(shot/f'{i:05d}.png')
    export_to_video(frames,str(root/'generated.mp4'),fps=16)
    (root/'provenance.json').write_text(json.dumps(dict(source=source,prompt=args.prompt,
        seed=args.seed,frames=args.frames,fps=16,steps=args.steps,lora=adapter_info,
        kind='synthetic generated video; requires review before training',
        compatible_frame_importer='train_v03.py --frames-dir; need multiple reviewed shots for split'),indent=2)+'\n')


if __name__ == '__main__':
    main()
