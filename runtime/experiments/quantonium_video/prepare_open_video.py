"""Download bounded open-movie samples and cache endpoint-only pretrained flow."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request
import zipfile

import numpy as np
import torch

from open_flow import FlowGuide, sha256, RAFT_SHA256
from train_v07 import atomic_json

CATALOG = Path(__file__).with_name('open_sources.json')


def download(url, path, maximum=100_000_000):
    """Bound transfer size; never publish a partial download as a complete file."""
    temp = path.with_suffix('.part')
    with urllib.request.urlopen(url,timeout=60) as response, temp.open('wb') as output:
        total = 0
        while block := response.read(1024*1024):
            total += len(block)
            if total > maximum:
                raise ValueError('Source exceeds the download size limit')
            output.write(block)
    temp.replace(path)


def reject_triplet(frames):
    """Conservative cut/blank heuristic, not a guarantee of semantic continuity."""
    f = frames.float()/255
    if any(image.std() < .025 for image in f):
        return True
    for a,b in zip(f,f[1:]):
        if (a-b).abs().mean() > .18:
            return True
        ha = torch.histc(a.mean(0),bins=32,min=0,max=1)/a.shape[-1]**2
        hb = torch.histc(b.mean(0),bins=32,min=0,max=1)/b.shape[-1]**2
        if (ha-hb).abs().sum() > .45:
            return True
    return False


def extract(video, stamp, size):
    command = ['ffmpeg','-v','error','-ss',str(stamp),'-i',str(video),
        '-an','-vf',f'fps=12,scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}',
        '-frames:v','3','-f','rawvideo','-pix_fmt','rgb24','-threads','1','pipe:1']
    raw = subprocess.run(command,check=True,capture_output=True,timeout=60).stdout
    if len(raw) != 3*size*size*3:
        raise ValueError('Source ended before a complete triplet')
    return torch.from_numpy(np.frombuffer(raw,dtype=np.uint8).copy().reshape(3,size,size,3)).permute(0,3,1,2)


def prepare(args):
    torch.set_num_threads(args.threads)
    root = Path(args.out).expanduser().resolve()
    if (root/'manifest.json').exists():
        raise ValueError('Dataset already prepared; choose a new output directory')
    root.mkdir(parents=True,exist_ok=True)
    catalog = json.loads(CATALOG.read_text())
    teacher = FlowGuide()
    records, sources = [],{}
    for name, source in catalog['videos'].items():
        local = root/(name+('.zip' if 'archive_member' in source else '.mp4'))
        if not local.exists():
            download(source['url'],local)
        if sha256(local) != source['download_sha256']:
            raise ValueError('Movie release changed: '+name)
        sources[name] = {**source,'download_sha256':sha256(local)}
        if 'archive_member' in source:
            video = root/(name+'.mp4')
            with zipfile.ZipFile(local) as archive:
                info = archive.getinfo(source['archive_member'])
                if info.file_size > 100_000_000:
                    raise ValueError('Expanded movie exceeds size limit')
                with archive.open(info) as src, video.open('wb') as dst:
                    shutil.copyfileobj(src,dst)
        else:
            video = local
        sources[name]['video_sha256'] = sha256(video)
        stamps = np.linspace(source['start_seconds'],source['end_seconds'],args.samples)
        for i,stamp in enumerate(stamps):
            frames = extract(video,float(stamp),args.size)
            if reject_triplet(frames):
                continue
            a,b = frames[[0,2]].float()/255
            ab,ba = teacher.endpoints(a[None],b[None])
            filename = f'{name}_{i:04d}.pt'
            torch.save(dict(frames=frames,ab=ab[0].clone(),ba=ba[0].clone()),root/filename)
            records.append(dict(file=filename,sha256=sha256(root/filename),source=name,
                                split=source['split'],start_seconds=float(stamp),fps=12,t=.5))
            print(name,i+1,'/',args.samples,flush=True)
    if not all(any(r['split']==s for r in records) for s in ('train','validation')):
        raise ValueError('Need nonempty train and validation films')
    manifest = dict(schema=1,size=args.size,teacher_sha256=RAFT_SHA256,
        sources=sources,records=records,preprocessing='RGB only; centered square crop; 12fps; target midpoint',
        cut_filter='MAD <= .18, histogram L1 <= .45, std >= .025; heuristic, manually review samples',
        split_scope='entire different films; validation is never used for gradient updates')
    atomic_json(root/'manifest.json',manifest)
    (root/'ATTRIBUTION.txt').write_text('\n'.join(
        s['attribution']+' | '+s['license']+' | '+s['license_source'] for s in sources.values())+'\n')
    print('PREPARED',len(records),root/'manifest.json',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    p.add_argument('--size',type=int,default=128)
    p.add_argument('--samples',type=int,default=32,help='Candidate triplets per film')
    p.add_argument('--threads',type=int,default=4)
    a = p.parse_args()
    if a.size < 128 or a.size > 256 or a.size%8 or not 2 <= a.samples <= 512 or a.threads < 1:
        p.error('size 128..256 divisible by 8; samples 2..512; threads positive')
    prepare(a)
