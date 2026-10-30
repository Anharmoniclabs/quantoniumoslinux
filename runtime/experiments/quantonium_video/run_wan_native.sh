#!/usr/bin/env bash
# Does not switch branches, install packages, reset files, or start training.
set -euo pipefail
WORK=/workspaces/quantonium-wan-cpu-1d7a4b6ce30d
PY="$WORK/.venv-wan-cpu/bin/python"
PIN=889b4d6119e8656e77f52b18cd46bdc462e7d4d4
PATH_IN_REPO=runtime/experiments/quantonium_video/wan_native_kernel_render.py

test -x "$PY"
git -C "$WORK" cat-file -e "$PIN^{commit}"
ADAPTER=$(mktemp "$WORK/.wan-native-XXXXXX.py")
git -C "$WORK" show "$PIN:$PATH_IN_REPO" > "$ADAPTER"

"$PY" -u - "$WORK" "$ADAPTER" <<'PY'
import fcntl,json,os,shutil,sys
from pathlib import Path
work,adapter=map(Path,sys.argv[1:])
old=Path('/home/codespace/.cache')
renderer=old/'qos-wan-render-dwOmKC.py'
if not renderer.is_file():
    raise SystemExit('Existing renderer missing; original files were not changed.')
scratch=Path('/tmp')/f'qos-wan-{os.getuid()}'
scratch.mkdir(mode=0o700,exist_ok=True)
if scratch.is_symlink() or scratch.stat().st_uid!=os.getuid():
    raise SystemExit('Unexpected scratch directory ownership or symlink.')
cache=scratch/'.cache';cache.mkdir(exist_ok=True)
if shutil.disk_usage(cache).free<10*(1<<30):
    raise SystemExit('Need 10 GiB free on the scratch filesystem. No cleanup performed.')
# Keep a process-owned lock across exec so a repeated paste cannot start two renders.
lock=os.open(scratch/'native-render.lock',os.O_CREAT|os.O_RDWR,0o600)
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit('A native render launched this way is already active.')
os.set_inheritable(lock,True)

def reuse(src,dst):
    if not src.exists():raise SystemExit(f'Missing preserved input: {src}')
    if dst.is_symlink():
        if dst.resolve()!=src.resolve():raise SystemExit(f'Unexpected link: {dst}')
    elif not dst.exists():
        dst.symlink_to(src,target_is_directory=src.is_dir())

for name in ('qos-wan-scale-ASlSXB','qos-wan-block0-lzg183le'):
    reuse(old/name,cache/name)
source=json.loads((work/'runtime/experiments/quantonium_video/open_sources.json').read_text())['weights']['wan']
name='qos-wan-block01-'+source['revision']
destination=cache/name
if destination.is_symlink():raise SystemExit('Block-cache parent must stay on the scratch disk.')
destination.mkdir(exist_ok=True)
for file in (old/name).glob('block*.safetensors'):
    receipt=file.with_suffix('.json')
    if receipt.is_file():
        # Existing real scratch files are retained and verified by the downloader.
        reuse(file,destination/file.name)
        reuse(receipt,destination/receipt.name)
print('Launching Wan with native RFTMW linear kernels and staged weight management.',flush=True)
print('No package reinstall. Existing checkpoints and original render are unchanged.',flush=True)
os.execv(sys.executable,[sys.executable,'-u',str(adapter),'--work',str(work),
    '--renderer',str(renderer),'--cache',str(cache),'--threads','2',
    '--frames','17','--size','256','--steps','30'])
PY
