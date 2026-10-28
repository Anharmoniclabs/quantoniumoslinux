#!/usr/bin/env python3
"""Extract original Jayc/J.C. full-body art from a LOCAL PDF character bible.
Never uploads source or output. Reference-only: inspect and approve the cutout.
"""
import argparse
from collections import deque
from pathlib import Path
import hashlib
import json
import re
import numpy as np
from PIL import Image

APPROVED = {
    'name': 'JC / Jayc (Juan Carlos Cabral)',
    'hair': 'close Caesar haircut, dark waves and taper, no afro',
    'eyes': 'large vertically elongated eyes, white sclera and slim dark pupils',
    'wardrobe': 'black urban streetwear, black/red headphones',
    'props': 'gold pendant separate from copper story coin',
    'style': 'clean fine 1990s anime outlines, no heavy ink or game-style shading',
    'rule': 'Preserve supplied pixels: never synthesize a replacement face.'
}

def find_bible():
    roots = [Path.home() / 'Downloads', Path.home() / 'Documents',
             Path('/mnt/c/Users/Anharoniclabs/Downloads'),
             Path('/mnt/c/Users/Anharoniclabs/Documents'),
             Path('/mnt/c/Users/Anharoniclabs/Desktop')]
    matches=[]
    for root in roots:
        if not root.is_dir(): continue
        for pattern in ('*Character*Bible*.pdf', '*Character*Image*Bible*.pdf',
                        '*Full_Cast*.pdf', '*Cemi*Bible*.pdf'):
            matches.extend(p for p in root.glob(pattern) if p.is_file())
    if not matches:
        raise SystemExit('Bible PDF not found in Downloads/Documents/Desktop. Pass --bible "/path/to/file.pdf"')
    matches=list({str(p.resolve()):p for p in matches}.values())
    matches.sort(key=lambda p: (0 if 'Full_Cast_Character_Bible' in p.name else 1,
                                0 if 'Character_Image_Bible' in p.name else 1,
                                -p.stat().st_mtime))
    print('FOUND BIBLES:')
    for p in matches[:10]: print(' ',p)
    print('USING',matches[0])
    return matches[0]

def find_page(doc, requested):
    if requested != 'auto':
        i=int(requested)-1
        if not 0<=i<len(doc): raise ValueError('Page out of range')
        return i
    if len(doc)==99: return 7 # approved JC v2 in 99-page image inventory
    if len(doc)==64: return 0 # JC on p1 of 64-sheet cast bible
    for i in range(min(25,len(doc))):
        text=doc[i].get_text().lower()
        if re.search(r'\b(?:g002|jayc|juan carlos|\bjc\b)',text):
            return i
    raise SystemExit('Cannot locate JC page. Supply --page 1 (64-page Bible) or --page 8 (99-page image inventory).')

def crop_default(doc,i):
    if len(doc)==99 and i==7:
        return (.174,.164,.366,.806)
    # Conservative left full-body region for other Bible layouts; check review.
    return (.075,.125,.435,.895)

def remove_exterior_white(im, threshold=239):
    """Erase only WHITE pixels connected to outside; keep white eyes and shoes."""
    rgb=np.asarray(im.convert('RGB'),dtype=np.uint8).copy()
    h,w=rgb.shape[:2]
    white=np.all(rgb>=threshold,axis=2)
    exterior=np.zeros((h,w),dtype=bool)
    q=deque()
    for x in range(w):
        if white[0,x]: exterior[0,x]=True;q.append((0,x))
        if white[h-1,x]: exterior[h-1,x]=True;q.append((h-1,x))
    for y in range(h):
        if white[y,0]: exterior[y,0]=True;q.append((y,0))
        if white[y,w-1]: exterior[y,w-1]=True;q.append((y,w-1))
    while q:
        y,x=q.popleft()
        for ny,nx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            if 0<=ny<h and 0<=nx<w and white[ny,nx] and not exterior[ny,nx]:
                exterior[ny,nx]=True;q.append((ny,nx))
    alpha=(~exterior).astype(np.uint8)*255
    return Image.fromarray(np.dstack([rgb,alpha]),'RGBA')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bible',default='auto',help='Local PDF path or auto-search Downloads')
    p.add_argument('--page',default='auto',help='1-indexed page, or auto (64p->1, 99p->8)')
    p.add_argument('--crop',type=float,nargs=4,metavar=('X0','Y0','X1','Y1'),
                   help='Normalized page crop 0..1; adjust if art does not fill crop')
    p.add_argument('--out',default='~/quantonium-video-runs/v05/jayc_private',
                   help='Private output, kept outside git')
    p.add_argument('--height',type=int,default=900)
    p.add_argument('--white-threshold',type=int,default=239)
    a=p.parse_args()
    try: import fitz
    except ImportError: raise SystemExit('Missing PyMuPDF: python -m pip install pymupdf')
    path=find_bible() if a.bible=='auto' else Path(a.bible).expanduser()
    if not path.is_file(): raise SystemExit('No Bible at '+str(path))
    doc=fitz.open(path)
    i=find_page(doc,a.page)
    bounds=a.crop or crop_default(doc,i)
    x0,y0,x1,y1=bounds
    if not (0<=x0<x1<=1 and 0<=y0<y1<=1):
        raise SystemExit('--crop values must be normalized 0..1')
    out=Path(a.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    page=doc[i]
    zoom=max(1.0,min(4.0,a.height/max(1,page.rect.height*(y1-y0))))
    pix=page.get_pixmap(matrix=fitz.Matrix(zoom,zoom),alpha=False)
    full=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
    rect=(round(x0*full.width),round(y0*full.height),
          round(x1*full.width),round(y1*full.height))
    raw=full.crop(rect)
    raw.save(out/'jayc_crop_review.png')
    sprite=remove_exterior_white(raw,a.white_threshold)
    bbox=sprite.getchannel('A').getbbox()
    if not bbox: raise SystemExit('No foreground found. Inspect crop and change --crop.')
    sprite=sprite.crop(bbox)
    aspect=sprite.width/max(1,sprite.height)
    if not .12<aspect<.95:
        raise SystemExit(f'Unexpected sprite aspect ratio {aspect:.2f}; inspect and retry with --crop')
    sprite.save(out/'jayc_cutout.png')
    checksum=hashlib.sha256(path.read_bytes()).hexdigest()
    meta={'reference':'private local Bible','source_name':path.name,
          'pdf_sha256':checksum,'page_1_indexed':i+1,'page_count':len(doc),
          'crop_normalized':list(bounds),'sprite_size_px':list(sprite.size),
          'crop_review':'jayc_crop_review.png','cutout':'jayc_cutout.png',
          'approved_model_spec':APPROVED,
          'qc':'VISUAL APPROVAL REQUIRED: verify face, eyes, pendant, full body, shoes and clean alpha.'}
    (out/'jayc_asset_manifest.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False))
    print('PAGE',i+1,'CROP',bounds,'CUTOUT',sprite.size)
    print('SAVED',out/'jayc_cutout.png')
    print('REVIEW',out/'jayc_crop_review.png')
    print('Never publish private Bible artwork or automatically replace the character.',flush=True)

if __name__=='__main__':
    main()
