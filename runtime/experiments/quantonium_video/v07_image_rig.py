"""Image-to-joint fitting for the Blender prepass. No image upload/service call.

Known JC cels use source-bound 2D annotations (estimated joint centers under
clothing, NOT creator-approved anatomy). Unseen images use local Keypoint R-CNN
with a confidence gate; a detector cannot establish hidden 3D anatomy.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from v07_scene import contact_point

def pixel_hash(image):
    return hashlib.sha256(np.asarray(image.convert('RGBA')).tobytes()).hexdigest()


def fit_image(path, reference_assets=None, force_detector=False):
    path = Path(path).expanduser().resolve()
    with Image.open(path) as source:
        rgba = source.convert('RGBA')
    pixels = np.asarray(rgba).copy()/255.
    if pixels[:,:,3].max() == 0:
        raise ValueError('Image-to-rig requires a nonempty full-body image')
    has_alpha = pixels[:,:,3].min() < 1
    inferred_support = None
    digest = pixel_hash(rgba)
    known = None
    if reference_assets and not force_detector:
        root = Path(reference_assets)
        # Verify file bytes against the existing published manifest first.
        sums = {p:h for h,p in (line.split() for line in (root/'SHA256SUMS').read_text().splitlines())}
        for i in range(8):
            file = root/f'pose_{i+1:02d}.webp'
            if hashlib.sha256(file.read_bytes()).hexdigest() != sums.get(file.name):
                raise ValueError('Reference image hash mismatch: '+str(file))
            with Image.open(file) as ref:
                if pixel_hash(ref) == digest:
                    known = i
                    break
    if known is not None:
        records = json.loads(Path(__file__).with_name('assets').joinpath('jc_v07_rig_landmarks.json').read_text())
        record = next((r for r in records if r['pixel_sha256'] == digest),None)
        if record is None:
            raise ValueError('No source-bound landmark record for this image')
        points = np.array(record['points'][:13],dtype=float)
        method = 'cached local pose detection + inspected joint corrections (estimated)'
        confidence = dict(person=record['person_score'],min_limb_logit=record['min_limb_logit'],review=record['review'])
    else:
        try:
            from torchvision.models.detection import keypointrcnn_resnet50_fpn
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError('New images need a matching torch/torchvision installation for local pose detection') from exc
        # Explicit upstream revision/hash. Safe tensor-only loading, no pickle fallback.
        url = 'https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-fc266e95.pth'
        state = torch.hub.load_state_dict_from_url(url,map_location='cpu',
                                                  check_hash=True,weights_only=True)
        model = keypointrcnn_resnet50_fpn(weights=None,weights_backbone=None,
                                        min_size=512,max_size=768).eval()
        model.load_state_dict(state,strict=True)
        rgb = pixels[:,:,:3]*pixels[:,:,3:4]+.75*(1-pixels[:,:,3:4])
        tensor = torch.from_numpy(rgb).permute(2,0,1).float()
        with torch.inference_mode():
            result = model([tensor])[0]
        keep = torch.nonzero(result['scores']>.85).flatten()
        if len(keep) != 1:
            raise ValueError('Image-to-rig needs exactly one confidently detected full-body person')
        k = int(keep[0])
        xy = result['keypoints'][k,:,:2].numpy()
        scores = result['keypoints_scores'][k].numpy()
        if not np.isfinite(xy).all() or (scores[5:] < 2.).any():
            raise ValueError('Uncertain limb joints: provide a clearer full-body image or reviewed landmarks')
        # Keep anatomical sides consistent through the whole shot; source annotation
        # names refer to the two image limbs, not anatomical identity guarantees.
        sides = (5,6) if xy[5,0] < xy[6,0] else (6,5)
        left,right = sides
        points = np.array([xy[0],(xy[5]+xy[6])/2,(xy[11]+xy[12])/2,
            xy[left],xy[left+2],xy[left+4],xy[right],xy[right+2],xy[right+4],
            xy[left+8],xy[left+10],xy[right+8],xy[right+10]])
        method = 'Keypoint R-CNN COCO_V1 fc266e95; estimated anatomy'
        confidence = dict(person=float(result['scores'][k]),min_limb_logit=float(scores[5:].min()))
        if not has_alpha:
            ankle = xy[15:17][np.argmax(xy[15:17,1])]
            inferred_support = [float(ankle[0]),min(rgba.height-1,float(result['boxes'][k,3])-1)]
    support = contact_point(pixels[:,:,3]) if has_alpha else inferred_support
    points = np.vstack((points,support))
    if (points[:,0]<0).any() or (points[:,0]>=rgba.width).any() or (points[:,1]<0).any() or (points[:,1]>=rgba.height).any():
        raise ValueError('Joint estimate outside source image; refusing a misregistered rig')
    return dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                pixel_sha256=digest,dimensions=list(rgba.size),points=points.tolist(),
                method=method,confidence=confidence,support_method='opaque sole band' if has_alpha else 'estimated from person box and ankle',
                limitation='2D joint estimates; depth and proxy volume are inferred, not a reconstructed JC mesh')
