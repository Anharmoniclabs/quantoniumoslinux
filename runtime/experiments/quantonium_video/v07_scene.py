"""Image registration, projected rig controls and contact geometry for V0.7."""

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

FLOOR = .87
SUPPORT_X = .56
KEY_TIMES = (0., .16, .28, .40, .50, .62, .78, 1.)
LIFT = (0., 0., 55., 92., 100., 65., 0., 0.)
JOINT_NAMES = ('head','neck','hip','shoulder_l','elbow_l','wrist_l',
               'shoulder_r','elbow_r','wrist_r','knee_l','ankle_l','knee_r','ankle_r','support')
BONES = ((0,1),(1,2),(1,3),(3,4),(4,5),(1,6),(6,7),(7,8),
         (2,9),(9,10),(2,11),(11,12))


def frame_times(count):
    """Include all eight source keys exactly in any supported preview length."""
    keys = np.rint(np.array(KEY_TIMES)*(count-1)).astype(int)
    if len(set(keys)) != 8:
        raise ValueError('Frame count cannot represent all eight key poses')
    return np.interp(np.arange(count),keys,KEY_TIMES).tolist()


def pose_guide(points, size):
    image = Image.new('RGB',(size,size))
    d = ImageDraw.Draw(image)
    colors = ((240,100,90),(90,190,240),(210,220,140))
    for i,(a,b) in enumerate(BONES):
        d.line((*points[a],*points[b]),fill=colors[i%3],width=max(1,size//48))
    for x,y in points:
        r = max(1,size//80)
        d.ellipse((x-r,y-r,x+r,y+r),fill=(255,255,255))
    return torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/255


def rig_flow(source, target, size):
    """Backward deformation prior from the Blender-projected joint controls.

    Local inverse-distance weights preserve a common translation exactly. This
    is an image deformation prior, not a claim of recovered surface anatomy.
    """
    source,target = np.asarray(source),np.asarray(target)
    yy,xx = np.mgrid[:size,:size]
    pixels = np.stack((xx,yy),-1)
    distances = ((pixels[:,:,None,:]-target[None,None,:,:])**2).sum(-1)
    weights = 1/(distances+(size*.035)**2)**2
    weights /= weights.sum(-1,keepdims=True)
    flow = np.einsum('hwj,jc->hwc',weights,source-target)
    return torch.from_numpy(flow.astype('float32')).permute(2,0,1)


def conditioning(start, middle, end, size):
    return torch.cat((pose_guide(start,size),pose_guide(middle,size),pose_guide(end,size),
                      rig_flow(start,middle,size),rig_flow(end,middle,size)),0)


def grounded(t):
    return t <= KEY_TIMES[1] or t >= KEY_TIMES[6]


def contact_point(alpha):
    """Lowest opaque sole band, ignoring transparent margins and faint crop residue."""
    ys, xs = np.nonzero(np.asarray(alpha) > .5)
    if not len(ys):
        raise ValueError('No opaque character pixels')
    bottom = int(ys.max())
    return float(np.median(xs[ys >= bottom - 1])), float(bottom)


def shift(image, dx, dy):
    """Translate premultiplied RGBA by pixels with transparent padding."""
    n, _, h, w = image.shape
    theta = image.new_tensor([[[1, 0, 0], [0, 1, 0]]]).repeat(n, 1, 1)
    theta[:, 0, 2] = -2 * torch.as_tensor(dx, device=image.device) / w
    theta[:, 1, 2] = -2 * torch.as_tensor(dy, device=image.device) / h
    grid = F.affine_grid(theta, image.shape, align_corners=False)
    return F.grid_sample(image, grid, align_corners=False, padding_mode='zeros')


def pin_contact(frame, target_x, target_y):
    """Inference-only root correction. Never applied to airborne frames.

    Returns the measured correction as well as the image: a constraint is not
    evidence that the network has learned contact. The preview reports both.
    """
    x, y = contact_point(frame[0, 3].detach().cpu().numpy())
    dx, dy = target_x - x, target_y - y
    return shift(frame, dx, dy), (dx, dy)


def over(background_rgb, rgba):
    return rgba[:, :3] + background_rgb * (1-rgba[:, 3:4])
