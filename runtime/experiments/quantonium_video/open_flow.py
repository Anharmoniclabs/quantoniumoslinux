"""Frozen, endpoint-only RAFT guidance for the existing RGBA interpolator.

No target frame enters the teacher. Flow is an approximate constant-velocity
backward field, not a ground-truth intermediate flow or a trained video generator.
"""
import hashlib
from pathlib import Path
import torch

RAFT_SHA256 = '01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


class FlowGuide:
    def __init__(self):
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        weights = Raft_Small_Weights.C_T_V2
        # Download verifies the publisher's hash prefix; additionally pin full hash.
        state = weights.get_state_dict(progress=True, check_hash=True)
        path = Path(torch.hub.get_dir())/'checkpoints'/Path(weights.url).name
        if sha256(path) != RAFT_SHA256:
            raise ValueError('RAFT weights differ from the verified release')
        self.model = raft_small(weights=None).eval().requires_grad_(False)
        self.model.load_state_dict(state, strict=True)
        self.transform = weights.transforms()

    @torch.inference_mode()
    def endpoints(self, a, b):
        if a.shape != b.shape or a.ndim != 4 or a.shape[1] != 3:
            raise ValueError('Expected equal NCHW RGB endpoint tensors')
        if min(a.shape[-2:]) < 128 or any(n % 8 for n in a.shape[-2:]):
            raise ValueError('RAFT requires dimensions >=128 and divisible by 8')
        x,y = self.transform(a,b)
        # Predict both directions in one batch, using endpoints exclusively.
        flow = self.model(torch.cat((x,y)),torch.cat((y,x)),num_flow_updates=12)[-1]
        if not torch.isfinite(flow).all():
            raise FloatingPointError('Nonfinite teacher flow')
        return flow[:len(a)],flow[len(a):]


def guides_from_flows(ab, ba, t):
    if ab.shape != ba.shape or ab.ndim != 4 or ab.shape[1] != 2:
        raise ValueError('Expected equal N2HW flow tensors')
    t = torch.as_tensor(t,device=ab.device,dtype=ab.dtype).reshape(-1,1,1,1)
    if (t < 0).any() or (t > 1).any():
        raise ValueError('Interpolation time must be in [0,1]')
    empty = ab.new_zeros((len(ab),9,*ab.shape[-2:]))
    return torch.cat((empty,-t*ab,-(1-t)*ba),1)


@torch.inference_mode()
def interpolate(model, teacher, a, b, t=.5):
    """Inference on opaque RGB plates using the same guidance as open-data training."""
    ab,ba = teacher.endpoints(a,b)
    alpha = torch.ones_like(a[:,:1])
    pred,*_ = model(torch.cat((a,alpha),1),torch.cat((b,alpha),1),
                   t,0.,guides_from_flows(ab,ba,t))
    return pred[:,:3]
