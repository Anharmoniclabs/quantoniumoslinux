"""Read-only FP32 forward/backward preflight for the actual saved model.

A passed CPU result is not GPU evidence. CUDA uses the available GPU (including
T4); no L4, BF16, mixed precision, optimizer update or model export is required.
"""
import gc
import hashlib
import json
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

from chat_recovery_train import atomic_json
from moe_model import MoEConfig, PhiStateMoELM

CONTEXT = 512
LOGIT_ATOL, LOGIT_RTOL = .003, .001
GRAD_ATOL, GRAD_RTOL = .0001, .005


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def _identity(checkpoint, tokenizer_path, device):
    lab = Path(__file__).parent
    result = {'checkpoint_sha256': _sha(checkpoint), 'tokenizer_sha256': _sha(tokenizer_path),
              'torch_version': str(torch.__version__), 'cuda_version': torch.version.cuda,
              'device': str(device), 'source_sha256': {
                  name: _sha(lab/name) for name in ('device_preflight.py', 'phistate.py', 'moe_model.py')},
              'context': CONTEXT, 'dtype': 'float32'}
    if device.type == 'cuda' and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        result['gpu'] = dict(name=props.name, capability=[props.major, props.minor],
                             total_memory_bytes=props.total_memory,
                             uuid=str(getattr(props, 'uuid', 'unavailable')))
    return result


def _forward_backward(model, ids):
    model.train()
    model.zero_grad(set_to_none=True)
    logits = model(ids[:, :-1])
    loss = F.cross_entropy(logits.flatten(0, 1), ids[:, 1:].flatten())
    penalty, _ = model.router_penalty()
    objective = loss + penalty
    if not torch.isfinite(logits).all() or not torch.isfinite(objective):
        raise FloatingPointError('Nonfinite preflight forward/loss')
    objective.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError('Nonfinite gradient: '+name)
    return logits.detach().cpu(), float(objective.detach().cpu())


def run_preflight(checkpoint, tokenizer_path, output_path, device='cuda'):
    """Write and return a cached identity-bound result; caller must require passed.

    Numerical disagreement or unavailable CUDA returns passed=False. Model weights
    are never saved or updated. Configuration/tokenizer errors are also reported
    as failures, so a caller can halt using the same explicit success condition.
    """
    checkpoint, tokenizer_path, output_path = map(Path, (checkpoint, tokenizer_path, output_path))
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Preflight supports cpu or cuda')
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device() if torch.cuda.is_available() else 0)
    identity = _identity(checkpoint, tokenizer_path, device)
    if output_path.is_file():
        cached = json.loads(output_path.read_text())
        if cached.get('identity') == identity and cached.get('passed') is True:
            return {**cached, 'cache_hit': True}
    report = dict(identity=identity, passed=False, cache_hit=False,
                  scope='CPU vs CUDA FP32' if device.type == 'cuda' else 'CPU consistency only; GPU not tested',
                  gpu_tested=False, checkpoint_unchanged=False,
                  tolerances=dict(logit_atol=LOGIT_ATOL, logit_rtol=LOGIT_RTOL,
                                  gradient_atol=GRAD_ATOL, gradient_rtol=GRAD_RTOL))
    if 'gpu' in identity:
        report.update(gpu_name=identity['gpu']['name'], capability=identity['gpu']['capability'],
                      total_gpu_memory_bytes=identity['gpu']['total_memory_bytes'])
    cpu_model = target_model = state = None
    target_parameters = other = None
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    try:
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable; select a GPU runtime')
        cuda_devices = [device.index] if device.type == 'cuda' else []
        # Model construction consumes randomness; restore it on success and error.
        with torch.random.fork_rng(devices=cuda_devices):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if state.get('format') != 'PhiStateMoE-v1':
                raise ValueError('Unsupported checkpoint format')
            if state.get('tokenizer_sha256') != identity['tokenizer_sha256']:
                raise ValueError('Checkpoint tokenizer hash mismatch')
            cfg = MoEConfig(**state['config']); cfg.validate()
            tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
            if cfg.vocab != tokenizer.vocab_size():
                raise ValueError('Tokenizer/model vocabulary mismatch')
            encoded = tokenizer.encode('<|general|>\nA small bird landed beside the window. ' * 256)
            if len(encoded) < CONTEXT+1 or min(encoded) < 0 or max(encoded) >= cfg.vocab:
                raise ValueError('Invalid preflight token sequence')
            ids = torch.tensor([encoded[:CONTEXT+1]], dtype=torch.long)
            cpu_model = PhiStateMoELM(cfg).float()
            cpu_model.load_state_dict(state['state_dict'], strict=True)
            reference, reference_loss = _forward_backward(cpu_model, ids)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            target_model = PhiStateMoELM(cfg).float().to(device)
            target_model.load_state_dict(state['state_dict'], strict=True)
            actual, actual_loss = _forward_backward(target_model, ids.to(device))
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
                report['gpu_tested'] = True
            report['forward_max_error'] = float((reference-actual).abs().max())
            report['objective_cpu'] = reference_loss
            report['objective_target'] = actual_loss
            report['loss_absolute_error'] = abs(reference_loss-actual_loss)
            forward_ok = torch.allclose(reference, actual, atol=LOGIT_ATOL, rtol=LOGIT_RTOL)
            grad_max, failed = 0., []
            target_parameters = dict(target_model.named_parameters())
            for name, parameter in cpu_model.named_parameters():
                other = target_parameters[name]
                expected = parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
                observed = other.grad.detach().cpu() if other.grad is not None else torch.zeros_like(parameter)
                grad_max = max(grad_max, float((expected-observed).abs().max()))
                if not torch.allclose(expected, observed, atol=GRAD_ATOL, rtol=GRAD_RTOL):
                    failed.append(name)
            report['gradient_max_error'] = grad_max
            report['gradient_mismatch_parameters'] = failed
            report['passed'] = bool(forward_ok and not failed and abs(reference_loss-actual_loss) <= .003)
            if not report['passed']:
                report['error'] = 'CPU/device numerical mismatch; halt before training (routing ties can also cause disagreement)'
    except Exception as error:
        report['passed'] = False
        report['error'] = type(error).__name__+': '+str(error)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn
        report['checkpoint_unchanged'] = _sha(checkpoint) == identity['checkpoint_sha256']
        if not report['checkpoint_unchanged']:
            report['passed'] = False
            report['error'] = 'Checkpoint changed during preflight'
        cpu_model = target_model = state = target_parameters = other = None
        gc.collect()
        if device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(output_path, report)
    return report
