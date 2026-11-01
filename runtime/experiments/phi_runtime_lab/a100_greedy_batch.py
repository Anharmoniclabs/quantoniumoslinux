"""Batched FP32 greedy decoding compatible with coherence_quality.generate_answer."""
from contextlib import contextmanager

import torch

from coherence_quality import generate_answer, generation_prompt_text


class _SerialFallbackAnswers(list):
    """Keep answer dictionaries identical to serial output and expose run metadata."""
    decode_backend = 'serial_after_batch_parity_mismatch'


@contextmanager
def _fp32_eval(model, device):
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Batched FP32 evaluation supports CPU and CUDA models')
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('Batched greedy evaluation requires FP32 model weights')
    was_training = model.training
    tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    tf32_cudnn = torch.backends.cudnn.allow_tf32
    try:
        if device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        model.eval()
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            yield
    finally:
        model.train(was_training)
        if device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = tf32_matmul
            torch.backends.cudnn.allow_tf32 = tf32_cudnn


def _chunk(model, tokenizer, batch, device):
    eos = tokenizer.eos_id()
    batch_n = len(batch)
    prefix_sizes = [len(ids) for _, ids, _ in batch]
    prefix_max = max(prefix_sizes)
    prefix_min = min(prefix_sizes)
    limits = [limit for _, _, limit in batch]
    # Storage beyond the prefix is never read as a prompt token. Every real
    # prompt begins at cache position zero, with no left padding.
    prefixes = torch.tensor([ids + [eos]*(prefix_max-len(ids)) for _, ids, _ in batch], device=device)
    length = torch.tensor(prefix_sizes, device=device)
    limit = torch.tensor(limits, device=device)
    output = torch.full((batch_n, max(limits)), eos, device=device, dtype=torch.long)
    finished = torch.zeros(batch_n, device=device, dtype=torch.bool)
    ended_on_eos = torch.zeros_like(finished)
    nonfinite = torch.zeros_like(finished)
    cache = None
    token = prefixes[:, 0]
    max_steps = max(n + lim - 1 for n, lim in zip(prefix_sizes, limits))
    for position in range(max_steps):
        logits, cache = model.step(token.reshape(batch_n, 1), cache)
        scores = logits[:, -1, :]
        if scores.dtype != torch.float32:
            raise RuntimeError('Model returned non-FP32 greedy logits')
        nonfinite |= ~finished & ~torch.isfinite(scores).all(dim=-1)
        if position+1 < prefix_min:
            token = prefixes[:, position+1]
            continue
        next_id = scores.argmax(dim=-1)
        output_pos = position + 1 - length
        emit = ~finished & (output_pos >= 0)
        index = output_pos.clamp(0, output.shape[1]-1).unsqueeze(1)
        previous = output.gather(1, index).squeeze(1)
        output.scatter_(1, index, torch.where(emit, next_id, previous).unsqueeze(1))
        ended_on_eos |= emit & (next_id == eos)
        finished |= ended_on_eos | (emit & (output_pos+1 == limit))
        # Poll by chunk, rather than synchronizing CUDA on every token. Rows
        # which have finished consume EOS only and have no influence on outputs.
        if (position+1) % 8 == 0 and bool(finished.all()):
            break
        token = next_id
        if position+1 < prefix_max:
            token = torch.where(position+1 < length, prefixes[:, position+1], next_id)
        token = torch.where(finished, eos, token)
    bad_rows = nonfinite.nonzero().flatten().tolist()
    if bad_rows:
        original_indices = [batch[i][0] for i in bad_rows]
        raise FloatingPointError(f'Nonfinite generation logits for active input indices {original_indices}')
    ended = ended_on_eos.tolist()
    answers = []
    for (item_index, _, max_tokens), ids, eos_seen in zip(batch, output.tolist(), ended):
        produced = ids[:max_tokens]
        if eos_seen:
            produced = produced[:produced.index(eos)]
        answers.append((item_index, dict(text=tokenizer.decode(produced), tokens=len(produced), eos=eos_seen)))
    return answers


def generate_batch(model, tokenizer, items, batch_size=16, verify=0, serial_on_mismatch=False):
    """Return one greedy answer dict per item, preserving the input order.

    Each item supplies `max_tokens` plus either a user string in `prompt` or a
    conversation ending in a user turn in `messages`. `verify` selects up to that
    many representative prompt/budget lengths and requires text, token count,
    and EOS to agree with the original single-row FP32 implementation. When
    `serial_on_mismatch=True`, a differing answer discards the entire batched
    result and reruns this complete suite in original order on the serial path.
    Numerical and execution faults still propagate to the caller.
    """
    if type(batch_size) is not int or not 1 <= batch_size <= 128:
        raise ValueError('batch_size must be an integer from 1 through 128')
    if type(verify) is not int or verify < 0:
        raise ValueError('verify must be a nonnegative integer')
    if type(serial_on_mismatch) is not bool:
        raise ValueError('serial_on_mismatch must be a boolean')
    items = list(items)
    if not items:
        return []
    device = next(model.parameters()).device
    encoded, prompts = [], []
    for index, item in enumerate(items):
        if ('prompt' in item) == ('messages' in item):
            raise ValueError(f'Expected exactly one of prompt and messages at input {index}')
        key = 'messages' if 'messages' in item else 'prompt'
        prompt, limit = item[key], item['max_tokens']
        if not isinstance(prompt, list if key == 'messages' else str) or type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError(f'Invalid prompt or generation limit at input {index}')
        ids = tokenizer.encode(generation_prompt_text(prompt))
        if not ids or len(ids) + limit > 512:
            raise ValueError(f'Empty prompt or more than 512 prompt+generation tokens at input {index}')
        encoded.append((index, ids, limit))
        prompts.append(prompt)
    # Similar lengths minimize wasted work on completed rows while the index
    # mapping keeps grading, reports, and examples in their original order.
    encoded.sort(key=lambda row: (len(row[1])+row[2], len(row[1]), row[0]))
    results = [None] * len(items)
    with _fp32_eval(model, device):
        for start in range(0, len(encoded), batch_size):
            for index, answer in _chunk(model, tokenizer, encoded[start:start+batch_size], device):
                results[index] = answer
        sample_count = min(verify, len(items))
        sample_indices = ([len(items)//2] if sample_count == 1 else
                          [round(i*(len(items)-1)/(sample_count-1)) for i in range(sample_count)])
        for position in sample_indices:
            index = encoded[position][0]
            item = items[index]
            expected = generate_answer(model, tokenizer, prompts[index], max_tokens=item['max_tokens'])
            if results[index] != expected:
                if serial_on_mismatch:
                    print(f'BATCH_PARITY_MISMATCH at input {index}. Discarding all {len(items)} '
                          'batched answers and recomputing the complete suite serially.', flush=True)
                    serial = _SerialFallbackAnswers(
                        generate_answer(model, tokenizer, prompt, max_tokens=row['max_tokens'])
                        for row, prompt in zip(items, prompts)
                    )
                    serial.mismatch_index = index
                    return serial
                raise RuntimeError(f'Batched vs single-row greedy generation mismatch at input {index}: '
                                   f'batched={results[index]!r}, single={expected!r}. Stop this comparison '
                                   'and use the single-row decoder for both candidates.')
    return results
