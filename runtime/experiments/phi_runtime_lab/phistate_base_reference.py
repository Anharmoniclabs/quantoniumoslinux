"""Conventional dense, full-causal-attention next-token reference for the PhiState corpus trial."""
import argparse
import hashlib
import itertools
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6).to(x.dtype)


class DenseBlock(nn.Module):
    def __init__(self, width, heads, hidden):
        super().__init__()
        self.heads = heads
        self.attn_norm = RMSNorm(width)
        self.ff_norm = RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.gate_value = nn.Linear(width, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x, cos, sin):
        b, t, w = x.shape
        q, k, v = [tensor.view(b, t, self.heads, w // self.heads).transpose(1, 2)
                   for tensor in self.qkv(self.attn_norm(x)).chunk(3, dim=-1)]
        z = F.scaled_dot_product_attention(_rotate(q, cos, sin), _rotate(k, cos, sin), v,
                                           dropout_p=0.0, is_causal=True)
        x = x + self.proj(z.transpose(1, 2).reshape(b, t, w))
        gate, value = self.gate_value(self.ff_norm(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * value)


def _rotate(x, cos, sin):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)


class DenseCausalLM(nn.Module):
    """Dense SwiGLU Transformer with tied output embeddings, RoPE and full causal attention."""
    def __init__(self, vocab=8192, width=320, layers=6, heads=8, context=128, hidden=1920):
        super().__init__()
        if min(vocab, width, layers, heads, context, hidden) < 1 or width % (heads * 2):
            raise ValueError('Width must contain a positive even number of channels per attention head.')
        self.vocab, self.context = vocab, context
        self.embedding = nn.Embedding(vocab, width)
        self.blocks = nn.ModuleList(DenseBlock(width, heads, hidden) for _ in range(layers))
        self.final_norm = RMSNorm(width)
        freqs = 10000.0 ** (-torch.arange(0, width // heads, 2, dtype=torch.float32) / (width // heads))
        angles = torch.arange(context, dtype=torch.float32)[:, None] * freqs[None, :]
        self.register_buffer('cos', angles.cos(), persistent=False)
        self.register_buffer('sin', angles.sin(), persistent=False)
        for module in self.modules():
            if isinstance(module, (nn.Embedding, nn.Linear)):
                nn.init.normal_(module.weight, std=0.02)
        for layer in self.blocks:
            for proj in (layer.proj, layer.down):
                nn.init.normal_(proj.weight, std=0.02 / math.sqrt(2 * layers))

    def forward(self, ids):
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.context:
            raise ValueError(f'Expected [batch, 1..{self.context}] token IDs.')
        x = self.embedding(ids)
        cos, sin = self.cos[:ids.shape[1]].to(x.dtype), self.sin[:ids.shape[1]].to(x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        return F.linear(self.final_norm(x), self.embedding.weight)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def _symbols(tokenizer, alphabet):
    output = []
    for symbol in alphabet:
        ids = tokenizer.encode(symbol)
        if len(ids) == 1 and tokenizer.decode(ids) == symbol and ids[0] not in output:
            output.append(ids[0])
    if len(output) < 8:
        raise ValueError('Diagnostic requires eight distinct one-token symbols.')
    return output[:8]


def _synthetic_windows(tokenizer, confirm=False):
    """Hold out complete (key, distractor1, distractor2) tuples; each occurs with every value."""
    digits = _symbols(tokenizer, '0123456789')
    keys = _symbols(tokenizer, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    if set(keys) & set(digits):
        raise ValueError('Overlapping key and value alphabets.')
    filler = keys
    eos = tokenizer.eos_id()
    start = tokenizer.encode('=')
    if eos < 0 or not start:
        raise ValueError('A start marker and EOS are required.')
    # The colon is the identical final input token in every diagnostic example.
    suffix = tokenizer.encode(':')
    if not suffix:
        raise ValueError('Missing marker.')
    groups = list(itertools.product(range(8), repeat=3))
    random.Random(90719).shuffle(groups)
    # The first pilot inspected groups[448:]; confirmatory models exclude those entirely.
    splits = ({'train': groups[:384], 'verify': groups[384:448]} if confirm else
              {'train': groups[:448], 'verify': groups[448:]})
    result = {}
    for split, triples in splits.items():
        windows = [[*start, keys[k], value, filler[i], filler[j], *suffix, value, eos]
                   for k, i, j in triples for value in digits]
        result[split] = torch.tensor(windows)
    return result


@torch.inference_mode()
def _probe(model, windows):
    model.eval()
    nll = hits = swapped_hits = 0
    count = len(windows)
    for rows in windows.split(32):
        inputs, targets = rows[:, :-1], rows[:, 1:]
        logits = model(inputs)
        nll += float(F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction='sum'))
        pred = logits[:, -2].argmax(-1)
        hits += int(pred.eq(rows[:, -2]).sum())
        # Each octet shares the entire context except the earlier stated value.
        # Rotation exchanges that earlier value for a different value, but keeps
        # each original target fixed. Predictions come from the same forward pass.
        shifted = pred.view(-1, 8).roll(1, 1).flatten()
        swapped_hits += int(shifted.eq(rows[:, -2]).sum())
    return {'windows': count, 'bpt': nll / (count * (windows.shape[1] - 1) * math.log(2)),
            'correct_generated_value': hits, 'value_accuracy': hits/count,
            'accuracy_after_changing_only_earlier_value': swapped_hits/count}


def cpu_positive_control(tokenizer, seeds=None, confirm=True):
    """Prerecorded narrow next-token success criterion. The production test bins are never used."""
    seeds = tuple(seeds) if seeds is not None else ((20261109, 20261110) if confirm else (20261105, 20261106))
    steps = 1536 if confirm else 256
    pools = _synthetic_windows(tokenizer, confirm=confirm)
    runs = []
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            for seed in seeds:
                torch.manual_seed(seed)
                model = DenseCausalLM(tokenizer.vocab_size(), width=24, layers=2, heads=3,
                                      context=32, hidden=48)
                rng = torch.Generator().manual_seed(seed + 1)
                opt = torch.optim.AdamW(model.parameters(), lr=0.0025, weight_decay=0.01)
                before = _probe(model, pools['verify'])
                model.train()
                for _ in range(steps):
                    rows = pools['train'][torch.randint(len(pools['train']), (32,), generator=rng)]
                    logits = model(rows[:, :-1])
                    loss = F.cross_entropy(logits.flatten(0, 1), rows[:, 1:].flatten())
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    opt.step()
                after = _probe(model, pools['verify'])
                runs.append({'seed': seed, 'parameters': model.parameter_count(), 'updates': steps,
                             'scored_training_tokens': steps*32*(pools['train'].shape[1]-1),
                             'before': before, 'after': after,
                             'passed': after['value_accuracy'] >= .90 and
                             after['accuracy_after_changing_only_earlier_value'] <= .20
                             and after['bpt'] < before['bpt']})
    finally:
        torch.set_num_threads(threads)
    return {'qualified': len(runs) >= 2 and all(r['passed'] for r in runs), 'runs': runs,
            'criteria': {'value_accuracy_min': .9, 'wrong_prefix_accuracy_max': .2,
                         'mean_next_token_bpt_must_fall': True, 'all_seeds_must_pass': True},
            'split': {'training_windows': len(pools['train']), 'verify_windows': len(pools['verify']),
                      'verify_sha256': hashlib.sha256(pools['verify'].numpy().tobytes()).hexdigest()},
            'tokenizer_sha256': hashlib.sha256(tokenizer.serialized_model_proto()).hexdigest(),
            'protocol': 'dense-real-token-causal-positive-control-confirm-v2' if confirm else 'dense-real-token-causal-positive-control-v1',
            'module_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope': 'Tiny dense architecture positive control on generated token windows; predicts all tokens. '
                     'Verification holds out context triplets; the last input marker is invariant, so producing '
                     'the correct digit requires an earlier token. No production code/general corpus or assistant '
                     'generalization claim is implied.'}


if __name__ == '__main__':
    import sentencepiece as spm
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tok = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    receipt = cpu_positive_control(tok)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False))
    print(json.dumps(receipt))
