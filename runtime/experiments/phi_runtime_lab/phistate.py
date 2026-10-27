"""PhiState: CPU-friendly experimental local-attention + diagonal SSM language model.

Research prototype, not a pretrained foundation model. Phase-only FFT is *not*
claimed to be the canonical RFT. The canonical phi-grid transform is used
specifically in ChannelMixer; recurrent frequencies are a separate hypothesis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

PHI = (1.0 + math.sqrt(5.0)) / 2.0


@lru_cache(maxsize=16)
def spectral_unitary(n: int, kind: str) -> torch.Tensor:
    """Return n-by-n unitary U, with samples indexing rows and modes columns.

    Phi: Phi_ik = exp(2pi*i*i*frac((k+1)*phi))/sqrt(n)
         U = Phi (Phi^H Phi)^(-1/2)
    DFT: U_ik = exp(2pi*i*i*k/n)/sqrt(n)
    """
    t = torch.arange(n, dtype=torch.float64)
    if kind == "phi":
        freqs = torch.remainder((t + 1) * PHI, 1.0)
    elif kind == "dft":
        freqs = t / n
    else:
        raise ValueError(f"Unknown unitary basis: {kind}")
    raw = torch.exp(2j * math.pi * t[:, None] * freqs[None, :]) / math.sqrt(n)
    if kind == "phi":
        gram = raw.mH @ raw
        lam, vec = torch.linalg.eigh(gram)
        if lam.min().item() <= 1e-10:
            raise ArithmeticError(f"Phi Gram is ill-conditioned at n={n}: {lam.min().item()}")
        invroot = (vec * torch.rsqrt(lam).unsqueeze(0)) @ vec.mH
        raw = raw @ invroot
    return raw.to(dtype=torch.complex64)


@dataclass
class ModelConfig:
    mode: str = "phi"  # phi | dft | local (recurrent memory frequency family)
    mixer_basis: str = "auto"  # auto (= mode) | phi | dft, for 2x2 factorial ablations
    width: int = 96
    layers: int = 2
    heads: int = 4
    window: int = 16
    mix_group: int = 16
    mlp_ratio: int = 2
    vocab: int = 256

    def validate(self) -> None:
        if self.mode not in ("phi", "dft", "local"):
            raise ValueError("mode must be phi, dft, or local")
        if self.mixer_basis not in ("auto", "phi", "dft"):
            raise ValueError("mixer_basis must be auto, phi, or dft")
        if self.width <= 0 or self.layers <= 0 or self.heads <= 0:
            raise ValueError("width, layers, heads must be positive")
        if self.width % self.heads or self.width % self.mix_group:
            raise ValueError("width must be divisible by heads and mix_group")
        if self.window < 1 or self.mlp_ratio < 1:
            raise ValueError("window and mlp_ratio must be positive")
        if self.vocab != 256:
            raise ValueError("this prototype uses UTF-8 bytes, vocab=256")


class RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x.square(), dim=-1, keepdim=True) + 1e-5) * self.weight


class LocalAttention(nn.Module):
    """True O(T * window * width) causal windowed attention, without a T*T mask."""
    def __init__(self, width: int, heads: int, window: int) -> None:
        super().__init__()
        self.heads, self.head_dim, self.window = heads, width // heads, window
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.bias = nn.Parameter(torch.zeros(heads, window))

    def _qkv(self, x: torch.Tensor):
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def shape(y):
            return y.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        return shape(q), shape(k), shape(v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self._qkv(x)
        # F.pad works on the temporal axis for [B,H,T,Dh].
        k_win = F.pad(k, (0, 0, self.window-1, 0)).unfold(2, self.window, 1)
        v_win = F.pad(v, (0, 0, self.window-1, 0)).unfold(2, self.window, 1)
        # unfold returns [B,H,T,Dh,W], swap the last two dimensions.
        k_win, v_win = k_win.transpose(-1, -2), v_win.transpose(-1, -2)
        score = (q.unsqueeze(-2) * k_win).sum(-1) / math.sqrt(self.head_dim)
        score = score + self.bias.view(1, self.heads, 1, self.window)
        now = torch.arange(t, device=x.device)[:, None]
        shift = torch.arange(self.window, device=x.device)[None, :] - (self.window - 1)
        valid = now + shift >= 0
        score = score.masked_fill(~valid.view(1, 1, t, self.window), -1e4)
        probs = torch.softmax(score.float(), dim=-1).to(dtype=x.dtype)
        out = (probs.unsqueeze(-1) * v_win).sum(-2)
        return self.proj(out.transpose(1, 2).reshape(b, t, d))

    def step(self, x: torch.Tensor, state: dict[str, torch.Tensor] | None):
        # x: [B,1,D] ; cache keys/values store the last <= window positions.
        b, _, d = x.shape
        q, k, v = self._qkv(x)
        if state is not None:
            k = torch.cat((state["k"], k), dim=2)
            v = torch.cat((state["v"], v), dim=2)
        k, v = k[:, :, -self.window:, :], v[:, :, -self.window:, :]
        score = (q * k).sum(-1) / math.sqrt(self.head_dim)
        score = score + self.bias[:, -k.shape[2]:].unsqueeze(0)
        prob = torch.softmax(score.float(), dim=-1).to(dtype=x.dtype)
        out = (prob.unsqueeze(-1) * v).sum(2).reshape(b, 1, d)
        return self.proj(out), {"k": k, "v": v}


class GoldenState(nn.Module):
    """Diagonal complex recurrent memory with FFT convolution training.

    h[t,c] = a[c] * h[t-1,c] + u[t,c]
    y[t,c] = real(readout[c] * h[t,c])
    a[c] = exp(-softplus(decay[c]) + 1j * (theta[c] + bounded_offset[c]))

    This is inspired by S4D; the golden-frequency initialization is experimental.
    It is NOT an exact canonical RFT transform.
    """
    def __init__(self, width: int, mode: str) -> None:
        super().__init__()
        if mode not in ("phi", "dft"):
            raise ValueError("GoldenState mode must be phi or dft")
        k = torch.arange(width, dtype=torch.float64)
        frac = torch.remainder((k + 1) * PHI, 1.0) if mode == "phi" else k / width
        self.register_buffer("theta", (2 * math.pi * frac).float(), persistent=False)
        init_decay = torch.linspace(0.02, 0.25, width)
        self.decay_raw = nn.Parameter(torch.log(torch.expm1(init_decay)))
        self.phase_delta = nn.Parameter(torch.zeros(width))
        self.read_re = nn.Parameter(torch.full((width,), 0.15))
        self.read_im = nn.Parameter(torch.zeros(width))
        self.input_proj = nn.Linear(width, width, bias=False)
        self.input_gate = nn.Linear(width, width, bias=False)
        self.output_proj = nn.Linear(width, width, bias=False)

    def parameters_for_kernel(self):
        decay = F.softplus(self.decay_raw) + 1e-4
        theta = self.theta + 0.10 * torch.tanh(self.phase_delta)
        a = torch.exp(torch.complex(-decay, theta))
        c = torch.complex(self.read_re, self.read_im)
        return a, c

    def _input(self, x: torch.Tensor) -> torch.Tensor:
        return self.input_proj(x) * torch.sigmoid(self.input_gate(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        u = self._input(x)  # [B,T,D]
        a, c = self.parameters_for_kernel()
        powers = torch.arange(t, device=x.device, dtype=a.real.dtype)[:, None]
        kernel = ((a[None, :] ** powers) * c[None, :]).real  # [T,D]
        fft_len = 1 << (2*t - 1).bit_length()
        yf = (torch.fft.rfft(u.transpose(1, 2), n=fft_len, dim=-1) *
              torch.fft.rfft(kernel.transpose(0, 1), n=fft_len, dim=-1).unsqueeze(0))
        y = torch.fft.irfft(yf, n=fft_len, dim=-1)[:, :, :t].transpose(1, 2)
        return self.output_proj(y)

    def step(self, x: torch.Tensor, prev: torch.Tensor | None):
        u = self._input(x).squeeze(1)
        a, c = self.parameters_for_kernel()
        if prev is None:
            prev = torch.zeros_like(u, dtype=a.dtype)
        state = a.unsqueeze(0) * prev + u.to(dtype=a.dtype)
        y = (state * c.unsqueeze(0)).real.unsqueeze(1)
        return self.output_proj(y), state


class CanonicalChannelMixer(nn.Module):
    """Blockwise canonical phi-RFT or DFT diagonal spectral gain, real projection.

    Coeffs = U^H x;  Y = Re[U diag(gain) U^H x].
    The gains initialize to 1, so the layer initializes as the identity.
    """
    def __init__(self, group: int, kind: str) -> None:
        super().__init__()
        self.group = group
        U = spectral_unitary(group, kind)
        self.register_buffer("basis", U, persistent=True)
        self.gain_raw = nn.Parameter(torch.zeros(group))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gain = 1.0 + 0.5 * torch.tanh(self.gain_raw)
        U = self.basis
        M = ((U.conj() * gain.unsqueeze(0)) @ U.T).real
        return torch.matmul(x.reshape(*x.shape[:-1], -1, self.group), M).reshape_as(x)


class SwiGLU(nn.Module):
    def __init__(self, width: int, ratio: int) -> None:
        super().__init__()
        mid = width * ratio
        self.fc = nn.Linear(width, 2 * mid, bias=False)
        self.out = nn.Linear(mid, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc(x).chunk(2, dim=-1)
        return self.out(a * F.silu(b))


class PhiBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.width
        self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
        self.attn = LocalAttention(d, cfg.heads, cfg.window)
        self.memory = GoldenState(d, cfg.mode) if cfg.mode != "local" else None
        self.ff = SwiGLU(d, cfg.mlp_ratio)
        self.norm3 = RMSNorm(d) if cfg.mode != "local" else None
        mix_basis = cfg.mode if cfg.mixer_basis == "auto" else cfg.mixer_basis
        self.mixer = CanonicalChannelMixer(cfg.mix_group, mix_basis) if cfg.mode != "local" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm1(x)
        path = self.attn(z)
        if self.memory is not None:
            path = (path + self.memory(z)) * (1 / math.sqrt(2.0))
        x = x + 0.7 * path
        x = x + 0.5 * self.ff(self.norm2(x))
        if self.mixer is not None:
            z = self.norm3(x)
            x = x + 0.5 * (self.mixer(z) - z)
        return x

    def step(self, x: torch.Tensor, state: dict[str, Any] | None):
        z = self.norm1(x)
        at, att_state = self.attn.step(z, None if state is None else state["attention"])
        mem_state = None
        if self.memory is not None:
            mt, mem_state = self.memory.step(z, None if state is None else state["memory"])
            at = (at + mt) * (1 / math.sqrt(2.0))
        x = x + 0.7 * at
        x = x + 0.5 * self.ff(self.norm2(x))
        if self.mixer is not None:
            z = self.norm3(x)
            x = x + 0.5 * (self.mixer(z) - z)
        return x, {"attention": att_state, "memory": mem_state}


class PhiStateLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab, cfg.width)
        self.blocks = nn.ModuleList([PhiBlock(cfg) for _ in range(cfg.layers)])
        self.norm = RMSNorm(cfg.width)
        # Tie input/output weights to save parameters and training memory.
        nn.init.normal_(self.embedding.weight, std=0.05)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        return F.linear(self.norm(x), self.embedding.weight)

    def step(self, token: torch.Tensor, cache: list | None = None):
        """One-token inference; token shape [batch,1]."""
        if token.ndim != 2 or token.shape[1] != 1:
            raise ValueError("step input must be [batch,1]")
        if cache is None:
            cache = [None] * len(self.blocks)
        x = self.embedding(token)
        updated = []
        for block, state in zip(self.blocks, cache):
            x, state = block.step(x, state)
            updated.append(state)
        return F.linear(self.norm(x), self.embedding.weight), updated

    @staticmethod
    def from_checkpoint_config(raw: dict) -> "PhiStateLM":
        return PhiStateLM(ModelConfig(**raw))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
