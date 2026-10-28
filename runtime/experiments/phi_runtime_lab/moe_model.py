"""Sparse routed PhiState hybrid language model (training research prototype).

Uses the *existing* phistate.py GoldenState FFT/recurrent path and local causal
attention. Independent per-token routed SwiGLU experts; only top-k experts are
executed for each token. This does not yet implement native RFTMW MoE inference.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from phistate import PhiStateLM, PhiBlock, RMSNorm, SwiGLU, ModelConfig, count_parameters


@dataclass
class MoEConfig(ModelConfig):
    experts: int = 8
    top_k: int = 2
    expert_ratio: int = 2
    shared_ratio: int = 1
    router_aux_weight: float = 0.01
    router_z_weight: float = 0.001

    def validate(self) -> None:
        # Do NOT call ModelConfig.validate(): the original model requires
        # vocab=256, while MoE supports byte (256) OR SentencePiece BPE.
        if self.mode not in ("phi", "dft", "local"):
            raise ValueError("mode must be phi, dft, or local")
        if self.mixer_basis not in ("auto", "phi", "dft"):
            raise ValueError("mixer_basis must be auto, phi, or dft")
        if not (self.width > 0 and self.layers > 0 and self.heads > 0):
            raise ValueError("width, layers, heads must be positive")
        if self.width % self.heads or self.width % self.mix_group:
            raise ValueError("width must be divisible by heads and mix_group")
        if self.window < 1 or self.mlp_ratio < 1 or self.mix_group < 1:
            raise ValueError("window, mix-group and mlp-ratio must be positive")
        if not (256 <= self.vocab <= 65535):
            raise ValueError("vocab must be in [256,65535]")
        if self.experts < 2 or self.top_k < 1 or self.top_k > self.experts:
            raise ValueError("require experts>=2 and 1 <= top_k <= experts")
        if self.expert_ratio < 1 or self.shared_ratio < 0:
            raise ValueError("expert_ratio >=1 and shared_ratio >=0")
        if self.router_aux_weight < 0 or self.router_z_weight < 0:
            raise ValueError("router losses must be >=0")


class SparseMoE(nn.Module):
    """Top-k token routing, no capacity drops. No dense all-expert forward pass.

    Compute scales with selected experts, but a CPU/PyTorch token dispatcher
    and sparse per-expert minibatches may still be slower than dense FFNs.
    """
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.top_k = cfg.top_k
        self.expert_count = cfg.experts
        self.router = nn.Linear(cfg.width, cfg.experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.01)
        self.experts = nn.ModuleList([SwiGLU(cfg.width, cfg.expert_ratio) for _ in range(cfg.experts)])
        self.shared = SwiGLU(cfg.width, cfg.shared_ratio) if cfg.shared_ratio else None
        self.last_aux: Optional[torch.Tensor] = None
        self.last_z: Optional[torch.Tensor] = None
        self.register_buffer("last_routing", torch.zeros(cfg.experts), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, x.shape[-1])
        scores = self.router(flat).float()  # FP32 router, even under mixed precision
        probabilities = torch.softmax(scores, dim=-1)
        raw_top, routes = torch.topk(scores, self.top_k, dim=-1)
        # With top-1, normalize against all experts so router still receives
        # predictive gradients. With top-2, use normalized selected probabilities.
        if self.top_k == 1:
            gates = probabilities.gather(1, routes)
        else:
            gates = torch.softmax(raw_top, dim=-1)
        gates = gates.to(flat.dtype)

        # Differentiable importance; hard assignments counted without grad.
        with torch.no_grad():
            assignment_share = torch.bincount(routes.reshape(-1),
                                              minlength=self.expert_count).float()
            assignment_share /= max(1, routes.numel())
            self.last_routing.copy_(assignment_share)
        importance = probabilities.mean(dim=0)
        self.last_aux = self.expert_count * (importance * assignment_share).sum()
        self.last_z = torch.logsumexp(scores, dim=-1).square().mean()

        routed = torch.zeros_like(flat)
        # These loops only execute selected (token, expert) pairs.
        # No token is dropped due to arbitrary expert capacity limits.
        for index, expert in enumerate(self.experts):
            token_index, slot_index = torch.where(routes == index)
            if token_index.numel() == 0:
                continue
            selected = flat.index_select(0, token_index)
            response = expert(selected) * gates[token_index, slot_index, None]
            routed.index_add_(0, token_index, response)
        if self.shared is not None:
            routed = routed + self.shared(flat)
        return routed.reshape(original_shape)


class MoEPhiBlock(PhiBlock):
    def __init__(self, cfg: MoEConfig):
        super().__init__(cfg)
        self.ff = SparseMoE(cfg)


class PhiStateMoELM(PhiStateLM):
    def __init__(self, cfg: MoEConfig):
        # Explicitly initialize the architecture to avoid constructing an unused
        # dense FFN at 100M scale. Forward/step are inherited from PhiStateLM.
        nn.Module.__init__(self)
        cfg.validate()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab, cfg.width)
        self.blocks = nn.ModuleList([MoEPhiBlock(cfg) for _ in range(cfg.layers)])
        self.norm = RMSNorm(cfg.width)
        nn.init.normal_(self.embedding.weight, std=0.05)

    def router_penalty(self) -> tuple[torch.Tensor, dict]:
        aux = torch.stack([b.ff.last_aux for b in self.blocks if b.ff.last_aux is not None])
        z = torch.stack([b.ff.last_z for b in self.blocks if b.ff.last_z is not None])
        device = self.embedding.weight.device
        aux_mean = aux.mean() if aux.numel() else torch.zeros((), device=device)
        z_mean = z.mean() if z.numel() else torch.zeros((), device=device)
        weighted = self.cfg.router_aux_weight * aux_mean + self.cfg.router_z_weight * z_mean
        counts = torch.stack([b.ff.last_routing for b in self.blocks]).mean(0)
        report = {
            "router_load_loss": float(aux_mean.detach().item()),
            "router_z_loss": float(z_mean.detach().item()),
            "expert_selection": [round(float(v), 4) for v in counts.detach().tolist()],
        }
        return weighted, report

    def estimate_active_parameters(self) -> int:
        total = count_parameters(self)
        extra = 0
        for b in self.blocks:
            moe = b.ff
            # Each token only runs top_k experts, though all expert weights
            # still occupy RAM and optimizer storage.
            per_expert = count_parameters(moe.experts[0])
            extra += (moe.expert_count - moe.top_k) * per_expert
        return total - extra


PROFILES = {
    "mini": {"width": 32, "layers": 2, "heads": 4, "window": 8, "mix_group": 16,
             "experts": 4, "top_k": 2, "expert_ratio": 2, "shared_ratio": 1},
    "35m": {"width": 320, "layers": 6, "heads": 8, "window": 32, "mix_group": 16,
             "experts": 8, "top_k": 2, "expert_ratio": 2, "shared_ratio": 1},
    "100m": {"width": 480, "layers": 8, "heads": 8, "window": 32, "mix_group": 16,
              "experts": 8, "top_k": 2, "expert_ratio": 2, "shared_ratio": 1},
}


def config_for(profile: str, vocab: int, mode: str = "dft", **overrides) -> MoEConfig:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    params = {**PROFILES[profile], "mode": mode, "mixer_basis": "dft" if mode == "dft" else "phi", "vocab": vocab}
    params.update({k: v for k, v in overrides.items() if v is not None})
    return MoEConfig(**params)
