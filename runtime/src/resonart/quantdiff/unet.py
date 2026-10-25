"""
QuantDiff Lightweight U-Net.

A custom ~12M-parameter pixel-space denoising U-Net built entirely in
PyTorch with no dependency on HuggingFace diffusers or any external
model library.

Architecture
------------
  in_ch=3  →  InitConv(64)
  Down0: ResBlock(64→64)  + CondBlock  →  stride-2 downsample
  Down1: ResBlock(64→128) + CondBlock  →  stride-2 downsample
  Down2: ResBlock(128→256)+ CondBlock
  Mid:   ResBlock(256→256) → SelfAttention → ResBlock(256→256)
  Up2:   ResBlock(512→128) → bilinear ×2
  Up1:   ResBlock(256→64)  → bilinear ×2
  Up0:   ResBlock(128→64)
  Out:   GroupNorm → SiLU → Conv(64→3)

Time conditioning : sinusoidal embedding → 2-layer MLP → injected via
                    additive shift in each ResBlock.
Prompt conditioning: learned linear projection of a vocab-embedding
                     vector → scale+shift (adaGN) in each ResBlock.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
# ---------------------------------------------------------------------------

def sinusoidal_embedding(timesteps: "torch.Tensor", dim: int) -> "torch.Tensor":
    """Map integer timesteps → sinusoidal embedding of size *dim*."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000) * torch.arange(half, device=timesteps.device).float() / (half - 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class ResBlock(nn.Module):
        """Residual block with time-embedding injection."""

        def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.1):
            super().__init__()
            groups_in = min(32, in_ch)
            groups_out = min(32, out_ch)
            self.norm1 = nn.GroupNorm(groups_in, in_ch)
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.time_proj = nn.Linear(time_dim, out_ch)
            self.norm2 = nn.GroupNorm(groups_out, out_ch)
            self.dropout = nn.Dropout(dropout)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
            self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
            self.act = nn.SiLU()

        def forward(
            self,
            x: "torch.Tensor",
            t_emb: "torch.Tensor",
        ) -> "torch.Tensor":
            h = self.act(self.norm1(x))
            h = self.conv1(h)
            h = h + self.time_proj(self.act(t_emb))[:, :, None, None]
            h = self.act(self.norm2(h))
            h = self.dropout(h)
            h = self.conv2(h)
            return h + self.skip(x)

    class SelfAttention(nn.Module):
        """Multi-head self-attention over spatial feature maps."""

        def __init__(self, channels: int, heads: int = 4):
            super().__init__()
            self.heads = heads
            head_dim = channels // heads
            self.scale = head_dim**-0.5
            self.norm = nn.GroupNorm(min(32, channels), channels)
            self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
            self.proj = nn.Conv2d(channels, channels, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            B, C, H, W = x.shape
            h = self.norm(x)
            qkv = self.qkv(h)
            q, k, v = qkv.chunk(3, dim=1)
            # Reshape to (B, heads, head_dim, HW)
            hd = C // self.heads
            q = q.reshape(B, self.heads, hd, H * W)
            k = k.reshape(B, self.heads, hd, H * W)
            v = v.reshape(B, self.heads, hd, H * W)
            attn = torch.einsum("bhdn,bhdm->bhnm", q, k) * self.scale
            attn = attn.softmax(dim=-1)
            out = torch.einsum("bhnm,bhdm->bhdn", attn, v).reshape(B, C, H, W)
            return x + self.proj(out)

    class CondBlock(nn.Module):
        """Conditioning injection: adaGN scale+shift from cond vector."""

        def __init__(self, channels: int, cond_dim: int):
            super().__init__()
            self.norm = nn.GroupNorm(min(32, channels), channels)
            self.proj = nn.Linear(cond_dim, channels * 2)
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

        def forward(
            self, x: "torch.Tensor", cond: "torch.Tensor"
        ) -> "torch.Tensor":
            h = self.norm(x)
            params = self.proj(cond)
            scale, shift = params.chunk(2, dim=-1)
            return h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]

    # -----------------------------------------------------------------------
    # Main U-Net
    # -----------------------------------------------------------------------

    class QuantDiffUNet(nn.Module):
        """
        QuantoniumOS custom pixel-space denoising U-Net (~12M parameters).

        Parameters
        ----------
        in_channels   : image channels (3 for RGB)
        base_channels : channel width at the widest (encoder level 0)
        channel_mults : tuple of multipliers for each encoder level
        time_dim      : sinusoidal + MLP time-embedding dimension
        cond_dim      : conditioning vector dimension
        dropout       : dropout rate inside ResBlocks
        """

        def __init__(
            self,
            in_channels: int = 3,
            base_channels: int = 64,
            channel_mults: Tuple[int, ...] = (1, 2, 4),
            time_dim: int = 256,
            cond_dim: int = 128,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.time_dim = time_dim
            self.cond_dim = cond_dim

            ch = [base_channels * m for m in channel_mults]  # [64, 128, 256]

            # Time MLP
            self.time_mlp = nn.Sequential(
                nn.Linear(time_dim, time_dim * 4),
                nn.SiLU(),
                nn.Linear(time_dim * 4, time_dim),
            )

            # Initial projection
            self.init_conv = nn.Conv2d(in_channels, ch[0], 3, padding=1)

            # ---- Encoder ----
            self.enc0 = ResBlock(ch[0], ch[0], time_dim, dropout)
            self.cond0 = CondBlock(ch[0], cond_dim)
            self.down0 = nn.Conv2d(ch[0], ch[0], 4, stride=2, padding=1)

            self.enc1 = ResBlock(ch[0], ch[1], time_dim, dropout)
            self.cond1 = CondBlock(ch[1], cond_dim)
            self.down1 = nn.Conv2d(ch[1], ch[1], 4, stride=2, padding=1)

            self.enc2 = ResBlock(ch[1], ch[2], time_dim, dropout)
            self.cond2 = CondBlock(ch[2], cond_dim)

            # ---- Bottleneck ----
            self.mid1 = ResBlock(ch[2], ch[2], time_dim, dropout)
            self.mid_attn = SelfAttention(ch[2])
            self.mid2 = ResBlock(ch[2], ch[2], time_dim, dropout)

            # ---- Decoder ----
            self.dec2 = ResBlock(ch[2] + ch[2], ch[1], time_dim, dropout)
            self.up2 = nn.ConvTranspose2d(ch[1], ch[1], 4, stride=2, padding=1)

            self.dec1 = ResBlock(ch[1] + ch[1], ch[0], time_dim, dropout)
            self.up1 = nn.ConvTranspose2d(ch[0], ch[0], 4, stride=2, padding=1)

            self.dec0 = ResBlock(ch[0] + ch[0], ch[0], time_dim, dropout)

            # Output head
            self.out_norm = nn.GroupNorm(min(32, ch[0]), ch[0])
            self.out_conv = nn.Conv2d(ch[0], in_channels, 3, padding=1)

        def forward(
            self,
            x: "torch.Tensor",
            timesteps: "torch.Tensor",
            cond: Optional["torch.Tensor"] = None,
        ) -> "torch.Tensor":
            """
            x         : (B, 3, H, W) noisy image in [-1, 1]
            timesteps : (B,) integer training timesteps
            cond      : (B, cond_dim) conditioning vector (zeros if None)
            Returns   : (B, 3, H, W) predicted noise εθ
            """
            B = x.shape[0]
            if cond is None:
                cond = torch.zeros(B, self.cond_dim, device=x.device, dtype=x.dtype)

            # Time embedding
            t_emb = sinusoidal_embedding(timesteps, self.time_dim)
            t_emb = self.time_mlp(t_emb)

            # Encoder
            h0 = self.init_conv(x)
            h0 = self.enc0(h0, t_emb)
            h0 = self.cond0(h0, cond)
            h0_ds = self.down0(h0)

            h1 = self.enc1(h0_ds, t_emb)
            h1 = self.cond1(h1, cond)
            h1_ds = self.down1(h1)

            h2 = self.enc2(h1_ds, t_emb)
            h2 = self.cond2(h2, cond)

            # Bottleneck
            h = self.mid1(h2, t_emb)
            h = self.mid_attn(h)
            h = self.mid2(h, t_emb)

            # Decoder
            h = self.dec2(torch.cat([h, h2], dim=1), t_emb)
            h = self.up2(h)

            h = self.dec1(torch.cat([h, h1], dim=1), t_emb)
            h = self.up1(h)

            h = self.dec0(torch.cat([h, h0], dim=1), t_emb)

            # Output
            h = F.silu(self.out_norm(h))
            return self.out_conv(h)

        def param_count(self) -> int:
            return sum(p.numel() for p in self.parameters())

else:
    # Provide a stub so imports succeed even without torch
    class QuantDiffUNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch is required to instantiate QuantDiffUNet")

        def param_count(self) -> int:
            return 0
