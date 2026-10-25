"""
QuantDiff — QuantoniumOS proprietary diffusion inference engine.

A fully custom pixel-space diffusion pipeline:
  - DDIM noise scheduler (no HuggingFace diffusers dependency)
  - Lightweight U-Net with self-attention and conditioning
  - Warm-start from Resonart engine when model weights are absent
  - 4K LANCZOS upscaling post-step
  - Training script to distill from Resonart output
"""

from .engine import QuantDiffEngine
from .scheduler import DDIMScheduler

__all__ = ["QuantDiffEngine", "DDIMScheduler"]
