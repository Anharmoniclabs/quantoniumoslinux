"""
AetherCanvas — backward-compat shim. Use src.resonart instead.
"""

from .engine import ResonartEngine, AetherCanvasEngine, GenerateRequest, GenerateResult, NanobananaEngine

__all__ = ["ResonartEngine", "AetherCanvasEngine", "GenerateRequest", "GenerateResult", "NanobananaEngine"]