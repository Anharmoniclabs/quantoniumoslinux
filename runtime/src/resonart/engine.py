"""Compatibility wrapper exposing the engine under the Resonart name."""

from src.aethercanvas.engine import AetherCanvasEngine, GenerateRequest, GenerateResult

ResonartEngine = AetherCanvasEngine
NanobananaEngine = AetherCanvasEngine

__all__ = [
    "ResonartEngine",
    "GenerateRequest",
    "GenerateResult",
    "AetherCanvasEngine",
    "NanobananaEngine",
]
