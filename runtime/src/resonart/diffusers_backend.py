from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import Optional

from src.aethercanvas.engine import GenerateRequest, GenerateResult

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", message=r".*Siglip2ImageProcessorFast.*")
warnings.filterwarnings("ignore", message=r".*local_dir_use_symlinks.*")
warnings.filterwarnings("ignore", message=r".*enable_vae_slicing.*")

try:
    import torch
except Exception as exc:  # pragma: no cover - import error path is environment-specific
    torch = None
    _TORCH_ERROR = exc
else:
    _TORCH_ERROR = None

try:
    from diffusers import AutoPipelineForText2Image
except Exception as exc:  # pragma: no cover - import error path is environment-specific
    AutoPipelineForText2Image = None
    _DIFFUSERS_ERROR = exc
else:
    _DIFFUSERS_ERROR = None


_PIPELINE = None
_PIPELINE_KEY: tuple[str, str] | None = None


def _pick_device() -> str:
    if torch is None:
        raise RuntimeError(f"torch unavailable: {_TORCH_ERROR}")
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pick_model(device: str) -> str:
    explicit = os.getenv("QUANTONIUM_DIFFUSERS_MODEL")
    if explicit:
        return explicit
    if device == "cuda":
        return os.getenv("QUANTONIUM_DIFFUSERS_MODEL_GPU", "stabilityai/sd-turbo")
    return os.getenv("QUANTONIUM_DIFFUSERS_MODEL_CPU", "runwayml/stable-diffusion-v1-5")


def _torch_dtype_for(device: str):
    if torch is None:
        raise RuntimeError(f"torch unavailable: {_TORCH_ERROR}")
    if device == "cuda":
        return torch.float16
    return torch.float32


def _load_pipeline(model_id: str, device: str):
    global _PIPELINE, _PIPELINE_KEY

    if AutoPipelineForText2Image is None:
        raise RuntimeError(f"diffusers unavailable: {_DIFFUSERS_ERROR}")

    key = (model_id, device)
    if _PIPELINE is not None and _PIPELINE_KEY == key:
        return _PIPELINE

    local_only = os.getenv("QUANTONIUM_DIFFUSERS_LOCAL_ONLY", "0") == "1"
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=_torch_dtype_for(device),
        local_files_only=local_only,
    )

    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(device)

    for fn_name in ("enable_attention_slicing", "enable_vae_slicing"):
        fn = getattr(pipe, fn_name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass

    _PIPELINE = pipe
    _PIPELINE_KEY = key
    return pipe


def _round_dimension(value: int) -> int:
    """Clamp generation dimension — SD 1.5 runs best at 512; upscale after."""
    clamped = max(256, min(int(value), 768))
    rounded = max(256, (clamped // 64) * 64)
    return rounded


def _target_dimension(value: int) -> int:
    """Final output dimension after upscaling — up to 4K."""
    return max(256, min(int(value), 4096))


def _upscale_image(image, target_w: int, target_h: int):
    """High-quality LANCZOS upscale to target resolution."""
    from PIL import Image as _PilImage
    if image.width == target_w and image.height == target_h:
        return image
    return image.resize((target_w, target_h), _PilImage.LANCZOS)


def _steps_for(req: GenerateRequest, model_id: str, device: str) -> int:
    requested = max(4, min(int(req.steps), 40))
    turbo = any(tag in model_id.lower() for tag in ("turbo", "lightning"))
    if turbo:
        return min(requested, 8 if device == "cuda" else 6)
    if device == "cpu":
        return min(requested, 20)
    return requested


def _guidance_for(model_id: str) -> float:
    lowered = model_id.lower()
    if "turbo" in lowered or "lightning" in lowered:
        return 0.0
    return 7.0


def _style_suffix(style: str) -> str:
    return {
        "photo": "photorealistic, 8K UHD, ultra detailed, cinematic lighting, natural textures, realistic depth of field, masterpiece, sharp focus",
        "anime": "anime illustration, clean line art, vivid color, polished shading",
        "oil_paint": "oil painting, rich brushwork, painterly texture",
        "neon": "cinematic neon lighting, glowing reflections, high contrast",
        "abstract": "abstract composition, bold forms, expressive color",
        "sketch": "illustrated sketch, pencil lines, monochrome shading",
        "pixel": "pixel art, sprite-like shapes, crisp low resolution aesthetic",
    }.get(style, "high detail image")


def _palette_suffix(palette_hint: str) -> str:
    return {
        "forest": "lush green palette, natural foliage tones",
        "ocean": "cool blue palette, oceanic tones",
        "desert": "warm sandy palette, golden desert tones",
        "snow": "cold bright palette, snow and ice tones",
        "volcano": "molten orange and charcoal palette",
        "neon": "neon magenta cyan palette",
        "cyberpunk": "cyberpunk palette, teal and magenta light",
        "city": "urban palette, asphalt, steel, window glow",
        "space": "cosmic dark palette, star-lit highlights",
        "sunset": "sunset palette, orange pink sky",
        "racing": "performance car palette, vivid reds and metallic highlights",
        "dog": "natural dog fur tones, outdoor pet photography colors",
        "golden": "golden retriever fur tones, warm sunlight",
        "cat": "soft feline fur tones, gentle contrast",
    }.get(palette_hint, "")


def _compose_prompt(req: GenerateRequest) -> str:
    parts = [req.prompt.strip()]
    style_suffix = _style_suffix(req.style)
    if style_suffix:
        parts.append(style_suffix)
    if req.scene_hint:
        parts.append(f"{req.scene_hint.replace('_', ' ')} composition")
    palette_suffix = _palette_suffix(req.palette_hint)
    if palette_suffix:
        parts.append(palette_suffix)
    if req.quality == "ultra":
        parts.append("highly detailed, refined textures")
    return ", ".join(part for part in parts if part)


def _seed_generator(seed: Optional[int], device: str):
    if torch is None or seed is None:
        return None
    gen_device = "cpu" if device == "mps" else device
    return torch.Generator(device=gen_device).manual_seed(int(seed))


class DiffusersEngine:
    def __init__(self, output_dir: str = "generated/images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, req: GenerateRequest) -> GenerateResult:
        if torch is None:
            return GenerateResult(success=False, error=f"torch unavailable: {_TORCH_ERROR}")
        if AutoPipelineForText2Image is None:
            return GenerateResult(success=False, error=f"diffusers unavailable: {_DIFFUSERS_ERROR}")

        try:
            device = _pick_device()
            model_id = _pick_model(device)
            pipe = _load_pipeline(model_id, device)

            width = _round_dimension(req.width)
            height = _round_dimension(req.height)
            steps = _steps_for(req, model_id, device)
            guidance = _guidance_for(model_id)
            prompt = _compose_prompt(req)
            generator = _seed_generator(req.seed, device)

            kwargs = {
                "prompt": prompt,
                "negative_prompt": req.negative_prompt or None,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "width": width,
                "height": height,
            }
            if generator is not None:
                kwargs["generator"] = generator

            image = pipe(**kwargs).images[0]

            # Upscale to requested resolution (up to 4K) using LANCZOS
            target_w = _target_dimension(req.width)
            target_h = _target_dimension(req.height)
            if target_w != width or target_h != height:
                image = _upscale_image(image, target_w, target_h)

            slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in req.prompt).strip("_")[:40] or "image"
            filename = f"{int(time.time())}_{slug}.png"
            out_path = self.output_dir / filename
            image.save(out_path, optimize=False, compress_level=1)

            return GenerateResult(
                success=True,
                path=str(out_path),
                prompt=req.prompt,
                style=req.style,
                seed=int(req.seed or 0),
                width=target_w,
                height=target_h,
            )
        except Exception as exc:
            return GenerateResult(success=False, error=str(exc))
