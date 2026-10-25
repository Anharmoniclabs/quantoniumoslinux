"""
QuantDiff Engine — QuantoniumOS proprietary diffusion inference.

Two operating modes
-------------------
1. **Warm-start (no weights)**: Resonart generates a structured base
   image; our DDIM loop lightly refines it using random-initialized U-Net
   weights at very low noise strength (~5%).  Output is always structured
   and identical to Resonart until the model is trained.

2. **Full DDIM (trained weights present)**: starts from pure Gaussian noise
   and denoises completely with the trained QuantDiffUNet.  Produces novel
   photorealistic output matching the learned distribution.

Model weights are loaded from ``models/quantdiff/unet.pt`` by default
(override via env var ``QUANTDIFF_MODEL_PATH``).

4K output is achieved by generating at 64×64 → LANCZOS upscale to the
requested resolution (up to 4096 px).
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.aethercanvas.engine import GenerateRequest, GenerateResult, ResonartEngine
from .scheduler import DDIMScheduler
from .unet import QuantDiffUNet, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NATIVE_SIZE = 64          # U-Net native spatial resolution (pixels)
BASE_CHANNELS = 64
TIME_DIM = 256
COND_DIM = 128

# Conditioning vocabulary (must stay in sync with training)
_STYLE_VOCAB = ["photo", "anime", "oil_paint", "neon", "abstract", "sketch", "pixel"]
_SCENE_VOCAB = [
    "sportscar", "dog", "cat", "tree", "landscape",
    "portrait", "city", "space", "building", "vehicle",
]
_PALETTE_VOCAB = [
    "forest", "ocean", "desert", "snow", "volcano", "neon", "cyberpunk",
    "city", "space", "sunset", "racing", "dog", "golden", "cat", "default",
]
_VOCAB = _STYLE_VOCAB + _SCENE_VOCAB + _PALETTE_VOCAB

# Warm-start noise strength when no trained weights exist.
# At 0.05 the DDIM changes only ~5% of the AetherCanvas image.
_DEFAULT_WARMSTART_STRENGTH = float(
    os.getenv("QUANTDIFF_WARMSTART_STRENGTH", "0.05")
)
# Full DDIM strength when trained weights are available.
_DEFAULT_FULL_STRENGTH = float(
    os.getenv("QUANTDIFF_STRENGTH", "1.0")
)

# ---------------------------------------------------------------------------
# Prompt conditioning encoder
# ---------------------------------------------------------------------------

def _encode_prompt(req: GenerateRequest) -> "torch.Tensor":
    """
    Map a GenerateRequest → (1, COND_DIM) conditioning tensor.

    Uses vocab one-hot + word-overlap scoring; no CLIP or external encoder
    required.  The same encoding is used during training so inference is
    always consistent with training-time conditioning.
    """
    vec = np.zeros(COND_DIM, dtype=np.float32)
    prompt_lower = req.prompt.lower()
    words = set(re.findall(r"\w+", prompt_lower))

    # Style one-hot
    if req.style in _STYLE_VOCAB:
        vec[_STYLE_VOCAB.index(req.style)] = 1.0

    # Scene one-hot
    base = len(_STYLE_VOCAB)
    if req.scene_hint in _SCENE_VOCAB:
        vec[base + _SCENE_VOCAB.index(req.scene_hint)] = 1.0

    # Palette one-hot
    base2 = base + len(_SCENE_VOCAB)
    if req.palette_hint in _PALETTE_VOCAB:
        vec[base2 + _PALETTE_VOCAB.index(req.palette_hint)] = 1.0

    # Soft word-overlap for remaining dimensions
    offset = base2 + len(_PALETTE_VOCAB)
    remaining = COND_DIM - offset
    for i, term in enumerate(_VOCAB[:remaining]):
        if any(term in w or w in term for w in words):
            vec[offset + i] = 0.5

    return torch.from_numpy(vec).unsqueeze(0)  # (1, COND_DIM)


# ---------------------------------------------------------------------------
# PIL helpers
# ---------------------------------------------------------------------------

def _img_to_tensor(img) -> "torch.Tensor":
    """PIL Image (RGB) → (1, 3, H, W) float32 in [-1, 1]."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


def _tensor_to_img(t: "torch.Tensor"):
    """(1, 3, H, W) or (3, H, W) float32 [-1,1] → PIL Image (RGB, uint8)."""
    from PIL import Image as _PIL
    arr = t.squeeze(0).detach().float().numpy()          # (3, H, W)
    arr = ((arr.transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return _PIL.fromarray(arr, "RGB")


def _upscale(img, w: int, h: int):
    """LANCZOS upscale to (w, h)."""
    from PIL import Image as _PIL
    if img.width == w and img.height == h:
        return img
    return img.resize((w, h), _PIL.LANCZOS)


# ---------------------------------------------------------------------------
# QuantDiffEngine
# ---------------------------------------------------------------------------

class QuantDiffEngine:
    """
    QuantoniumOS proprietary diffusion inference engine.

    Implements DDIM sampling with our own scheduler and U-Net — no
    HuggingFace diffusers dependency at inference time.

    Parameters
    ----------
    output_dir  : directory where generated PNGs are saved
    model_path  : path to UNet state-dict (.pt).  If None, resolved from
                  env var QUANTDIFF_MODEL_PATH or models/quantdiff/unet.pt
    native_size : spatial resolution of the U-Net (default 64 px)
    """

    def __init__(
        self,
        output_dir: str = "generated/images",
        model_path: Optional[str] = None,
        native_size: int = NATIVE_SIZE,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.native_size = native_size

        # Resolve model path
        _env_path = os.getenv("QUANTDIFF_MODEL_PATH")
        if model_path:
            self._model_path: Optional[Path] = Path(model_path)
        elif _env_path:
            self._model_path = Path(_env_path)
        else:
            # Default location relative to repo root
            self._model_path = Path(__file__).parents[4] / "models" / "quantdiff" / "unet.pt"

        self._model: Optional[QuantDiffUNet] = None
        self._trained = False
        self._scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            clip_sample=True,
        )
        self._aether = ResonartEngine(output_dir=output_dir)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> tuple["QuantDiffUNet", bool]:
        """
        Returns (model, is_trained).

        If trained weights exist at model_path the model is loaded from
        disk.  Otherwise a randomly-initialized model is returned and
        is_trained=False (warm-start mode).
        """
        if self._model is not None:
            return self._model, self._trained

        model = QuantDiffUNet(
            in_channels=3,
            base_channels=BASE_CHANNELS,
            time_dim=TIME_DIM,
            cond_dim=COND_DIM,
        )

        trained = False
        if self._model_path and self._model_path.exists():
            state = torch.load(str(self._model_path), map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            trained = True

        model.eval()
        self._model = model
        self._trained = trained
        return model, trained

    # ------------------------------------------------------------------
    # Core DDIM denoising loop
    # ------------------------------------------------------------------

    def _ddim_denoise(
        self,
        x: "torch.Tensor",
        model: "QuantDiffUNet",
        cond: "torch.Tensor",
        steps: int,
        start_t: Optional[int] = None,
        eta: float = 0.0,
    ) -> "torch.Tensor":
        """
        Run the DDIM reverse process.

        x       : initial noisy sample (1, 3, H, W)
        start_t : if not None, only run timesteps ≤ start_t (img2img)
        Returns : denoised sample (1, 3, H, W)
        """
        self._scheduler.set_timesteps(steps)
        timesteps = self._scheduler.timesteps

        # For img2img: only run the subset of timesteps ≤ start_t
        if start_t is not None:
            timesteps = timesteps[timesteps <= start_t]

        with torch.no_grad():
            for t in timesteps:
                t_batch = torch.tensor([int(t)], dtype=torch.long)
                # Ensure input is always float32 (scheduler may return float64 otherwise)
                noise_pred = model(x.float(), t_batch, cond.float()).float()
                x = torch.from_numpy(
                    self._scheduler.step(noise_pred.numpy(), int(t), x.numpy(), eta=eta)
                )

        return x.float()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, req: GenerateRequest) -> GenerateResult:
        if not TORCH_AVAILABLE:
            # Graceful fallback: AetherCanvas with no torch
            return self._aether.generate(req)

        try:
            from PIL import Image as _PIL  # noqa: F401
        except ImportError:
            return GenerateResult(success=False, error="Pillow not available")

        try:
            seed = req.seed if req.seed is not None else int(time.time()) & 0xFFFF_FFFF
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed & 0x7FFF_FFFF)

            model, trained = self._load_model()

            steps = max(4, min(int(req.steps), 50))
            target_w = max(64, min(int(req.width), 4096))
            target_h = max(64, min(int(req.height), 4096))
            H = W = self.native_size

            cond = _encode_prompt(req)

            if trained:
                # --------------------------------------------------------
                # Full DDIM from pure Gaussian noise
                # --------------------------------------------------------
                x = torch.randn(1, 3, H, W)
                x = self._ddim_denoise(x, model, cond, steps, eta=0.0)
            else:
                # --------------------------------------------------------
                # Warm-start: AetherCanvas → low-strength DDIM refinement
                # --------------------------------------------------------
                # 1. Generate AetherCanvas base at native_size
                warm_req = GenerateRequest(
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt,
                    steps=max(8, steps),
                    style=req.style,
                    width=H,
                    height=W,
                    quality=req.quality,
                    scene_hint=req.scene_hint,
                    palette_hint=req.palette_hint,
                    seed=seed,
                    output_dir=str(self.output_dir),
                )
                base_result = self._aether.generate(warm_req)
                if not base_result.success:
                    return base_result  # fallback failed

                from PIL import Image as _PIL
                base_img = _PIL.open(base_result.path).convert("RGB").resize(
                    (W, H), _PIL.LANCZOS
                )
                x0 = _img_to_tensor(base_img).numpy()  # (1,3,H,W) float32 [-1,1]

                # 2. Add small amount of noise (warm-start strength)
                strength = _DEFAULT_WARMSTART_STRENGTH
                noisy_x, start_t = self._scheduler.noise_image_strength(x0, strength, rng)
                x = torch.from_numpy(noisy_x)

                # 3. Partial DDIM reverse from start_t → 0
                x = self._ddim_denoise(x, model, cond, steps, start_t=start_t, eta=0.0)

            # Convert to PIL and upscale
            img = _tensor_to_img(x)
            if target_w != W or target_h != H:
                img = _upscale(img, target_w, target_h)

            slug = (
                "".join(c.lower() if c.isalnum() else "_" for c in req.prompt)
                .strip("_")[:40]
                or "qdiff"
            )
            filename = f"qdiff_{int(time.time())}_{slug}.png"
            out_path = self.output_dir / filename
            img.save(str(out_path), optimize=False, compress_level=1)

            return GenerateResult(
                success=True,
                path=str(out_path),
                prompt=req.prompt,
                style=req.style,
                seed=int(seed),
                width=target_w,
                height=target_h,
            )

        except Exception as exc:
            return GenerateResult(success=False, error=str(exc))
