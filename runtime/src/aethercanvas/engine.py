"""
Resonart Python image engine.

Local procedural renderer — the Python implementation of the Resonart engine
(native Rust port lives in engines/resonart.rs).
Combines:
- spectral texture synthesis for rich backgrounds (IQ fBM, octave rotation)
- semantic prompt parsing for recognizable subjects
- supersampled composition and cinematic post-processing
- style-aware finishing tuned for local CPU execution
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class GenerateRequest:
    prompt: str
    negative_prompt: str = ""
    steps: int = 20
    style: str = "pixel"
    width: int = 1024
    height: int = 1024
    quality: str = "balanced"
    scene_hint: str = ""
    palette_hint: str = ""
    spectral_complexity: float = 0.0
    background_path: str = ""
    seed: Optional[int] = None
    output_dir: str = "generated/images"


@dataclass
class GenerateResult:
    success: bool
    path: str = ""
    prompt: str = ""
    style: str = ""
    seed: int = 0
    width: int = 0
    height: int = 0
    error: str = ""


PALETTE_MAP = {
    "forest": [(34, 85, 34), (60, 120, 40), (100, 160, 60), (180, 210, 100), (240, 230, 180)],
    "ocean": [(10, 30, 80), (20, 60, 140), (40, 120, 180), (80, 180, 200), (200, 240, 255)],
    "desert": [(180, 130, 60), (220, 170, 90), (240, 200, 120), (255, 230, 160), (255, 245, 210)],
    "snow": [(200, 220, 240), (220, 235, 250), (240, 245, 255), (255, 255, 255), (180, 200, 230)],
    "volcano": [(80, 10, 0), (160, 30, 10), (220, 80, 20), (255, 140, 40), (255, 220, 100)],
    "neon": [(255, 0, 128), (0, 255, 180), (180, 0, 255), (255, 220, 0), (0, 200, 255)],
    "cyberpunk": [(20, 0, 40), (60, 0, 120), (0, 200, 255), (255, 50, 150), (255, 200, 0)],
    "retro": [(255, 100, 50), (255, 200, 50), (50, 200, 100), (50, 100, 200), (200, 50, 200)],
    "pastel": [(255, 200, 210), (200, 220, 255), (200, 255, 220), (255, 255, 200), (230, 200, 255)],
    "dark": [(10, 10, 20), (30, 20, 40), (50, 40, 60), (80, 60, 80), (120, 100, 120)],
    "midnight": [(5, 5, 20), (15, 10, 40), (30, 20, 80), (50, 40, 120), (100, 80, 160)],
    "city": [(40, 40, 60), (80, 80, 100), (160, 160, 180), (200, 180, 120), (255, 200, 80)],
    "space": [(0, 0, 10), (10, 5, 30), (20, 10, 60), (80, 40, 120), (255, 255, 255)],
    "sky": [(100, 160, 240), (140, 190, 255), (180, 220, 255), (220, 240, 255), (255, 255, 255)],
    "sunset": [(255, 80, 30), (255, 140, 60), (255, 180, 80), (255, 220, 120), (200, 100, 180)],
    "racing": [(110, 12, 12), (190, 24, 24), (238, 60, 48), (245, 180, 72), (250, 236, 220)],
    "dog": [(126, 92, 60), (158, 122, 76), (201, 168, 119), (230, 210, 173), (98, 72, 48)],
    "golden": [(166, 124, 48), (202, 156, 76), (226, 188, 116), (244, 220, 170), (92, 60, 30)],
    "cat": [(100, 100, 110), (145, 145, 152), (190, 182, 170), (240, 226, 210), (70, 68, 76)],
    "default": [(60, 80, 140), (100, 130, 180), (160, 180, 220), (200, 210, 230), (240, 245, 255)],
}

STYLE_POSTPROCESS = {
    "pixel": {"pixelate": 8, "contrast": 1.35, "saturation": 1.55, "sharpen": 1.2},
    "photo": {"blur": 0.25, "contrast": 1.15, "saturation": 1.05, "sharpen": 1.5, "vignette": 0.28},
    "anime": {"blur": 0.15, "contrast": 1.35, "saturation": 1.85, "sharpen": 1.3},
    "sketch": {"greyscale": True, "contrast": 1.9, "edge": True},
    "oil_paint": {"blur": 1.1, "contrast": 1.18, "saturation": 1.3, "paint": True},
    "neon": {"blur": 0.35, "contrast": 1.9, "saturation": 2.4, "neon": True, "vignette": 0.24},
    "abstract": {"contrast": 1.45, "saturation": 2.0, "grain": 0.05},
}


class ResonartEngine:
    def __init__(self, output_dir: str = "generated/images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, req: GenerateRequest) -> GenerateResult:
        if not PIL_AVAILABLE:
            return GenerateResult(success=False, error="Pillow not installed. Run: pip install Pillow")

        try:
            seed = req.seed if req.seed is not None else self._prompt_seed(req.prompt)
            rng = np.random.default_rng(seed)
            style = req.style if req.style in STYLE_POSTPROCESS else "photo"
            steps = max(8, min(req.steps, 60))
            quality = req.quality if req.quality in {"draft", "balanced", "ultra"} else "balanced"

            width = max(256, min(req.width, 2048))
            height = max(256, min(req.height, 2048))

            quality_bonus = {"draft": 0, "balanced": 1, "ultra": 2}[quality]
            base_scale = 2 if style in {"photo", "oil_paint", "neon", "anime"} else 1
            render_scale = min(4, base_scale + quality_bonus)

            tokens = self._tokenize(req.prompt)
            if req.scene_hint == "dog":
                tokens |= {"dog", "canine"}
            elif req.scene_hint == "dog_portrait":
                tokens |= {"dog", "canine", "portrait", "headshot"}
            elif req.scene_hint == "cat":
                tokens |= {"cat", "feline"}
            elif req.scene_hint == "tree":
                tokens |= {"tree", "forest"}
            elif req.scene_hint == "cityscape":
                tokens |= {"city", "skyline", "urban"}
            elif req.scene_hint == "sportscar":
                tokens |= {"sportscar", "car", "vehicle", "road", "coupe", "racing"}

            negative_tokens = self._tokenize(req.negative_prompt)
            palette = self._extract_palette(req.prompt, negative_tokens, req.palette_hint)

            render_w = width * render_scale
            render_h = height * render_scale

            # Phase-2: use Rust-rendered background if provided; fall back to Python synthesis.
            if req.background_path and Path(req.background_path).exists():
                bg_img = Image.open(req.background_path).convert("RGB")
                if bg_img.size != (render_w, render_h):
                    bg_img = bg_img.resize((render_w, render_h), Image.Resampling.LANCZOS)
                img = bg_img
            else:
                canvas = self._spectral_background(rng, render_w, render_h, palette, steps, tokens)
                img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), "RGB")

            img = self._paint_environment(img, rng, tokens, negative_tokens, palette)
            img = self._render_semantic_subject(img, rng, tokens, negative_tokens, req.prompt, palette, style)

            # Portrait-specific: DOF background blur then 3-point lighting
            is_portrait = bool({"portrait", "closeup", "headshot"} & tokens) or req.scene_hint in {"dog_portrait"}
            if is_portrait and style in {"photo", "oil_paint"}:
                img = self._portrait_dof(img, render_w, render_h)
                img = self._portrait_lighting(img)

            img = self._add_depth_and_lighting(img, style)
            img = self._postprocess(img, STYLE_POSTPROCESS[style], style)

            if render_scale > 1:
                img = img.resize((width, height), Image.Resampling.LANCZOS)

            img = self._stamp(img, style, negative_tokens)

            slug = self._slugify(req.prompt)[:40]
            filename = f"{int(time.time())}_{slug}.png"
            out_path = self.output_dir / filename
            img.save(str(out_path), "PNG")

            return GenerateResult(
                success=True,
                path=str(out_path),
                prompt=req.prompt,
                style=style,
                seed=int(seed),
                width=width,
                height=height,
            )
        except Exception as exc:
            return GenerateResult(success=False, error=str(exc))

    def _spectral_background(
        self,
        rng: np.random.Generator,
        width: int,
        height: int,
        palette: List[Tuple[int, int, int]],
        steps: int,
        tokens: set[str],
    ) -> np.ndarray:
        """fBM background using IQ's technique: octave rotation + quintic smoothstep.

        Key parameters (Inigo Quilez, 2019):
          G = 0.5  (gain, H=1 'yellow noise') — isotropic self-similarity, best for nature
          frequency doubles each octave (f *= 2.01, slightly detuned to prevent superposition)
          domain rotated ~22.5° per octave to break directional artifacts
          quintic smoothstep t³(6t²−15t+10) instead of linear for smooth cell transitions
        """
        canvas = np.zeros((height, width, 3), dtype=np.float64)
        # Scale to 4π so we get ~2 full cycles across image (richer than 1 cycle)
        xs = np.linspace(0, 4 * math.pi, width)
        ys = np.linspace(0, 4 * math.pi, height)
        xx_base, yy_base = np.meshgrid(xs, ys)

        # fBM parameters
        G = 0.5          # amplitude gain per octave (H=1, yellow noise)
        octaves = min(8, steps // 3 + 4)
        freq_bias = 0.5 if {"portrait", "dog", "cat"} & tokens else 1.0

        f = freq_bias    # starting frequency
        a = 1.0          # starting amplitude
        total_a = 0.0
        angle = 0.0      # domain rotation angle, accumulates 22.5° per octave

        for index in range(octaves):
            c1 = np.array(palette[index % len(palette)], dtype=np.float64)
            c2 = np.array(palette[(index + 1) % len(palette)], dtype=np.float64)

            phase_x = rng.uniform(0, 2 * math.pi)
            phase_y = rng.uniform(0, 2 * math.pi)

            # Rotate coordinate domain each octave (IQ: breaks directional artifacts)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            rx = cos_a * xx_base - sin_a * yy_base
            ry = sin_a * xx_base + cos_a * yy_base

            # Sinusoidal wave at this octave's frequency
            wave = np.sin(f * rx + phase_x) * np.cos(f * ry + phase_y)

            # Quintic smoothstep on alpha: t³(6t²−15t+10) — smooth cell boundaries
            t = (wave + 1.0) * 0.5          # map [-1,1] → [0,1]
            t_smooth = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

            alpha = t_smooth[:, :, np.newaxis]
            layer = (1.0 - alpha) * c1 + alpha * c2
            canvas += a * layer
            total_a += a

            f *= 2.01        # slightly detuned to avoid exact harmonic superposition
            a *= G           # amplitude halves each octave
            angle += 0.3926990  # 22.5° rotation per octave

        canvas /= total_a   # normalise accumulated amplitudes

        vertical = np.linspace(0, 1, height)[:, None, None]
        top = np.array(palette[0], dtype=np.float64)
        bottom = np.array(palette[min(2, len(palette) - 1)], dtype=np.float64)
        canvas = canvas * 0.55 + ((1 - vertical) * top + vertical * bottom) * 0.45

        lo, hi = canvas.min(), canvas.max()
        if hi > lo:
            canvas = (canvas - lo) / (hi - lo) * 255.0
        return canvas

    def _paint_environment(
        self,
        img: "Image.Image",
        rng: np.random.Generator,
        tokens: set[str],
        negative_tokens: set[str],
        palette: List[Tuple[int, int, int]],
    ) -> "Image.Image":
        draw = ImageDraw.Draw(img, "RGBA")
        width, height = img.size
        car_tokens = {"car", "sportscar", "vehicle", "supercar", "coupe", "road", "street", "highway", "track", "race", "racing"}
        horizon = int(height * (0.62 if {"dog", "cat", "portrait"} & tokens else (0.6 if tokens & car_tokens else 0.55)))

        if tokens & {"field", "grass", "park", "meadow", "dog", "cat"} and not negative_tokens & {"field", "grass", "park", "meadow", "dog", "cat"}:
            ground = [palette[min(2, len(palette) - 1)][0] // 2, 130, 72, 255]
            draw.rectangle((0, horizon, width, height), fill=tuple(ground))
            for _ in range(width // 8):
                x = int(rng.integers(0, width))
                blade_h = int(rng.integers(height // 30, height // 12))
                draw.line((x, height, x + int(rng.integers(-3, 4)), height - blade_h), fill=(90, 175, 70, 170), width=1)

        if tokens & {"city", "skyline", "building", "urban"} and not negative_tokens & {"city", "skyline", "building", "urban"}:
            base_y = int(height * 0.68)
            for x in range(0, width, max(18, width // 24)):
                w = int(rng.integers(width // 28, width // 14))
                h = int(rng.integers(height // 8, height // 2))
                draw.rectangle((x, base_y - h, x + w, base_y), fill=(24, 28, 46, 215))
                for wx in range(x + 4, min(x + w - 2, width - 2), 8):
                    for wy in range(base_y - h + 4, base_y - 4, 10):
                        if rng.random() > 0.55:
                            draw.rectangle((wx, wy, wx + 3, wy + 4), fill=(255, 214, 120, 200))

        if tokens & car_tokens and not negative_tokens & car_tokens:
            road_top = int(height * 0.6)
            road = [
                (0, height),
                (width, height),
                (int(width * 0.72), road_top),
                (int(width * 0.28), road_top),
            ]
            draw.polygon(road, fill=(42, 44, 52, 238))
            draw.line((int(width * 0.29), road_top, int(width * 0.71), road_top), fill=(115, 120, 132, 110), width=max(2, width // 280))
            draw.line((int(width * 0.15), height - 2, int(width * 0.32), road_top), fill=(84, 88, 96, 160), width=max(2, width // 180))
            draw.line((int(width * 0.85), height - 2, int(width * 0.68), road_top), fill=(84, 88, 96, 160), width=max(2, width // 180))
            stripe_w = max(4, width // 140)
            stripe_h = max(26, height // 12)
            lane_x = width // 2
            for idx in range(5):
                t0 = idx / 5.0
                t1 = (idx + 0.52) / 5.0
                y0 = int(road_top + (height - road_top) * t0)
                y1 = int(road_top + (height - road_top) * min(1.0, t1))
                perspective = 1.0 + idx * 0.28
                w = int(stripe_w * perspective)
                draw.polygon(
                    [
                        (lane_x - w, y1),
                        (lane_x + w, y1),
                        (lane_x + max(1, w // 2), y0),
                        (lane_x - max(1, w // 2), y0),
                    ],
                    fill=(248, 232, 168, 210),
                )
            for light_x in [int(width * 0.18), int(width * 0.82)]:
                for ring in range(3, 0, -1):
                    rr = ring * max(8, width // 100)
                    draw.ellipse((light_x - rr, road_top - rr // 2, light_x + rr, road_top + rr // 2), fill=(255, 180, 96, 22 * ring))

        if tokens & {"space", "night", "stars", "galaxy", "cosmic"} and not negative_tokens & {"space", "night", "stars", "galaxy", "cosmic", "stars"}:
            for _ in range(width // 3):
                x = int(rng.integers(0, width))
                y = int(rng.integers(0, int(height * 0.75)))
                r = int(rng.integers(1, 3))
                draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, int(rng.integers(90, 210))))

        if tokens & {"sun", "glow", "sunset", "golden"} and not negative_tokens & {"sun", "glow", "sunset", "golden"}:
            radius = width // 10
            cx = int(width * 0.75)
            cy = int(height * 0.22)
            for ring in range(5, 0, -1):
                alpha = 18 * ring
                rr = radius * ring // 2
                draw.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=(255, 220, 140, alpha))

        return img

    def _render_semantic_subject(
        self,
        img: "Image.Image",
        rng: np.random.Generator,
        tokens: set[str],
        negative_tokens: set[str],
        prompt: str,
        palette: List[Tuple[int, int, int]],
        style: str,
    ) -> "Image.Image":
        if tokens & {"dog", "puppy", "retriever", "canine"} and not negative_tokens & {"dog", "puppy", "retriever", "canine"}:
            return self._render_dog(img, rng, tokens, prompt, style)
        if tokens & {"cat", "kitten", "feline"} and not negative_tokens & {"cat", "kitten", "feline"}:
            return self._render_cat(img, rng, tokens, style)
        if tokens & {"car", "sportscar", "vehicle", "supercar", "coupe"} and not negative_tokens & {"car", "sportscar", "vehicle", "supercar", "coupe"}:
            return self._render_sportscar(img, rng, tokens, style)
        if tokens & {"tree", "forest"} and not negative_tokens & {"tree", "forest"}:
            return self._render_tree(img, rng, palette)
        return img

    def _render_sportscar(self, img: "Image.Image", rng: np.random.Generator, tokens: set[str], style: str) -> "Image.Image":
        width, height = img.size
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        glow_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer, "RGBA")
        glow_draw = ImageDraw.Draw(glow_layer, "RGBA")

        if tokens & {"yellow", "gold", "golden"}:
            body = (238, 194, 48, 255)
        elif tokens & {"blue", "azure", "cobalt"}:
            body = (42, 112, 220, 255)
        elif tokens & {"white", "silver", "chrome"}:
            body = (224, 228, 236, 255)
        elif tokens & {"black", "midnight", "dark"}:
            body = (34, 34, 42, 255)
        elif tokens & {"green", "emerald"}:
            body = (32, 138, 96, 255)
        else:
            body = (214, 42, 38, 255)

        shadow = tuple(max(0, channel - 70) for channel in body[:3]) + (255,)
        mid = tuple(min(255, int(channel * 0.82 + 24)) for channel in body[:3]) + (255,)
        highlight = tuple(min(255, channel + 55) for channel in body[:3]) + (210,)

        ground = int(height * 0.78)
        car_w = int(width * 0.54)
        car_h = int(height * 0.16)
        cx = width // 2
        x0 = cx - car_w // 2
        x1 = cx + car_w // 2
        roof_y = ground - int(car_h * 1.45)
        nose_y = ground - int(car_h * 0.78)
        deck_y = ground - int(car_h * 1.02)
        rear_x = x0 + int(car_w * 0.26)
        front_x = x0 + int(car_w * 0.77)
        wheel_r = max(20, car_h // 2)
        wheel_y = ground - wheel_r // 4

        draw.ellipse((x0 + 20, ground - 8, x1 - 12, ground + 28), fill=(18, 16, 18, 95))

        body_points = [
            (x0 + int(car_w * 0.03), ground - int(car_h * 0.18)),
            (x0 + int(car_w * 0.1), deck_y),
            (x0 + int(car_w * 0.27), roof_y + int(car_h * 0.06)),
            (x0 + int(car_w * 0.47), roof_y),
            (x0 + int(car_w * 0.66), roof_y + int(car_h * 0.12)),
            (x0 + int(car_w * 0.83), nose_y),
            (x1 - int(car_w * 0.02), ground - int(car_h * 0.2)),
            (x1 - int(car_w * 0.08), ground - int(car_h * 0.05)),
            (x0 + int(car_w * 0.11), ground - int(car_h * 0.03)),
        ]
        draw.polygon(body_points, fill=body)

        side_scoop = [
            (x0 + int(car_w * 0.4), ground - int(car_h * 0.48)),
            (x0 + int(car_w * 0.63), ground - int(car_h * 0.5)),
            (x0 + int(car_w * 0.58), ground - int(car_h * 0.25)),
            (x0 + int(car_w * 0.38), ground - int(car_h * 0.22)),
        ]
        draw.polygon(side_scoop, fill=shadow)
        draw.polygon(
            [
                (x0 + int(car_w * 0.06), ground - int(car_h * 0.12)),
                (x1 - int(car_w * 0.08), ground - int(car_h * 0.12)),
                (x1 - int(car_w * 0.12), ground - int(car_h * 0.02)),
                (x0 + int(car_w * 0.12), ground - int(car_h * 0.02)),
            ],
            fill=mid,
        )

        cabin = [
            (x0 + int(car_w * 0.27), roof_y + int(car_h * 0.1)),
            (x0 + int(car_w * 0.46), roof_y + int(car_h * 0.02)),
            (x0 + int(car_w * 0.65), roof_y + int(car_h * 0.18)),
            (x0 + int(car_w * 0.71), ground - int(car_h * 0.63)),
            (x0 + int(car_w * 0.39), ground - int(car_h * 0.6)),
        ]
        draw.polygon(cabin, fill=(44, 60, 84, 235))
        draw.line((x0 + int(car_w * 0.5), roof_y + int(car_h * 0.05), x0 + int(car_w * 0.47), ground - int(car_h * 0.61)), fill=(190, 210, 232, 150), width=max(2, width // 320))

        draw.line(
            [
                (x0 + int(car_w * 0.14), ground - int(car_h * 0.74)),
                (x0 + int(car_w * 0.43), roof_y + int(car_h * 0.08)),
                (x0 + int(car_w * 0.79), nose_y + int(car_h * 0.02)),
            ],
            fill=highlight,
            width=max(2, width // 300),
            joint="curve",
        )
        draw.line(
            [
                (x0 + int(car_w * 0.19), ground - int(car_h * 0.42)),
                (x0 + int(car_w * 0.45), ground - int(car_h * 0.56)),
                (x0 + int(car_w * 0.75), ground - int(car_h * 0.44)),
            ],
            fill=(255, 255, 255, 90),
            width=max(1, width // 420),
            joint="curve",
        )

        for wheel_x in [rear_x, front_x]:
            draw.ellipse((wheel_x - wheel_r, wheel_y - wheel_r, wheel_x + wheel_r, wheel_y + wheel_r), fill=(20, 20, 24, 255))
            rim_r = int(wheel_r * 0.58)
            draw.ellipse((wheel_x - rim_r, wheel_y - rim_r, wheel_x + rim_r, wheel_y + rim_r), fill=(172, 176, 186, 255))
            hub_r = max(4, wheel_r // 6)
            draw.ellipse((wheel_x - hub_r, wheel_y - hub_r, wheel_x + hub_r, wheel_y + hub_r), fill=(110, 112, 118, 255))
            for spoke in range(5):
                angle = spoke * (math.pi / 2.5) + 0.18
                sx = int(math.cos(angle) * (rim_r - 3))
                sy = int(math.sin(angle) * (rim_r - 3))
                draw.line((wheel_x, wheel_y, wheel_x + sx, wheel_y + sy), fill=(228, 230, 236, 220), width=max(1, width // 600))

        draw.arc((rear_x - wheel_r - 8, wheel_y - wheel_r - 10, rear_x + wheel_r + 8, wheel_y + wheel_r - 8), start=188, end=354, fill=shadow, width=max(5, width // 170))
        draw.arc((front_x - wheel_r - 10, wheel_y - wheel_r - 12, front_x + wheel_r + 10, wheel_y + wheel_r - 8), start=186, end=354, fill=shadow, width=max(5, width // 170))

        headlight = (x1 - int(car_w * 0.03), ground - int(car_h * 0.28))
        taillight = (x0 + int(car_w * 0.05), ground - int(car_h * 0.35))
        draw.ellipse((headlight[0] - 12, headlight[1] - 6, headlight[0] + 8, headlight[1] + 4), fill=(245, 250, 255, 240))
        draw.ellipse((taillight[0] - 8, taillight[1] - 5, taillight[0] + 8, taillight[1] + 5), fill=(255, 70, 52, 220))
        for ring in range(4, 0, -1):
            glow_r = ring * max(8, width // 130)
            glow_draw.ellipse((headlight[0] - glow_r, headlight[1] - glow_r // 2, headlight[0] + glow_r, headlight[1] + glow_r // 2), fill=(255, 244, 220, 20 * ring))
        if tokens & {"neon", "night", "city", "cyberpunk"}:
            for ring in range(3, 0, -1):
                rr = ring * max(6, width // 160)
                glow_draw.ellipse((taillight[0] - rr, taillight[1] - rr // 2, taillight[0] + rr, taillight[1] + rr // 2), fill=(255, 64, 80, 18 * ring))

        if style in {"photo", "neon", "oil_paint"}:
            streak_y = ground - int(car_h * 0.18)
            for idx in range(6):
                streak_x = x0 - int(car_w * 0.08) - idx * max(10, width // 80)
                streak_len = int(car_w * (0.08 + idx * 0.02))
                draw.line((streak_x, streak_y + idx * 3, streak_x + streak_len, streak_y + idx), fill=(255, 255, 255, max(24, 82 - idx * 10)), width=max(1, width // 700))

        composite = Image.alpha_composite(img.convert("RGBA"), layer)
        composite = Image.alpha_composite(composite, glow_layer)
        return composite.convert("RGB")

    def _render_dog(self, img: "Image.Image", rng: np.random.Generator, tokens: set[str], prompt: str, style: str) -> "Image.Image":
        width, height = img.size

        # Portrait prompts benefit from a dedicated close-up composition.
        if {"portrait", "closeup", "close-up", "headshot"} & tokens:
            return self._render_dog_portrait(img, rng, tokens, style)

        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        fur_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer, "RGBA")
        fur_draw = ImageDraw.Draw(fur_layer, "RGBA")

        golden = bool({"golden", "retriever"} & tokens)
        sitting = bool({"sit", "sitting", "portrait"} & tokens)
        fur = (224, 188, 122, 255) if golden else (176, 130, 84, 255)
        fur_mid = (202, 156, 95, 255) if golden else (152, 108, 66, 255)
        fur_shadow = (132, 90, 50, 255) if golden else (112, 76, 44, 255)
        fur_light = (246, 226, 188, 230) if golden else (228, 198, 158, 230)

        cx = int(width * 0.5)
        ground = int(height * 0.74)
        body_w = int(width * (0.32 if sitting else 0.36))
        body_h = int(height * (0.25 if sitting else 0.2))
        torso_top = ground - body_h

        # Ground shadow anchors the subject and stops the "floating blob" effect.
        draw.ellipse(
            (cx - body_w // 2 - 35, ground - 12, cx + body_w // 2 + 28, ground + 18),
            fill=(25, 20, 16, 95),
        )

        # Torso, chest and rear quarter in clearer side profile.
        body_cx = cx + (10 if sitting else 0)
        draw.ellipse((body_cx - body_w // 2, torso_top + 14, body_cx + body_w // 2, ground - 6), fill=fur)
        draw.ellipse((body_cx - body_w // 2 - 24, torso_top + 22, body_cx - body_w // 8, ground + 8), fill=fur_light)
        draw.ellipse((body_cx + body_w // 8, torso_top + 18, body_cx + body_w // 2 + 16, ground + 2), fill=fur_shadow)

        # Neck and head (left-facing profile).
        neck_x0 = body_cx - body_w // 2 - 10
        neck_x1 = body_cx - body_w // 6
        neck_y0 = torso_top + 22
        neck_y1 = torso_top + body_h // 2 + 10
        draw.rounded_rectangle((neck_x0, neck_y0, neck_x1, neck_y1), radius=10, fill=fur_mid)

        head_cx = neck_x0 - int(width * 0.065)
        head_cy = torso_top + int(body_h * 0.4)
        head_rx = int(width * 0.085)
        head_ry = int(height * 0.08)
        draw.ellipse((head_cx - head_rx, head_cy - head_ry, head_cx + head_rx, head_cy + head_ry), fill=fur)

        # Snout projects left.
        muzzle_w = int(head_rx * 1.0)
        muzzle_h = int(head_ry * 0.65)
        muzzle_x0 = head_cx - head_rx - muzzle_w // 2
        muzzle_y0 = head_cy + head_ry // 4
        draw.ellipse((muzzle_x0, muzzle_y0, muzzle_x0 + muzzle_w, muzzle_y0 + muzzle_h), fill=(242, 226, 198, 238))
        draw.ellipse((muzzle_x0 - 6, muzzle_y0 + muzzle_h // 3, muzzle_x0 + 8, muzzle_y0 + muzzle_h // 3 + 12), fill=(20, 18, 18, 255))
        draw.line((muzzle_x0 + 2, muzzle_y0 + muzzle_h // 2 + 8, muzzle_x0 + 10, muzzle_y0 + muzzle_h // 2 + 12), fill=(30, 24, 20, 220), width=2)

        # Eye and brow ridge.
        eye_x = head_cx - 4
        eye_y = head_cy - 4
        draw.ellipse((eye_x - 4, eye_y - 3, eye_x + 3, eye_y + 4), fill=(24, 16, 12, 255))
        draw.ellipse((eye_x - 3, eye_y - 2, eye_x - 1, eye_y), fill=(255, 255, 255, 190))
        draw.arc((eye_x - 10, eye_y - 8, eye_x + 8, eye_y + 2), start=190, end=355, fill=fur_shadow, width=2)

        # Retriever ear: one visible floppy ear on profile side.
        ear = [
            (head_cx + 2, head_cy - head_ry + 6),
            (head_cx + 22, head_cy - 8),
            (head_cx + 10, head_cy + head_ry + 12),
            (head_cx - 8, head_cy + 10),
        ]
        draw.polygon(ear, fill=fur_shadow)

        # Legs and paws.
        leg_top = torso_top + body_h // 2 + 16
        if sitting:
            front_x = [body_cx - body_w // 4, body_cx - body_w // 9]
            for lx in front_x:
                draw.rounded_rectangle((lx, leg_top, lx + 18, ground + body_h // 2), radius=8, fill=fur_mid)
                draw.ellipse((lx - 2, ground + body_h // 2 - 5, lx + 22, ground + body_h // 2 + 6), fill=(72, 52, 30, 255))
            rear_x = body_cx + body_w // 7
            draw.ellipse((rear_x - 8, ground - 2, rear_x + 34, ground + 26), fill=fur_shadow)
            draw.ellipse((rear_x - 2, ground + 16, rear_x + 30, ground + 34), fill=(70, 52, 34, 255))
        else:
            for lx in [body_cx - body_w // 3, body_cx - body_w // 8, body_cx + body_w // 8, body_cx + body_w // 3]:
                draw.rounded_rectangle((lx, leg_top, lx + 16, ground + body_h // 3), radius=8, fill=fur_mid)
                draw.ellipse((lx - 2, ground + body_h // 3 - 5, lx + 20, ground + body_h // 3 + 5), fill=(72, 52, 32, 255))

        # Tail with fluffy taper.
        tail_base = (body_cx + body_w // 2 - 8, torso_top + 28)
        tail_tip = (body_cx + body_w // 2 + 42, torso_top - 12)
        draw.line([tail_base, ((tail_base[0] + tail_tip[0]) // 2, tail_tip[1] + 8), tail_tip], fill=fur_shadow, width=max(8, width // 56), joint="curve")
        draw.ellipse((tail_tip[0] - 10, tail_tip[1] - 6, tail_tip[0] + 8, tail_tip[1] + 10), fill=fur_light)

        # Fur texture strokes concentrated around silhouette and chest.
        stroke_count = 420 if style in {"photo", "oil_paint"} else 210
        x0 = body_cx - body_w // 2 - 24
        x1 = body_cx + body_w // 2 + 30
        y0 = torso_top - 10
        y1 = ground + body_h // 2
        for _ in range(stroke_count):
            px = int(rng.integers(x0, x1))
            py = int(rng.integers(y0, y1))
            dx = int(rng.integers(-3, 4))
            dy = int(rng.integers(3, 11))
            shade = fur_light if rng.random() > 0.55 else fur_shadow
            fur_draw.line((px, py, px + dx, py + dy), fill=shade, width=1)

        # Whisker dots and subtle muzzle whiskers.
        for wx in [muzzle_x0 + 8, muzzle_x0 + 14, muzzle_x0 + 20]:
            draw.ellipse((wx, muzzle_y0 + muzzle_h // 2 + 2, wx + 2, muzzle_y0 + muzzle_h // 2 + 4), fill=(86, 66, 46, 180))
            draw.line((wx + 2, muzzle_y0 + muzzle_h // 2 + 3, wx + 14, muzzle_y0 + muzzle_h // 2 + 1), fill=(130, 110, 90, 120), width=1)

        composite = Image.alpha_composite(img.convert("RGBA"), layer)
        composite = Image.alpha_composite(composite, fur_layer)
        return composite.convert("RGB")

    def _render_dog_portrait(self, img: "Image.Image", rng: np.random.Generator, tokens: set[str], style: str) -> "Image.Image":
        width, height = img.size
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        fur_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer, "RGBA")
        fur_draw = ImageDraw.Draw(fur_layer, "RGBA")

        golden = bool({"golden", "retriever"} & tokens)
        fur = (224, 188, 122, 255) if golden else (176, 130, 84, 255)
        fur_mid = (202, 156, 95, 255) if golden else (152, 108, 66, 255)
        fur_shadow = (132, 90, 50, 255) if golden else (112, 76, 44, 255)
        fur_light = (246, 226, 188, 230) if golden else (228, 198, 158, 230)

        cx = width // 2
        cy = int(height * 0.56)
        head_r = int(min(width, height) * 0.16)

        # Chest/neck fluff base
        draw.ellipse((cx - head_r - 40, cy + head_r // 2 - 10, cx + head_r + 40, cy + head_r + 100), fill=fur_light)
        draw.ellipse((cx - head_r - 20, cy + head_r // 2 + 20, cx + head_r + 20, cy + head_r + 120), fill=fur)

        # Floppy ears
        left_ear = [
            (cx - head_r + 8, cy - head_r + 20),
            (cx - head_r - 70, cy - 10),
            (cx - head_r - 34, cy + head_r + 24),
            (cx - head_r + 14, cy + head_r // 3),
        ]
        right_ear = [
            (cx + head_r - 8, cy - head_r + 20),
            (cx + head_r + 70, cy - 10),
            (cx + head_r + 34, cy + head_r + 24),
            (cx + head_r - 14, cy + head_r // 3),
        ]
        draw.polygon(left_ear, fill=fur_shadow)
        draw.polygon(right_ear, fill=fur_shadow)

        # Head and forehead
        draw.ellipse((cx - head_r, cy - head_r, cx + head_r, cy + head_r), fill=fur)
        draw.ellipse((cx - head_r + 12, cy - head_r + 10, cx + head_r - 12, cy), fill=fur_mid)

        # Snout
        muzzle_w = int(head_r * 1.35)
        muzzle_h = int(head_r * 0.9)
        muzzle_x0 = cx - muzzle_w // 2
        muzzle_y0 = cy + head_r // 5
        draw.ellipse((muzzle_x0, muzzle_y0, muzzle_x0 + muzzle_w, muzzle_y0 + muzzle_h), fill=(243, 226, 198, 238))

        # Nose and mouth
        nose_x = cx
        nose_y = muzzle_y0 + muzzle_h // 3
        draw.ellipse((nose_x - 14, nose_y - 8, nose_x + 14, nose_y + 10), fill=(22, 18, 18, 255))
        draw.line((nose_x, nose_y + 6, nose_x, nose_y + 22), fill=(32, 22, 20, 220), width=2)
        draw.arc((nose_x - 22, nose_y + 14, nose_x + 2, nose_y + 36), start=300, end=80, fill=(34, 24, 20, 220), width=2)
        draw.arc((nose_x - 2, nose_y + 14, nose_x + 22, nose_y + 36), start=100, end=240, fill=(34, 24, 20, 220), width=2)

        # Eyes — sclera → amber iris → pupil → dual specular (key + fill lights)
        eye_y = cy - head_r // 4
        left_eye_x = cx - head_r // 3
        right_eye_x = cx + head_r // 3
        er = head_r // 5  # eye radius
        iris_color = (118, 72, 28, 255) if golden else (66, 48, 30, 255)  # amber/dark-brown
        for ex in [left_eye_x, right_eye_x]:
            # Sclera (warm white)
            draw.ellipse((ex - er - 2, eye_y - er + 1, ex + er + 2, eye_y + er - 1), fill=(238, 232, 218, 255))
            # Iris (amber for golden, dark brown otherwise)
            draw.ellipse((ex - er + 1, eye_y - er + 2, ex + er - 1, eye_y + er - 2), fill=iris_color)
            # Pupil
            pu = max(3, er // 2)
            draw.ellipse((ex - pu, eye_y - pu, ex + pu, eye_y + pu), fill=(10, 8, 6, 255))
            # Key-light specular (bright, upper-right — matches warm key light)
            draw.ellipse((ex + pu // 2, eye_y - pu - 1, ex + pu + 3, eye_y - 1), fill=(255, 255, 255, 230))
            # Fill-light specular (softer, lower-left — matches cool fill)
            draw.ellipse((ex - pu - 1, eye_y + 2, ex - 1, eye_y + pu), fill=(190, 210, 235, 130))
            # Brow ridge shadow
            draw.arc((ex - er - 6, eye_y - er - 8, ex + er + 6, eye_y + 2), start=195, end=345, fill=fur_shadow, width=2)

        # Fur texture: noise-driven flow field (strokes follow fur grain, not random).
        # Direction is based on body position: strokes radiate outward from spine,
        # then fBM noise adds natural variation (IQ fBM technique applied to flow).
        stroke_count = 560 if style in {"photo", "oil_paint"} else 280
        x0 = cx - head_r - 70
        x1 = cx + head_r + 70
        y0 = cy - head_r - 10
        y1 = cy + head_r + 120
        # Pre-compute noise table for flow variation (avoids per-pixel noise calls)
        noise_t = np.sin(np.arange(stroke_count, dtype=np.float64) * 1.6180339 * 0.7)
        noise_t2 = np.cos(np.arange(stroke_count, dtype=np.float64) * 2.3999632 * 0.5)
        for i in range(stroke_count):
            px = int(rng.integers(x0, x1))
            py = int(rng.integers(y0, y1))
            # Fur grows downward-outward from the head centre spine
            spine_x, spine_y = cx, cy - head_r // 2
            dx_raw = px - spine_x
            dy_raw = py - spine_y
            base_len = math.sqrt(dx_raw ** 2 + dy_raw ** 2) or 1.0
            base_ang = math.atan2(dy_raw, dx_raw)
            # fBM-style noise: two sinusoidal octaves modulate the flow angle
            noise_ang = noise_t[i] * 0.38 + noise_t2[i] * 0.18
            flow_ang = base_ang + noise_ang
            stroke_len = int(rng.integers(7, 15))
            dx = int(math.cos(flow_ang) * stroke_len)
            dy = int(math.sin(flow_ang) * stroke_len)
            shade = fur_light if rng.random() > 0.50 else fur_shadow
            fur_draw.line((px, py, px + dx, py + dy), fill=shade, width=1)

        # Subsurface scattering on muzzle: warm glow pass (skin/tissue translucency)
        sss_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        sss_draw = ImageDraw.Draw(sss_layer, "RGBA")
        sss_cx = cx
        sss_cy = muzzle_y0 + muzzle_h // 2
        for ring in range(4, 0, -1):
            r_sss = int(muzzle_w * 0.55 * ring / 4)
            alpha_sss = int(28 * (5 - ring))
            sss_draw.ellipse((sss_cx - r_sss, sss_cy - r_sss // 2, sss_cx + r_sss, sss_cy + r_sss // 2),
                             fill=(255, 190, 140, alpha_sss))

        # Subtle whiskers
        for wx in [cx - 16, cx - 6, cx + 4, cx + 14]:
            draw.ellipse((wx, muzzle_y0 + muzzle_h // 2 + 4, wx + 2, muzzle_y0 + muzzle_h // 2 + 6), fill=(88, 68, 48, 180))

        composite = Image.alpha_composite(img.convert("RGBA"), layer)
        composite = Image.alpha_composite(composite, fur_layer)
        composite = Image.alpha_composite(composite, sss_layer)
        return composite.convert("RGB")

    def _render_cat(self, img: "Image.Image", rng: np.random.Generator, tokens: set[str], style: str) -> "Image.Image":
        width, height = img.size
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer, "RGBA")
        base = (160, 158, 162, 255)
        shadow = (96, 92, 100, 255)
        cx = width // 2
        ground = int(height * 0.74)
        draw.ellipse((cx - 90, ground - 100, cx + 50, ground + 10), fill=base)
        draw.ellipse((cx - 130, ground - 146, cx - 36, ground - 58), fill=base)
        draw.polygon([(cx - 110, ground - 120), (cx - 126, ground - 168), (cx - 84, ground - 136)], fill=shadow)
        draw.polygon([(cx - 58, ground - 120), (cx - 40, ground - 168), (cx - 14, ground - 132)], fill=shadow)
        draw.line((cx + 42, ground - 74, cx + 120, ground - 160), fill=shadow, width=12)
        composite = Image.alpha_composite(img.convert("RGBA"), layer)
        return composite.convert("RGB")

    def _render_tree(self, img: "Image.Image", rng: np.random.Generator, palette: List[Tuple[int, int, int]]) -> "Image.Image":
        width, height = img.size
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer, "RGBA")
        x = width // 2
        trunk_w = width // 22
        draw.rounded_rectangle((x - trunk_w, height // 2, x + trunk_w, height - height // 8), radius=trunk_w // 2, fill=(92, 56, 28, 255))
        for _ in range(18):
            r = int(rng.integers(width // 16, width // 8))
            cx = int(x + rng.integers(-width // 8, width // 8))
            cy = int(height // 2 + rng.integers(-height // 5, height // 12))
            leaf = palette[int(rng.integers(0, len(palette)))]
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(*leaf, 180))
        composite = Image.alpha_composite(img.convert("RGBA"), layer)
        return composite.convert("RGB")

    def _portrait_dof(self, img: "Image.Image", width: int, height: int) -> "Image.Image":
        """Simulate bokeh depth-of-field: blur background, keep subject centre sharp."""
        cx = width // 2
        cy = int(height * 0.44)
        subject_r = int(min(width, height) * 0.34)

        blur_r = max(4, width // 60)
        blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_r))

        xs = np.linspace(0, width - 1, width)[np.newaxis, :]
        ys = np.linspace(0, height - 1, height)[:, np.newaxis]
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

        # Smooth quintic falloff (same function from IQ fBM research)
        t = np.clip((dist - subject_r * 0.55) / (subject_r * 0.75), 0.0, 1.0)
        mask_blur = (t * t * t * (t * (t * 6.0 - 15.0) + 10.0))[:, :, np.newaxis]

        sharp = np.asarray(img).astype(np.float64)
        soft = np.asarray(blurred).astype(np.float64)
        result = sharp * (1.0 - mask_blur) + soft * mask_blur
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), "RGB")

    def _portrait_lighting(self, img: "Image.Image") -> "Image.Image":
        """3-point portrait lighting: key (warm upper-left), fill (cool right), rim (top edge)."""
        width, height = img.size
        arr = np.asarray(img).astype(np.float64)

        xs = np.linspace(0.0, 1.0, width)[np.newaxis, :]   # (1, W)
        ys = np.linspace(0.0, 1.0, height)[:, np.newaxis]  # (H, 1)

        # Key light: warm, upper-left (simulates window or sun)
        key = np.exp(-((xs - 0.22) ** 2 / 0.12 + (ys - 0.20) ** 2 / 0.12)) * 24.0
        arr[:, :, 0] += key          # warm R
        arr[:, :, 1] += key * 0.72   # warm G (slightly less than R)

        # Fill light: cool blue, right side (sky/ambient)
        fill = np.exp(-((xs - 0.92) ** 2 / 0.18 + (ys - 0.55) ** 2 / 0.28)) * 14.0
        arr[:, :, 2] += fill         # cool B
        arr[:, :, 1] += fill * 0.45  # slight green in fill

        # Rim light: bright edge along top-right contour (back lighting separation)
        rim = ((xs * 0.55 + (1.0 - ys) * 0.45) ** 3) * 18.0
        arr[:, :, 0] += rim * 0.85
        arr[:, :, 1] += rim * 0.78
        arr[:, :, 2] += rim * 0.55

        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

    def _add_depth_and_lighting(self, img: "Image.Image", style: str) -> "Image.Image":
        width, height = img.size
        overlay = Image.new("RGB", img.size, (0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for row in range(height):
            alpha = int(90 * (row / max(1, height - 1)))
            draw.line((0, row, width, row), fill=(255, 190, 120) if row < height * 0.45 else (24, 32, 50), width=1)
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=max(8, width // 40)))
        blend = 0.08 if style == "photo" else 0.05
        return Image.blend(img, overlay, blend)

    def _postprocess(self, img: "Image.Image", cfg: dict, style: str) -> "Image.Image":
        if cfg.get("greyscale"):
            img = img.convert("L").convert("RGB")
        blur_r = cfg.get("blur", 0.0)
        if blur_r > 0:
            img = img.filter(ImageFilter.GaussianBlur(blur_r))
        if cfg.get("edge"):
            edges = img.filter(ImageFilter.FIND_EDGES)
            edges = ImageEnhance.Contrast(edges).enhance(3.0)
            img = Image.blend(img, edges, 0.6)
        if cfg.get("paint"):
            img = img.filter(ImageFilter.ModeFilter(size=5))
        contrast = cfg.get("contrast", 1.0)
        if contrast != 1.0:
            img = ImageEnhance.Contrast(img).enhance(contrast)
        saturation = cfg.get("saturation", 1.0)
        if saturation != 1.0:
            img = ImageEnhance.Color(img).enhance(saturation)
        sharpen = cfg.get("sharpen", 1.0)
        if sharpen and sharpen != 1.0:
            img = ImageEnhance.Sharpness(img).enhance(sharpen)
        pixelate = cfg.get("pixelate", 0)
        if pixelate > 0:
            reduced = img.resize((max(1, img.width // pixelate), max(1, img.height // pixelate)), Image.Resampling.NEAREST)
            img = reduced.resize((img.width, img.height), Image.Resampling.NEAREST)
        if cfg.get("neon"):
            bloom = img.filter(ImageFilter.GaussianBlur(6))
            img = Image.blend(img, bloom, 0.45)
        vignette = cfg.get("vignette", 0.0)
        if vignette > 0:
            img = self._apply_vignette(img, vignette)
        grain = cfg.get("grain", 0.0)
        if grain > 0:
            img = self._apply_grain(img, grain)
        return img

    def _apply_vignette(self, img: "Image.Image", strength: float) -> "Image.Image":
        width, height = img.size
        xs = np.linspace(-1.0, 1.0, width)
        ys = np.linspace(-1.0, 1.0, height)
        xx, yy = np.meshgrid(xs, ys)
        dist = np.sqrt(xx * xx + yy * yy)
        mask = np.clip(1.0 - np.maximum(0.0, dist - 0.2) * strength * 1.8, 0.55, 1.0)
        arr = np.asarray(img).astype(np.float64)
        arr *= mask[:, :, None]
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

    def _apply_grain(self, img: "Image.Image", strength: float) -> "Image.Image":
        arr = np.asarray(img).astype(np.float64)
        noise = np.random.default_rng(12345).normal(0.0, 255.0 * strength, size=arr.shape)
        arr = np.clip(arr + noise, 0, 255)
        return Image.fromarray(arr.astype(np.uint8), "RGB")

    def _stamp(self, img: "Image.Image", style: str, negative_tokens: set[str]) -> "Image.Image":
        if negative_tokens & {"watermark", "logo", "text", "signature", "caption"}:
            return img
        try:
            draw = ImageDraw.Draw(img)
            draw.text((8, img.height - 18), f"Resonart · {style}", fill=(255, 255, 255, 110))
        except Exception:
            pass
        return img

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"\w+", text.lower()))

    @staticmethod
    def _prompt_seed(prompt: str) -> int:
        digest = hashlib.sha256(prompt.encode()).digest()
        return int.from_bytes(digest[:4], "big")

    @staticmethod
    def _extract_palette(
        prompt: str,
        negative_tokens: Optional[set[str]] = None,
        palette_hint: str = "",
    ) -> List[Tuple[int, int, int]]:
        tokens = set(re.findall(r"\w+", prompt.lower()))
        blocked = negative_tokens or set()
        if palette_hint and palette_hint in PALETTE_MAP and palette_hint not in blocked:
            return PALETTE_MAP[palette_hint]
        for key, palette in PALETTE_MAP.items():
            if key in tokens and key not in blocked:
                return palette
        for key, palette in PALETTE_MAP.items():
            if key in blocked:
                continue
            if any((key in token or token in key) and token not in blocked for token in tokens):
                return palette
        return PALETTE_MAP["default"]

    @staticmethod
    def _slugify(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


# Backward-compat aliases
AetherCanvasEngine = ResonartEngine
NanobananaEngine = ResonartEngine