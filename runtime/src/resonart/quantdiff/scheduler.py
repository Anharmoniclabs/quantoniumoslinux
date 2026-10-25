"""
QuantDiff DDIM noise scheduler.

Implements DDIM (Denoising Diffusion Implicit Models) sampling entirely
in NumPy/PyTorch — zero dependency on HuggingFace diffusers or any
third-party scheduling library.

Reference: Song et al., "Denoising Diffusion Implicit Models", ICLR 2021.
           https://arxiv.org/abs/2010.02502
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


class DDIMScheduler:
    """
    DDIM noise scheduler for fast deterministic sampling.

    Supports linear and cosine (scaled-linear) beta schedules.
    Uses the DDIM update rule (eta=0 → fully deterministic).
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.prediction_type = prediction_type

        if beta_schedule == "linear":
            betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        elif beta_schedule == "scaled_linear":
            # Stable Diffusion default: sqrt(β) linearly spaced, then squared.
            betas = (
                np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=np.float32)
                ** 2
            )
        elif beta_schedule == "cosine":
            # Nichol & Dhariwal 2021 cosine schedule.
            steps = num_train_timesteps + 1
            x = np.linspace(0, num_train_timesteps, steps, dtype=np.float32)
            ac = np.cos(((x / num_train_timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
            ac = ac / ac[0]
            betas = np.clip(1 - ac[1:] / ac[:-1], 0.0, 0.999).astype(np.float32)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule!r}")

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod: np.ndarray = np.cumprod(self.alphas)
        # ᾱ at t=-1 defined as 1.0 (no noise at the very start)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        self._timesteps: Optional[np.ndarray] = None
        self._num_inference_steps: Optional[int] = None

    # ------------------------------------------------------------------
    # Scheduler configuration
    # ------------------------------------------------------------------

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Compute the sub-sampled DDIM timestep sequence."""
        self._num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        # Descending: T-1 … 0
        ts = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy()
        self._timesteps = ts.astype(np.int64)

    @property
    def timesteps(self) -> np.ndarray:
        if self._timesteps is None:
            raise RuntimeError("Call set_timesteps() before iterating.")
        return self._timesteps

    # ------------------------------------------------------------------
    # Noise addition (forward process)
    # ------------------------------------------------------------------

    def add_noise(
        self,
        original_samples: np.ndarray,
        noise: np.ndarray,
        timestep: int,
    ) -> np.ndarray:
        """q(x_t | x_0): add noise at training timestep *t*."""
        sa = np.float32(self.sqrt_alphas_cumprod[timestep])
        sb = np.float32(self.sqrt_one_minus_alphas_cumprod[timestep])
        return (sa * original_samples + sb * noise).astype(np.float32)

    def noise_image_strength(
        self, x0: np.ndarray, strength: float, rng: Optional[np.random.Generator] = None
    ) -> tuple[np.ndarray, int]:
        """
        Add noise corresponding to a given diffusion *strength* ∈ (0, 1].

        Returns (noisy_x, start_timestep) so the denoising loop can begin
        at the correct point for img2img-style refinement.
        """
        t = int(round(strength * (self.num_train_timesteps - 1)))
        t = max(0, min(t, self.num_train_timesteps - 1))
        noise = (rng.standard_normal(x0.shape) if rng else np.random.randn(*x0.shape)).astype(
            np.float32
        )
        return self.add_noise(x0, noise, t), t

    # ------------------------------------------------------------------
    # DDIM denoising step (reverse process)
    # ------------------------------------------------------------------

    def step(
        self,
        model_output: np.ndarray,
        timestep: int,
        sample: np.ndarray,
        eta: float = 0.0,
    ) -> np.ndarray:
        """
        DDIM reverse step: compute x_{t-1} from x_t and the model's
        predicted noise (epsilon prediction) or predicted x_0.

        Parameters
        ----------
        model_output : predicted noise εθ(x_t, t)   [shape: same as sample]
        timestep     : current timestep index t
        sample       : current noisy sample x_t
        eta          : stochasticity parameter (0 = fully deterministic DDIM)

        Returns
        -------
        x_{t-1}
        """
        t = int(timestep)
        prev_t = self._prev_timestep(t)

        alpha_t = float(self.alphas_cumprod[t])
        alpha_prev = float(self.alphas_cumprod[prev_t] if prev_t >= 0 else 1.0)
        beta_t = 1.0 - alpha_t

        # --- Predict x_0 from model output ---------------------------------
        if self.prediction_type == "epsilon":
            pred_x0 = (sample - np.float32(math.sqrt(beta_t)) * model_output) / np.float32(math.sqrt(alpha_t))
        elif self.prediction_type == "v_prediction":
            # v-parameterization: v = sqrt(α) * ε - sqrt(1-α) * x_0
            pred_x0 = np.float32(math.sqrt(alpha_t)) * sample - np.float32(math.sqrt(beta_t)) * model_output
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type!r}")

        if self.clip_sample:
            pred_x0 = np.clip(pred_x0, -1.0, 1.0)

        # --- DDIM direction toward x_t -------------------------------------
        # σ_t for the stochastic component
        sigma_t = (
            eta
            * math.sqrt((1 - alpha_prev) / (1 - alpha_t))
            * math.sqrt(1 - alpha_t / alpha_prev)
            if alpha_prev < alpha_t
            else 0.0
        )
        coeff_xt = np.float32(math.sqrt(max(1 - alpha_prev - sigma_t**2, 0.0)))
        prev_sample = np.float32(math.sqrt(alpha_prev)) * pred_x0 + coeff_xt * model_output

        if eta > 0:
            noise = np.random.randn(*sample.shape).astype(np.float32)
            prev_sample = prev_sample + np.float32(sigma_t) * noise

        return prev_sample.astype(np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prev_timestep(self, t: int) -> int:
        if self._num_inference_steps is None:
            return max(0, t - 1)
        step_size = self.num_train_timesteps // self._num_inference_steps
        return max(t - step_size, 0)
