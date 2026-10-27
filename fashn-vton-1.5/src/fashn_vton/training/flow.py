"""Rectified-flow training helpers."""

import math
from typing import Optional

import torch
import torch.nn.functional as F

from fashn_vton.tryon_mmdit import TryOnModel


def time_shift(mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
    """Same schedule biasing as inference (`utils.sampling.time_shift`)."""
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def sample_timesteps(
    batch_size: int,
    device: torch.device,
    *,
    time_shift_mu: Optional[float] = 1.5,
) -> torch.Tensor:
    """
    Sample training timesteps in (0, 1].

    Uses uniform noise in (0, 1] then optional time shift to emphasize mid/high noise.
    """
    t = torch.rand(batch_size, device=device).clamp(min=1e-4, max=1.0)
    if time_shift_mu is not None:
        t = time_shift(time_shift_mu, 1.0, t)
    return t


def rectified_flow_loss(
    model: TryOnModel,
    target: torch.Tensor,
    *,
    ca_images: torch.Tensor,
    garment_images: torch.Tensor,
    person_poses: torch.Tensor,
    garment_poses: torch.Tensor,
    garment_categories: torch.Tensor,
    cond_mask: torch.Tensor,
    time_shift_mu: Optional[float] = 1.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Flow-matching loss aligned with Euler sampling in `TryOnPipeline._sample`.

    Straight-path interpolation: x_t = (1 - t) * noise + t * target, v* = target - noise.
    """
    noise = torch.randn_like(target)
    t = sample_timesteps(target.shape[0], target.device, time_shift_mu=time_shift_mu)
    t_view = t.view(-1, 1, 1, 1)

    x_t = (1.0 - t_view) * noise + t_view * target
    v_target = target - noise

    pred = model(
        x_t,
        t,
        ca_images=ca_images,
        garment_images=garment_images,
        person_poses=person_poses,
        garment_poses=garment_poses,
        garment_categories=garment_categories,
        mask=cond_mask,
    )["x"]

    loss = F.mse_loss(pred, v_target)
    metrics = {
        "loss": float(loss.detach().cpu()),
        "t_mean": float(t.mean().detach().cpu()),
    }
    return loss, metrics
