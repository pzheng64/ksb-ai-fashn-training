"""Exponential moving average of model weights."""

from copy import deepcopy

import torch
from torch import nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999, device: torch.device | None = None):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = decay
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
        if device is not None:
            self.ema_model.to(device)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_params = dict(self.ema_model.named_parameters())
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            ema_param = ema_params[name]
            ema_param.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        return self.ema_model.state_dict()
