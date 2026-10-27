"""Full training checkpoints (model, optimizer, scheduler, step)."""

import json
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from fashn_vton.utils.checkpoint import save_checkpoint


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler],
    global_step: int,
    epoch: int,
    args: dict[str, Any],
    ema_state: Optional[dict] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "global_step": global_step,
        "epoch": epoch,
        "args": args,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "ema": ema_state,
    }
    torch.save(payload, path)

    save_checkpoint(model.state_dict(), str(path.with_suffix(".model.safetensors")))
    if ema_state is not None:
        save_checkpoint(ema_state, str(path.with_suffix(".ema.safetensors")))

    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump({"global_step": global_step, "epoch": epoch, "args": args}, f, indent=2)


def load_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
    device: torch.device,
) -> dict[str, Any]:
    path = Path(path)
    payload = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])

    return {
        "global_step": int(payload.get("global_step", 0)),
        "epoch": int(payload.get("epoch", 0)),
        "args": payload.get("args", {}),
        "ema": payload.get("ema"),
    }
