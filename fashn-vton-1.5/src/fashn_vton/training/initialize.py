"""Weight initialization for training TryOnModel from scratch."""

import torch
from torch import nn

from fashn_vton.tryon_mmdit import LastLayer, TryOnModel


def init_tryon_model(model: TryOnModel) -> None:
    """
    Initialize a fresh TryOnModel for pretraining.

    Uses Xavier/uniform init on linear layers, small normal init on embeddings,
    and zero init on the final prediction layer (common for flow/diffusion heads).
    PatchEmbed conv weights are already initialized in the module constructor.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    for module in model.modules():
        if isinstance(module, LastLayer):
            nn.init.zeros_(module.linear.weight)
            nn.init.zeros_(module.linear.bias)

    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            if module.elementwise_affine and module.weight is not None:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    if hasattr(model, "t_embedder"):
        for param in model.t_embedder.parameters():
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)
