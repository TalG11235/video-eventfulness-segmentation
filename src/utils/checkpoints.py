from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def save_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict, path: Path
) -> None:
    """Save model checkpoint with optimizer state and metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(model: nn.Module, path: Path, strict: bool = True) -> dict:
    """Load model checkpoint and return metadata."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=strict)
    return {
        "epoch": ckpt.get("epoch"),
        "metrics": ckpt.get("metrics"),
    }
