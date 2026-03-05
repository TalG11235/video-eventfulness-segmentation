from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def save_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict, path: Path,
    feature_mean: float | None = None, feature_std: float | None = None,
) -> None:
    """Save model checkpoint with optimizer state, metrics, and normalization stats."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
        },
        path,
    )


def load_checkpoint(model: nn.Module, path: Path, strict: bool = True) -> dict:
    """Load model checkpoint and return metadata including normalization stats."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=strict)
    return {
        "epoch": ckpt.get("epoch"),
        "metrics": ckpt.get("metrics"),
        "feature_mean": ckpt.get("feature_mean"),
        "feature_std": ckpt.get("feature_std"),
    }
