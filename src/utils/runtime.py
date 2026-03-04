from __future__ import annotations

import random
import subprocess

import numpy as np
import torch

from src.configs import Config
from src.datasets import compute_feature_normalization_stats


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str) -> torch.device:
    """Get torch device, falling back to CPU if CUDA unavailable."""
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


def get_git_commit() -> str | None:
    """Get current git commit hash."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def ensure_feature_normalization_stats(
    cfg: Config,
    train_manifest_path: str | None = None,
    log: bool = False,
) -> None:
    if (
        cfg.model.input_type != "features"
        or not cfg.data.normalize_features
        or (cfg.data.feature_mean is not None and cfg.data.feature_std is not None)
    ):
        return

    manifest_path = train_manifest_path or cfg.data.manifest_train
    if manifest_path is None:
        raise ValueError(
            "normalize_features=true but stats are missing and manifest_train unavailable. "
            "Please ensure stats are pre-computed or manifest_train is in config."
        )

    cfg.data.feature_mean, cfg.data.feature_std = compute_feature_normalization_stats(
        manifest_path,
        feature_dim=cfg.model.feature_dim,
    )
    if log:
        print(
            "Computed feature normalization stats from train manifest: "
            f"mean={cfg.data.feature_mean:.6f}, std={cfg.data.feature_std:.6f}"
        )
