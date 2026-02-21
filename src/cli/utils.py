import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.backbones import ResNetFeatureExtractor
from src.configs import Config
from src.models.ms_tcn import MSTCN, MSTCNConfig


class MSTCNWithBackbone(nn.Module):
    """MS-TCN model with optional ResNet feature extraction backbone."""

    def __init__(self, cfg: Config):
        super().__init__()
        model_cfg = cfg.model
        if model_cfg.input_type == "frames":
            self.backbone = ResNetFeatureExtractor(
                name=model_cfg.resnet_name,
                pretrained=model_cfg.resnet_pretrained,
                out_dim=model_cfg.feature_dim,
            )
        else:
            self.backbone = nn.Identity()
        ms_cfg = MSTCNConfig(
            num_stages=model_cfg.num_stages,
            num_layers=model_cfg.num_layers,
            num_f_maps=model_cfg.num_f_maps,
            input_dim=model_cfg.feature_dim,
            num_classes=model_cfg.num_classes,
        )
        self.ms_tcn = MSTCN(ms_cfg, dropout=model_cfg.dropout)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(inputs)
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        return self.ms_tcn(feats)


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


def get_git_commit() -> str | None:
    """Get current git commit hash."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def write_metadata(cfg: Config, out_dir: Path) -> None:
    """Write training metadata (config + git commit) to JSON."""
    meta = {
        "model": cfg.model.__dict__,
        "data": cfg.data.__dict__,
        "training": cfg.training.__dict__,
        "ddtr": cfg.ddtr.__dict__,
        "git_commit": get_git_commit(),
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
