from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.configs import Config
from src.datasets import VideoDataset, pad_collate
from src.utils.checkpoints import load_checkpoint
from src.utils.model_factory import MSTCNWithBackbone
from src.utils.runtime import get_device
from src.utils import load_config

def infer_batch(
    config_path: str, checkpoint_path: str, manifest_path: str, output_dir: str
) -> dict:
    """Run inference on a manifest of videos."""
    cfg = Config.from_dict(load_config(config_path))
    device = get_device(cfg.training.device)
    
    model = MSTCNWithBackbone(cfg).to(device)
    load_checkpoint(model, Path(checkpoint_path), strict=True)
    model.eval()

    dataset = VideoDataset(
        manifest_path,
        input_type=cfg.model.input_type,
        feature_dim=cfg.model.feature_dim,
        with_labels=False,
        normalize_features=cfg.data.normalize_features,
        feature_mean=cfg.data.feature_mean,
        feature_std=cfg.data.feature_std,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=pad_collate,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    num_videos = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            mask = batch["mask"].to(device)
            logits, _ = model(inputs)
            probs = torch.softmax(logits, dim=-1)
            length = int(mask[0].sum().item())
            video_id = batch["video_ids"][0]
            np.save(
                out_dir / f"{video_id}_probs.npy",
                probs[0, :length].cpu().numpy().astype(np.float32),
            )
            np.save(
                out_dir / f"{video_id}_logits.npy",
                logits[0, :length].cpu().numpy().astype(np.float32),
            )
            num_videos += 1

    return {"num_videos": num_videos, "output_dir": str(out_dir)}
