from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.configs import Config
from src.datasets import VideoDataset, pad_collate
from src.utils import load_config
from src.utils.checkpoints import load_checkpoint
from src.utils.model_factory import MSTCNWithBackbone
from src.utils.runtime import get_device


def _infer_num_classes_from_checkpoint(checkpoint_path: Path) -> int | None:
    """Infer output classes from saved MS-TCN checkpoint weights."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state", ckpt)
    if not isinstance(state_dict, dict):
        return None

    # Prefer stage1 head, which is guaranteed in MSTCN checkpoints.
    preferred_suffixes = ("stage1.conv_out.weight", "stage1.conv_out.bias")
    for suffix in preferred_suffixes:
        for key, value in state_dict.items():
            if not key.endswith(suffix):
                continue
            if not isinstance(value, torch.Tensor) or value.numel() == 0:
                continue
            return int(value.shape[0])

    # Fallback: any conv_out head tensor.
    for key, value in state_dict.items():
        if "conv_out" not in key:
            continue
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            continue
        return int(value.shape[0])
    return None


def run_ddtr_inference(
    config_path: str,
    checkpoint_path: str,
    manifest_path: str,
    device: str | None = None,
) -> dict[str, Any]:
    """Run model inference on a manifest and return per-video DDTR-ready predictions.
    
    Normalization stats are loaded from the checkpoint if available, ensuring we use
    the exact stats that were computed during training.
    """
    cfg = Config.from_dict(load_config(config_path))
    inferred_num_classes = _infer_num_classes_from_checkpoint(Path(checkpoint_path))
    if inferred_num_classes is not None and inferred_num_classes != int(cfg.model.num_classes):
        print(
            "Adjusting cfg.model.num_classes from "
            f"{cfg.model.num_classes} to {inferred_num_classes} based on checkpoint"
        )
        cfg.model.num_classes = inferred_num_classes

    runtime_device = get_device(device or cfg.training.device)

    model = MSTCNWithBackbone(cfg).to(runtime_device)
    ckpt_metadata = load_checkpoint(model, Path(checkpoint_path), strict=True)
    
    # Load normalization stats from checkpoint if available
    if cfg.data.normalize_features and ckpt_metadata.get("feature_mean") is not None:
        cfg.data.feature_mean = ckpt_metadata["feature_mean"]
        cfg.data.feature_std = ckpt_metadata["feature_std"]
    
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

    predictions: list[dict[str, Any]] = []
    num_labeled_videos = 0

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(runtime_device)
            mask = batch["mask"].to(runtime_device)
            logits, _ = model(inputs)
            probs = torch.softmax(logits, dim=-1)

            has_labels = "labels" in batch
            for i, video_id in enumerate(batch["video_ids"]):
                length = int(mask[i].sum().item())
                video_probs = probs[i, :length].cpu().numpy().astype(np.float32, copy=False)
                pred_labels = np.argmax(video_probs, axis=-1).astype(np.int64, copy=False)

                pred_item: dict[str, Any] = {
                    "video_id": video_id,
                    "probs": video_probs,
                    "pred_labels": pred_labels,
                }
                if has_labels:
                    labels = batch["labels"][i, :length].cpu().numpy().astype(np.int64, copy=False)
                    pred_item["labels"] = labels
                    num_labeled_videos += 1
                predictions.append(pred_item)

    return {
        "predictions": predictions,
        "num_videos": len(predictions),
        "num_labeled_videos": num_labeled_videos,
        "num_classes": int(cfg.model.num_classes),
    }
