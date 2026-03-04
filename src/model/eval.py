from __future__ import annotations

"""Evaluation utilities for temporal action localization."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.configs import Config
from src.datasets import VideoDataset, pad_collate
from src.model.ms_tcn import f1_score, edit_score, frame_accuracy

# scipy is an optional dependency used for simple post‑processing of predictions
try:
    from scipy.signal import medfilt
except ImportError:  # pragma: no cover - optional
    medfilt = None
from src.utils.checkpoints import load_checkpoint
from src.utils.model_factory import MSTCNWithBackbone
from src.utils.runtime import get_device
from src.utils import load_config

def evaluate_model(
    config_path: str, checkpoint_path: str, manifest_path: str, output_dir: str | None = None
) -> dict:
    """Evaluate model on validation/test manifest."""
    cfg = Config.from_dict(load_config(config_path))
    device = get_device(cfg.training.device)

    model = MSTCNWithBackbone(cfg).to(device)
    load_checkpoint(model, Path(checkpoint_path), strict=True)
    model.eval()

    dataset = VideoDataset(
        manifest_path,
        input_type=cfg.model.input_type,
        feature_dim=cfg.model.feature_dim,
        with_labels=True,
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

    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0
    f1_10_scores = []
    f1_25_scores = []
    f1_50_scores = []
    edit_scores = []
    video_metrics = []

    # read optional post‑processing settings from config
    median_kernel = cfg.get("eval", {}).get("median_filter", 0)
    if median_kernel and medfilt is None:
        print("warning: median_filter requested but scipy not installed; skipping")
        median_kernel = 0

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)
            video_id = batch["video_ids"][0]

            logits, _ = model(inputs)
            acc = frame_accuracy(logits, labels, mask)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labs = labels.cpu().numpy()
            masks = mask.cpu().numpy()

            total_acc += float(acc)
            total_batches += 1

            length = int(masks[0].sum())
            if length > 0:
                pred_seq = preds[0, :length].tolist()
                gt_seq = labs[0, :length].tolist()
                # optionally smooth the prediction sequence before scoring
                if median_kernel and medfilt is not None:
                    import numpy as _np
                    pred_seq = medfilt(_np.array(pred_seq), kernel_size=median_kernel).astype(int).tolist()

                f1_10 = f1_score(pred_seq, gt_seq, 0.1)
                f1_25 = f1_score(pred_seq, gt_seq, 0.25)
                f1_50 = f1_score(pred_seq, gt_seq, 0.5)
                edit = edit_score(pred_seq, gt_seq)

                f1_10_scores.append(f1_10)
                f1_25_scores.append(f1_25)
                f1_50_scores.append(f1_50)
                edit_scores.append(edit)

                video_metrics.append(
                    {
                        "video_id": video_id,
                        "f1@10": f1_10,
                        "f1@25": f1_25,
                        "f1@50": f1_50,
                        "edit": edit,
                    }
                )

    metrics = {
        "frame_acc": total_acc / max(total_batches, 1),
        "f1@10": sum(f1_10_scores) / max(len(f1_10_scores), 1),
        "f1@25": sum(f1_25_scores) / max(len(f1_25_scores), 1),
        "f1@50": sum(f1_50_scores) / max(len(f1_50_scores), 1),
        "edit": sum(edit_scores) / max(len(edit_scores), 1),
    }

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "metrics.json").open("w") as f:
            json.dump(metrics, f, indent=2)
        with (out_dir / "video_metrics.jsonl").open("w") as f:
            for vm in video_metrics:
                f.write(json.dumps(vm) + "\n")

    return metrics
