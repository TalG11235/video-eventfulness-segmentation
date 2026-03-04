from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.configs import Config
from src.datasets import VideoDataset, compute_feature_normalization_stats, pad_collate
from src.model.ms_tcn import ms_tcn_loss, frame_accuracy, f1_score, edit_score
from src.utils.checkpoints import save_checkpoint
from src.utils.metadata import write_metadata
from src.utils.model_factory import MSTCNWithBackbone
from src.utils.runtime import get_device, set_seed
from src.utils import load_config, save_two_row_stripe_plot

def build_loader(cfg: Config, split: str) -> DataLoader | None:
    """Build train/val dataloader from manifest in config."""
    data_cfg = cfg.data
    manifest = getattr(data_cfg, f"manifest_{split}", None)
    if manifest is None:
        return None
    dataset = VideoDataset(
        manifest,
        input_type=cfg.model.input_type,
        feature_dim=cfg.model.feature_dim,
        with_labels=True,
        normalize_features=data_cfg.normalize_features,
        feature_mean=data_cfg.feature_mean,
        feature_std=data_cfg.feature_std,
    )
    generator = torch.Generator()
    generator.manual_seed(cfg.training.seed)
    return DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        shuffle=split == "train",
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        collate_fn=pad_collate,
        generator=generator,
    )


def write_val_comparisons(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    comparisons_root: Path,
    epoch: int,
    num_classes: int,
    max_videos: int = 3,
    max_width_px: int = 60000,
    pixels_per_frame: int = 2,
) -> None:
    """Visualize predictions vs ground truth for first validation samples."""
    if not callable(save_two_row_stripe_plot):
        return
    comp_dir = comparisons_root / f"epoch{epoch:03d}"
    comp_dir.mkdir(parents=True, exist_ok=True)
    max_frames_per_plot = max(1, max_width_px // max(pixels_per_frame, 1))
    plotted_videos = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if plotted_videos >= max_videos:
                break
            if "labels" not in batch:
                continue
            inputs = batch["inputs"].to(device)
            labels = batch["labels"].cpu().numpy()
            mask = batch["mask"].cpu().numpy()
            logits, _ = model(inputs)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            for i, video_id in enumerate(batch["video_ids"]):
                if plotted_videos >= max_videos:
                    break
                length = int(mask[i].sum())
                if length <= 0:
                    continue
                gt = labels[i, :length]
                pr = preds[i, :length]
                num_parts = max(1, (length + max_frames_per_plot - 1) // max_frames_per_plot)
                for part_idx in range(num_parts):
                    start = part_idx * max_frames_per_plot
                    end = min(length, (part_idx + 1) * max_frames_per_plot)
                    gt_part = gt[start:end]
                    pr_part = pr[start:end]
                    if gt_part.size == 0:
                        continue
                    suffix = f"_part{part_idx + 1:03d}" if num_parts > 1 else ""
                    out_path = comp_dir / f"{video_id}_epoch{epoch:03d}{suffix}.png"
                    title = f"{video_id} [{start}:{end}]"
                    save_two_row_stripe_plot(
                        gt_part,
                        pr_part,
                        out_path,
                        num_classes=num_classes,
                        title=title,
                    )
                plotted_videos += 1


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    smoothing_weight: float,
    compute_tas_metrics: bool = False,
    max_steps: int | None = None,
) -> dict:
    """Run training or validation epoch."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0
    tas_f1_10 = []
    tas_f1_25 = []
    tas_f1_50 = []
    tas_edit = []

    data_iter = iter(loader)
    step = 0
    while True:
        if max_steps is not None and step >= max_steps:
            break
        step += 1
        try:
            batch = next(data_iter)
        except StopIteration:
            break
        inputs = batch["inputs"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device)

        logits, stage_outputs = model(inputs)
        loss = ms_tcn_loss(stage_outputs, labels, mask, smoothing_weight=smoothing_weight)
        acc = frame_accuracy(logits, labels, mask)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_acc += float(acc)
        total_batches += 1

        if compute_tas_metrics:
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labs = labels.cpu().numpy()
            masks = mask.cpu().numpy()
            for i in range(preds.shape[0]):
                length = int(masks[i].sum())
                pred_seq = preds[i, :length].tolist()
                gt_seq = labs[i, :length].tolist()
                tas_edit.append(edit_score(pred_seq, gt_seq))
                tas_f1_10.append(f1_score(pred_seq, gt_seq, 0.1))
                tas_f1_25.append(f1_score(pred_seq, gt_seq, 0.25))
                tas_f1_50.append(f1_score(pred_seq, gt_seq, 0.5))
    metrics = {
        "loss": total_loss / max(total_batches, 1),
        "accuracy": total_acc / max(total_batches, 1),
    }
    if compute_tas_metrics:
        metrics.update(
            {
                "f1@10": sum(tas_f1_10) / max(len(tas_f1_10), 1),
                "f1@25": sum(tas_f1_25) / max(len(tas_f1_25), 1),
                "f1@50": sum(tas_f1_50) / max(len(tas_f1_50), 1),
                "edit": sum(tas_edit) / max(len(tas_edit), 1),
            }
        )
    return metrics


def _print_epoch_table_header(has_val: bool) -> None:
    columns = [
        ("epoch", 10),
        ("train_loss", 12),
        ("train_acc", 11),
    ]
    if has_val:
        columns.extend(
            [
                ("val_loss", 12),
                ("val_acc", 10),
                ("val_f1@10", 11),
                ("val_f1@25", 11),
                ("val_f1@50", 11),
                ("val_edit", 11),
            ]
        )

    header = " ".join(f"{name:>{width}}" for name, width in columns)
    divider = " ".join("-" * width for _, width in columns)
    print(header)
    print(divider)


def _print_epoch_table_row(
    epoch: int, total_epochs: int, train_metrics: dict, val_metrics: dict | None
) -> None:
    values = [
        (f"{epoch}/{total_epochs}", 10),
        (f"{train_metrics['loss']:.4f}", 12),
        (f"{train_metrics['accuracy']:.4f}", 11),
    ]
    if val_metrics is not None:
        values.extend(
            [
                (f"{val_metrics['loss']:.4f}", 12),
                (f"{val_metrics['accuracy']:.4f}", 10),
                (f"{val_metrics['f1@10']:.2f}", 11),
                (f"{val_metrics['f1@25']:.2f}", 11),
                (f"{val_metrics['f1@50']:.2f}", 11),
                (f"{val_metrics['edit']:.2f}", 11),
            ]
        )

    print(" ".join(f"{value:>{width}}" for value, width in values))


def train_model(
    config_path: str, output_dir: str | None = None, comparisons_dir: str | None = None
) -> dict:
    """Train MS-TCN model on training data."""
    cfg = Config.from_dict(load_config(config_path))
    if (
        cfg.model.input_type == "features"
        and cfg.data.normalize_features
        and (cfg.data.feature_mean is None or cfg.data.feature_std is None)
        and cfg.data.manifest_train is not None
    ):
        cfg.data.feature_mean, cfg.data.feature_std = compute_feature_normalization_stats(
            cfg.data.manifest_train,
            feature_dim=cfg.model.feature_dim,
        )
        print(
            "Computed feature normalization stats from train manifest: "
            f"mean={cfg.data.feature_mean:.6f}, std={cfg.data.feature_std:.6f}"
        )
    set_seed(cfg.training.seed)
    device = get_device(cfg.training.device)
    print(f"device: {device}, input_type={cfg.model.input_type}")

    model = MSTCNWithBackbone(cfg).to(device)
    train_loader = build_loader(cfg, "train")
    val_loader = build_loader(cfg, "val")
    if train_loader is None:
        raise ValueError("Training manifest is missing (data.manifest_train)")
    print(f"train: {len(train_loader.dataset)} samples, batch_size={cfg.data.batch_size}")
    if val_loader is not None:
        print(f"val: {len(val_loader.dataset)} samples, batch_size={cfg.data.batch_size}")
    if val_loader is not None and not callable(save_two_row_stripe_plot):
        print("save_two_row_stripe_plot unavailable; skipping validation comparison plots.")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )

    # optional learning‑rate scheduler defined in config.training.lr_scheduler
    scheduler = None
    sched_cfg = cfg.training.lr_scheduler
    if sched_cfg is not None:
        typ = sched_cfg.type
        if typ == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=sched_cfg.factor,
                patience=sched_cfg.patience,
                mode=sched_cfg.mode,
            )
        elif typ == "step_lr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=sched_cfg.step_size,
                gamma=sched_cfg.gamma,
            )
        # add other schedulers here if needed

    out_dir = Path(output_dir or cfg.training.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comparisons_root = Path(comparisons_dir) if comparisons_dir is not None else out_dir.parent / "comparisons"
    comparisons_root.mkdir(parents=True, exist_ok=True)
    write_metadata(cfg, out_dir)
    metrics_path = out_dir / "metrics.jsonl"
    printed_epoch_header = False

    # choose which validation metric to use for checkpointing; loss by default
    checkpoint_metric = cfg.training.checkpoint_metric
    # if the chosen metric should be maximized set this to True (edit, f1 etc.)
    checkpoint_metric_higher_is_better = cfg.training.checkpoint_metric_higher_is_better

    best_val = None
    for epoch in range(1, cfg.training.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            cfg.model.smoothing_weight,
            compute_tas_metrics=False,
        )
        val_metrics = None
        if val_loader is not None:
            val_metrics = run_epoch(
                model,
                val_loader,
                None,
                device,
                cfg.model.smoothing_weight,
                compute_tas_metrics=True,
            )

        if not printed_epoch_header:
            _print_epoch_table_header(val_metrics is not None)
            printed_epoch_header = True
        _print_epoch_table_row(epoch, cfg.training.epochs, train_metrics, val_metrics)

        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        if val_metrics is not None:
            # update scheduler if we have one
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics.get("loss", 0.0))
                else:
                    scheduler.step()

            score = val_metrics.get(checkpoint_metric, val_metrics.get("loss"))
            if best_val is None or (
                score > best_val if checkpoint_metric_higher_is_better else score < best_val
            ):
                best_val = score
                save_checkpoint(model, optimizer, epoch, val_metrics, out_dir / "best.pt")

        save_checkpoint(model, optimizer, epoch, train_metrics, out_dir / "last.pt")

        if val_loader is not None:
            write_val_comparisons(
                model, val_loader, device, comparisons_root, epoch, cfg.model.num_classes
            )

    return {"best_val": best_val, "output_dir": str(out_dir)}
