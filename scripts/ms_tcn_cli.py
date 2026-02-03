import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.backbones import ResNetFeatureExtractor
from src.configs import Config
from src.datasets import VideoDataset, pad_collate
from src.models.ms_tcn import MSTCN, MSTCNConfig, edit_score, f1_score, frame_accuracy, ms_tcn_loss
from src.utils import load_config


class MSTCNWithBackbone(nn.Module):
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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


def _build_loader(cfg: Config, split: str) -> DataLoader | None:
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


def _save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict, path: Path):
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


def _get_git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _write_metadata(cfg: Config, out_dir: Path) -> None:
    meta = {
        "model": cfg.model.__dict__,
        "data": cfg.data.__dict__,
        "training": cfg.training.__dict__,
        "ddtr": cfg.ddtr.__dict__,
        "git_commit": _get_git_commit(),
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    smoothing_weight: float,
    compute_tas_metrics: bool = False,
    max_steps: int | None = None,
) -> dict:
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


def train(args: argparse.Namespace) -> None:
    cfg = Config.from_dict(load_config(args.config))
    _set_seed(cfg.training.seed)
    device = _get_device(cfg.training.device)
    print(f"device: {device}, input_type={cfg.model.input_type}")

    model = MSTCNWithBackbone(cfg).to(device)
    train_loader = _build_loader(cfg, "train")
    val_loader = _build_loader(cfg, "val")
    if train_loader is None:
        raise ValueError("Training manifest is missing (data.manifest_train)")
    print(f"train: {len(train_loader.dataset)} samples, batch_size={cfg.data.batch_size}")
    if val_loader is not None:
        print(f"val: {len(val_loader.dataset)} samples, batch_size={cfg.data.batch_size}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    out_dir = Path(cfg.training.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_metadata(cfg, out_dir)
    metrics_path = out_dir / "metrics.jsonl"

    best_val = None
    for epoch in range(1, cfg.training.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            cfg.model.smoothing_weight,
            compute_tas_metrics=False,
        )
        val_metrics = None
        if val_loader is not None:
            val_metrics = _run_epoch(
                model,
                val_loader,
                None,
                device,
                cfg.model.smoothing_weight,
                compute_tas_metrics=True,
            )

        msg = (
            f"epoch {epoch}/{cfg.training.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f}"
        )
        if val_metrics is not None:
            msg += (
                f" val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_f1@10={val_metrics['f1@10']:.2f} "
                f"val_f1@25={val_metrics['f1@25']:.2f} "
                f"val_f1@50={val_metrics['f1@50']:.2f} "
                f"val_edit={val_metrics['edit']:.2f}"
            )
        print(msg)

        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        if val_metrics is not None:
            score = val_metrics["loss"]
            if best_val is None or score < best_val:
                best_val = score
                _save_checkpoint(model, optimizer, epoch, val_metrics, out_dir / "best.pt")

        _save_checkpoint(model, optimizer, epoch, train_metrics, out_dir / "last.pt")


def infer(args: argparse.Namespace) -> None:
    cfg = Config.from_dict(load_config(args.config))
    device = _get_device(cfg.training.device)
    model = MSTCNWithBackbone(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=True)
    model.eval()

    dataset = VideoDataset(
        args.manifest,
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

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            mask = batch["mask"].to(device)
            logits, _ = model(inputs)
            probs = torch.softmax(logits, dim=-1)
            length = int(mask[0].sum().item())
            video_id = batch["video_ids"][0]
            np.save(out_dir / f"{video_id}_probs.npy", probs[0, :length].cpu().numpy().astype(np.float32))
            np.save(out_dir / f"{video_id}_logits.npy", logits[0, :length].cpu().numpy().astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_p = sub.add_parser("train")
    train_p.add_argument("--config", required=True)

    infer_p = sub.add_parser("infer")
    infer_p.add_argument("--config", required=True)
    infer_p.add_argument("--checkpoint", required=True)
    infer_p.add_argument("--manifest", required=True)
    infer_p.add_argument("--output_dir", required=True)

    args = parser.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "infer":
        infer(args)


if __name__ == "__main__":
    main()
