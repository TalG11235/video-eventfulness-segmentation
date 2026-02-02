import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backbones import TemporalBackbone
from src.datasets import VideoDataset, pad_collate
from src.heads import DDTRLogHead
from src.utils import load_config


class DDTRTrainModel(nn.Module):
    def __init__(
        self,
        input_type: str,
        feature_dim: int,
        temporal_dim: int,
        num_actions: int,
        kernel_sizes=(3, 5, 7),
        dilations=(1, 2, 3),
        dropout: float = 0.1,
        resnet_name: str = "resnet18",
        resnet_pretrained: bool = False,
    ):
        super().__init__()
        self.backbone = TemporalBackbone(
            input_type=input_type,
            feature_dim=feature_dim,
            temporal_dim=temporal_dim,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            dropout=dropout,
            resnet_name=resnet_name,
            resnet_pretrained=resnet_pretrained,
        )
        self.head = DDTRLogHead(temporal_dim, num_actions, dropout=dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(inputs)
        return self.head(feats)


def _get_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


def _build_loader(cfg: dict, split: str):
    data_cfg = cfg["data"]
    manifest_key = f"manifest_{split}"
    if manifest_key not in data_cfg:
        return None
    dataset = VideoDataset(
        data_cfg[manifest_key],
        input_type=data_cfg["input_type"],
        feature_dim=cfg["model"]["feature_dim"],
        with_labels=True,
    )
    return DataLoader(
        dataset,
        batch_size=data_cfg.get("batch_size", 1),
        shuffle=split == "train",
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=True,
        collate_fn=pad_collate,
    )


def _ddtr_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    b, t, a = logits.shape
    logits = logits.view(b * t, a)
    labels = labels.view(b * t)
    mask = mask.view(b * t)
    labels = labels.masked_fill(~mask, -100)
    return nn.CrossEntropyLoss(ignore_index=-100)(logits, labels)


def _ddtr_accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == labels) & mask
    return correct.sum().float() / mask.sum().clamp(min=1)


def _run_epoch(model, loader, optimizer, device):
    if loader is None:
        return None
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device)

        logits = model(inputs)
        loss = _ddtr_loss(logits, labels, mask)
        acc = _ddtr_accuracy(logits, labels, mask)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_acc += acc.item()
        total_batches += 1

    return {
        "loss": total_loss / max(total_batches, 1),
        "accuracy": total_acc / max(total_batches, 1),
    }


def _save_checkpoint(model, optimizer, epoch, metrics, path: Path):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = _get_device(cfg.get("training", {}).get("device", "cpu"))

    model_cfg = cfg["model"]
    model = DDTRTrainModel(
        input_type=model_cfg.get("input_type", "features"),
        feature_dim=model_cfg["feature_dim"],
        temporal_dim=model_cfg.get("temporal_dim", 256),
        num_actions=model_cfg["num_actions"],
        kernel_sizes=tuple(model_cfg.get("ms_kernel_sizes", [3, 5, 7])),
        dilations=tuple(model_cfg.get("ms_dilations", [1, 2, 3])),
        dropout=model_cfg.get("dropout", 0.1),
        resnet_name=model_cfg.get("resnet_name", "resnet18"),
        resnet_pretrained=model_cfg.get("resnet_pretrained", False),
    ).to(device)

    train_loader = _build_loader(cfg, "train")
    val_loader = _build_loader(cfg, "val")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("training", {}).get("lr", 1e-4),
        weight_decay=cfg.get("training", {}).get("weight_decay", 1e-4),
    )

    out_dir = Path(cfg.get("training", {}).get("save_dir", "outputs/train"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2)
    metrics_path = out_dir / "metrics.jsonl"

    best_val = None
    epochs = cfg.get("training", {}).get("epochs", 1)
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, device)
        val_metrics = _run_epoch(model, val_loader, None, device)

        if val_metrics is None:
            print(
                f"epoch {epoch}/{epochs} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f}"
            )
        else:
            print(
                f"epoch {epoch}/{epochs} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f}"
            )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        if val_metrics is not None:
            score = val_metrics["loss"]
            if best_val is None or score < best_val:
                best_val = score
                _save_checkpoint(model, optimizer, epoch, val_metrics, out_dir / "best.pt")

        _save_checkpoint(model, optimizer, epoch, train_metrics, out_dir / "last.pt")


if __name__ == "__main__":
    main()
