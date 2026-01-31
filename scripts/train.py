import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import get_dataset, IGNORE_INDEX
from src.models import EventSegmentationModel


def load_config(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_loader(cfg: dict, split: str):
    data_cfg = cfg["data"]
    dataset_name = data_cfg["dataset"]
    
    # Handle datasets with different initialization signatures
    if dataset_name == "gtea_hf":
        # GTEA HF uses cross-validation splits instead of manifest files
        cv_split = data_cfg.get("cv_split", 1)
        ds = get_dataset(
            dataset_name,
            split=split,
            cv_split=cv_split,
            T=data_cfg["clip_len"],
            random_start=data_cfg.get("random_start", True) if split == "train" else False,
            binary=data_cfg.get("binary", True),
            bg_id=data_cfg.get("bg_id", 0),
            seed=data_cfg.get("seed", 0),
        )
    elif dataset_name == "fall_detection":
        # Fall detection uses manifest files, supports both features and frames
        manifest_key = f"manifest_{split}"
        if manifest_key not in data_cfg:
            return None
        ds = get_dataset(
            dataset_name,
            manifest_path=data_cfg[manifest_key],
            clip_len=data_cfg["clip_len"],
            random_start=data_cfg.get("random_start", True) if split == "train" else False,
            input_type=data_cfg["input_type"],
            seed=data_cfg.get("seed", 0),
        )
    else:
        # Manifest-based datasets (ddtr_logs, fall_seg)
        manifest_key = f"manifest_{split}"
        if manifest_key not in data_cfg:
            return None
        ds = get_dataset(
            dataset_name,
            manifest_path=data_cfg[manifest_key],
            clip_len=data_cfg["clip_len"],
            random_start=data_cfg.get("random_start", True) if split == "train" else False,
            input_type=data_cfg["input_type"],
            seed=data_cfg.get("seed", 0),
        )
    
    return DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=split == "train",
        num_workers=cfg["training"].get("num_workers", 0),
        pin_memory=True,
    )
    
    return DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=split == "train",
        num_workers=cfg["training"].get("num_workers", 0),
        pin_memory=True,
    )


def ddtr_loss(logits, labels, mask):
    b, t, a = logits.shape
    logits = logits.view(b * t, a)
    labels = labels.view(b * t)
    mask = mask.view(b * t)
    labels = labels.masked_fill(~mask, IGNORE_INDEX)
    return nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)(logits, labels)


def fall_loss(logits, labels, mask):
    # logits/labels: [B, T]
    labels = labels.float()
    loss = nn.BCEWithLogitsLoss(reduction="none")(logits, labels)
    loss = loss * mask.float()
    return loss.sum() / mask.sum().clamp(min=1)


def ddtr_accuracy(logits, labels, mask):
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == labels) & mask
    return correct.sum().float() / mask.sum().clamp(min=1)


def fall_accuracy(logits, labels, mask):
    preds = (torch.sigmoid(logits) > 0.5).long()
    correct = (preds == labels) & mask
    return correct.sum().float() / mask.sum().clamp(min=1)


def run_epoch(model, loader, task, optimizer=None, device="cpu"):
    if loader is None:
        return None
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    for batch in loader:
        # Handle different batch formats
        if "inputs" in batch:
            inputs = batch["inputs"].to(device)
        elif "features" in batch:
            inputs = batch["features"].to(device)
        else:
            raise KeyError("Batch must contain either 'inputs' or 'features'")
        
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device)

        logits = model(inputs)
        if task == "ddtr":
            loss = ddtr_loss(logits, labels, mask)
            acc = ddtr_accuracy(logits, labels, mask)
        else:
            # fall, gtea_hf, and fall_detection are all binary segmentation tasks
            loss = fall_loss(logits, labels, mask)
            acc = fall_accuracy(logits, labels, mask)

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


def save_checkpoint(model, optimizer, epoch, metrics, path: Path):
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

    # Set device, falling back to CPU if CUDA is not available
    device_str = cfg["training"].get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    model_cfg = cfg["model"]
    model = EventSegmentationModel(
        task=cfg["task"],
        input_type=model_cfg["input_type"],
        feature_dim=model_cfg["feature_dim"],
        resnet_name=model_cfg.get("resnet_name", "resnet18"),
        resnet_pretrained=model_cfg.get("resnet_pretrained", False),
        temporal_dim=model_cfg.get("temporal_dim", 256),
        ms_kernel_sizes=tuple(model_cfg.get("ms_kernel_sizes", [3, 5, 7])),
        ms_dilations=tuple(model_cfg.get("ms_dilations", [1, 2, 3])),
        num_actions=model_cfg.get("num_actions"),
        dropout=model_cfg.get("dropout", 0.1),
    ).to(device)

    train_loader = build_loader(cfg, "train")
    val_loader = build_loader(cfg, "val")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"].get("lr", 1e-4),
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )

    out_dir = Path(cfg["training"].get("save_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2)

    best_val = None
    epochs = cfg["training"].get("epochs", 1)
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, cfg["task"], optimizer, device)
        val_metrics = run_epoch(model, val_loader, cfg["task"], None, device)

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

        if val_metrics is not None:
            score = val_metrics["loss"]
            if best_val is None or score < best_val:
                best_val = score
                save_checkpoint(model, optimizer, epoch, val_metrics, out_dir / "best.pt")

        save_checkpoint(model, optimizer, epoch, train_metrics, out_dir / "last.pt")


if __name__ == "__main__":
    main()
