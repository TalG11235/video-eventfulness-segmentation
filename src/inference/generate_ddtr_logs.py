import argparse
from pathlib import Path

import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.backbones import TemporalBackbone
from src.datasets import VideoDataset, pad_collate
from src.heads import DDTRLogHead
from src.utils import load_config, load_pickle, save_pickle


class DDTRLogGenerator(nn.Module):
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


def _load_checkpoint(model: nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            state = ckpt["model_state"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)


def _compute_stats(probs: np.ndarray, mask: np.ndarray) -> dict:
    valid = probs[mask]
    if valid.size == 0:
        return {"mean": 0.0, "var": 0.0, "entropy": 0.0}

    mean = float(valid.mean())
    var = float(valid.var())
    eps = 1e-8
    entropy = float((-valid * np.log(valid + eps)).sum(axis=-1).mean())
    return {"mean": mean, "var": var, "entropy": entropy}


def _format_output(output: dict, cfg: dict) -> dict:
    ddtr_cfg = cfg.get("ddtr", {})
    key_map = ddtr_cfg.get("key_map", {})
    remapped = {}
    for key, value in output.items():
        if value is None:
            continue
        remapped[key_map.get(key, key)] = value

    template_path = ddtr_cfg.get("template_pkl")
    if template_path:
        template = load_pickle(template_path)
        if isinstance(template, dict):
            template.update(remapped)
            return template
    return remapped


def _validate_output(output: dict) -> None:
    required = ["video_ids", "lengths", "probs"]
    for key in required:
        if key not in output:
            raise KeyError(f"Missing required key in output: {key}")

    video_ids = output["video_ids"]
    lengths = output["lengths"]
    probs = output["probs"]

    if len(video_ids) != len(lengths):
        raise ValueError("video_ids and lengths length mismatch")
    if probs.shape[0] != len(video_ids):
        raise ValueError("probs batch dim must match number of videos")


def _build_manifest_from_dir(input_dir: Path) -> Path:
    manifest_path = input_dir / "__autogen_manifest__.jsonl"
    entries = []
    for path in sorted(input_dir.glob("*.npy")):
        entries.append({"video_id": path.stem, "features_path": str(path)})
    for path in sorted(input_dir.glob("*.pt")):
        entries.append({"video_id": path.stem, "features_path": str(path)})
    for path in sorted(input_dir.glob("*.pth")):
        entries.append({"video_id": path.stem, "features_path": str(path)})

    if not entries:
        raise ValueError(f"No feature files found in {input_dir}")

    with manifest_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")

    return manifest_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    input_path = Path(args.input)
    if input_path.is_dir():
        manifest_path = _build_manifest_from_dir(input_path)
        input_type = "features"
    else:
        manifest_path = input_path
        input_type = data_cfg.get("input_type", model_cfg.get("input_type", "features"))

    dataset = VideoDataset(str(manifest_path), input_type=input_type)
    loader = DataLoader(
        dataset,
        batch_size=data_cfg.get("batch_size", 1),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 0),
        collate_fn=pad_collate,
    )

    device_str = data_cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    model = DDTRLogGenerator(
        input_type=input_type,
        feature_dim=model_cfg.get("feature_dim", 512),
        temporal_dim=model_cfg.get("temporal_dim", 256),
        num_actions=model_cfg["num_actions"],
        kernel_sizes=tuple(model_cfg.get("ms_kernel_sizes", [3, 5, 7])),
        dilations=tuple(model_cfg.get("ms_dilations", [1, 2, 3])),
        dropout=model_cfg.get("dropout", 0.1),
        resnet_name=model_cfg.get("resnet_name", "resnet18"),
        resnet_pretrained=model_cfg.get("resnet_pretrained", False),
    ).to(device)

    _load_checkpoint(model, args.checkpoint)
    model.eval()

    all_probs = []
    all_logits = []
    all_video_ids = []
    all_lengths = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            mask = batch["mask"].to(device)
            logits = model(inputs)
            probs = DDTRLogHead.logits_to_probs(logits)

            for i in range(inputs.shape[0]):
                length = int(mask[i].sum().item())
                all_lengths.append(length)
                all_video_ids.append(batch["video_ids"][i])
                all_probs.append(probs[i, :length].cpu().numpy().astype(np.float32))
                all_logits.append(logits[i, :length].cpu().numpy().astype(np.float32))

    if not all_lengths:
        raise ValueError("No videos found in input")

    max_len = max(all_lengths)
    num_actions = all_probs[0].shape[-1]
    probs_padded = np.zeros((len(all_probs), max_len, num_actions), dtype=np.float32)
    logits_padded = np.zeros((len(all_logits), max_len, num_actions), dtype=np.float32)
    mask = np.zeros((len(all_probs), max_len), dtype=bool)

    for i, (p, l) in enumerate(zip(all_probs, all_lengths)):
        probs_padded[i, :l] = p
        logits_padded[i, :l] = all_logits[i]
        mask[i, :l] = True

    output = {
        "video_ids": all_video_ids,
        "lengths": np.asarray(all_lengths, dtype=np.int64),
        "probs": probs_padded,
        "mask": mask,
        "logits": logits_padded,
    }

    stats = _compute_stats(probs_padded, mask)
    print(
        "log_stats: "
        f"mean={stats['mean']:.6f} "
        f"var={stats['var']:.6f} "
        f"entropy={stats['entropy']:.6f}"
    )

    _validate_output(output)
    formatted = _format_output(output, cfg)

    save_pickle(formatted, args.output)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
