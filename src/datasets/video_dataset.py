import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        input_type: str,
        feature_dim: int | None = None,
        with_labels: bool = False,
        normalize_features: bool = False,
        feature_mean: float | None = None,
        feature_std: float | None = None,
    ):
        if input_type not in {"frames", "features"}:
            raise ValueError(f"Unknown input_type: {input_type}")

        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        self.input_type = input_type
        self.feature_dim = feature_dim
        self.with_labels = with_labels
        self.normalize_features = normalize_features
        self.feature_mean = feature_mean
        self.feature_std = feature_std

        self.items = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                self.items.append(json.loads(line))

    def __len__(self):
        return len(self.items)

    def _load_array(self, path: str) -> np.ndarray:
        path = Path(path)
        if path.suffix == ".npy":
            return np.load(path)
        if path.suffix in {".pt", ".pth"}:
            return torch.load(path).cpu().numpy()
        raise ValueError(f"Unsupported file: {path}")

    def _load_inputs(self, item):
        if self.input_type == "frames":
            key = "frames_path"
        else:
            key = "features_path"
        if key not in item:
            raise KeyError(f"Missing {key} in manifest entry")
        arr = self._load_array(item[key]).astype(np.float32)
        if self.input_type == "features" and arr.ndim == 2 and self.feature_dim is not None:
            if arr.shape[0] == self.feature_dim and arr.shape[1] != self.feature_dim:
                arr = arr.T
            elif arr.shape[1] != self.feature_dim:
                raise ValueError(
                    f"Feature dim mismatch for {item[key]}: got {arr.shape}, "
                    f"expected feature_dim={self.feature_dim}"
                )
        # Apply feature normalization if enabled
        if self.normalize_features and self.input_type == "features":
            if self.feature_mean is not None and self.feature_std is not None:
                arr = (arr - self.feature_mean) / self.feature_std
        return arr

    def _load_labels(self, item):
        if "labels_path" not in item:
            raise KeyError("Missing labels_path in manifest entry")
        arr = self._load_array(item["labels_path"]).astype(np.int64)
        if arr.ndim != 1:
            raise ValueError(f"Labels must be 1D, got {arr.shape} for {item['labels_path']}")
        return arr

    def __getitem__(self, idx):
        item = self.items[idx]
        inputs = self._load_inputs(item)
        output = {
            "inputs": torch.from_numpy(inputs),
            "meta": {
                "video_id": item.get("video_id", str(idx)),
            },
        }
        if self.with_labels or "labels_path" in item:
            labels = self._load_labels(item)
            output["labels"] = torch.from_numpy(labels)
        return output


def pad_collate(batch):
    lengths = [item["inputs"].shape[0] for item in batch]
    max_len = max(lengths) if lengths else 0
    video_ids = [item["meta"]["video_id"] for item in batch]
    has_labels = bool(batch) and "labels" in batch[0]

    if not batch:
        output = {
            "inputs": torch.empty(0),
            "mask": torch.empty(0, dtype=torch.bool),
            "lengths": torch.empty(0, dtype=torch.long),
            "video_ids": [],
        }
        if has_labels:
            output["labels"] = torch.empty(0, dtype=torch.long)
        return output

    first = batch[0]["inputs"]
    dims = first.dim()
    if dims == 2:
        feat_dim = first.shape[1]
        padded = torch.zeros((len(batch), max_len, feat_dim), dtype=first.dtype)
    elif dims == 4:
        c, h, w = first.shape[1:]
        padded = torch.zeros((len(batch), max_len, c, h, w), dtype=first.dtype)
    else:
        raise ValueError(f"Unsupported input dims: {dims}")

    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    labels = None
    if has_labels:
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        inp = item["inputs"]
        t = inp.shape[0]
        padded[i, :t] = inp
        mask[i, :t] = True
        if has_labels:
            lab = item["labels"]
            if lab.shape[0] != t:
                raise ValueError(f"Label length {lab.shape[0]} != input length {t}")
            labels[i, :t] = lab

    output = {
        "inputs": padded,
        "mask": mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "video_ids": video_ids,
    }
    if has_labels:
        output["labels"] = labels
    return output
