import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, manifest_path: str, input_type: str):
        if input_type not in {"frames", "features"}:
            raise ValueError(f"Unknown input_type: {input_type}")

        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        self.input_type = input_type

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
        return self._load_array(item[key]).astype(np.float32)

    def __getitem__(self, idx):
        item = self.items[idx]
        inputs = self._load_inputs(item)
        return {
            "inputs": torch.from_numpy(inputs),
            "meta": {
                "video_id": item.get("video_id", str(idx)),
            },
        }


def pad_collate(batch):
    lengths = [item["inputs"].shape[0] for item in batch]
    max_len = max(lengths) if lengths else 0
    video_ids = [item["meta"]["video_id"] for item in batch]

    if not batch:
        return {
            "inputs": torch.empty(0),
            "mask": torch.empty(0, dtype=torch.bool),
            "lengths": torch.empty(0, dtype=torch.long),
            "video_ids": [],
        }

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

    for i, item in enumerate(batch):
        inp = item["inputs"]
        t = inp.shape[0]
        padded[i, :t] = inp
        mask[i, :t] = True

    return {
        "inputs": padded,
        "mask": mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "video_ids": video_ids,
    }
