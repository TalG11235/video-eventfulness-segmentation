import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.clip_sampler import sample_clip

IGNORE_INDEX = -100


class ManifestVideoDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        clip_len: int,
        random_start: bool,
        input_type: str,
        pad_value: int = IGNORE_INDEX,
        seed: int = 0,
    ):
        if input_type not in {"frames", "features"}:
            raise ValueError(f"Unknown input_type: {input_type}")

        self.manifest_path = Path(manifest_path)
        self.clip_len = clip_len
        self.random_start = random_start
        self.input_type = input_type
        self.pad_value = pad_value
        self.rng = random.Random(seed)

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

    def _load_frames_or_features(self, item):
        if self.input_type == "frames":
            key = "frames_path"
        else:
            key = "features_path"
        if key not in item:
            raise KeyError(f"Missing {key} in manifest entry")
        return self._load_array(item[key]).astype(np.float32)

    def _load_labels(self, item):
        if "labels_path" not in item:
            raise KeyError("Missing labels_path in manifest entry")
        return self._load_array(item["labels_path"]).astype(np.int64)

    def _sample(self, features, labels):
        if self.input_type == "frames":
            length = min(len(features), len(labels))
            features = features[:length]
            labels = labels[:length]
            if length >= self.clip_len:
                start = self.rng.randint(0, length - self.clip_len) if self.random_start else 0
                end = start + self.clip_len
                feat = features[start:end]
                lab = labels[start:end]
                mask = np.ones((self.clip_len,), dtype=bool)
            else:
                start = 0
                pad = self.clip_len - length
                feat = np.concatenate(
                    [features, np.zeros((pad,) + features.shape[1:], dtype=features.dtype)],
                    axis=0,
                )
                lab = np.concatenate(
                    [labels, np.full((pad,), self.pad_value, dtype=labels.dtype)], axis=0
                )
                mask = np.concatenate(
                    [np.ones((length,), dtype=bool), np.zeros((pad,), dtype=bool)], axis=0
                )
            return feat, lab, mask, start, length

        return sample_clip(
            features,
            labels,
            self.clip_len,
            self.random_start,
            self.pad_value,
            self.rng,
        )

    def __getitem__(self, idx):
        item = self.items[idx]
        inputs = self._load_frames_or_features(item)
        labels = self._load_labels(item)
        inputs, labels, mask, start, length = self._sample(inputs, labels)

        return {
            "inputs": torch.from_numpy(inputs),
            "labels": torch.from_numpy(labels),
            "mask": torch.from_numpy(mask),
            "meta": {
                "video_id": item.get("video_id", str(idx)),
                "start": start,
                "orig_len": length,
            },
        }
