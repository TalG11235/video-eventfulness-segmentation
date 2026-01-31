"""
Fall Detection Dataset Loader

Supports:
1. Pre-extracted features (numpy/torch files)
2. Raw video frames (extracts features on-the-fly using ResNet)
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


class FallDetectionDataset(Dataset):
    """
    Load fall detection data from manifest files.
    
    Manifest format (JSONL):
    {
        "video_id": "video_001",
        "frames_path": "path/to/frames.npy" or "path/to/video.mp4",
        "features_path": "path/to/features.npy",  # Optional, if using pre-extracted
        "labels_path": "path/to/labels.npy",
        "duration": 300  # frames or seconds
    }
    """

    def __init__(
        self,
        manifest_path: str,
        clip_len: int = 128,
        random_start: bool = True,
        input_type: str = "features",
        pad_value: int = IGNORE_INDEX,
        seed: int = 0,
    ):
        """
        Args:
            manifest_path: Path to JSONL manifest file
            clip_len: Window size in frames
            random_start: Whether to use random window position
            input_type: "features" (pre-extracted) or "frames" (raw video)
            pad_value: Value to use for padding short sequences
            seed: Random seed
        """
        if input_type not in {"frames", "features"}:
            raise ValueError(f"Unknown input_type: {input_type}")

        self.manifest_path = Path(manifest_path)
        self.clip_len = clip_len
        self.random_start = random_start
        self.input_type = input_type
        self.pad_value = pad_value
        self.rng = random.Random(seed)

        # Load manifest
        self.items = []
        if self.manifest_path.exists():
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    self.items.append(json.loads(line))
        else:
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}\n"
                f"Expected format: JSONL with 'video_id', 'labels_path', "
                f"and either 'features_path' (for features) or 'frames_path' (for raw frames)"
            )

    def __len__(self):
        return len(self.items)

    def _load_array(self, path: str) -> np.ndarray:
        """Load numpy or torch array."""
        path = Path(path)
        if path.suffix == ".npy":
            return np.load(path)
        elif path.suffix in {".pt", ".pth"}:
            return torch.load(path).cpu().numpy()
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def _extract_features_from_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        Extract features from raw video frames using ResNet.
        
        Args:
            frames: [T, H, W, C] video frames
            
        Returns:
            features: [T, 512] ResNet features
        """
        try:
            from src.models.backbone import ResNetFeatureExtractor

            if not hasattr(self, "_feature_extractor"):
                self._feature_extractor = ResNetFeatureExtractor(
                    name="resnet18", pretrained=True, out_dim=512
                )
                self._feature_extractor.eval()

            # Convert frames to tensor and normalize
            frames_tensor = torch.from_numpy(frames).float() / 255.0  # [T, H, W, C]
            frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # [T, C, H, W]

            # Extract features
            with torch.no_grad():
                features = self._feature_extractor(frames_tensor)  # [T, 512]

            return features.numpy()
        except Exception as e:
            raise RuntimeError(
                f"Failed to extract features from frames: {e}\n"
                f"Make sure raw frames are in shape [T, H, W, C]"
            )

    def _load_input(self, item: dict) -> np.ndarray:
        """Load either pre-extracted features or raw frames."""
        if self.input_type == "features":
            if "features_path" not in item:
                raise KeyError(
                    "Missing 'features_path' in manifest entry. "
                    "For input_type='features', provide pre-extracted feature paths."
                )
            return self._load_array(item["features_path"]).astype(np.float32)
        else:
            # Raw frames
            if "frames_path" not in item:
                raise KeyError(
                    "Missing 'frames_path' in manifest entry. "
                    "For input_type='frames', provide raw video frame paths."
                )
            frames = self._load_array(item["frames_path"])
            return self._extract_features_from_frames(frames)

    def _load_labels(self, item: dict) -> np.ndarray:
        """Load binary fall labels."""
        if "labels_path" not in item:
            raise KeyError("Missing 'labels_path' in manifest entry")
        return self._load_array(item["labels_path"]).astype(np.int64)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        video_id = item.get("video_id", f"video_{idx}")

        # Load features/frames and labels
        features = self._load_input(item)  # [L, D]
        labels = self._load_labels(item)  # [L]

        # Ensure same length
        L = min(len(features), len(labels))
        features = features[:L]
        labels = labels[:L]

        # Create fixed-length window
        if L >= self.clip_len:
            start = (
                self.rng.randint(0, L - self.clip_len)
                if self.random_start
                else 0
            )
            feat = features[start : start + self.clip_len]
            lab = labels[start : start + self.clip_len]
            mask = np.ones((self.clip_len,), dtype=bool)
        else:
            # Pad short sequences
            start = 0
            pad = self.clip_len - L
            feat = np.concatenate(
                [features, np.zeros((pad, features.shape[1]), dtype=features.dtype)],
                axis=0,
            )
            lab = np.concatenate(
                [labels, np.full((pad,), self.pad_value, dtype=labels.dtype)], axis=0
            )
            mask = np.concatenate(
                [np.ones((L,), dtype=bool), np.zeros((pad,), dtype=bool)], axis=0
            )

        return {
            "features": torch.from_numpy(feat),  # [T, D]
            "labels": torch.from_numpy(lab),  # [T]
            "mask": torch.from_numpy(mask),  # [T]
            "meta": {"video_id": video_id, "start": start, "orig_len": L},
        }
