import pickle
from pathlib import Path

import numpy as np
import torch


def save_pickle(obj, path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str):
    path = Path(path)
    with path.open("rb") as handle:
        return pickle.load(handle)


class DDTRDatasetGenerator:
    """Generate DDTR-format pickle datasets from model predictions."""

    def __init__(self, config_path: str, checkpoint_path: str, model, device):
        """
        Initialize the generator.

        Args:
            config_path: Path to model config
            checkpoint_path: Path to trained model checkpoint
            model: Initialized model instance (with backbone + ms_tcn)
            device: Torch device
        """
        from src.utils import load_config

        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device

        # Load config (lazy import to avoid circular dependencies)
        config_dict = load_config(config_path)
        # Convert dict to simple namespace-like object or keep as dict
        self.config_dict = config_dict

        self.model = model.to(self.device)
        self.model.eval()

    def generate(self, manifest_path: str, output_path: str) -> dict:
        """
        Generate DDTR pickle file from model predictions on manifest.

        Args:
            manifest_path: Path to data manifest (jsonl)
            output_path: Path to save pickle file

        Returns:
            Dictionary with 'num_videos' and 'output_path'
        """
        from torch.utils.data import DataLoader
        from src.datasets import VideoDataset, pad_collate

        cfg = self.config_dict
        
        dataset = VideoDataset(
            manifest_path,
            input_type=cfg.get("model", {}).get("input_type", "features"),
            feature_dim=cfg.get("model", {}).get("feature_dim", 512),
            with_labels=True,
            normalize_features=cfg.get("data", {}).get("normalize_features", False),
            feature_mean=cfg.get("data", {}).get("feature_mean"),
            feature_std=cfg.get("data", {}).get("feature_std"),
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.get("data", {}).get("num_workers", 0),
            collate_fn=pad_collate,
        )

        stochastic = []
        target = []
        num_classes = cfg.get("model", {}).get("num_classes", 50)

        with torch.no_grad():
            for batch in loader:
                inputs = batch["inputs"].to(self.device)
                labels = batch["labels"].cpu().numpy()
                mask = batch["mask"].cpu().numpy()

                logits, _ = self.model(inputs)
                probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)

                length = int(mask[0].sum())
                stochastic.append(probs[0, :length])
                target.append(self._one_hot(labels[0, :length], num_classes))

        output = {"stochastic": stochastic, "target": target}

        # Write pickle
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_pickle(output, str(out_path))

        return {"num_videos": len(stochastic), "output_path": str(out_path)}

    @staticmethod
    def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
        """Convert labels to one-hot encoding."""
        out = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
        out[np.arange(labels.shape[0]), labels] = 1.0
        return out
