import random
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

IGNORE_INDEX = -100

class GTEAHFFeatureDataset(Dataset):
    def __init__(
        self,
        split: str,          # "train", "test", or "val" (val maps to test)
        cv_split: int = 1,   # 1..4  (name="split{cv_split}")
        T: int = 128,
        random_start: bool = True,
        binary: bool = True,
        bg_id: int = 0,
        seed: int = 0,
    ):
        # Map "val" to "test" since GTEA only has train/test splits
        if split == "val":
            split = "test"
        
        assert split in {"train", "test"}
        self.T = T
        self.random_start = random_start
        self.binary = binary
        self.bg_id = bg_id
        self.rng = random.Random(seed)

        # Try to load from HuggingFace with fallback
        try:
            # Try with trust_remote_code for older datasets versions
            ds = load_dataset(
                "dinggd/gtea",
                name=f"split{cv_split}",
                trust_remote_code=True,
            )
            self.ds = ds[split]
        except Exception as e1:
            # If that fails, try without trust_remote_code
            try:
                ds = load_dataset(
                    "dinggd/gtea",
                    name=f"split{cv_split}",
                )
                self.ds = ds[split]
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load GTEA dataset. The HuggingFace dataset 'dinggd/gtea' "
                    f"uses a deprecated loading script format.\n"
                    f"Error 1 (with trust_remote_code): {e1}\n"
                    f"Error 2 (without trust_remote_code): {e2}\n\n"
                    f"Try installing an older version of datasets:\n"
                    f"  pip install 'datasets==2.14.7'\n"
                    f"Or contact the dataset author to convert to standard Parquet format."
                ) from e2
        
        # Set format to python to avoid PyArrow formatting issues
        self.ds.set_format("python")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        vid = ex["video_id"]

        feat = np.asarray(ex["video_feature"], dtype=np.float32)  # [L, D]
        lab  = np.asarray(ex["video_label"], dtype=np.int64)      # [L]
        L = min(len(feat), len(lab))
        feat, lab = feat[:L], lab[:L]

        if self.binary:
            lab = (lab != self.bg_id).astype(np.int64)  # 0/1

        # fixed-length window
        if L >= self.T:
            start = self.rng.randint(0, L - self.T) if self.random_start else 0
            feat = feat[start:start + self.T]
            lab  = lab[start:start + self.T]
            mask = np.ones((self.T,), dtype=bool)
        else:
            start = 0
            pad = self.T - L
            feat = np.concatenate([feat, np.zeros((pad, feat.shape[1]), np.float32)], axis=0)
            lab  = np.concatenate([lab, np.full((pad,), IGNORE_INDEX, np.int64)], axis=0)
            mask = np.concatenate([np.ones((L,), bool), np.zeros((pad,), bool)], axis=0)

        return {
            "features": torch.from_numpy(feat),   # [T, D]
            "labels": torch.from_numpy(lab),      # [T]
            "mask": torch.from_numpy(mask),       # [T]
            "meta": {"video_id": vid, "start": start, "orig_len": L},
        }
