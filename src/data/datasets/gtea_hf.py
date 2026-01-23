import random
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

IGNORE_INDEX = -100

class GTEAHFFeatureDataset(Dataset):
    def __init__(
        self,
        split: str,          # "train" or "test"
        cv_split: int = 1,   # 1..4  (name="split{cv_split}")
        T: int = 128,
        random_start: bool = True,
        binary: bool = True,
        bg_id: int = 0,
        seed: int = 0,
        trust_remote_code: bool = True,  # because dataset uses a loading script
    ):
        assert split in {"train", "test"}
        self.T = T
        self.random_start = random_start
        self.binary = binary
        self.bg_id = bg_id
        self.rng = random.Random(seed)

        ds = load_dataset(
            "dinggd/gtea",
            name=f"split{cv_split}",
            trust_remote_code=trust_remote_code,
        )
        self.ds = ds[split]

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
