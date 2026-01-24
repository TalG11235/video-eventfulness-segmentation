import bisect
import random

from torch.utils.data import Dataset


class MultiDataset(Dataset):
    def __init__(self, datasets, weights=None, seed: int = 0):
        if not datasets:
            raise ValueError("datasets must be non-empty")
        self.datasets = datasets
        if weights is None:
            weights = [1.0] * len(datasets)
        if len(weights) != len(datasets):
            raise ValueError("weights must match datasets length")
        total = sum(weights)
        self.weights = [w / total for w in weights]
        self.rng = random.Random(seed)
        self._length = sum(len(ds) for ds in datasets)
        self._cumulative = []
        acc = 0.0
        for w in self.weights:
            acc += w
            self._cumulative.append(acc)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        _ = idx  # draw stochastically by weight
        r = self.rng.random()
        ds_idx = bisect.bisect_left(self._cumulative, r)
        ds = self.datasets[ds_idx]
        sample_idx = self.rng.randrange(len(ds))
        return ds[sample_idx]
