import random
from torch.utils.data import Dataset

class MultiDataset(Dataset):
    """
    Samples from multiple datasets. Supports:
      - round-robin (deterministic)
      - weighted random (recommended)
    """
    def __init__(self, datasets, weights=None, mode="weighted"):
        self.datasets = datasets
        self.mode = mode
        self.weights = weights or [1.0] * len(datasets)
        self.lengths = [len(d) for d in datasets]
        self.total = sum(self.lengths)

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        if self.mode == "round_robin":
            d_idx = idx % len(self.datasets)
            local = (idx // len(self.datasets)) % len(self.datasets[d_idx])
            return self.datasets[d_idx][local]

        # weighted random
        d_idx = random.choices(range(len(self.datasets)), weights=self.weights, k=1)[0]
        local = random.randint(0, len(self.datasets[d_idx]) - 1)
        return self.datasets[d_idx][local]
