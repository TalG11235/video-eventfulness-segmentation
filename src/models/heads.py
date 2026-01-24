import torch
from torch import nn


class DDTRActionHead(nn.Module):
    def __init__(self, in_dim: int, num_actions: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_dim, num_actions)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, T, D]
        x = self.dropout(feats)
        return self.classifier(x)  # logits [B, T, A]

    @staticmethod
    def build_distribution(logits: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=logits)


class FallSegHead(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_dim, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, T, D]
        x = self.dropout(feats)
        return self.classifier(x).squeeze(-1)  # logits [B, T]
