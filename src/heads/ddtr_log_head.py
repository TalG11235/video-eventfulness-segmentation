import torch
from torch import nn


class MSTCNHead(nn.Module):
    def __init__(self, num_stages: int = 4):
        super().__init__()
        self.num_stages = num_stages

    def forward(self, stage_outputs: torch.Tensor) -> torch.Tensor:
        # stage_outputs: [num_stages, B, num_classes, T]
        # Return the final stage output: [B, T, num_classes]
        return stage_outputs[-1].transpose(1, 2)  # [B, num_classes, T] -> [B, T, num_classes]

    @staticmethod
    def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)

    @staticmethod
    def build_distribution(logits: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=logits)
