from typing import Optional

import torch
from torch import nn

from src.models.backbone import ResNetFeatureExtractor, MultiScaleTemporalConv
from src.models.heads import DDTRActionHead, FallSegHead


class EventSegmentationModel(nn.Module):
    def __init__(
        self,
        task: str,
        input_type: str,
        feature_dim: int,
        resnet_name: str = "resnet18",
        resnet_pretrained: bool = False,
        temporal_dim: int = 256,
        ms_kernel_sizes=(3, 5, 7),
        ms_dilations=(1, 2, 3),
        num_actions: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if task not in {"ddtr", "fall"}:
            raise ValueError(f"Unknown task: {task}")
        if input_type not in {"frames", "features"}:
            raise ValueError(f"Unknown input_type: {input_type}")

        self.task = task
        self.input_type = input_type
        self.feature_dim = feature_dim

        if input_type == "frames":
            self.backbone = ResNetFeatureExtractor(
                name=resnet_name,
                pretrained=resnet_pretrained,
                out_dim=feature_dim,
            )
            temporal_in = feature_dim
        else:
            self.backbone = nn.Identity()
            temporal_in = feature_dim

        self.temporal = MultiScaleTemporalConv(
            in_dim=temporal_in,
            hidden_dim=temporal_dim,
            kernel_sizes=ms_kernel_sizes,
            dilations=ms_dilations,
            dropout=dropout,
        )

        if task == "ddtr":
            if num_actions is None:
                raise ValueError("num_actions is required for ddtr task")
            self.head = DDTRActionHead(temporal_dim, num_actions, dropout=dropout)
        else:
            self.head = FallSegHead(temporal_dim, dropout=dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: frames [B, T, C, H, W] or features [B, T, D]
        feats = self.backbone(inputs)
        feats = self.temporal(feats)
        return self.head(feats)
