from __future__ import annotations

import torch
from torch import nn

from src.configs import Config
from src.model.backbone import ResNetFeatureExtractor
from src.model.ms_tcn import MSTCN, MSTCNConfig


class MSTCNWithBackbone(nn.Module):
    """MS-TCN model with optional ResNet feature extraction backbone."""

    def __init__(self, cfg: Config):
        super().__init__()
        model_cfg = cfg.model
        if model_cfg.input_type == "frames":
            self.backbone = ResNetFeatureExtractor(
                name=model_cfg.resnet_name,
                pretrained=model_cfg.resnet_pretrained,
                out_dim=model_cfg.feature_dim,
            )
        else:
            self.backbone = nn.Identity()
        ms_cfg = MSTCNConfig(
            num_stages=model_cfg.num_stages,
            num_layers=model_cfg.num_layers,
            num_f_maps=model_cfg.num_f_maps,
            input_dim=model_cfg.feature_dim,
            num_classes=model_cfg.num_classes,
        )
        self.ms_tcn = MSTCN(ms_cfg, dropout=model_cfg.dropout)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(inputs)
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        return self.ms_tcn(feats)
