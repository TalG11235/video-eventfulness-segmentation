import torch
from torch import nn


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, name: str = "resnet18", pretrained: bool = False, out_dim: int = 512):
        super().__init__()
        try:
            import torchvision.models as models
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("torchvision is required for ResNetFeatureExtractor") from exc

        if not hasattr(models, name):
            raise ValueError(f"Unknown resnet model: {name}")

        base = getattr(models, name)(pretrained=pretrained)
        layers = [
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            base.avgpool,
        ]
        self.backbone = nn.Sequential(*layers)
        in_features = base.fc.in_features
        self.proj = nn.Linear(in_features, out_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [B, T, C, H, W]
        b, t, c, h, w = frames.shape
        x = frames.reshape(b * t, c, h, w)
        x = self.backbone(x)
        x = x.view(b * t, -1)
        x = self.proj(x)
        return x.view(b, t, -1)


class MultiScaleTemporalConv(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        kernel_sizes=(3, 5, 7),
        dilations=(1, 2, 3),
        dropout: float = 0.1,
    ):
        super().__init__()
        if len(kernel_sizes) != len(dilations):
            raise ValueError("kernel_sizes and dilations must have same length")

        self.convs = nn.ModuleList()
        for k, d in zip(kernel_sizes, dilations):
            padding = (k // 2) * d
            self.convs.append(
                nn.Conv1d(in_dim, hidden_dim, kernel_size=k, dilation=d, padding=padding)
            )
        self.fuse = nn.Conv1d(hidden_dim * len(kernel_sizes), hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, T, D]
        x = feats.transpose(1, 2)  # [B, D, T]
        multi = [self.act(conv(x)) for conv in self.convs]
        x = torch.cat(multi, dim=1)
        x = self.fuse(self.dropout(x))
        x = self.act(x)
        return x.transpose(1, 2)


class TemporalBackbone(nn.Module):
    def __init__(
        self,
        input_type: str,
        feature_dim: int,
        temporal_dim: int,
        kernel_sizes=(3, 5, 7),
        dilations=(1, 2, 3),
        dropout: float = 0.1,
        resnet_name: str = "resnet18",
        resnet_pretrained: bool = False,
    ):
        super().__init__()
        if input_type not in {"frames", "features"}:
            raise ValueError(f"Unknown input_type: {input_type}")

        self.input_type = input_type
        if input_type == "frames":
            self.frame_backbone = ResNetFeatureExtractor(
                name=resnet_name,
                pretrained=resnet_pretrained,
                out_dim=feature_dim,
            )
            temporal_in = feature_dim
        else:
            self.frame_backbone = nn.Identity()
            temporal_in = feature_dim

        self.temporal = MultiScaleTemporalConv(
            in_dim=temporal_in,
            hidden_dim=temporal_dim,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            dropout=dropout,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: frames [B, T, C, H, W] or features [B, T, D]
        feats = self.frame_backbone(inputs)
        return self.temporal(feats)
