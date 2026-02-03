import torch
from torch import nn

try:
    from torchvision import models
except Exception as exc:  # pragma: no cover - optional dependency
    models = None
    _import_error = exc


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, name: str = "resnet18", pretrained: bool = False, out_dim: int | None = None):
        super().__init__()
        if models is None:  # pragma: no cover
            raise ImportError(f"torchvision is required for ResNetFeatureExtractor: {_import_error}")

        if not hasattr(models, name):
            raise ValueError(f"Unknown ResNet model: {name}")

        model_fn = getattr(models, name)
        try:
            if pretrained:
                weights = model_fn(weights="DEFAULT").weights
                backbone = model_fn(weights=weights)
            else:
                backbone = model_fn(weights=None)
        except TypeError:
            backbone = model_fn(pretrained=pretrained)

        self.out_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.project = None
        if out_dim is not None and out_dim != self.out_dim:
            self.project = nn.Linear(self.out_dim, out_dim, bias=False)
            self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If x is already features, return as-is.
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            return x

        if x.dim() == 5:
            b, t, c, h, w = x.shape
            x = x.reshape(b * t, c, h, w)
            feats = self.backbone(x)
            if self.project is not None:
                feats = self.project(feats)
            feats = feats.reshape(b, t, -1)
            return feats

        if x.dim() == 4:
            feats = self.backbone(x)
            if self.project is not None:
                feats = self.project(feats)
            return feats

        raise ValueError(f"Unsupported input shape for ResNetFeatureExtractor: {x.shape}")

    def extract(self, video_tensor: torch.Tensor) -> torch.Tensor:
        feats = self.forward(video_tensor)
        if feats.dim() == 3:
            return feats[0]
        return feats
