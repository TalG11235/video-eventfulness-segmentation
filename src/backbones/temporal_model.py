import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x):
        out = self.stage1(x)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


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
        num_stages: int = 4,
        num_layers: int = 10,
        num_f_maps: int = 64,
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

        self.temporal = MultiStageModel(num_stages, num_layers, num_f_maps, temporal_in, temporal_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: frames [B, T, C, H, W] or features [B, T, D]
        feats = self.frame_backbone(inputs)
        # feats: [B, D, T] - need to transpose for MS-TCN
        feats = feats.transpose(1, 2)  # [B, T, D] -> [B, D, T]
        return self.temporal(feats)  # Returns [num_stages, B, num_classes, T]
