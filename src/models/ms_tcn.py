import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class MSTCNConfig:
    num_stages: int = 4
    num_layers: int = 10
    num_f_maps: int = 64
    input_dim: int = 2048
    num_classes: int = 0
    smoothing_weight: float = 0.15


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation: int, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels, 3, padding=dilation, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class SingleStageModel(nn.Module):
    def __init__(self, num_layers: int, num_f_maps: int, dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [
                copy.deepcopy(DilatedResidualLayer(2**i, num_f_maps, num_f_maps, dropout))
                for i in range(num_layers)
            ]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class MultiStageModel(nn.Module):
    def __init__(
        self,
        num_stages: int,
        num_layers: int,
        num_f_maps: int,
        dim: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes, dropout)
        self.stages = nn.ModuleList(
            [
                copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes, dropout))
                for _ in range(num_stages - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stage1(x)
        outputs = out.unsqueeze(0)
        for stage in self.stages:
            out = stage(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class MSTCN(nn.Module):
    def __init__(self, cfg: MSTCNConfig, dropout: float = 0.5):
        super().__init__()
        if cfg.num_classes <= 0:
            raise ValueError("MSTCNConfig.num_classes must be set to a positive value")
        self.cfg = cfg
        self.model = MultiStageModel(
            cfg.num_stages, cfg.num_layers, cfg.num_f_maps, cfg.input_dim, cfg.num_classes, dropout
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError(f"Expected input of shape [B, T, D] or [T, D], got {x.shape}")
        x = x.transpose(1, 2)
        stage_outputs = self.model(x)
        logits = stage_outputs[-1].transpose(1, 2)
        return logits, stage_outputs

    @staticmethod
    def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)


def ms_tcn_loss(
    stage_outputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    smoothing_weight: float = 0.15,
) -> torch.Tensor:
    # stage_outputs: [S, B, C, T], targets: [B, T], mask: [B, T]
    num_stages, batch, num_classes, seq_len = stage_outputs.shape
    targets = targets.clone()
    targets = targets.masked_fill(~mask, -100)

    ce_loss = 0.0
    for s in range(num_stages):
        logits = stage_outputs[s].transpose(1, 2).reshape(batch * seq_len, num_classes)
        ce_loss = ce_loss + nn.CrossEntropyLoss(ignore_index=-100)(logits, targets.reshape(-1))
    ce_loss = ce_loss / num_stages

    smoothing = 0.0
    for s in range(num_stages):
        log_probs = F.log_softmax(stage_outputs[s], dim=1)
        diff = log_probs[:, :, 1:] - log_probs[:, :, :-1]
        mse = torch.clamp(diff.pow(2), min=0.0, max=16.0)
        valid = mask[:, 1:].unsqueeze(1).float()
        smoothing = smoothing + (mse * valid).mean()
    smoothing = smoothing / num_stages

    return ce_loss + smoothing_weight * smoothing


def frame_accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == labels) & mask
    return (correct.sum().float() / mask.sum().clamp(min=1)).item()


def _segment_labels(labels: list[int]) -> list[tuple[int, int, int]]:
    segments = []
    if not labels:
        return segments
    prev = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != prev:
            segments.append((prev, start, i - 1))
            prev = labels[i]
            start = i
    segments.append((prev, start, len(labels) - 1))
    return segments


def edit_score(pred: list[int], gt: list[int]) -> float:
    pred_seq = [s[0] for s in _segment_labels(pred)]
    gt_seq = [s[0] for s in _segment_labels(gt)]
    n = len(pred_seq)
    m = len(gt_seq)
    if n == 0 and m == 0:
        return 100.0
    if n == 0 or m == 0:
        return 0.0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if pred_seq[i - 1] == gt_seq[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    dist = dp[n][m]
    score = (1.0 - dist / max(n, m)) * 100.0
    return max(score, 0.0)


def f1_score(pred: list[int], gt: list[int], overlap_threshold: float) -> float:
    pred_segs = _segment_labels(pred)
    gt_segs = _segment_labels(gt)
    if not pred_segs and not gt_segs:
        return 100.0
    if not pred_segs or not gt_segs:
        return 0.0
    matched_gt = [False] * len(gt_segs)
    tp = 0
    fp = 0
    for p_label, p_start, p_end in pred_segs:
        best_iou = 0.0
        best_idx = -1
        for i, (g_label, g_start, g_end) in enumerate(gt_segs):
            if matched_gt[i] or p_label != g_label:
                continue
            inter = max(0, min(p_end, g_end) - max(p_start, g_start) + 1)
            union = (p_end - p_start + 1) + (g_end - g_start + 1) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_iou >= overlap_threshold:
            tp += 1
            matched_gt[best_idx] = True
        else:
            fp += 1
    fn = matched_gt.count(False)
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return 100.0 * (2 * tp) / denom
