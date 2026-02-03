import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.configs import Config
from src.datasets import VideoDataset, pad_collate
from src.models.ms_tcn import MSTCN, MSTCNConfig
from src.backbones import ResNetFeatureExtractor
from src.utils import load_config


class _ModelWrapper(torch.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.model.input_type == "frames":
            self.backbone = ResNetFeatureExtractor(
                name=cfg.model.resnet_name,
                pretrained=cfg.model.resnet_pretrained,
                out_dim=cfg.model.feature_dim,
            )
        else:
            self.backbone = torch.nn.Identity()
        ms_cfg = MSTCNConfig(
            num_stages=cfg.model.num_stages,
            num_layers=cfg.model.num_layers,
            num_f_maps=cfg.model.num_f_maps,
            input_dim=cfg.model.feature_dim,
            num_classes=cfg.model.num_classes,
        )
        self.ms_tcn = MSTCN(ms_cfg, dropout=cfg.model.dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(inputs)
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        logits, _ = self.ms_tcn(feats)
        return logits


def _load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)


def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = Config.from_dict(load_config(args.config))
    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    dataset = VideoDataset(
        args.manifest,
        input_type=cfg.model.input_type,
        feature_dim=cfg.model.feature_dim,
        with_labels=True,
        normalize_features=cfg.data.normalize_features,
        feature_mean=cfg.data.feature_mean,
        feature_std=cfg.data.feature_std,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=pad_collate,
    )

    model = _ModelWrapper(cfg).to(device)
    _load_checkpoint(model, args.checkpoint)
    model.eval()

    stochastic: list[np.ndarray] = []
    target: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            labels = batch["labels"].cpu().numpy()
            mask = batch["mask"].cpu().numpy()
            logits = model(inputs)
            probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)

            length = int(mask[0].sum())
            stochastic.append(probs[0, :length])
            target.append(_one_hot(labels[0, :length], cfg.model.num_classes))

    output = {"stochastic": stochastic, "target": target}

    if args.dry_run:
        sample = output["stochastic"][0]
        sample_target = output["target"][0]
        print("dry_run sample:")
        print("  stochastic", type(sample), sample.dtype, sample.shape)
        print("  target", type(sample_target), sample_target.dtype, sample_target.shape)
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
