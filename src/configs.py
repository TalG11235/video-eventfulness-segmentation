from __future__ import annotations

"""Typed configuration objects for CLI entry points."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    input_type: str = "features"
    feature_dim: int = 2048
    resnet_name: str = "resnet18"
    resnet_pretrained: bool = False
    num_stages: int = 4
    num_layers: int = 10
    num_f_maps: int = 64
    num_classes: int = 1
    dropout: float = 0.5
    smoothing_weight: float = 0.15


@dataclass
class DataConfig:
    dataset: str = ""
    data_dir: str = ""
    split: Any = None
    manifest_train: str | None = None
    manifest_val: str | None = None
    manifest_test: str | None = None
    normalize_features: bool = False
    feature_mean: Any = None
    feature_std: Any = None
    batch_size: int = 4
    num_workers: int = 0
    seed: int = 0
    splits: Any = None


@dataclass
class TrainingConfig:
    batch_size: int = 4
    epochs: int = 50
    lr: float = 5e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    device: str = "cpu"
    save_dir: str = "outputs"
    seed: int = 0


@dataclass
class DDTRConfig:
    smoothing_weight: float = 0.15


@dataclass
class Config:
    task: str = "ddtr"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ddtr: DDTRConfig = field(default_factory=DDTRConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        model = ModelConfig(**raw.get("model", {}))
        data = DataConfig(**raw.get("data", {}))
        training = TrainingConfig(**raw.get("training", {}))
        ddtr = DDTRConfig(**raw.get("ddtr", {}))
        task = raw.get("task", "ddtr")
        return cls(task=task, model=model, data=data, training=training, ddtr=ddtr)
