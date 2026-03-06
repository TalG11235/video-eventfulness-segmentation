from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import DatasetConverter
from .fiftysalads import FiftySaladsConverter

_CONVERTER_INSTANCES: tuple[DatasetConverter, ...] = (FiftySaladsConverter(),)

_CONVERTERS: dict[str, DatasetConverter] = {}
for _converter in _CONVERTER_INSTANCES:
    for _dataset_name in _converter.dataset_names:
        normalized_name = _dataset_name.strip().lower()
        if normalized_name in _CONVERTERS:
            raise ValueError(f"Duplicate converter registration for dataset {normalized_name!r}")
        _CONVERTERS[normalized_name] = _converter


def prepare_split_manifests(cfg_split: dict[str, Any], split_idx: int, split_out_dir: Path) -> None:
    data_cfg = cfg_split.setdefault("data", {})
    dataset_name = str(data_cfg.get("dataset", "")).strip().lower()
    if not dataset_name:
        raise ValueError("Missing data.dataset; set it to choose a dataset converter")

    if dataset_name not in _CONVERTERS:
        known = ", ".join(sorted(_CONVERTERS))
        raise ValueError(
            f"No dataset converter registered for {dataset_name!r}. Available converters: {known}"
        )

    _CONVERTERS[dataset_name].prepare_split_manifests(cfg_split, split_idx, split_out_dir)


__all__ = ["DatasetConverter", "prepare_split_manifests"]
