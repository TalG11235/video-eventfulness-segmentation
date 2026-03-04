from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .fiftysalads import prepare_split_manifests as prepare_fiftysalads_split_manifests

SplitManifestPreparer = Callable[[dict[str, Any], int, Path], None]

_CONVERTERS: dict[str, SplitManifestPreparer] = {
    "50salads": prepare_fiftysalads_split_manifests,
    "50_salads": prepare_fiftysalads_split_manifests,
}


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

    _CONVERTERS[dataset_name](cfg_split, split_idx, split_out_dir)


__all__ = ["prepare_split_manifests"]
