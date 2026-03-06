from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class DatasetConverter(ABC):
    dataset_names: tuple[str, ...]

    @abstractmethod
    def prepare_split_manifests(
        self, cfg_split: dict[str, Any], split_idx: int, split_out_dir: Path
    ) -> None:
        """Populate split-specific manifests and update cfg_split in-place."""

