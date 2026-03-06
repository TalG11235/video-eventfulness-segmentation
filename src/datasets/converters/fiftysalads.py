from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import DatasetConverter


def _load_label_mapping(mapping_path: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    with mapping_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            idx_str, label = line.split(maxsplit=1)
            mapping[label] = int(idx_str)
    return mapping


def _read_split_video_ids(bundle_path: Path) -> list[str]:
    video_ids = []
    with bundle_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            video_ids.append(Path(line).stem)
    return video_ids


def _ensure_labels_npy(
    video_id: str, ground_truth_dir: Path, labels_dir: Path, label_to_idx: dict[str, int]
) -> Path:
    import numpy as np

    labels_out = labels_dir / f"{video_id}.npy"
    if labels_out.exists():
        return labels_out

    gt_path = ground_truth_dir / f"{video_id}.txt"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")

    label_names = []
    with gt_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            label = line.strip()
            if label:
                label_names.append(label)

    missing = sorted({name for name in label_names if name not in label_to_idx})
    if missing:
        raise ValueError(f"Unknown labels in {gt_path}: {missing[:5]}")

    label_ids = np.asarray([label_to_idx[name] for name in label_names], dtype=np.int64)
    labels_dir.mkdir(parents=True, exist_ok=True)
    np.save(labels_out, label_ids)
    return labels_out


def _write_manifest(
    video_ids: list[str],
    features_dir: Path,
    ground_truth_dir: Path,
    labels_dir: Path,
    label_to_idx: dict[str, int],
    manifest_path: Path,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for video_id in video_ids:
            features_path = features_dir / f"{video_id}.npy"
            if not features_path.exists():
                raise FileNotFoundError(f"Feature file not found: {features_path}")
            labels_path = _ensure_labels_npy(video_id, ground_truth_dir, labels_dir, label_to_idx)
            item = {
                "video_id": video_id,
                "features_path": str(features_path),
                "labels_path": str(labels_path),
            }
            handle.write(json.dumps(item) + "\n")


class FiftySaladsConverter(DatasetConverter):
    dataset_names: tuple[str, ...] = ("50salads", "50_salads")

    def prepare_split_manifests(
        self, cfg_split: dict[str, Any], split_idx: int, split_out_dir: Path
    ) -> None:
        data_cfg = cfg_split.setdefault("data", {})
        model_cfg = cfg_split.setdefault("model", {})
        data_dir = Path(data_cfg.get("data_dir", "data/50salads"))
        features_dir = data_dir / "features"
        ground_truth_dir = data_dir / "groundTruth"
        splits_dir = data_dir / "splits"
        mapping_path = data_dir / "mapping.txt"

        train_bundle = splits_dir / f"train.split{split_idx}.bundle"
        val_bundle = splits_dir / f"test.split{split_idx}.bundle"
        required = [features_dir, ground_truth_dir, train_bundle, val_bundle, mapping_path]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing required 50Salads files for manifest generation: " + ", ".join(missing)
            )

        label_to_idx = _load_label_mapping(mapping_path)
        inferred_num_classes = max(label_to_idx.values()) + 1 if label_to_idx else 0
        configured_num_classes = model_cfg.get("num_classes")
        if (
            configured_num_classes is not None
            and int(configured_num_classes) != inferred_num_classes
        ):
            raise ValueError(
                "Configured model.num_classes does not match 50Salads mapping.txt: "
                f"got {configured_num_classes}, inferred {inferred_num_classes}"
            )
        model_cfg["num_classes"] = inferred_num_classes

        train_ids = _read_split_video_ids(train_bundle)
        val_ids = _read_split_video_ids(val_bundle)

        manifests_dir = split_out_dir / "manifests"
        labels_dir = split_out_dir / "labels"
        train_manifest = manifests_dir / "train.jsonl"
        val_manifest = manifests_dir / "val.jsonl"

        _write_manifest(
            train_ids, features_dir, ground_truth_dir, labels_dir, label_to_idx, train_manifest
        )
        _write_manifest(
            val_ids, features_dir, ground_truth_dir, labels_dir, label_to_idx, val_manifest
        )

        data_cfg["manifest_train"] = str(train_manifest)
        data_cfg["manifest_val"] = str(val_manifest)
        data_cfg["manifest_test"] = str(val_manifest)


_CONVERTER = FiftySaladsConverter()


def prepare_split_manifests(cfg_split: dict[str, Any], split_idx: int, split_out_dir: Path) -> None:
    _CONVERTER.prepare_split_manifests(cfg_split, split_idx, split_out_dir)
