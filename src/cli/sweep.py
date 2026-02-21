"""Multi-split cross-validation sweep."""

import copy
import json
from pathlib import Path
from typing import Any

from src.utils import load_config

from .train import train_model
from .infer import infer_batch
from .eval import evaluate_model


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


def _ensure_split_manifests(cfg_split: dict[str, Any], split_idx: int, split_out_dir: Path) -> None:
    data_cfg = cfg_split.setdefault("data", {})
    data_dir = Path(data_cfg.get("data_dir", "data/50salads"))
    features_dir = data_dir / "features"
    ground_truth_dir = data_dir / "groundTruth"
    splits_dir = data_dir / "splits"
    mapping_path = data_dir / "mapping.txt"

    train_bundle = splits_dir / f"train.split{split_idx}.bundle"
    val_bundle = splits_dir / f"test.split{split_idx}.bundle"
    required = [features_dir, ground_truth_dir, train_bundle, val_bundle, mapping_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required 50Salads files for manifest generation: " + ", ".join(missing)
        )

    label_to_idx = _load_label_mapping(mapping_path)
    train_ids = _read_split_video_ids(train_bundle)
    val_ids = _read_split_video_ids(val_bundle)

    manifests_dir = split_out_dir / "manifests"
    labels_dir = split_out_dir / "labels"
    train_manifest = manifests_dir / "train.jsonl"
    val_manifest = manifests_dir / "val.jsonl"

    _write_manifest(train_ids, features_dir, ground_truth_dir, labels_dir, label_to_idx, train_manifest)
    _write_manifest(val_ids, features_dir, ground_truth_dir, labels_dir, label_to_idx, val_manifest)

    data_cfg["manifest_train"] = str(train_manifest)
    data_cfg["manifest_val"] = str(val_manifest)
    data_cfg["manifest_test"] = str(val_manifest)


def sweep_splits(
    config_path: str,
    splits: list[int] = None,
    output_base_dir: str = "outputs/cv",
    parallel: bool = False,
) -> dict:
    """
    Run training/inference/evaluation on all cross-validation splits.
    
    Args:
        config_path: Path to base config (will be updated per split)
        splits: List of split indices (e.g. [1,2,3,4,5])
        output_base_dir: Base output directory for all splits
        parallel: Whether to parallelize splits (not yet implemented)
    
    Returns:
        Dictionary with aggregated results across splits
    """
    if splits is None:
        splits = [1, 2, 3, 4, 5]

    base_cfg = load_config(config_path)
    output_base = Path(output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    results = {
        "splits": {},
        "aggregate": {},
    }

    for split_idx in splits:
        print(f"\n{'='*60}")
        print(f"Split {split_idx}/{len(splits)}")
        print("=" * 60)

        # Create split-specific config (in-memory)
        cfg_split = copy.deepcopy(base_cfg)
        
        # Update config for this split
        if "data" not in cfg_split:
            cfg_split["data"] = {}
        cfg_split["data"]["split"] = split_idx
        
        # Update output paths
        split_out_dir = output_base / f"split_{split_idx}"
        if "training" not in cfg_split:
            cfg_split["training"] = {}
        cfg_split["training"]["save_dir"] = str(split_out_dir / "checkpoints")
        _ensure_split_manifests(cfg_split, split_idx, split_out_dir)

        # Write split config temporarily
        split_config_path = split_out_dir / "config.yaml"
        split_config_path.parent.mkdir(parents=True, exist_ok=True)
        import yaml

        with open(split_config_path, "w") as f:
            yaml.dump(cfg_split, f)

        # Train
        print(f"Training split {split_idx}...")
        train_result = train_model(str(split_config_path), str(split_out_dir / "checkpoints"))

        # Infer
        print(f"Inferring split {split_idx}...")
        if "data" in cfg_split and "manifest_val" in cfg_split["data"]:
            infer_result = infer_batch(
                str(split_config_path),
                str(split_out_dir / "checkpoints" / "best.pt"),
                cfg_split["data"]["manifest_val"],
                str(split_out_dir / "predictions"),
            )
        else:
            infer_result = {"num_videos": 0}

        # Evaluate
        print(f"Evaluating split {split_idx}...")
        if "data" in cfg_split and "manifest_val" in cfg_split["data"]:
            eval_result = evaluate_model(
                str(split_config_path),
                str(split_out_dir / "checkpoints" / "best.pt"),
                cfg_split["data"]["manifest_val"],
                str(split_out_dir / "eval"),
            )
        else:
            eval_result = {}

        results["splits"][f"split_{split_idx}"] = {
            "config": cfg_split,
            "train": train_result,
            "infer": infer_result,
            "eval": eval_result,
        }

    # Aggregate results
    eval_results = [
        r["eval"] for r in results["splits"].values() if r["eval"]
    ]
    if eval_results:
        num_splits = len(eval_results)
        metrics = ["frame_acc", "f1@10", "f1@25", "f1@50", "edit"]
        for metric in metrics:
            values = [r.get(metric, 0) for r in eval_results]
            results["aggregate"][metric] = {
                "mean": sum(values) / len(values),
                "std": _std(values),
                "values": values,
            }

    # Write results
    results_path = output_base / "results.json"
    with open(results_path, "w") as f:
        # Convert to serializable format
        results_out = copy.deepcopy(results)
        for split_key in results_out["splits"]:
            results_out["splits"][split_key].pop("config", None)  # Remove config (too large)
        json.dump(results_out, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Sweep complete. Results saved to {results_path}")
    print(json.dumps(results["aggregate"], indent=2))

    return results


def _std(values: list[float]) -> float:
    """Compute standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def main():
    """CLI entry point for cross-validation sweep."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Run cross-validation sweep on specified splits"
    )
    parser.add_argument("config", help="Path to config file (YAML)")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Get splits from config
    splits_config = cfg.get("data", {}).get("splits", "all")

    if splits_config == "all":
        splits = [1, 2, 3, 4, 5]
    elif isinstance(splits_config, list):
        splits = splits_config
    else:
        print(f"Error: data.splits must be 'all' or a list, got {splits_config}")
        sys.exit(1)

    print(f"Running sweep on splits: {splits}")

    # Run sweep
    sweep_splits(
        config_path=args.config,
        splits=splits,
        output_base_dir=cfg.get("training", {}).get("save_dir", "outputs"),
        parallel=False,
    )

    print("✓ Sweep complete")


if __name__ == "__main__":
    main()
