"""Multi-split cross-validation sweep."""

import copy
import json
from pathlib import Path
from typing import Any

from src.utils import load_config

from .train import train_model
from .infer import infer_batch
from .eval import evaluate_model


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
