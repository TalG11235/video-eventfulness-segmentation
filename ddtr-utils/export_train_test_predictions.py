#!/usr/bin/env python3
import argparse
import json

import numpy as np

from src.export.ddtr_exporter import run_ddtr_inference
from src.export.ddtr_format import save_as_ddtr_pickle
from src.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export train/test manifests into a DDTR-compatible pickle."
    )
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--train-manifest-path", required=True)
    parser.add_argument("--test-manifest-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or None

    train_result = run_ddtr_inference(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        manifest_path=args.train_manifest_path,
        device=device,
    )
    test_result = run_ddtr_inference(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        manifest_path=args.test_manifest_path,
        device=device,
    )

    train_items = list(train_result["predictions"])
    test_items = list(test_result["predictions"])
    train_count = len(train_items)
    test_count = len(test_items)
    total_count = train_count + test_count

    if total_count == 0:
        raise ValueError("No predictions exported from train/test manifests.")

    # DDTR always applies a random train_test_split() internally.
    # Arrange samples so DDTR's split (seed=42, train_percent=0.8) maps
    # exactly to the original train/test manifests.
    train_percent = train_count / total_count
    seed = 42
    n_train = train_count
    n_test = total_count - n_train
    rng = np.random.RandomState(seed)
    perm = rng.permutation(total_count)
    ddtr_test_idx = perm[:n_test]
    ddtr_train_idx = perm[n_test : (n_test + n_train)]

    if len(ddtr_train_idx) != train_count or len(ddtr_test_idx) != test_count:
        raise ValueError(
            f"Unexpected DDTR split sizes: train={len(ddtr_train_idx)} test={len(ddtr_test_idx)} "
            f"expected train={train_count} test={test_count}"
        )

    combined_predictions = [None] * total_count
    for idx, item in zip(ddtr_train_idx.tolist(), train_items):
        combined_predictions[idx] = item
    for idx, item in zip(ddtr_test_idx.tolist(), test_items):
        combined_predictions[idx] = item

    if any(item is None for item in combined_predictions):
        raise ValueError("Failed to assign all combined prediction slots.")

    export_result = save_as_ddtr_pickle(
        predictions={"predictions": combined_predictions},
        config=load_config(args.config_path),
        output_path=args.output_path,
    )

    summary = {
        "train_manifest": args.train_manifest_path,
        "test_manifest": args.test_manifest_path,
        "checkpoint": args.checkpoint_path,
        "device": device,
        "train_videos": train_count,
        "test_videos": test_count,
        "num_videos": export_result["num_videos"],
        "num_classes": export_result["num_classes"],
        "num_targets_from_labels": export_result["num_targets_from_labels"],
        "num_targets_from_predictions": export_result["num_targets_from_predictions"],
        "train_percent_for_ddtr_cfg": train_percent,
        "seed_for_ddtr_cfg": seed,
        "output_path": export_result["output_path"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
