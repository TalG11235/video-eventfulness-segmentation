#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from src.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a DDTR config JSON derived from a split config/manifests."
    )
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--train-manifest-path", required=True)
    parser.add_argument("--test-manifest-path", default="")
    parser.add_argument("--pickle-path", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--ddtr-config-path", required=True)
    parser.add_argument("--ddtr-device", required=True)
    parser.add_argument("--num-epochs", required=True, type=int)
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--num-workers", required=True, type=int)
    parser.add_argument("--num-timesteps", required=True, type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--train-percent", type=float, default=None)
    parser.add_argument("--test-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config_path)
    train_manifest_path = Path(args.train_manifest_path)
    test_manifest_path = Path(args.test_manifest_path) if args.test_manifest_path else None
    pickle_path = Path(args.pickle_path)
    summary_path = Path(args.summary_path)
    ddtr_config_path = Path(args.ddtr_config_path)
    report_path = Path(args.report_path)

    cfg = load_config(str(config_path))
    data_cfg = cfg.get("data", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    data_dir = Path(data_cfg.get("data_dir", ""))
    mapping_path = data_dir / "mapping.txt"

    activity_names = {}
    if mapping_path.exists():
        with mapping_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                idx, label = line.split(maxsplit=1)
                activity_names[str(int(idx))] = label

    base_num_classes = model_cfg.get("num_classes")
    if base_num_classes is None:
        if activity_names:
            base_num_classes = max(int(k) for k in activity_names) + 1
        else:
            raise ValueError("Could not resolve num_classes from config or mapping.txt")
    base_num_classes = int(base_num_classes)

    if not activity_names:
        activity_names = {str(i): f"class_{i}" for i in range(base_num_classes)}

    # DDTR dataset preprocessing appends one extra "pad" class channel.
    ddtr_num_classes = base_num_classes + 1
    activity_names[str(base_num_classes)] = "pad"

    with train_manifest_path.open("r", encoding="utf-8") as train_handle:
        train_count = sum(1 for _ in train_handle)
    if test_manifest_path is not None and test_manifest_path.exists():
        with test_manifest_path.open("r", encoding="utf-8") as test_handle:
            test_count = sum(1 for _ in test_handle)
    else:
        test_count = 0

    total_count = train_count + test_count
    if total_count == 0 and args.train_percent is None:
        raise ValueError("Both train/test manifests are empty.")
    if args.train_percent is not None and not 0.0 < args.train_percent < 1.0:
        raise ValueError("--train-percent must satisfy 0 < train_percent < 1")

    resolved_train_percent = (
        args.train_percent if args.train_percent is not None else train_count / total_count
    )

    payload = {
        "data_path": str(pickle_path),
        "summary_path": str(summary_path),
        "device": args.ddtr_device,
        "parallelize": False,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "num_timesteps": args.num_timesteps,
        "train_percent": resolved_train_percent,
        "num_workers": args.num_workers,
        "test_every": args.test_every,
        "num_classes": ddtr_num_classes,
        "batch_size": args.batch_size,
        "conditional_dropout": 0.1,
        "matrix_dropout": 0,
        "eval_train": False,
        "mode": "cond",
        "predict_on": "original",
        "seed": 42,
        "enable_matrix": True,
        "matrix_type": "pm",
        "activity_names": activity_names,
        "process_discovery_method": "inductive",
    }

    ddtr_config_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.mkdir(parents=True, exist_ok=True)
    with ddtr_config_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    report_path.write_text(
        f"base_num_classes={base_num_classes}\n"
        f"ddtr_num_classes={ddtr_num_classes}\n"
        f"train_manifest_count={train_count}\n"
        f"test_manifest_count={test_count}\n"
        f"train_percent={resolved_train_percent}\n"
        f"mapping_path={mapping_path if mapping_path.exists() else '<not-found>'}\n",
        encoding="utf-8",
    )
    print(f"DDTR config written: {ddtr_config_path}")
    print(f"base_num_classes={base_num_classes}")
    print(f"ddtr_num_classes={ddtr_num_classes}")
    print(f"train_manifest_count={train_count}")
    print(f"test_manifest_count={test_count}")
    print(f"learning_rate={args.learning_rate}")
    print(f"train_percent={resolved_train_percent}")
    print(f"test_every={args.test_every}")
    print(f"batch_size={args.batch_size}")
    print(f"num_workers={args.num_workers}")
    print(f"num_timesteps={args.num_timesteps}")


if __name__ == "__main__":
    main()
