#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.export.ddtr_exporter import run_ddtr_inference
from src.export.ddtr_format import save_as_ddtr_pickle
from src.utils import load_config


def _parse_splits(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("--splits must not be empty")
    out = [int(p) for p in parts]
    if any(s <= 0 for s in out):
        raise ValueError(f"--splits must contain positive integers, got {out}")
    return out


def _resolve_best_checkpoint(checkpoints_dir: Path) -> Path:
    best = checkpoints_dir / "best.pt"
    if best.exists():
        return best

    metadata = checkpoints_dir / "metadata.json"
    if metadata.exists():
        try:
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            cand = payload.get("best_checkpoint")
            if cand:
                cand_path = Path(cand)
                if not cand_path.is_absolute():
                    cand_path = checkpoints_dir / cand_path
                if cand_path.exists():
                    return cand_path
        except json.JSONDecodeError:
            pass

    pts = sorted(checkpoints_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if pts:
        return pts[0]

    raise FileNotFoundError(f"No checkpoint found in {checkpoints_dir}")


def _apply_stride(arr: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return arr
    sliced = arr[::stride]
    if sliced.shape[0] == 0:
        return arr[:1]
    return sliced


def _apply_stride_to_item(item: dict[str, Any], stride: int) -> dict[str, Any]:
    if stride <= 1:
        return item
    out = dict(item)
    for key in ("probs", "labels", "pred_labels"):
        if key in out and out[key] is not None:
            out[key] = _apply_stride(np.asarray(out[key]), stride)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build DDTR pickle from an existing segmentation outputs directory by "
            "running split-wise inference on each split val manifest (OOF predictions)."
        )
    )
    parser.add_argument("--outputs-dir", required=True, help="Existing outputs dir, e.g. outputs/50salads-...")
    parser.add_argument("--ddtr-pickle-path", required=True, help="Output DDTR pickle path")
    parser.add_argument("--splits", default="1,2,3,4,5", help="Comma-separated split ids")
    parser.add_argument("--device", default="", help="Optional inference device override")
    parser.add_argument("--temporal-stride", type=int, default=1, help="Downsample factor for probs/labels")
    parser.add_argument("--expected-videos", type=int, default=50, help="Expected total OOF videos")
    parser.add_argument(
        "--trace-index-path",
        default="",
        help="Optional JSONL path recording video->split/checkpoint mapping",
    )
    parser.add_argument(
        "--summary-path",
        default="",
        help="Optional summary JSON path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = _parse_splits(args.splits)
    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists():
        raise FileNotFoundError(f"--outputs-dir does not exist: {outputs_dir}")
    if args.temporal_stride < 1:
        raise ValueError("--temporal-stride must be >= 1")

    device = args.device or None
    all_predictions: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    video_to_split: dict[str, int] = {}
    first_config_path: Path | None = None

    for split_idx in splits:
        split_dir = outputs_dir / f"split_{split_idx}"
        config_path = split_dir / "config.yaml"
        val_manifest_path = split_dir / "manifests" / "val.jsonl"
        checkpoints_dir = split_dir / "checkpoints"

        if not config_path.exists():
            raise FileNotFoundError(f"Missing split config: {config_path}")
        if not val_manifest_path.exists():
            raise FileNotFoundError(f"Missing split val manifest: {val_manifest_path}")
        if not checkpoints_dir.exists():
            raise FileNotFoundError(f"Missing split checkpoints dir: {checkpoints_dir}")

        if first_config_path is None:
            first_config_path = config_path

        checkpoint_path = _resolve_best_checkpoint(checkpoints_dir)
        infer_result = run_ddtr_inference(
            config_path=str(config_path),
            checkpoint_path=str(checkpoint_path),
            manifest_path=str(val_manifest_path),
            device=device,
        )

        split_preds = infer_result["predictions"]
        if len(split_preds) == 0:
            raise ValueError(f"Split {split_idx} produced no predictions from {val_manifest_path}")

        for item in split_preds:
            video_id = str(item["video_id"])
            if video_id in video_to_split:
                prev = video_to_split[video_id]
                raise ValueError(
                    f"Duplicate OOF prediction for {video_id}: split_{prev} and split_{split_idx}"
                )
            video_to_split[video_id] = split_idx

            out_item = _apply_stride_to_item(item, args.temporal_stride)
            probs = np.asarray(out_item["probs"])
            if probs.ndim != 2:
                raise ValueError(f"Invalid probs shape for {video_id}: {probs.shape}")
            if "labels" in out_item and out_item["labels"] is not None:
                labels = np.asarray(out_item["labels"]).reshape(-1)
                if labels.shape[0] != probs.shape[0]:
                    raise ValueError(
                        f"Length mismatch for {video_id}: probs T={probs.shape[0]} labels T={labels.shape[0]}"
                    )

            all_predictions.append(out_item)
            trace_rows.append(
                {
                    "video_id": video_id,
                    "split": split_idx,
                    "checkpoint_path": str(checkpoint_path),
                    "config_path": str(config_path),
                    "manifest_path": str(val_manifest_path),
                    "num_frames": int(probs.shape[0]),
                    "num_classes": int(probs.shape[1]),
                    "temporal_stride": args.temporal_stride,
                }
            )

        print(
            f"split_{split_idx}: checkpoint={checkpoint_path.name} "
            f"val_videos={len(split_preds)}"
        )

    all_predictions.sort(key=lambda x: str(x["video_id"]))
    trace_rows.sort(key=lambda x: str(x["video_id"]))

    if args.expected_videos > 0 and len(all_predictions) != args.expected_videos:
        raise ValueError(
            f"Expected {args.expected_videos} videos, got {len(all_predictions)} from splits={splits}"
        )

    if first_config_path is None:
        raise RuntimeError("No split config found")
    cfg = load_config(str(first_config_path))
    export_result = save_as_ddtr_pickle(
        predictions={"predictions": all_predictions},
        config=cfg,
        output_path=args.ddtr_pickle_path,
    )

    trace_index_path = (
        Path(args.trace_index_path)
        if args.trace_index_path
        else Path(args.ddtr_pickle_path).with_suffix(".trace_index.jsonl")
    )
    trace_index_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_index_path.open("w", encoding="utf-8") as handle:
        for row in trace_rows:
            handle.write(json.dumps(row) + "\n")

    summary = {
        "outputs_dir": str(outputs_dir),
        "splits": splits,
        "expected_videos": args.expected_videos,
        "collected_videos": len(all_predictions),
        "temporal_stride": args.temporal_stride,
        "device_override": args.device,
        "ddtr_export": export_result,
        "trace_index_path": str(trace_index_path),
    }
    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else Path(args.ddtr_pickle_path).with_suffix(".summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Done.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
