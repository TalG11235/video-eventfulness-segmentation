from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from src.configs import Config


def _resolve_prediction_items(predictions: Any) -> list[dict[str, Any]]:
    if isinstance(predictions, Mapping):
        raw_items = predictions.get("predictions")
        if raw_items is None:
            raise KeyError("Predictions mapping must contain a 'predictions' key.")
    else:
        raw_items = predictions

    if not isinstance(raw_items, Iterable):
        raise TypeError("Predictions must be an iterable of per-video prediction dictionaries.")

    items = list(raw_items)
    if not all(isinstance(item, Mapping) for item in items):
        raise TypeError("Each prediction item must be a mapping.")
    return [dict(item) for item in items]


def _resolve_num_classes(config: Any) -> int | None:
    if isinstance(config, Config):
        return int(config.model.num_classes)
    if isinstance(config, Mapping):
        model_cfg = config.get("model")
        if isinstance(model_cfg, Mapping):
            num_classes = model_cfg.get("num_classes")
            if num_classes is not None:
                return int(num_classes)
    return None


def _to_one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    if labels.ndim != 1:
        raise ValueError(f"Labels must be 1D, got shape {labels.shape}.")
    if labels.size == 0:
        return np.zeros((0, num_classes), dtype=np.float64)
    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError(
            f"Labels out of bounds for num_classes={num_classes}: "
            f"min={int(labels.min())}, max={int(labels.max())}"
        )

    one_hot = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
    one_hot[np.arange(labels.shape[0]), labels] = 1.0
    return one_hot


def save_as_ddtr_pickle(predictions: Any, config: Any, output_path: str) -> dict[str, Any]:
    """
    Save predictions in DDTR pickle format.

    Output schema:
    - stochastic: list[np.ndarray[T, C], dtype=float32]
    - target: list[np.ndarray[T, C], dtype=float64]
    """
    items = _resolve_prediction_items(predictions)
    config_num_classes = _resolve_num_classes(config)

    stochastic: list[np.ndarray] = []
    target: list[np.ndarray] = []

    used_gt_labels = 0
    used_pred_labels = 0
    resolved_num_classes = config_num_classes

    for item in items:
        if "probs" not in item:
            raise KeyError("Each prediction item must contain a 'probs' field.")

        probs = np.asarray(item["probs"], dtype=np.float32)
        if probs.ndim != 2:
            raise ValueError(f"Prediction probs must be 2D [T, C], got shape {probs.shape}.")

        if resolved_num_classes is None:
            resolved_num_classes = int(probs.shape[1])
        if probs.shape[1] != resolved_num_classes:
            raise ValueError(
                f"Class dimension mismatch: expected {resolved_num_classes}, got {probs.shape[1]}."
            )

        stochastic.append(probs.astype(np.float32, copy=False))

        if item.get("labels") is not None:
            labels = np.asarray(item["labels"], dtype=np.int64).reshape(-1)
            used_gt_labels += 1
        else:
            fallback = item.get("pred_labels")
            if fallback is None:
                fallback = np.argmax(probs, axis=-1)
            labels = np.asarray(fallback, dtype=np.int64).reshape(-1)
            used_pred_labels += 1

        if labels.shape[0] != probs.shape[0]:
            raise ValueError(
                f"Label length mismatch for item {item.get('video_id', '<unknown>')}: "
                f"labels={labels.shape[0]}, probs={probs.shape[0]}"
            )
        target.append(_to_one_hot(labels, resolved_num_classes))

    if resolved_num_classes is None:
        raise ValueError("Could not resolve num_classes from config or prediction tensors.")

    payload = {"stochastic": stochastic, "target": target}
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return {
        "output_path": str(out_path),
        "num_videos": len(stochastic),
        "num_classes": resolved_num_classes,
        "num_targets_from_labels": used_gt_labels,
        "num_targets_from_predictions": used_pred_labels,
    }
