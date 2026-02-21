"""CLI module for training, inference, and evaluation."""

from .train import train_model
from .infer import infer_batch
from .eval import evaluate_model
from .sweep import sweep_splits
from src.utils.pickle_io import DDTRDatasetGenerator

__all__ = [
    "train_model",
    "infer_batch",
    "evaluate_model",
    "sweep_splits",
    "DDTRDatasetGenerator",
]
