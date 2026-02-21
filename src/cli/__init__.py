"""CLI module for training, inference, and evaluation."""

__all__ = [
    "train_model",
    "infer_batch",
    "evaluate_model",
    "sweep_splits",
    "DDTRDatasetGenerator",
]


def __getattr__(name):
    if name == "train_model":
        from .train import train_model

        return train_model
    if name == "infer_batch":
        from .infer import infer_batch

        return infer_batch
    if name == "evaluate_model":
        from .eval import evaluate_model

        return evaluate_model
    if name == "sweep_splits":
        from .sweep import sweep_splits

        return sweep_splits
    if name == "DDTRDatasetGenerator":
        from src.utils.pickle_io import DDTRDatasetGenerator

        return DDTRDatasetGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
