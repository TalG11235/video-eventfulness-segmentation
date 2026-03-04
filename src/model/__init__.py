"""Model package with training, inference, and evaluation entry points."""

__all__ = [
    "train_model",
    "infer_batch",
    "evaluate_model",
    "run_crossval",
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
    if name == "run_crossval":
        from .crossval import run_crossval

        return run_crossval
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
