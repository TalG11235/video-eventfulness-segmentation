import pickle
from pathlib import Path


def save_pickle(obj, path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str):
    path = Path(path)
    with path.open("rb") as handle:
        return pickle.load(handle)
