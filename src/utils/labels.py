import json
from pathlib import Path


def load_label_map(path: str):
    path = Path(path)
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    labels = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                labels.append(line)
    return labels
