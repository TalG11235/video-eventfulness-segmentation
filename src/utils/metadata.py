from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.configs import Config

from .runtime import get_git_commit


def write_metadata(cfg: Config, out_dir: Path) -> None:
    """Write training metadata (config + git commit) to JSON."""
    meta = {
        "model": asdict(cfg.model),
        "data": asdict(cfg.data),
        "training": asdict(cfg.training),
        "eval": asdict(cfg.eval),
        "git_commit": get_git_commit(),
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
