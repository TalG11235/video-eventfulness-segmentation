from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def _build_cmap(num_classes: int) -> ListedColormap:
    if num_classes <= 20:
        base = plt.get_cmap("tab20")
    elif num_classes <= 256:
        base = plt.get_cmap("turbo")
    else:
        base = plt.get_cmap("hsv")
    colors = base(np.linspace(0.0, 1.0, num_classes))
    return ListedColormap(colors)


def save_two_row_stripe_plot(
    gt: np.ndarray,
    pred: np.ndarray,
    out_path: str | Path,
    num_classes: int | None = None,
    title: str | None = None,
    dpi: int = 150,
) -> None:
    gt = np.asarray(gt).astype(np.int64)
    pred = np.asarray(pred).astype(np.int64)
    length = min(gt.shape[0], pred.shape[0])
    if length <= 0:
        return
    gt = gt[:length]
    pred = pred[:length]

    if num_classes is None:
        num_classes = int(max(gt.max(), pred.max()) + 1)
    if num_classes <= 0:
        return

    cmap = _build_cmap(num_classes)
    stripes = np.stack([gt, pred], axis=0)

    fig_w = max(6.0, length / 40.0)
    fig, ax = plt.subplots(figsize=(fig_w, 1.8), dpi=dpi)
    ax.imshow(
        stripes,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=num_classes - 1,
    )
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["GT", "Pred"])
    ax.set_xticks([])
    if title:
        ax.set_title(title)

    fig.tight_layout(pad=0.2)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
