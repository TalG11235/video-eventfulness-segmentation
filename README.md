# Event Segmentation (MS-TCN)

This project trains and evaluates a temporal event segmentation model (MS-TCN) for frame-level action labeling in videos.

## The Pipeline

`src.model.crossval` runs the full segmentation workflow per split:

1. Builds split manifests from dataset metadata.
2. Trains MS-TCN on train videos.
3. Runs inference on validation videos.
4. Evaluates frame accuracy, F1@{10,25,50}, and edit score.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run The Model

Example (local run):

```bash
python3 -m src.model.crossval configs/50salads.yaml
```

Example (explicit output directory):

```bash
python3 -m src.model.crossval configs/50salads.yaml --output_dir outputs/50salads-run
```

Example (Slurm):

```bash
sbatch scripts/crossval_event_segmentation.sh
```

## Config Notes

- `data.splits`: use a list like `[1,2,3,4,5]` or `"all"`.
- `training.device`: set to `cuda` or `cpu`.
- `training.save_dir`: base path for run outputs.

## Outputs

Each split directory contains:

- `checkpoints/` (including `best.pt`)
- `predictions/` (`*_logits.npy`, `*_probs.npy`)
- `eval/metrics.json` and `eval/video_metrics.jsonl`
- `comparisons/` qualitative prediction vs. label stripe plots
