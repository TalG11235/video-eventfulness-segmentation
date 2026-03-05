#!/bin/bash
#SBATCH --job-name=ddtr-only
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -p l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/tal.gorbunov/projects/video-eventfulness-segmentation
#SBATCH --output=/home/tal.gorbunov/projects/video-eventfulness-segmentation/logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT="/home/tal.gorbunov/projects/video-eventfulness-segmentation"
MAIN_VENV="${PROJECT_ROOT}/.venv"
DDTR_DIR="${PROJECT_ROOT}/ddtr/DDTR"
DDTR_VENV="${DDTR_DIR}/venv"

EXPORT_DEVICE=""
DDTR_DEVICE="cuda:0"

# Static run selection (edit here if you want a different dataset/split).
DATASET_PREFIX="50salads"
SPLIT_ID="5"
OUTPUT_DIR="${PROJECT_ROOT}/outputs_ddtr/${DATASET_PREFIX}"
NUM_EPOCHS=2000
DDTR_BATCH_SIZE=1
DDTR_NUM_WORKERS=0
DDTR_NUM_TIMESTEPS=50

LATEST_RUN_DIR="$(ls -1dt "${PROJECT_ROOT}/outputs/${DATASET_PREFIX}-"* 2>/dev/null | head -1 || true)"
if [[ -z "${LATEST_RUN_DIR}" ]]; then
    echo "ERROR: Could not find any ${DATASET_PREFIX} runs under ${PROJECT_ROOT}/outputs."
    exit 1
fi

SPLIT_DIR="${LATEST_RUN_DIR}/split_${SPLIT_ID}"
CHECKPOINT_PATH="${SPLIT_DIR}/checkpoints/best.pt"
TRAIN_MANIFEST_PATH="${SPLIT_DIR}/manifests/train.jsonl"
TEST_MANIFEST_PATH="${SPLIT_DIR}/manifests/val.jsonl"
CONFIG_PATH="${SPLIT_DIR}/config.yaml"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT_PATH}"
    exit 1
fi
if [[ ! -f "${TRAIN_MANIFEST_PATH}" ]]; then
    echo "ERROR: Train manifest not found: ${TRAIN_MANIFEST_PATH}"
    exit 1
fi
if [[ ! -f "${TEST_MANIFEST_PATH}" ]]; then
    echo "ERROR: Test manifest not found: ${TEST_MANIFEST_PATH}"
    exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "ERROR: Config not found: ${CONFIG_PATH}"
    exit 1
fi
if [[ ! -x "${MAIN_VENV}/bin/python3" ]]; then
    echo "ERROR: Main venv missing python: ${MAIN_VENV}/bin/python3"
    exit 1
fi
if [[ ! -x "${DDTR_VENV}/bin/python3" ]]; then
    echo "ERROR: DDTR venv missing python: ${DDTR_VENV}/bin/python3"
    exit 1
fi

mkdir -p "${PROJECT_ROOT}/logs"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${OUTPUT_DIR}/run-${TIMESTAMP}"
mkdir -p "${RUN_DIR}"

DDTR_PICKLE="${RUN_DIR}/predictions.pkl"
DDTR_CONFIG="${RUN_DIR}/ddtr_config.json"
DDTR_SUMMARY_PATH="${RUN_DIR}/ddtr_config_summary.txt"
DDTR_RUNS_DIR="${RUN_DIR}/ddtr_runs"
RUN_LOG="${RUN_DIR}/run.log"

# Mirror all script output into the run directory log.
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "=== DDTR-only pipeline ==="
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Train manifest: ${TRAIN_MANIFEST_PATH}"
echo "Test manifest: ${TEST_MANIFEST_PATH}"
echo "Config: ${CONFIG_PATH}"
echo "Latest run dir: ${LATEST_RUN_DIR}"
echo "Run dir: ${RUN_DIR}"
echo "Run log: ${RUN_LOG}"
echo "Export device: ${EXPORT_DEVICE:-<config-default>}"
echo "DDTR device: ${DDTR_DEVICE}"
echo "DDTR batch size: ${DDTR_BATCH_SIZE}"
echo "DDTR workers: ${DDTR_NUM_WORKERS}"
echo "DDTR timesteps: ${DDTR_NUM_TIMESTEPS}"

echo "=== Step 1: Exporting train/test predictions to DDTR format ==="
cd "${PROJECT_ROOT}"
source "${MAIN_VENV}/bin/activate"

python3 - "${CONFIG_PATH}" "${CHECKPOINT_PATH}" "${TRAIN_MANIFEST_PATH}" "${TEST_MANIFEST_PATH}" "${DDTR_PICKLE}" "${EXPORT_DEVICE}" <<'PY'
import json
import sys

import numpy as np

from src.export.ddtr_exporter import run_ddtr_inference
from src.export.ddtr_format import save_as_ddtr_pickle
from src.utils import load_config

config_path = sys.argv[1]
checkpoint_path = sys.argv[2]
train_manifest_path = sys.argv[3]
test_manifest_path = sys.argv[4]
output_path = sys.argv[5]
device = sys.argv[6] if sys.argv[6] else None

train_result = run_ddtr_inference(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    manifest_path=train_manifest_path,
    device=device,
)
test_result = run_ddtr_inference(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    manifest_path=test_manifest_path,
    device=device,
)

train_items = list(train_result["predictions"])
test_items = list(test_result["predictions"])
train_count = len(train_items)
test_count = len(test_items)
total_count = train_count + test_count

if total_count == 0:
    raise ValueError("No predictions exported from train/test manifests.")

# DDTR always applies a random train_test_split() internally.
# Arrange samples so DDTR's split (seed=42, train_percent=0.8) maps
# exactly to the original train/test manifests.
train_percent = train_count / total_count
seed = 42
n_train = train_count
n_test = total_count - n_train
rng = np.random.RandomState(seed)
perm = rng.permutation(total_count)
ddtr_test_idx = perm[:n_test]
ddtr_train_idx = perm[n_test:(n_test + n_train)]

if len(ddtr_train_idx) != train_count or len(ddtr_test_idx) != test_count:
    raise ValueError(
        f"Unexpected DDTR split sizes: train={len(ddtr_train_idx)} test={len(ddtr_test_idx)} "
        f"expected train={train_count} test={test_count}"
    )

combined_predictions = [None] * total_count
for idx, item in zip(ddtr_train_idx.tolist(), train_items):
    combined_predictions[idx] = item
for idx, item in zip(ddtr_test_idx.tolist(), test_items):
    combined_predictions[idx] = item

if any(item is None for item in combined_predictions):
    raise ValueError("Failed to assign all combined prediction slots.")

export_result = save_as_ddtr_pickle(
    predictions={"predictions": combined_predictions},
    config=load_config(config_path),
    output_path=output_path,
)

summary = {
    "train_manifest": train_manifest_path,
    "test_manifest": test_manifest_path,
    "checkpoint": checkpoint_path,
    "device": device,
    "train_videos": train_count,
    "test_videos": test_count,
    "num_videos": export_result["num_videos"],
    "num_classes": export_result["num_classes"],
    "num_targets_from_labels": export_result["num_targets_from_labels"],
    "num_targets_from_predictions": export_result["num_targets_from_predictions"],
    "train_percent_for_ddtr_cfg": train_percent,
    "seed_for_ddtr_cfg": seed,
    "output_path": export_result["output_path"],
}
print(json.dumps(summary, indent=2))
PY

echo "=== Step 2: Creating DDTR config ==="
python3 - "${CONFIG_PATH}" "${TRAIN_MANIFEST_PATH}" "${TEST_MANIFEST_PATH}" "${DDTR_PICKLE}" "${DDTR_RUNS_DIR}" "${DDTR_CONFIG}" "${DDTR_DEVICE}" "${NUM_EPOCHS}" "${DDTR_SUMMARY_PATH}" "${DDTR_BATCH_SIZE}" "${DDTR_NUM_WORKERS}" "${DDTR_NUM_TIMESTEPS}" <<'PY'
import json
import sys
from pathlib import Path

from src.utils import load_config

config_path = Path(sys.argv[1])
train_manifest_path = Path(sys.argv[2])
test_manifest_path = Path(sys.argv[3])
pickle_path = Path(sys.argv[4])
summary_path = Path(sys.argv[5])
ddtr_config_path = Path(sys.argv[6])
ddtr_device = sys.argv[7]
num_epochs = int(sys.argv[8])
report_path = Path(sys.argv[9])
batch_size = int(sys.argv[10])
num_workers = int(sys.argv[11])
num_timesteps = int(sys.argv[12])

cfg = load_config(str(config_path))
data_cfg = cfg.get("data", {}) or {}
model_cfg = cfg.get("model", {}) or {}
data_dir = Path(data_cfg.get("data_dir", ""))
mapping_path = data_dir / "mapping.txt"

activity_names = {}
if mapping_path.exists():
    with mapping_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            idx, label = line.split(maxsplit=1)
            activity_names[str(int(idx))] = label

base_num_classes = model_cfg.get("num_classes")
if base_num_classes is None:
    if activity_names:
        base_num_classes = max(int(k) for k in activity_names) + 1
    else:
        raise ValueError("Could not resolve num_classes from config or mapping.txt")
base_num_classes = int(base_num_classes)

if not activity_names:
    activity_names = {str(i): f"class_{i}" for i in range(base_num_classes)}

# DDTR dataset preprocessing appends one extra "pad" class channel.
ddtr_num_classes = base_num_classes + 1
activity_names[str(base_num_classes)] = "pad"

train_count = sum(1 for _ in train_manifest_path.open("r", encoding="utf-8"))
test_count = sum(1 for _ in test_manifest_path.open("r", encoding="utf-8"))
total_count = train_count + test_count
if total_count == 0:
    raise ValueError("Both train/test manifests are empty.")

payload = {
    "data_path": str(pickle_path),
    "summary_path": str(summary_path),
    "device": ddtr_device,
    "parallelize": False,
    "num_epochs": num_epochs,
    "learning_rate": 1e-5,
    "num_timesteps": num_timesteps,
    "train_percent": train_count / total_count,
    "num_workers": num_workers,
    "test_every": 25,
    "num_classes": ddtr_num_classes,
    "batch_size": batch_size,
    "conditional_dropout": 0.1,
    "matrix_dropout": 0,
    "eval_train": True,
    "mode": "cond",
    "predict_on": "original",
    "seed": 42,
    "enable_matrix": True,
    "matrix_type": "pm",
    "activity_names": activity_names,
    "process_discovery_method": "inductive",
}

ddtr_config_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.mkdir(parents=True, exist_ok=True)
with ddtr_config_path.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)

report_path.write_text(
    f"base_num_classes={base_num_classes}\n"
    f"ddtr_num_classes={ddtr_num_classes}\n"
    f"train_manifest_count={train_count}\n"
    f"test_manifest_count={test_count}\n"
    f"train_percent={train_count / total_count}\n"
    f"mapping_path={mapping_path if mapping_path.exists() else '<not-found>'}\n",
    encoding="utf-8",
)
print(f"DDTR config written: {ddtr_config_path}")
print(f"base_num_classes={base_num_classes}")
print(f"ddtr_num_classes={ddtr_num_classes}")
print(f"train_manifest_count={train_count}")
print(f"test_manifest_count={test_count}")
print(f"batch_size={batch_size}")
print(f"num_workers={num_workers}")
print(f"num_timesteps={num_timesteps}")
PY

echo "=== Step 3: Running DDTR ==="
cd "${DDTR_DIR}"
source "${DDTR_VENV}/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
python3 main.py --cfg_path "${DDTR_CONFIG}"

echo "=== Done ==="
echo "DDTR pickle: ${DDTR_PICKLE}"
echo "DDTR config: ${DDTR_CONFIG}"
echo "DDTR outputs: ${DDTR_RUNS_DIR}"
