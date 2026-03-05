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
NUM_EPOCHS=10
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
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python3 "${PROJECT_ROOT}/ddtr-utils/export_train_test_predictions.py" \
    --config-path "${CONFIG_PATH}" \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --train-manifest-path "${TRAIN_MANIFEST_PATH}" \
    --test-manifest-path "${TEST_MANIFEST_PATH}" \
    --output-path "${DDTR_PICKLE}" \
    --device "${EXPORT_DEVICE}"

echo "=== Step 2: Creating DDTR config ==="
python3 "${PROJECT_ROOT}/ddtr-utils/create_ddtr_config.py" \
    --config-path "${CONFIG_PATH}" \
    --train-manifest-path "${TRAIN_MANIFEST_PATH}" \
    --test-manifest-path "${TEST_MANIFEST_PATH}" \
    --pickle-path "${DDTR_PICKLE}" \
    --summary-path "${DDTR_RUNS_DIR}" \
    --ddtr-config-path "${DDTR_CONFIG}" \
    --ddtr-device "${DDTR_DEVICE}" \
    --num-epochs "${NUM_EPOCHS}" \
    --report-path "${DDTR_SUMMARY_PATH}" \
    --batch-size "${DDTR_BATCH_SIZE}" \
    --num-workers "${DDTR_NUM_WORKERS}" \
    --num-timesteps "${DDTR_NUM_TIMESTEPS}"

echo "=== Step 3: Running DDTR ==="
cd "${DDTR_DIR}"
source "${DDTR_VENV}/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
python3 main.py --cfg_path "${DDTR_CONFIG}"

echo "=== Done ==="
echo "DDTR pickle: ${DDTR_PICKLE}"
echo "DDTR config: ${DDTR_CONFIG}"
echo "DDTR outputs: ${DDTR_RUNS_DIR}"
