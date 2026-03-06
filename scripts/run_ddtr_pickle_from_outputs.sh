#!/bin/bash
#SBATCH --job-name=ddtr-from-outputs
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -p l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/tal.gorbunov/projects/video-eventfulness-segmentation
#SBATCH --output=/home/tal.gorbunov/projects/video-eventfulness-segmentation/logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT="/home/tal.gorbunov/projects/video-eventfulness-segmentation"
VENV_PATH="${PROJECT_ROOT}/.venv"

# Existing segmentation output directory that already has split_1..split_5.
SEG_OUTPUTS_DIR="${PROJECT_ROOT}/outputs/50salads-20260305-152746"
SPLITS="1,2,3,4,5"
TEMPORAL_STRIDE="2"
EXPECTED_VIDEOS="50"
DEVICE_OVERRIDE=""

RUN_BASE_DIR="${PROJECT_ROOT}/outputs_ddtr/from-existing-outputs"

mkdir -p "${PROJECT_ROOT}/logs" "${RUN_BASE_DIR}"
RUN_TIME="$(date +%H:%M:%S)"
RUN_DIR="${RUN_BASE_DIR}/run-${RUN_TIME}"
mkdir -p "${RUN_DIR}"
RUN_LOG="${RUN_DIR}/run.log"
DDTR_PICKLE_PATH="${RUN_DIR}/50salads_ddtr_input.pkl"
TRACE_INDEX_PATH="${RUN_DIR}/trace_index.jsonl"
SUMMARY_PATH="${RUN_DIR}/summary.json"

exec > >(tee -a "${RUN_LOG}") 2>&1

echo "=== Build DDTR pickle from existing outputs ==="
echo "Segmentation outputs: ${SEG_OUTPUTS_DIR}"
echo "Splits: ${SPLITS}"
echo "Temporal stride: ${TEMPORAL_STRIDE}"
echo "Expected videos: ${EXPECTED_VIDEOS}"
echo "Device override: ${DEVICE_OVERRIDE:-<config-default>}"
echo "Run dir: ${RUN_DIR}"
echo "Run log: ${RUN_LOG}"

cd "${PROJECT_ROOT}"
source "${VENV_PATH}/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python3 ddtr-utils/build_ddtr_pickle_from_outputs.py \
    --outputs-dir "${SEG_OUTPUTS_DIR}" \
    --ddtr-pickle-path "${DDTR_PICKLE_PATH}" \
    --splits "${SPLITS}" \
    --device "${DEVICE_OVERRIDE}" \
    --temporal-stride "${TEMPORAL_STRIDE}" \
    --expected-videos "${EXPECTED_VIDEOS}" \
    --trace-index-path "${TRACE_INDEX_PATH}" \
    --summary-path "${SUMMARY_PATH}"

echo "=== Done ==="
echo "DDTR pickle: ${DDTR_PICKLE_PATH}"
echo "Trace index: ${TRACE_INDEX_PATH}"
echo "Summary: ${SUMMARY_PATH}"
