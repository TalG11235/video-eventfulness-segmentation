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
DDTR_DIR="${PROJECT_ROOT}/ddtr/DDTR"
DDTR_VENV="${DDTR_DIR}/venv"

GENERATED_PICKLE_BASE="${PROJECT_ROOT}/outputs_ddtr/from-existing-outputs"
DDTR_CONFIG_PATH="${PROJECT_ROOT}/configs/ddtr.json"
DDTR_PICKLE_PATH="${PROJECT_ROOT}/pickle/50salads_ddtr_input.pkl"
OUTPUT_DIR="${PROJECT_ROOT}/outputs_ddtr/50salads"

LATEST_PICKLE_RUN_DIR="$(ls -1dt "${GENERATED_PICKLE_BASE}/run-"* 2>/dev/null | head -1 || true)"
if [[ -z "${LATEST_PICKLE_RUN_DIR}" ]]; then
    echo "ERROR: Could not find generated pickle runs under ${GENERATED_PICKLE_BASE}."
    exit 1
fi
SOURCE_DDTR_PICKLE="${LATEST_PICKLE_RUN_DIR}/50salads_ddtr_input.pkl"
if [[ ! -f "${SOURCE_DDTR_PICKLE}" ]]; then
    echo "ERROR: Missing generated DDTR pickle: ${SOURCE_DDTR_PICKLE}"
    exit 1
fi
if [[ ! -f "${DDTR_CONFIG_PATH}" ]]; then
    echo "ERROR: Missing DDTR config: ${DDTR_CONFIG_PATH}"
    exit 1
fi

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}" "$(dirname "${DDTR_PICKLE_PATH}")"
RUN_TIME="$(date +%H:%M:%S)"
RUN_DIR="${OUTPUT_DIR}/run-${RUN_TIME}"
mkdir -p "${RUN_DIR}"

RUN_LOG="${RUN_DIR}/run.log"

# Mirror all script output into the run directory log.
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "=== DDTR-only pipeline ==="
echo "DDTR config: ${DDTR_CONFIG_PATH}"
echo "Source DDTR pickle run dir: ${LATEST_PICKLE_RUN_DIR}"
echo "Source DDTR pickle: ${SOURCE_DDTR_PICKLE}"
echo "Target DDTR pickle: ${DDTR_PICKLE_PATH}"
echo "Run dir: ${RUN_DIR}"
echo "Run log: ${RUN_LOG}"

echo "=== Step 1: Copying latest generated DDTR pickle ==="
cp "${SOURCE_DDTR_PICKLE}" "${DDTR_PICKLE_PATH}"
echo "Copied DDTR pickle to: ${DDTR_PICKLE_PATH}"

echo "=== Step 2: Running DDTR ==="
cd "${DDTR_DIR}"
source "${DDTR_VENV}/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
python3 main.py --cfg_path "${DDTR_CONFIG_PATH}"

echo "=== Done ==="
echo "DDTR pickle: ${DDTR_PICKLE_PATH}"
echo "DDTR config: ${DDTR_CONFIG_PATH}"
