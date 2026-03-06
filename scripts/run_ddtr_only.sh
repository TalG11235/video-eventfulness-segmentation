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

DDTR_CONFIG_PATH="${PROJECT_ROOT}/configs/ddtr.json"
OUTPUT_DIR="${PROJECT_ROOT}/outputs_ddtr/50salads"

if [[ ! -f "${DDTR_CONFIG_PATH}" ]]; then
    echo "ERROR: Missing DDTR config: ${DDTR_CONFIG_PATH}"
    exit 1
fi

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}"
RUN_TIME="$(date +%H:%M:%S)"
RUN_DIR="${OUTPUT_DIR}/run-${RUN_TIME}"
mkdir -p "${RUN_DIR}"

RUN_LOG="${RUN_DIR}/run.log"

# Mirror all script output into the run directory log.
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "=== DDTR-only pipeline ==="
echo "DDTR config: ${DDTR_CONFIG_PATH}"
echo "DDTR pickle path comes from config data_path"
echo "Run dir: ${RUN_DIR}"
echo "Run log: ${RUN_LOG}"

echo "=== Step 1: Running DDTR ==="
cd "${DDTR_DIR}"
source "${DDTR_VENV}/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
python3 main.py --cfg_path "${DDTR_CONFIG_PATH}"

echo "=== Done ==="
echo "DDTR config: ${DDTR_CONFIG_PATH}"
