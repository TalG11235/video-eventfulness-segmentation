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
PARAMS_JSON="${PROJECT_ROOT}/configs/ddtr_pickle_from_outputs_50salads.json"

PICKLE_OUTPUT_DIR="${1:-${PROJECT_ROOT}/pickle}"
PICKLE_OUTPUT_PATH="${PICKLE_OUTPUT_DIR}/50salads_ddtr_input.pkl"

if [[ ! -f "${PARAMS_JSON}" ]]; then
    echo "ERROR: Missing params JSON: ${PARAMS_JSON}"
    exit 1
fi

echo "=== Build DDTR pickle from existing outputs ==="
echo "Params JSON: ${PARAMS_JSON}"
echo "Pickle output dir: ${PICKLE_OUTPUT_DIR}"
echo "Pickle output path: ${PICKLE_OUTPUT_PATH}"

cd "${PROJECT_ROOT}"
source "${VENV_PATH}/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${PICKLE_OUTPUT_DIR}"

python3 ddtr-utils/build_ddtr_pickle_from_outputs.py \
    --params-json "${PARAMS_JSON}" \
    --ddtr-pickle-path "${PICKLE_OUTPUT_PATH}" \
    --no-sidecars

echo "=== Done ==="
echo "DDTR pickle: ${PICKLE_OUTPUT_PATH}"
