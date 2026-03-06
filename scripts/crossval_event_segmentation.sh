#!/bin/bash
#SBATCH --job-name=50salads-cv-sweep
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -p l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --time=12:00:00

#SBATCH --output=logs/%x-%j.out

set -euo pipefail

mkdir -p logs

source .venv/bin/activate

CONFIG_PATH="configs/50salads2.yaml"
BASE_OUTPUT_DIR="$(python3 -c "import yaml; cfg=yaml.safe_load(open('${CONFIG_PATH}', 'r', encoding='utf-8')) or {}; print(cfg.get('training', {}).get('save_dir', 'outputs'))")"
RUN_TIME="$(date +%H:%M:%S)"
RUN_OUTPUT_DIR="${BASE_OUTPUT_DIR}-${RUN_TIME}"

mkdir -p "${RUN_OUTPUT_DIR}"
exec > >(tee -a "${RUN_OUTPUT_DIR}/run-${SLURM_JOB_ID:-local}.log") 2>&1

python3 -m src.commands.crossval "${CONFIG_PATH}" --output_dir "${RUN_OUTPUT_DIR}"
