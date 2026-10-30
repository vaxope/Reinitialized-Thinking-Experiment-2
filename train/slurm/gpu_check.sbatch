#!/bin/bash
# First GPU job, ~0.5 SU. From the repo root on a login node:
#   source train/slurm/config.sh && sbatch --account="$ACCOUNT" train/slurm/gpu_check.sbatch
# Uses ghx4, not ghx4-interactive (interactive is charged 2x).
#SBATCH --job-name=gpqa-gpucheck
#SBATCH --partition=ghx4
#SBATCH --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=16 --mem=100g
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -eo pipefail   # no -u: module and conda activate scripts read unset variables
cd "$SLURM_SUBMIT_DIR"
source train/slurm/config.sh
module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)" && conda activate "$ENV"
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
date; python train/gpu_check.py "$DATA/train.jsonl"
date
# GRPO smoke test: 2 optimizer steps at the real answer cap (peak memory), offline wandb as in train.sbatch.
export WANDB_PROJECT=gpqa-qwen35-smoke WANDB_DIR="$RUNS/smoke-$SLURM_JOB_ID"
mkdir -p "$WANDB_DIR"
python train/grpo.py --format gar --train "$DATA/train.jsonl" --out "$RUNS/smoke-$SLURM_JOB_ID" \
  --max-steps 2 --max-completion 4096
echo "GPU CHECK PASSED"
