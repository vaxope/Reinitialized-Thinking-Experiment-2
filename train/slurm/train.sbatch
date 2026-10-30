#!/bin/bash
# Stage A GRPO run on 1 GH200 (1 SU per hour). From the repo root on a login node:
#   source train/slurm/config.sh && sbatch --account="$ACCOUNT" train/slurm/train.sbatch gar 4096
# Args: FORMAT (gar|nothink|think), MAX_COMPLETION (from the pilot's p95). Resubmitting resumes from
# the last checkpoint. Stop early with: scancel <jobid>
#SBATCH --job-name=gpqa-grpo
#SBATCH --partition=ghx4
#SBATCH --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=16 --mem=100g
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
set -eo pipefail   # no -u: module and conda activate scripts read unset variables
FORMAT="${1:?format}"; MAXC="${2:?max completion tokens}"
cd "$SLURM_SUBMIT_DIR"
source train/slurm/config.sh
module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)" && conda activate "$ENV"
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 WANDB_PROJECT=gpqa-qwen35 WANDB_DIR="$RUNS/stageA-2b-$FORMAT"
mkdir -p "$WANDB_DIR"

echo "job $SLURM_JOB_ID  git $(git rev-parse HEAD)  format $FORMAT  max_completion $MAXC"
python train/grpo.py --format "$FORMAT" --train "$DATA/train.jsonl" --out "$RUNS/stageA-2b-$FORMAT" \
  --max-completion "$MAXC"
