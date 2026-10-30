#!/bin/bash
# Held-out WebInstruct eval, untrained vs trained (~0.5 SU). From the repo root on a login node:
#   sbatch --account="$ACCOUNT" train/slurm/heldout_eval.sbatch [trained_run_name]   (default stageA-2b-gar)
# Resubmitting skips formats that already finished.
#SBATCH --job-name=gpqa-heldout
#SBATCH --partition=ghx4
#SBATCH --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=16 --mem=100g
#SBATCH --time=00:50:00
#SBATCH --output=%x-%j.out
set -eo pipefail   # no -u: module and conda activate scripts read unset variables
RUN="${1:-stageA-2b-gar}"
cd "$SLURM_SUBMIT_DIR"
source train/slurm/config.sh
module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)" && conda activate "$ENV"
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1

echo "job $SLURM_JOB_ID  git $(git rev-parse HEAD)  trained $RUNS/$RUN/final"
date; python train/heldout_eval.py --model Qwen/Qwen3.5-2B --tag base --data "$DATA/heldout.jsonl" --out "$RUNS/heldout-eval"
date; python train/heldout_eval.py --model "$RUNS/$RUN/final" --tag "$RUN" --data "$DATA/heldout.jsonl" --out "$RUNS/heldout-eval"
date; echo "HELDOUT EVAL DONE"
