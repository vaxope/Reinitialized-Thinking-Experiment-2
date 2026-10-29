#!/bin/bash
# Run on a DeltaAI LOGIN node (no GPU, not billed), from the repo root: bash train/slurm/setup_env.sh [cu129]
# Installs prebuilt aarch64 wheels only (no compiling) and downloads the model while internet is available.
# Default wheels need CUDA 13 (driver >= 580). If gpu_check.sbatch reports no CUDA, rerun with: cu129
set -eo pipefail   # no -u: module and conda activate scripts read unset variables
source train/slurm/config.sh
case "$ACCOUNT$PROJ" in *CHANGE_ME*) echo "Fill in ACCOUNT and PROJ in train/slurm/config.sh first"; exit 1;; esac
VARIANT="${1:-cu130}"
mkdir -p "$PROJ" "$HF_HOME" "$DATA" "$RUNS"

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
[ -d "$ENV" ] || conda create -y -p "$ENV" python=3.12
conda activate "$ENV"

if [ "$VARIANT" = cu129 ]; then
  pip install torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu129
  pip install "https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0+cu129-cp38-abi3-manylinux_2_28_aarch64.whl"
else
  pip install torch==2.13.0 vllm==0.31.0
fi
# vllm 0.31 needs transformers <5.18; trl 1.16 needs vllm <=0.31. fla = fast Gated DeltaNet kernels (Triton).
pip install transformers==5.17.0 trl==1.16.0 flash-linear-attention==0.6.0 datasets wandb
pip freeze > "$ENV/freeze.txt"

hf download Qwen/Qwen3.5-2B
echo "Done. Env: $ENV  (package list: $ENV/freeze.txt)"
echo "Next: copy train.jsonl + heldout.jsonl to $DATA, then:"
echo "  sbatch --account=\"$ACCOUNT\" train/slurm/gpu_check.sbatch"
