# Copy to config.sh (gitignored) and fill in the two lines: cp train/slurm/config.example.sh train/slurm/config.sh
# `accounts` on a login node lists your Slurm account;
# PROJ should be on project storage (not $HOME: small quota), e.g. /projects/<proj>/$USER/gpqa-qwen35.
export ACCOUNT="CHANGE_ME"           # e.g. bxxx-dtai-gh
export PROJ="/projects/CHANGE_ME/$USER/gpqa-qwen35"

# Env and model on NVMe: imports from /work/hdd took ~12 min on DeltaAI (RUN-005), longer than the GPU check.
export FAST="/work/nvme/CHANGE_ME/$USER/gpqa-qwen35"
export ENV="$FAST/envs/gar"
export HF_HOME="$FAST/hf"
export DATA="$PROJ/data"              # train.jsonl / heldout.jsonl from data_prep.py (no GPQA)
export RUNS="$PROJ/runs"
export WANDB_MODE=offline             # sync later from a login node: wandb sync $RUNS/<run>/wandb/offline-run-*
