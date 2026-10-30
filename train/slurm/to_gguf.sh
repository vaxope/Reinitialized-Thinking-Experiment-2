#!/bin/bash
# Convert a Hugging Face model to a Q8_0 GGUF for the local Ollama harness (same quant as Phase 1).
# Run on a DeltaAI LOGIN node (CPU only, not billed), from the repo root:
#   bash train/slurm/to_gguf.sh Qwen/Qwen3.5-2B base-2b-q8_0.gguf
#   bash train/slurm/to_gguf.sh "$RUNS/stageA-2b-gar/final" gar-2b-q8_0.gguf
# Uses llama.cpp's converter with its bundled gguf-py, so nothing is installed into the env.
# Both models must go through this same script (and llama.cpp commit) for a fair comparison.
set -eo pipefail   # no -u: module and conda activate scripts read unset variables
source train/slurm/config.sh
module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)" && conda activate "$ENV"
LCPP="$PROJ/llama.cpp"
[ -d "$LCPP" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LCPP"
SRC="$1"; OUT="$PROJ/gguf/$2"; mkdir -p "$PROJ/gguf"
# a hub id becomes its local snapshot folder (already downloaded by setup_env.sh)
[ -d "$SRC" ] || SRC="$(HF_HUB_OFFLINE=1 python -c "import sys; from huggingface_hub import snapshot_download as s; print(s(sys.argv[1]))" "$SRC")"
python "$LCPP/convert_hf_to_gguf.py" "$SRC" --outtype q8_0 --outfile "$OUT"
echo "llama.cpp $(git -C "$LCPP" rev-parse --short HEAD)  source $SRC" | tee "$OUT.txt"
ls -l "$OUT"
