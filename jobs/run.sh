#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=nvidia_a16:1
#SBATCH --output=logs/%x-%j.out

# Submit with: sbatch --job-name=<name> --export=ALL,CONFIG=configs/<cfg>.toml jobs/run.sh
set -euo pipefail
# compute nodes cannot reach huggingface.co; every load would burn 5 retries before
# falling back to the (already populated) cache
export HF_HUB_OFFLINE=1
# full-length prompts allocate very unevenly sized attention buffers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# under sbatch $0 is a spooled copy, so the repo path has to come from the submit dir
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
uv run python -u src/run.py --config "$CONFIG"
