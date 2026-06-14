#!/bin/bash
#SBATCH --job-name=prm_band
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/cluster/tufts/hugheslab/kheuto01/code/daft/rl/slurm/logs/prm_band_%A_%a.out
#SBATCH --error=/cluster/tufts/hugheslab/kheuto01/code/daft/rl/slurm/logs/prm_band_%A_%a.err
#SBATCH --array=0-3

# Phase 1b: score the mis-selected band (+control) with Qwen2.5-Math-PRM-7B.
REPO=/cluster/tufts/hugheslab/kheuto01/code/daft
COMPLETIONS="${COMPLETIONS:-${REPO}/rl/analysis/outputs/band_completions.jsonl}"
OUT_DIR="${OUT_DIR:-${REPO}/rl/analysis/outputs/prm_scored}"
PRM_PATH="${PRM_PATH:-Qwen/Qwen2.5-Math-PRM-7B}"
NUM_SHARDS="${NUM_SHARDS:-4}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

# PRM lives in the shared HF cache; load offline so compute nodes don't hit the network.
export HF_HOME=/cluster/tufts/hugheslab/LLMs/HF_CACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "============================================================"
echo "PRM:      ${PRM_PATH}"
echo "Input:    ${COMPLETIONS}"
echo "Out:      ${OUT_DIR}/scored_shard${SHARD_INDEX}.jsonl"
echo "Shard:    ${SHARD_INDEX}/${NUM_SHARDS}"
echo "============================================================"

set -eo pipefail
source ~/.bashrc
mamba activate daft_rl

mkdir -p "${REPO}/rl/slurm/logs" "${OUT_DIR}"

cd "${REPO}" && \
python -m rl.score_prm \
    --completions "${COMPLETIONS}" \
    --out "${OUT_DIR}/scored_shard${SHARD_INDEX}.jsonl" \
    --prm_path "${PRM_PATH}" \
    --batch_size 8 \
    --shard_index "${SHARD_INDEX}" \
    --num_shards "${NUM_SHARDS}"

echo "Done. Shard ${SHARD_INDEX} -> ${OUT_DIR}/scored_shard${SHARD_INDEX}.jsonl"
