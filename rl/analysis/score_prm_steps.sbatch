#!/bin/bash
#SBATCH --job-name=prm_steps
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/cluster/tufts/hugheslab/kheuto01/code/daft/rl/slurm/logs/prm_steps_%A_%a.out
#SBATCH --error=/cluster/tufts/hugheslab/kheuto01/code/daft/rl/slurm/logs/prm_steps_%A_%a.err
#SBATCH --array=0-3

# Phase 1b de-risk: re-score the band WITH per-step PRM scores for an
# aggregation-strategy sweep (last/min/prod/mean) offline.
REPO=/cluster/tufts/hugheslab/kheuto01/code/daft
COMPLETIONS="${REPO}/rl/analysis/outputs/band_completions.jsonl"
OUT_DIR="${REPO}/rl/analysis/outputs/prm_scored_steps"
PRM_PATH="Qwen/Qwen2.5-Math-PRM-7B"
NUM_SHARDS=4
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

export HF_HOME=/cluster/tufts/hugheslab/LLMs/HF_CACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

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
    --num_shards "${NUM_SHARDS}" \
    --save_step_scores

echo "Done. Shard ${SHARD_INDEX} -> ${OUT_DIR}/scored_shard${SHARD_INDEX}.jsonl"
