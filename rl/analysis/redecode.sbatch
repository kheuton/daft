#!/bin/bash
#SBATCH --job-name=redecode_lp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/cluster/tufts/hugheslab/kheuto01/code/daft/rl/slurm/logs/redecode_%A_%a.out
#SBATCH --error=/cluster/tufts/hugheslab/kheuto01/code/daft/rl/slurm/logs/redecode_%A_%a.err
#SBATCH --array=0-3

# Phase 1a: re-decode grpo's mis-selected band (+control) WITH token logprobs.
# 4-shard array, 1 GPU each. Mirrors rl/slurm/quick_eval.sbatch conventions.

REPO=/cluster/tufts/hugheslab/kheuto01/code/daft
MODEL_PATH="${MODEL_PATH:-${REPO}/outputs/arm_a_grpo/final}"
TARGETS="${TARGETS:-${REPO}/rl/analysis/outputs/redecode_targets.jsonl}"
OUT_DIR="${OUT_DIR:-${REPO}/rl/analysis/outputs/redecode}"
NUM_SHARDS="${NUM_SHARDS:-4}"
N_SAMPLES="${N_SAMPLES:-64}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
SEED="${SEED:-0}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

echo "============================================================"
echo "Model:    ${MODEL_PATH}"
echo "Targets:  ${TARGETS}"
echo "Out:      ${OUT_DIR}"
echo "Shard:    ${SHARD_INDEX}/${NUM_SHARDS}  n=${N_SAMPLES} T=${TEMPERATURE}"
echo "============================================================"

# Fail loudly; activate via bashrc (conda.sh alone leaves mamba activate broken
# and python falls through to base env -> 'No module named pebble').
set -eo pipefail
source ~/.bashrc
mamba activate daft_rl

mkdir -p "${REPO}/rl/slurm/logs" "${OUT_DIR}"

cd "${REPO}" && \
python -m rl.analysis.redecode_logprobs \
    --model_path "${MODEL_PATH}" \
    --targets "${TARGETS}" \
    --out_dir "${OUT_DIR}" \
    --n "${N_SAMPLES}" \
    --temperature "${TEMPERATURE}" \
    --max_tokens "${MAX_TOKENS}" \
    --seed "${SEED}" \
    --shard_index "${SHARD_INDEX}" \
    --num_shards "${NUM_SHARDS}"

echo "Done. Shard ${SHARD_INDEX} -> ${OUT_DIR}/shard${SHARD_INDEX}.jsonl"
