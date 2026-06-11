#!/bin/bash
#SBATCH --job-name=score_prm
#SBATCH --partition=gpu
#SBATCH --constraint=a100-80G
# Fallback: remove --constraint and use --partition=hugheslab for A6000 nodes
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/cluster/tufts/hugheslab/kheuto01/code/daft/rl/slurm/logs/score_prm_%A_%a.out
#SBATCH --error=/cluster/tufts/hugheslab/kheuto01/code/daft/rl/slurm/logs/score_prm_%A_%a.err
#SBATCH --array=0-3  # 0..(NUM_SHARDS-1) — adjust per launch

# ============================================================
# Configuration: set these before submitting
# ============================================================
COMPLETIONS="${COMPLETIONS:-/cluster/tufts/hugheslab/kheuto01/code/daft/rl/eval_outputs/completions.jsonl}"
OUT_BASE="${OUT_BASE:-/cluster/tufts/hugheslab/kheuto01/code/daft/rl/eval_outputs/scored}"
PRM_PATH="${PRM_PATH:-Qwen/Qwen2.5-Math-PRM-7B}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_SHARDS="${NUM_SHARDS:-4}"

# ============================================================
# Derive shard index from array task id
# ============================================================
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
OUT="${OUT_BASE}/shard${SHARD_INDEX}.jsonl"

echo "============================================================"
echo "Completions:  ${COMPLETIONS}"
echo "Out:          ${OUT}"
echo "PRM path:     ${PRM_PATH}"
echo "Batch size:   ${BATCH_SIZE}"
echo "Shard:        ${SHARD_INDEX}/${NUM_SHARDS}"
echo "============================================================"

# Activate environment
source /cluster/tufts/hugheslab/kheuto01/mambaforge/etc/profile.d/conda.sh
mamba activate daft_rl

# Create log dir if needed
mkdir -p /cluster/tufts/hugheslab/kheuto01/code/daft/rl/slurm/logs
mkdir -p "$(dirname "${OUT}")"

# Run score_prm — invoke as a module from repo root so that rl/ is on sys.path.
export PYTHONPATH=/cluster/tufts/hugheslab/kheuto01/code/daft

cd /cluster/tufts/hugheslab/kheuto01/code/daft && \
python -m rl.score_prm \
    --completions "${COMPLETIONS}" \
    --out "${OUT}" \
    --prm_path "${PRM_PATH}" \
    --batch_size "${BATCH_SIZE}" \
    --shard_index "${SHARD_INDEX}" \
    --num_shards "${NUM_SHARDS}"

echo "Done. Shard ${SHARD_INDEX} output: ${OUT}"
