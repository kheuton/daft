#!/bin/bash
#SBATCH --job-name=daft_filter_sweep
#SBATCH --output=/cluster/tufts/hugheslab/kheuto01/code/daft/rl/slurm/logs/filter_sweep_%A_%a.out
#SBATCH --error=/cluster/tufts/hugheslab/kheuto01/code/daft/rl/slurm/logs/filter_sweep_%A_%a.err
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
# Preferred: partition=hugheslab for A6000 (48GB) pilot nodes.
# Full run: --partition=gpu --constraint=a100-80G
# Overflow: --partition=preempt --requeue

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

set -eo pipefail
set +u
source ~/.bashrc
set -u
mamba activate daft_rl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Hardcoded to avoid resolving via $0 (Slurm spools scripts into SlurmdSpoolDir).
# Submit this script from any directory; paths below are always correct.
RL_DIR="/cluster/tufts/hugheslab/kheuto01/code/daft/rl"
MODEL_PATH="${MODEL_PATH:-/cluster/tufts/hugheslab/kheuto01/code/daft/checkpoints/sft_vanilla}"
DATA_FILE="${DATA_FILE:-${RL_DIR}/data/math_train_decontam.jsonl}"
OUT_FILE="${OUT_FILE:-${RL_DIR}/data/sweep.jsonl}"

NUM_SHARDS=4  # Must match --array upper bound + 1

mkdir -p "${RL_DIR}/data"
mkdir -p "${RL_DIR}/slurm/logs"

echo "========================================================"
echo "SLURM Job: ${SLURM_JOB_ID}, Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "Shard: ${SLURM_ARRAY_TASK_ID} / ${NUM_SHARDS}"
echo "Model: ${MODEL_PATH}"
echo "Data:  ${DATA_FILE}"
echo "Out:   ${OUT_FILE}"
echo "========================================================"

cd /cluster/tufts/hugheslab/kheuto01/code/daft

python -m rl.filter_sweep \
    --model_path "${MODEL_PATH}" \
    --data "${DATA_FILE}" \
    --out "${OUT_FILE}" \
    --G 16 \
    --temperature 1.0 \
    --max_tokens 2048 \
    --shard_index "${SLURM_ARRAY_TASK_ID}" \
    --num_shards "${NUM_SHARDS}"

echo "Shard ${SLURM_ARRAY_TASK_ID} done."
