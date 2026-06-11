#!/bin/bash
#SBATCH --job-name=quick_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/cluster/tufts/hugheslab/kheuto01/code/daft/rl/slurm/logs/quick_eval_%A_%a.out
#SBATCH --error=/cluster/tufts/hugheslab/kheuto01/code/daft/rl/slurm/logs/quick_eval_%A_%a.err
#SBATCH --array=0-3  # 0..(NUM_SHARDS-1) — set per launch

# ============================================================
# Configuration: set these before submitting
# ============================================================
# MODELS is a space-separated list of model paths.
# The array index selects the model; sharding is done within each job.
# For shard-parallel eval, set NUM_SHARDS > 1 and launch one job per shard.

MODELS=(
    "/path/to/checkpoint/pi0_sft"
    "/path/to/checkpoint/arm_grpo"
    "/path/to/checkpoint/arm_pass_at_k"
    "/path/to/checkpoint/arm_vote_k"
)

DATA="${DATA:-/cluster/tufts/hugheslab/kheuto01/code/daft/eval/datasets/math500.jsonl}"
N_SAMPLES="${N_SAMPLES:-64}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
SEED="${SEED:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
BASE_OUT_DIR="${BASE_OUT_DIR:-/cluster/tufts/hugheslab/kheuto01/code/daft/rl/eval_outputs}"

# ============================================================
# PROMPT-TEMPLATE DECISION (pinned — do not change without updating DESIGN.md):
#
# quick_eval uses the Qwen tokenizer's built-in chat template with
# add_generation_prompt=True.  This matches the training side (filter_sweep.py
# and TRL's maybe_apply_chat_template both use add_generation_prompt=True).
#
# The historical sal PRM best_of_n / wmaj@n runs (pi_0 baseline numbers in
# DESIGN.md section 1) used sal's DEFAULT custom_chat_template (a Llama-3
# override) WITHOUT add_generation_prompt.  Those runs are NOT prompt-comparable
# to quick_eval metrics.
#
# Decision: quick_eval is the canonical eval harness for all RL arm comparisons
# (the pre-registered endpoints in DESIGN.md section 5).  When reproducing or
# extending the sal PRM eval (DESIGN.md section 5 reference eval), set
# custom_chat_template=null and disable add_generation_prompt in sal's config
# so both harnesses see identical prompts — OR document explicitly that the two
# metric sets are not prompt-comparable.
# ============================================================

# ============================================================
# Derive model from array index
# ============================================================
MODEL_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
# MODEL_PATH env var overrides the MODELS array (single-model submissions).
MODEL_PATH="${MODEL_PATH:-${MODELS[$MODEL_INDEX]}}"

if [[ -z "${MODEL_PATH}" ]]; then
    echo "ERROR: No model at index ${MODEL_INDEX}"
    exit 1
fi

MODEL_NAME=$(basename "${MODEL_PATH}")
OUT_DIR="${BASE_OUT_DIR}/${MODEL_NAME}/T${TEMPERATURE}_n${N_SAMPLES}"

if [[ "${NUM_SHARDS}" -gt 1 ]]; then
    OUT_DIR="${OUT_DIR}/shard${SHARD_INDEX}"
fi

echo "============================================================"
echo "Model:        ${MODEL_PATH}"
echo "Model name:   ${MODEL_NAME}"
echo "Data:         ${DATA}"
echo "n samples:    ${N_SAMPLES}"
echo "Temperature:  ${TEMPERATURE}"
echo "Max tokens:   ${MAX_TOKENS}"
echo "Seed:         ${SEED}"
echo "Shard:        ${SHARD_INDEX}/${NUM_SHARDS}"
echo "Output dir:   ${OUT_DIR}"
echo "============================================================"

# Activate environment
# Fail the job on any error (a bare `echo Done` tail otherwise masks failures
# with exit 0), and activate via bashrc — conda.sh alone leaves `mamba activate`
# broken and python falls through to the base env.
set -eo pipefail
source ~/.bashrc
mamba activate daft_rl

# Create log dir if needed
mkdir -p /cluster/tufts/hugheslab/kheuto01/code/daft/rl/slurm/logs
mkdir -p "${OUT_DIR}"

# Run quick eval — invoke as a module from repo root so that `rl` is on sys.path.
# Using `python /path/to/quick_eval.py` would put rl/ (not the repo root) at
# sys.path[0], causing `from rl.rewards import ...` and `from rl.advantages import ...`
# to raise ImportError with a silent all-incorrect fallback and a hanging
# brute-force maj@k computation.  Sibling scripts (filter_sweep.sbatch) use this
# same cd+module pattern.
cd /cluster/tufts/hugheslab/kheuto01/code/daft && \
python -m rl.quick_eval \
    --model_path "${MODEL_PATH}" \
    --data "${DATA}" \
    --out_dir "${OUT_DIR}" \
    --n "${N_SAMPLES}" \
    --temperature "${TEMPERATURE}" \
    --max_tokens "${MAX_TOKENS}" \
    --seed "${SEED}" \
    --shard_index "${SHARD_INDEX}" \
    --num_shards "${NUM_SHARDS}"

echo "Done. Output in ${OUT_DIR}"
