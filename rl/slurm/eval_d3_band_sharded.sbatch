#!/bin/bash
#SBATCH --job-name=eval_d3_sh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/cluster/tufts/hugheslab/kheuto01/code/daft/rl/slurm/logs/eval_d3sh_%A_%a.out
#SBATCH --error=/cluster/tufts/hugheslab/kheuto01/code/daft/rl/slurm/logs/eval_d3sh_%A_%a.err
#SBATCH --array=0-3
# ---------------------------------------------------------------------------
# Sharded D3 band eval: each shard generates ~220/878 problems (n=64) then
# PRM-scores them. Compare is run separately after all 4 shards finish.
#   sbatch rl/slurm/eval_d3_band_sharded.sbatch
# ---------------------------------------------------------------------------
set -euo pipefail
REPO=/cluster/tufts/hugheslab/kheuto01/code/daft
MODEL_PATH="${MODEL_PATH:-${REPO}/outputs/arm_d_prmmargin/final}"
DATA="${DATA:-${REPO}/rl/analysis/outputs/band_eval_data.jsonl}"
OUT="${OUT:-${REPO}/rl/eval_outputs/arm_d_prmmargin_band}"
N_SAMPLES="${N_SAMPLES:-64}"
NUM_SHARDS=4
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
PRM_PATH="/cluster/tufts/hugheslab/LLMs/HF_CACHE/models--Qwen--Qwen2.5-Math-PRM-7B/snapshots/0610740060112df12585d00a1c5f4624d2f59051"

set +u; source ~/.bashrc; set -u
mamba activate daft_rl
cd "${REPO}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME=/cluster/tufts/hugheslab/LLMs/HF_CACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
SHARD_OUT="${OUT}/shard${SHARD}"
mkdir -p "${SHARD_OUT}" "${OUT}/scored" rl/slurm/logs

echo "=== shard ${SHARD}/${NUM_SHARDS}: quick_eval ==="
python -m rl.quick_eval \
    --model_path "${MODEL_PATH}" --data "${DATA}" --out_dir "${SHARD_OUT}" \
    --n "${N_SAMPLES}" --temperature 1.0 --max_tokens 4096 --seed 0 \
    --shard_index "${SHARD}" --num_shards "${NUM_SHARDS}"

echo "=== shard ${SHARD}: score_prm (per-step) ==="
python -m rl.score_prm \
    --completions "${SHARD_OUT}/completions.jsonl" \
    --out "${OUT}/scored/scored_shard${SHARD}.jsonl" \
    --prm_path "${PRM_PATH}" --batch_size 8 --save_step_scores

echo "shard ${SHARD} done -> ${OUT}/scored/scored_shard${SHARD}.jsonl"
