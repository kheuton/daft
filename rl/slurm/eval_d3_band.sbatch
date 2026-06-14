#!/bin/bash
#SBATCH --job-name=eval_d3_band
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/cluster/tufts/hugheslab/kheuto01/code/daft/rl/slurm/logs/eval_d3_%j.out
#SBATCH --error=/cluster/tufts/hugheslab/kheuto01/code/daft/rl/slurm/logs/eval_d3_%j.err
# ---------------------------------------------------------------------------
# Post-training eval of the D3 (prm_margin) policy on the 878 band+control
# problems: quick_eval (n=64) -> score_prm (--save_step_scores) -> compare to
# arm_a under plain-maj and the deployed PRM-weighted-vote(last,p4,drop-empty).
#
#   sbatch rl/slurm/eval_d3_band.sbatch
#   (override: --export=ALL,MODEL_PATH=...,OUT=...,N_SAMPLES=64)
# ---------------------------------------------------------------------------
set -euo pipefail
REPO=/cluster/tufts/hugheslab/kheuto01/code/daft
MODEL_PATH="${MODEL_PATH:-${REPO}/outputs/arm_d_prmmargin/final}"
DATA="${DATA:-${REPO}/rl/analysis/outputs/band_eval_data.jsonl}"
OUT="${OUT:-${REPO}/rl/eval_outputs/arm_d_prmmargin_band}"
N_SAMPLES="${N_SAMPLES:-64}"
PRM_PATH="/cluster/tufts/hugheslab/LLMs/HF_CACHE/models--Qwen--Qwen2.5-Math-PRM-7B/snapshots/0610740060112df12585d00a1c5f4624d2f59051"

set +u; source ~/.bashrc; set -u
mamba activate daft_rl
cd "${REPO}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME=/cluster/tufts/hugheslab/LLMs/HF_CACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "${OUT}/scored" rl/slurm/logs

echo "=== [1/3] quick_eval ${MODEL_PATH} on $(basename "${DATA}") n=${N_SAMPLES} ==="
python -m rl.quick_eval \
    --model_path "${MODEL_PATH}" --data "${DATA}" --out_dir "${OUT}" \
    --n "${N_SAMPLES}" --temperature 1.0 --max_tokens 4096 --seed 0 \
    --shard_index 0 --num_shards 1

echo "=== [2/3] score_prm (per-step) on completions ==="
python -m rl.score_prm \
    --completions "${OUT}/completions.jsonl" \
    --out "${OUT}/scored/scored_shard0.jsonl" \
    --prm_path "${PRM_PATH}" --batch_size 8 --save_step_scores

echo "=== [3/3] compare D3 vs arm_a (plain-maj + PRM-wvote last/p4) ==="
python -m rl.analysis.eval_d3_compare --d3_dir "${OUT}/scored"

echo "eval_d3_band done -> ${OUT}"
