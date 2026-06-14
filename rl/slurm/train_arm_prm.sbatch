#!/bin/bash
#SBATCH --job-name=daft_prm_arm
#SBATCH --output=logs/train_prm_%j.out
#SBATCH --error=logs/train_prm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=a100-80G
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --requeue
# ---------------------------------------------------------------------------
# DAFT Phase-2 verifier-aware (PRM-in-loop) training launcher.
# Same as train_arm.sbatch + offline HF env so the 7B PRM loads from its local
# snapshot dir (trust_remote_code modeling files) with no hub resolution.
#
# Full run:
#   sbatch --export=ALL,CONFIG=rl/configs/arm_d_prmmargin.yaml,OUTDIR=outputs/arm_d_prmmargin \
#          rl/slurm/train_arm_prm.sbatch
# Smoke (1 GPU, 3 steps):
#   sbatch --gres=gpu:a100:1 --constraint=a100-80G --time=01:00:00 \
#          --export=ALL,CONFIG=rl/configs/arm_d_prmmargin_smoke.yaml,OUTDIR=outputs/arm_d_smoke,GPU_COUNT=1 \
#          rl/slurm/train_arm_prm.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

CONFIG="${CONFIG:-${1:-}}"
OUTDIR="${OUTDIR:-${2:-}}"

if [[ -z "${CONFIG}" ]]; then
    echo "ERROR: CONFIG not set. Pass via --export=ALL,CONFIG=... or as arg 1." >&2
    exit 2
fi

REPO_DIR="/cluster/tufts/hugheslab/kheuto01/code/daft"

if [[ -z "${OUTDIR}" ]]; then
    base="$(basename "${CONFIG}" .yaml)"
    OUTDIR="outputs/${base}"
fi

if [[ -n "${GPU_COUNT:-}" ]]; then
    NPROC="${GPU_COUNT}"
elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    NPROC="${SLURM_GPUS_ON_NODE}"
else
    NPROC=4
fi

# --- Environment ---
set +u
source ~/.bashrc
set -u
mamba activate daft_rl

cd "${REPO_DIR}"
mkdir -p logs "${OUTDIR}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# PRM loads from a local snapshot dir with trust_remote_code -> stay offline so
# transformers does not try to reach the hub for the modeling files.
export HF_HOME=/cluster/tufts/hugheslab/LLMs/HF_CACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "========================================================"
echo "SLURM Job:  ${SLURM_JOB_ID:-<none>}"
echo "Node:       $(hostname)"
echo "GPUs:       ${NPROC}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "Config:     ${CONFIG}"
echo "Output dir: ${OUTDIR}"
echo "========================================================"

accelerate launch \
    --config_file rl/slurm/accelerate_ddp.yaml \
    --num_processes "${NPROC}" \
    rl/train_grpo.py \
    --config "${CONFIG}" \
    --output_dir "${OUTDIR}" \
    --resume

echo "Training finished for ${CONFIG}."
