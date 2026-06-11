#!/bin/bash
#SBATCH --job-name=daft_train_arm
#SBATCH --output=logs/train_arm_%j.out
#SBATCH --error=logs/train_arm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=a100-80G
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --requeue
# ---------------------------------------------------------------------------
# DAFT GRPO single-arm training launcher (DESIGN sec 6/7).
#
# Defaults above target the full run: partition=gpu, 4x A100-80G, 12h.
# Override at submit time, e.g.:
#   sbatch --partition=hugheslab --gres=gpu:a6000:4 --constraint="" \
#          --time=01:00:00 \
#          --export=ALL,CONFIG=rl/configs/smoke.yaml,OUTDIR=outputs/smoke \
#          rl/slurm/train_arm.sbatch
#
# Args / env:
#   CONFIG  (required)  path to a YAML in rl/configs (relative to daft repo root)
#   OUTDIR  (optional)  output dir; defaults to outputs/<config-basename>
#   GPU_COUNT (optional) number of processes; defaults to SLURM gpu count or 4
#
# Auto-requeue/resume: --requeue is set; on requeue this script re-runs and
# train_grpo.py --resume picks up the latest checkpoint-* in OUTDIR. Set
# save_steps small enough (50) that little is lost on preemption.
# ---------------------------------------------------------------------------

set -euo pipefail

# --- positional fallback: $1=CONFIG $2=OUTDIR ---
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

# --- GPU count for accelerate --num_processes ---
if [[ -n "${GPU_COUNT:-}" ]]; then
    NPROC="${GPU_COUNT}"
elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    NPROC="${SLURM_GPUS_ON_NODE}"
else
    NPROC=4
fi

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# ~/.bashrc -> /etc/bashrc references the unset var BASHRCSOURCED, which aborts
# under `set -u`. Relax nounset across the source, then restore it.
set +u
source ~/.bashrc
set -u
mamba activate daft_rl

cd "${REPO_DIR}"
mkdir -p logs "${OUTDIR}"
# rl/train_grpo.py is launched by path, which puts rl/ (not the repo root) on
# sys.path; the repo root must be importable for `import rl.*`.
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
# PCIe P2P (P2P/CUMEM) hangs on the first NCCL collective on pax nodes
# (verified on pax119 A6000s); SHM fallback is fine at 1.5B scale.
export NCCL_P2P_DISABLE=1
# Bootstrap over loopback: connects to the node's external IP got ECONNABORTED
# on pax119. Single-node jobs only — multinode must use a real interface.
export NCCL_SOCKET_IFNAME=lo

echo "========================================================"
echo "SLURM Job:  ${SLURM_JOB_ID:-<none>}"
echo "Node:       $(hostname)"
echo "GPUs:       ${NPROC}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "Config:     ${CONFIG}"
echo "Output dir: ${OUTDIR}"
echo "========================================================"

# DDP + colocated vLLM. accelerate config is plain MULTI_GPU bf16 (no FSDP).
accelerate launch \
    --config_file rl/slurm/accelerate_ddp.yaml \
    --num_processes "${NPROC}" \
    rl/train_grpo.py \
    --config "${CONFIG}" \
    --output_dir "${OUTDIR}" \
    --resume

echo "Training finished for ${CONFIG}."
