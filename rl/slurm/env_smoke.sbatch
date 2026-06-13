#!/bin/bash
#SBATCH --job-name=daft_env_smoke
#SBATCH --partition=hugheslab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/cluster/tufts/hugheslab/kheuto01/code/daft/rl/slurm/logs/env_smoke_%j.out
#SBATCH --error=/cluster/tufts/hugheslab/kheuto01/code/daft/rl/slurm/logs/env_smoke_%j.err

# Validates the daft_rl env on a real GPU node: driver vs cu124 wheels,
# torch CUDA, and a minimal vLLM generate. (GRPO smoke run is a separate job.)
set -x
source ~/.bashrc
mamba activate daft_rl
cd /cluster/tufts/hugheslab/kheuto01/code/daft

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c "import torch; print('cuda_available', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python - <<'EOF'
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", gpu_memory_utilization=0.5, max_model_len=2048)
out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=10, temperature=0))
print("VLLM_OK:", out[0].outputs[0].text.strip()[:50])
EOF
echo "ENV_SMOKE_DONE rc=$?"
