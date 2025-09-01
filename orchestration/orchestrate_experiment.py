#!/usr/bin/env python3
"""
DAFT Experiment Orchestrator

This script orchestrates the complete pipeline:
1. Training -> 2. Model Upload -> 3. Evaluation -> 4. Results Processing

Usage:
    python orchestrate_experiment.py configs/experiment_config.yaml
    python orchestrate_experiment.py configs/experiment_config.yaml --step training  # Run only training
    python orchestrate_experiment.py configs/experiment_config.yaml --dry-run       # Show what would be submitted
"""

import argparse
import subprocess
import yaml
import os
import sys
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

class ExperimentOrchestrator:
    def __init__(self, config_path: str):
        """Initialize the orchestrator with configuration."""
        self.config_path = config_path
        self.config = self.load_config(config_path)
        self.experiment_id = self.generate_experiment_id()
        self.job_ids = {}  # Store job IDs for dependency management
        self.scripts_dir = Path(__file__).parent / "scripts"
        self.logs_dir = Path(__file__).parent / "logs" / self.experiment_id
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Initializing experiment: {self.config['experiment']['name']}")
        print(f"📊 Experiment ID: {self.experiment_id}")
        print(f"📁 Logs directory: {self.logs_dir}")
    
    def load_config(self, config_path: str) -> Dict:
        """Load the experiment configuration."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def generate_experiment_id(self) -> str:
        """Generate a unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = self.config['experiment']['name'].replace(' ', '_').lower()
        return f"{name}_{timestamp}"
    
    def substitute_variables(self, text: str, context: Dict[str, Any]) -> str:
        """Substitute variables in text using context."""
        if isinstance(text, str):
            # Replace variables like {variable_name} with values from context
            for key, value in context.items():
                text = text.replace(f"{{{key}}}", str(value))
        return text
    
    def create_training_script(self) -> str:
        """Create the training SLURM script."""
        config = self.config['training']
        slurm_config = config['slurm']
        
        # Build context for variable substitution
        context = {
            'experiment_id': self.experiment_id,
            'model_name': config['model_name'],
            'train_dataset_name': config['train_dataset_name'],
            'learning_rate': config['learning_rate'],
            'epochs': config['epochs'],
            'batch_size': config['batch_size'],
            'weight_decay': config['weight_decay'],
            'wandb_project': config['wandb_project'],
            'wandb_entity': config['wandb_entity'],
            'hub_model_id': config['hub_model_id'],
            'topk_k': config['topk_k'],
            'code_base': self.config['environment']['code_base'],
            'training_env': self.config['environment']['training_env']
        }
        
        # Prepare training config for embedding in JSON with escaped quotes
        training_config_escaped = yaml.dump(config, default_flow_style=False).replace('"', '\\"')
        
        script_content = f"""#!/bin/bash
#SBATCH --job-name={slurm_config['job_name']}_{self.experiment_id}
#SBATCH --nodes={slurm_config['nodes']}
#SBATCH --ntasks-per-node={slurm_config['ntasks_per_node']}
#SBATCH --cpus-per-task={slurm_config['cpus_per_task']}
#SBATCH --gres={slurm_config['gres']}
#SBATCH --mem={slurm_config['mem']}
#SBATCH --time={slurm_config['time']}
#SBATCH --partition={slurm_config['partition']}
#SBATCH --output={self.logs_dir}/training_%j.out
#SBATCH --error={self.logs_dir}/training_%j.err
"""
        
        if 'nodelist' in slurm_config and slurm_config['nodelist']:
            script_content += f"#SBATCH --nodelist={slurm_config['nodelist']}\n"
        
        script_content += f"""
# Environment setup
source ~/.bashrc
mamba activate {context['training_env']}
cd {context['code_base']}

# Create output directory for this experiment
mkdir -p ckpts/{self.experiment_id}

# Set environment variables
export WANDB_PROJECT="{context['wandb_project']}"
export WANDB_ENTITY="{context['wandb_entity']}"

# Get node information for distributed training
node_array=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nnodes=$(echo $node_array | wc -w)
head_node=($node_array)
head_node_ip=$(ssh $head_node hostname --ip-address)
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600

# Calculate gradient accumulation steps
gpu_count=$(nvidia-smi -L | wc -l)
grad_acc=$(({context['batch_size']}/(gpu_count * nnodes)))

echo "=== Training Configuration ==="
echo "Experiment ID: {self.experiment_id}"
echo "Model: {context['model_name']}"
echo "Dataset: {context['train_dataset_name']}"
echo "Learning rate: {context['learning_rate']}"
echo "Epochs: {context['epochs']}"
echo "Batch size: {context['batch_size']}"
echo "Weight decay: {context['weight_decay']}"
echo "Number of nodes: $nnodes"
echo "Number of GPUs per node: $gpu_count"
echo "Gradient accumulation steps: $grad_acc"
echo "=========================="

# Launch distributed training
torchrun \\
    --nnodes=$nnodes \\
    --nproc_per_node=$gpu_count \\
    --rdzv_id=$SLURM_JOB_ID \\
    --rdzv_backend=c10d \\
    --rdzv_endpoint=$head_node_ip:29500 \\
    s1/train/sft.py \\
    --block_size=32768 \\
    --per_device_train_batch_size=1 \\
    --per_device_eval_batch_size=1 \\
    --gradient_accumulation_steps=$grad_acc \\
    --num_train_epochs={context['epochs']} \\
    --train_file_path="simplescaling/{context['train_dataset_name']}" \\
    --model_name="{context['model_name']}" \\
    --use_custom_loss={str(config['use_custom_loss']).lower()} \\
    --loss_type={config['loss_type']} \\
    --topk_k={config['topk_k']} \\
    --topk_temperature={config['topk_temperature']} \\
    --warmup_ratio=0.05 \\
    --report_to="wandb" \\
    --fsdp="full_shard auto_wrap" \\
    --fsdp_config="s1/train/fsdp_config_qwen_cpu.json" \\
    --bf16=True \\
    --eval_strategy="no" \\
    --logging_steps=1 \\
    --save_strategy="no" \\
    --lr_scheduler_type="cosine" \\
    --learning_rate={context['learning_rate']} \\
    --weight_decay={context['weight_decay']} \\
    --adam_beta1=0.9 \\
    --adam_beta2=0.95 \\
    --output_dir="ckpts/{self.experiment_id}" \\
    --push_to_hub={str(config['push_to_hub']).lower()} \\
    --save_only_model=True \\
    --gradient_checkpointing=True \\
    --accelerator_config='{{"gradient_accumulation_kwargs": {{"sync_each_batch": true}}}}'

echo "Training completed for experiment {self.experiment_id}"

# Save experiment metadata
cat > ckpts/{self.experiment_id}/experiment_metadata.json << EOF
{{
    "experiment_id": "{self.experiment_id}",
    "config_path": "{self.config_path}",
    "training_config": {training_config_escaped},
    "completed_at": "$(date)",
    "slurm_job_id": "$SLURM_JOB_ID"
}}
EOF

echo "Experiment metadata saved to ckpts/{self.experiment_id}/experiment_metadata.json"
"""
        
        script_path = self.scripts_dir / f"train_{self.experiment_id}.slurm"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        return str(script_path)
    
    def create_upload_script(self) -> str:
        """Create the model upload script."""
        config = self.config['upload']
        training_config = self.config['training']
        
        context = {
            'experiment_id': self.experiment_id,
            'hub_model_id': training_config['hub_model_id'],
            'model_name': training_config['model_name'],
            'learning_rate': training_config['learning_rate'],
            'epochs': training_config['epochs'],
            'batch_size': training_config['batch_size'],
            'topk_k': training_config['topk_k'],
            'code_base': self.config['environment']['code_base'],
            'training_env': self.config['environment']['training_env']
        }
        
        # Create model card with substituted variables
        model_card = self.substitute_variables(config['model_card_template'], context)
        
        script_content = f"""#!/bin/bash
#SBATCH --job-name=upload_{self.experiment_id}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --partition=batch
#SBATCH --output={self.logs_dir}/upload_%j.out
#SBATCH --error={self.logs_dir}/upload_%j.err

# Environment setup
source ~/.bashrc
mamba activate {context['training_env']}
cd {context['code_base']}

echo "=== Model Upload Configuration ==="
echo "Experiment ID: {self.experiment_id}"
echo "Model path: ckpts/{self.experiment_id}"
echo "Hub model ID: {context['hub_model_id']}"
echo "=========================="

# Create model card
cat > ckpts/{self.experiment_id}/README.md << 'MODELCARD'
{model_card}
MODELCARD

# Upload model to HuggingFace Hub
python -c "
import os
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize HF API
api = HfApi()

# Create repository if it doesn't exist
try:
    create_repo('{context['hub_model_id']}', exist_ok=True)
    print('Repository created/verified: {context['hub_model_id']}')
except Exception as e:
    print(f'Repository creation warning: {{e}}')

# Load and upload model
print('Loading model and tokenizer...')
model = AutoModelForCausalLM.from_pretrained('ckpts/{self.experiment_id}')
tokenizer = AutoTokenizer.from_pretrained('{context['model_name']}')

print('Uploading model to hub...')
model.push_to_hub('{context['hub_model_id']}')
tokenizer.push_to_hub('{context['hub_model_id']}')

# Upload experiment metadata
api.upload_file(
    path_or_fileobj='ckpts/{self.experiment_id}/experiment_metadata.json',
    path_in_repo='experiment_metadata.json',
    repo_id='{context['hub_model_id']}',
    repo_type='model'
)

print('Model upload completed successfully!')
"

echo "Upload completed for experiment {self.experiment_id}"
"""
        
        script_path = self.scripts_dir / f"upload_{self.experiment_id}.slurm"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        return str(script_path)
    
    def create_evaluation_script(self) -> str:
        """Create the evaluation SLURM script."""
        config = self.config['evaluation']
        slurm_config = config['slurm']
        training_config = self.config['training']
        
        # Use the trained model for evaluation
        model_path = training_config['hub_model_id']
        
        context = {
            'experiment_id': self.experiment_id,
            'model_path': model_path,
            'prm_path': config['prm_path'],
            'approach': config['approach'],
            'n': config['n'],
            'dataset_start': config['dataset_start'],
            'dataset_end': config['dataset_end'],
            'seed': config['seed'],
            'hub_dataset_id': config['hub_dataset_id'],
            'code_base': self.config['environment']['code_base'],
            'evaluation_env': self.config['environment']['evaluation_env']
        }
        
        # Create evaluation config file
        eval_config = {
            'approach': config['approach'],
            'n': config['n'],
            'search_batch_size': config['search_batch_size'],
            'sort_completed': config['sort_completed'],
            'filter_duplicates': config['filter_duplicates'],
            'dataset_start': config['dataset_start'],
            'dataset_end': config['dataset_end'],
            'seed': config['seed'],
            'prm_path': config['prm_path'],
            'model_path': model_path,
            'dataset_name': config.get('dataset_name', 'eval/datasets/math500.jsonl'),  # Use config value or default
            # VLLM sampling parameters to match training behavior
            'temperature': config.get('temperature', 0.8),
            'top_k': config.get('top_k', -1),  # -1 means disabled
            'top_p': config.get('top_p', 1.0),  # 1.0 means disabled  
        }
        
        eval_config_path = self.scripts_dir / f"eval_config_{self.experiment_id}.yaml"
        with open(eval_config_path, 'w') as f:
            yaml.dump(eval_config, f, default_flow_style=False)
        
        script_content = f"""#!/bin/bash
#SBATCH --job-name={slurm_config['job_name']}_{self.experiment_id}
#SBATCH --array={slurm_config['array']}
#SBATCH --gres={slurm_config['gres']}
#SBATCH --output={self.logs_dir}/evaluation_%x-%j_%A_%a.out
#SBATCH --error={self.logs_dir}/evaluation_%x-%j_%A_%a.err
#SBATCH --time={slurm_config['time']}
#SBATCH --nodes={slurm_config['nodes']}
#SBATCH --cpus-per-task={slurm_config['cpus_per_task']}
#SBATCH --mem={slurm_config['mem']}
#SBATCH --partition={slurm_config['partition']}
"""
        
        if 'nodelist' in slurm_config and slurm_config['nodelist']:
            script_content += f"#SBATCH --nodelist={slurm_config['nodelist']}\n"
        
        if 'constraint' in slurm_config and slurm_config['constraint']:
            script_content += f"#SBATCH --constraint=\"{slurm_config['constraint']}\"\n"
        
        script_content += f"""
# Environment setup
source ~/.bashrc
mamba activate {context['evaluation_env']}
cd {context['code_base']}

# Define evaluation parameters
STEP=250
ENDPOINT=$((SLURM_ARRAY_TASK_COUNT * STEP - STEP))
STARTS=($(seq 0 $STEP $ENDPOINT))

DATASET_START=${{STARTS[$SLURM_ARRAY_TASK_ID-1]}}
DATASET_END=$((${{STARTS[$SLURM_ARRAY_TASK_ID-1]}}+$STEP))

# Create output directory for this task
DATE_STR=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="{self.config['experiment']['output_base_dir']}/{self.experiment_id}/eval_outputs/${{DATE_STR}}_task${{SLURM_ARRAY_TASK_ID}}"
mkdir -p "$OUTPUT_DIR"

echo "=== Evaluation Configuration ==="
echo "Experiment ID: {self.experiment_id}"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Dataset range: $DATASET_START to $DATASET_END"
echo "Model: {context['model_path']}"
echo "PRM: {context['prm_path']}"
echo "Approach: {context['approach']}"
echo "N: {context['n']}"
echo "Output dir: $OUTPUT_DIR"
echo "=========================="

# Run evaluation
python eval/test_time_compute.py {eval_config_path} \\
    --dataset_start=$DATASET_START \\
    --dataset_end=$DATASET_END \\
    --push_to_hub={str(config['push_results_to_hub']).lower()} \\
    --hub_dataset_id="{context['hub_dataset_id']}" \\
    --output_dir="$OUTPUT_DIR"

echo "Evaluation task $SLURM_ARRAY_TASK_ID completed for experiment {self.experiment_id}"
"""
        
        script_path = self.scripts_dir / f"eval_{self.experiment_id}.slurm"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        return str(script_path)
    
    def create_processing_script(self) -> str:
        """Create the results processing script."""
        config = self.config['processing']
        evaluation_config = self.config['evaluation']
        
        context = {
            'experiment_id': self.experiment_id,
            'outputs_base_dir': f"{self.config['experiment']['output_base_dir']}/{self.experiment_id}/eval_outputs",
            'hub_dataset_id': evaluation_config['hub_dataset_id'],
            'code_base': self.config['environment']['code_base'],
            'evaluation_env': self.config['environment']['evaluation_env']
        }
        
        script_content = f"""#!/bin/bash
#SBATCH --job-name=process_{self.experiment_id}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --partition=batch
#SBATCH --output={self.logs_dir}/processing_%j.out
#SBATCH --error={self.logs_dir}/processing_%j.err

# Environment setup
source ~/.bashrc
mamba activate {context['evaluation_env']}
cd {context['code_base']}

echo "=== Results Processing Configuration ==="
echo "Experiment ID: {self.experiment_id}"
echo "Outputs directory: {context['outputs_base_dir']}"
echo "Hub dataset ID: {context['hub_dataset_id']}"
echo "=========================="

# Create processing script
python -c "
import os
import re
import json
import yaml
import numpy as np
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from sal.utils.math import *
from sal.utils.grader import *
from sal.utils.qwen_math_parser import *
from sal.utils.data import get_dataset
from sal.config import Config
from collections import defaultdict
import matplotlib.pyplot as plt

print('Starting results processing...')

# Load experiment configuration
with open('{self.config_path}', 'r') as f:
    experiment_config = yaml.safe_load(f)

# Set up paths
outputs_base_dir = '{context['outputs_base_dir']}'
experiment_id = '{self.experiment_id}'

print(f'Looking for output files in: {{outputs_base_dir}}')

# Combine evaluation chunks
all_datasets = []
subdirs = [d for d in os.listdir(outputs_base_dir) if os.path.isdir(os.path.join(outputs_base_dir, d))]
subdirs.sort()

print(f'Found {{len(subdirs)}} output directories')

for subdir in subdirs:
    match = re.search(r'task(\d+)', subdir)
    if match:
        task_num = int(match.group(1))
        output_dir = os.path.join(outputs_base_dir, subdir)
        
        # Load batch files for this task
        for batch_num in range(5):  # batch_0.jsonl through batch_4.jsonl
            batch_file = os.path.join(output_dir, f'batch_{{batch_num}}.jsonl')
            if os.path.exists(batch_file):
                try:
                    dataset = load_dataset('json', data_files=batch_file, split='train')
                    all_datasets.append(dataset)
                    print(f'  Task {{task_num:2d}} batch_{{batch_num}}: {{len(dataset)}} samples')
                except Exception as e:
                    print(f'  Error loading {{batch_file}}: {{e}}')

# Combine all datasets
if all_datasets:
    print(f'Combining {{len(all_datasets)}} datasets...')
    combined_dataset = concatenate_datasets(all_datasets)
    print(f'Combined dataset size: {{len(combined_dataset)}} samples')
    
    # Save combined dataset
    output_base = '{self.config['experiment']['output_base_dir']}/{self.experiment_id}'
    os.makedirs(output_base, exist_ok=True)
    
    combined_file = os.path.join(output_base, '{context['hub_dataset_id'].split('/')[-1]}_combined.json')
    print(f'Saving combined dataset to: {{combined_file}}')
    
    # Convert to pandas DataFrame and save as JSON
    df = combined_dataset.to_pandas()
    df.to_json(combined_file, orient='records', lines=False, indent=2)
    
    # Compute evaluation metrics
    print('Computing evaluation metrics...')
    
    # Load ground truth
    config = Config()
    gt = get_dataset(config)
    gt_dict = {{item['unique_id']: item for item in gt}}
    data_dict = {{item['unique_id']: item for item in combined_dataset}}
    
    # Find common unique_ids
    common_ids = set(gt_dict.keys()) & set(data_dict.keys())
    print(f'Found {{len(common_ids)}} common samples for evaluation')
    
    # Define evaluation points
    n_completions = [2**i for i in range(8)]  # [1, 2, 4, 8, 16, 32, 64, 128]
    
    results = []
    for unique_id in common_ids:
        gt_sample = gt_dict[unique_id]
        data_sample = data_dict[unique_id]
        
        # Extract answers from all completions
        extracted_answers = [strip_string(extract_answer(completion, 'math')) for completion in data_sample['completions']]
        scores = np.array(data_sample['scores'])
        
        sample_result = {{
            'unique_id': unique_id,
            'gt_answer': gt_sample['answer'],
            'extracted_answers': extracted_answers,
            'scores': scores,
            'best_of_n_correct': {{}},
            'weighted_best_of_n_correct': {{}}
        }}
        
        # Calculate metrics for each n
        for n in n_completions:
            if n > len(extracted_answers):
                n_actual = len(extracted_answers)
            else:
                n_actual = n
                
            # Best-of-n
            best_idx = np.argmax(scores[:n_actual])
            best_answer = extracted_answers[best_idx]
            
            is_correct_best_of_n = math_equal(
                memoized_canonical_form(gt_sample['answer']), 
                memoized_canonical_form(best_answer)
            )
            sample_result['best_of_n_correct'][n] = is_correct_best_of_n
            
            # Weighted best-of-n
            answer_scores = defaultdict(float)
            for i in range(n_actual):
                canonical_answer = memoized_canonical_form(extracted_answers[i])
                answer_scores[canonical_answer] += scores[i]
            
            best_weighted_answer = max(answer_scores.keys(), key=lambda x: answer_scores[x])
            
            is_correct_weighted = math_equal(
                memoized_canonical_form(gt_sample['answer']), 
                best_weighted_answer
            )
            sample_result['weighted_best_of_n_correct'][n] = is_correct_weighted
        
        results.append(sample_result)
    
    # Calculate overall accuracies
    best_of_n_accuracies = {{}}
    weighted_best_of_n_accuracies = {{}}
    
    for n in n_completions:
        best_of_n_correct = sum(1 for r in results if r['best_of_n_correct'].get(n, False))
        weighted_best_of_n_correct = sum(1 for r in results if r['weighted_best_of_n_correct'].get(n, False))
        
        best_of_n_accuracies[n] = best_of_n_correct / len(results)
        weighted_best_of_n_accuracies[n] = weighted_best_of_n_correct / len(results)
    
    # Print results
    print('\\n' + '='*60)
    print('EVALUATION RESULTS')
    print('='*60)
    print(f'Total samples evaluated: {{len(results)}}')
    print()
    print('Best-of-n accuracies:')
    for n in n_completions:
        print(f'  n={{n:3d}}: {{best_of_n_accuracies[n]:.4f}} ({{best_of_n_accuracies[n]:.2%}})')
    
    print()
    print('Weighted best-of-n accuracies:')
    for n in n_completions:
        print(f'  n={{n:3d}}: {{weighted_best_of_n_accuracies[n]:.4f}} ({{weighted_best_of_n_accuracies[n]:.2%}})')
    
    # Save detailed results
    results_file = os.path.join(output_base, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump({{
            'experiment_id': experiment_id,
            'total_samples': len(results),
            'best_of_n_accuracies': best_of_n_accuracies,
            'weighted_best_of_n_accuracies': weighted_best_of_n_accuracies,
            'n_completions': n_completions
        }}, f, indent=2)
    
    print(f'\\nDetailed results saved to: {{results_file}}')
    
    # Generate plots
    try:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(n_completions, [best_of_n_accuracies[n] for n in n_completions], 'o-', label='Best-of-n', linewidth=2, markersize=6)
        plt.plot(n_completions, [weighted_best_of_n_accuracies[n] for n in n_completions], 's-', label='Weighted best-of-n', linewidth=2, markersize=6)
        plt.xlabel('Number of completions (n)')
        plt.ylabel('Accuracy')
        plt.title(f'Test-time Compute Scaling\\n{{experiment_id}}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log', base=2)
        
        plt.subplot(1, 2, 2)
        improvement = [weighted_best_of_n_accuracies[n] - best_of_n_accuracies[n] for n in n_completions]
        plt.plot(n_completions, improvement, 'ro-', linewidth=2, markersize=6)
        plt.xlabel('Number of completions (n)')
        plt.ylabel('Accuracy improvement')
        plt.title('Weighted vs Best-of-n Improvement')
        plt.grid(True, alpha=0.3)
        plt.xscale('log', base=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plot_file = os.path.join(output_base, 'evaluation_plots.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f'Plots saved to: {{plot_file}}')
        
    except Exception as e:
        print(f'Warning: Could not generate plots: {{e}}')
    
    print('\\nResults processing completed!')
else:
    print('No datasets found to process!')
"

echo "Results processing completed for experiment {self.experiment_id}"
"""
        
        script_path = self.scripts_dir / f"process_{self.experiment_id}.slurm"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        return str(script_path)
    
    def submit_job(self, script_path: str, dependencies: List[str] = None) -> str:
        """Submit a SLURM job and return the job ID."""
        cmd = ["sbatch"]
        
        if dependencies:
            dep_string = ":".join(dependencies)
            cmd.extend(["--dependency", f"afterok:{dep_string}"])
        
        cmd.append(script_path)
        
        print(f"📤 Submitting job: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Extract job ID from sbatch output (format: "Submitted batch job 12345")
            job_id = result.stdout.strip().split()[-1]
            print(f"✅ Job submitted successfully with ID: {job_id}")
            return job_id
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to submit job: {e}")
            print(f"Error output: {e.stderr}")
            raise
    
    def run_step(self, step_name: str, dry_run: bool = False) -> Optional[str]:
        """Run a specific step of the pipeline."""
        print(f"\n🔧 Preparing {step_name} step...")
        
        if step_name == "training":
            script_path = self.create_training_script()
            dependencies = []
        elif step_name == "upload":
            script_path = self.create_upload_script()
            dependencies = [self.job_ids.get("training")] if "training" in self.job_ids else []
        elif step_name == "evaluation":
            script_path = self.create_evaluation_script()
            dependencies = [self.job_ids.get("upload")] if "upload" in self.job_ids else []
        elif step_name == "processing":
            script_path = self.create_processing_script()
            dependencies = [self.job_ids.get("evaluation")] if "evaluation" in self.job_ids else []
        else:
            raise ValueError(f"Unknown step: {step_name}")
        
        print(f"📄 Script created: {script_path}")
        
        if dry_run:
            print(f"🔍 DRY RUN: Would submit {script_path} with dependencies: {dependencies}")
            return "DRY_RUN_JOB_ID"
        else:
            # Filter out None dependencies
            valid_dependencies = [dep for dep in dependencies if dep is not None]
            job_id = self.submit_job(script_path, valid_dependencies)
            self.job_ids[step_name] = job_id
            return job_id
    
    def run_full_pipeline(self, dry_run: bool = False):
        """Run the complete pipeline."""
        print(f"\n🚀 Starting full pipeline for experiment: {self.experiment_id}")
        
        steps = ["training", "upload", "evaluation", "processing"]
        
        for step in steps:
            self.run_step(step, dry_run)
            
        print(f"\n✅ Pipeline submission completed!")
        print(f"📊 Job IDs: {self.job_ids}")
        
        if not dry_run:
            print(f"\n📋 Monitor jobs with:")
            print(f"   squeue -u $USER")
            print(f"📁 Check logs in: {self.logs_dir}")
            print(f"📈 Results will be saved to: {self.config['experiment']['output_base_dir']}/{self.experiment_id}")

def main():
    parser = argparse.ArgumentParser(description="Orchestrate DAFT experiments")
    parser.add_argument("config", help="Path to experiment configuration file")
    parser.add_argument("--step", choices=["training", "upload", "evaluation", "processing"],
                       help="Run only a specific step")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be submitted without actually submitting")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)
    
    orchestrator = ExperimentOrchestrator(args.config)
    
    try:
        if args.step:
            orchestrator.run_step(args.step, args.dry_run)
        else:
            orchestrator.run_full_pipeline(args.dry_run)
    except Exception as e:
        print(f"❌ Error during orchestration: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
