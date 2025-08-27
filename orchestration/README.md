# DAFT Experiment Orchestration System

This orchestration system automates the complete pipeline for training and evaluating models using the DAFT (Data Augmentation with Fine-Tuning) methodology. It eliminates manual steps and makes it easy to run experiments at scale.

## 🚀 Quick Start

### 1. Test the System
```bash
# Navigate to the orchestration directory
cd /cluster/tufts/hugheslab/kheuto01/code/daft/orchestration

# Run a quick test (dry run)
python orchestrate_experiment.py configs/quick_test_config.yaml --dry-run

# Run the actual test experiment
python orchestrate_experiment.py configs/quick_test_config.yaml
```

### 2. Monitor Your Experiment
```bash
# Check all experiments
python manage_experiments.py list

# Watch a specific experiment in real-time
python manage_experiments.py watch <experiment_id>

# Check SLURM queue
squeue -u $USER
```

### 3. Run Full-Scale Experiments
```bash
# Production experiment
python orchestrate_experiment.py configs/experiment_config.yaml

# Large-scale study
python orchestrate_experiment.py configs/large_scale_config.yaml
```

## 📋 Pipeline Overview

The orchestration system automates these 4 steps:

1. **Training** (`s1` environment)
   - Trains model using custom S1 loss with configurable parameters
   - Saves model checkpoint locally
   - Automatically uploads to HuggingFace Hub

2. **Model Upload** (`s1` environment)
   - Creates model card with experiment metadata
   - Pushes trained model to HuggingFace Hub
   - Makes model available for evaluation

3. **Evaluation** (`sal` environment)
   - Runs test-time compute evaluation using the trained model
   - Generates completions using best-of-n approach
   - Saves evaluation chunks locally and optionally to Hub

4. **Results Processing** (`sal` environment)
   - Merges evaluation chunks into single dataset
   - Computes accuracy metrics (best-of-n and weighted best-of-n)
   - Generates scaling plots and analysis

Each step automatically waits for the previous step to complete using SLURM job dependencies.

## 📁 Directory Structure

```
orchestration/
├── configs/                    # Experiment configuration files
│   ├── experiment_config.yaml  # Main production configuration
│   ├── quick_test_config.yaml  # Quick test configuration
│   └── large_scale_config.yaml # Large-scale experiment configuration
├── scripts/                    # Generated SLURM scripts (auto-created)
├── logs/                      # Experiment logs (auto-created)
├── outputs/                   # Experiment outputs (auto-created)
├── orchestrate_experiment.py  # Main orchestration script
├── manage_experiments.py      # Experiment management utilities
└── README.md                  # This file
```

## ⚙️ Configuration

### Main Configuration File

Each experiment is defined by a YAML configuration file with these sections:

```yaml
experiment:
  name: "experiment_name"
  description: "Experiment description"
  output_base_dir: "/path/to/outputs"

training:
  model_name: "Qwen/Qwen2.5-1.5B-Instruct"
  learning_rate: 1e-5
  epochs: 5
  batch_size: 16
  # ... more training parameters

evaluation:
  approach: "best_of_n"
  n: 128
  dataset_start: 0
  dataset_end: 500
  # ... more evaluation parameters

# SLURM resource specifications for each step
# Environment specifications
# Dependency management
```

### Key Configuration Options

#### Training Parameters
- `model_name`: Base model from HuggingFace
- `train_dataset_name`: Dataset for training
- `use_custom_loss`: Enable custom S1 loss
- `topk_k`: Top-k parameter for custom loss
- `wandb_project`: W&B project for logging
- `hub_model_id`: Where to upload trained model

#### Evaluation Parameters
- `approach`: Evaluation method (`best_of_n`, `beam_search`, etc.)
- `n`: Number of completions to generate
- `prm_path`: Process reward model for scoring
- `hub_dataset_id`: Where to upload evaluation results

#### SLURM Configuration
Each step has its own SLURM configuration:
- Resource requirements (nodes, GPUs, memory)
- Time limits
- Partition and constraints
- Job dependencies (automatically managed)

## 🔧 Usage Examples

### Running Individual Steps

```bash
# Run only training
python orchestrate_experiment.py configs/experiment_config.yaml --step training

# Run only evaluation (assumes model is already available)
python orchestrate_experiment.py configs/experiment_config.yaml --step evaluation

# Dry run to see what would be submitted
python orchestrate_experiment.py configs/experiment_config.yaml --dry-run
```

### Managing Experiments

```bash
# List all experiments with status
python manage_experiments.py list

# Show detailed status for specific experiment
python manage_experiments.py status experiment_name_20250827_143022

# Cancel all jobs for an experiment
python manage_experiments.py cancel experiment_name_20250827_143022

# Clean up experiments older than 7 days
python manage_experiments.py cleanup --older-than 7

# Watch experiment progress in real-time
python manage_experiments.py watch experiment_name_20250827_143022
```

### Creating Custom Configurations

1. Copy an existing config file:
```bash
cp configs/experiment_config.yaml configs/my_experiment.yaml
```

2. Modify parameters for your experiment:
   - Change model size/type
   - Adjust training hyperparameters
   - Modify evaluation settings
   - Update SLURM resources based on your requirements

3. Run your experiment:
```bash
python orchestrate_experiment.py configs/my_experiment.yaml
```

## 📊 Monitoring and Results

### Real-time Monitoring

- **SLURM queue**: `squeue -u $USER`
- **Job details**: `scontrol show job <job_id>`
- **Live logs**: `tail -f logs/<experiment_id>/<step>_<job_id>.out`
- **Experiment watcher**: `python manage_experiments.py watch <experiment_id>`

### Results Location

After completion, results are saved in:
```
outputs/<experiment_id>/
├── evaluation_results.json      # Detailed metrics
├── evaluation_plots.png         # Scaling curves
├── <dataset_name>_combined.json # Combined evaluation dataset
└── eval_outputs/                # Raw evaluation chunks
    ├── <timestamp>_task1/
    ├── <timestamp>_task2/
    └── ...
```

### Key Metrics

The system computes:
- **Best-of-n accuracy**: Standard best-of-n scaling
- **Weighted best-of-n accuracy**: Score-weighted aggregation
- **Scaling curves**: Performance vs. number of completions
- **Improvement analysis**: Weighted vs. standard best-of-n

## 🔄 Experiment Lifecycle

1. **Configuration**: Define experiment parameters in YAML
2. **Submission**: Submit pipeline with automatic dependencies
3. **Training**: Model trains and uploads to HuggingFace Hub
4. **Evaluation**: Test-time compute evaluation runs in parallel
5. **Processing**: Results are merged and analyzed
6. **Completion**: All artifacts saved and metrics computed

## 🛠️ Troubleshooting

### Common Issues

1. **Job Dependencies Fail**
   - Check previous step completed successfully
   - Verify SLURM job logs for errors
   - Use `manage_experiments.py status` to diagnose

2. **Resource Allocation**
   - Adjust SLURM parameters in config
   - Check partition availability
   - Modify memory/GPU requirements

3. **Environment Issues**
   - Ensure `s1` and `sal` conda environments exist
   - Check if required packages are installed
   - Verify paths in configuration

4. **HuggingFace Upload Fails**
   - Check HuggingFace token is configured
   - Verify repository permissions
   - Ensure model ID doesn't conflict

### Log Analysis

Check logs in this order:
1. `logs/<experiment_id>/training_<job_id>.out`
2. `logs/<experiment_id>/upload_<job_id>.out`
3. `logs/<experiment_id>/evaluation_<job_id>_<array_id>.out`
4. `logs/<experiment_id>/processing_<job_id>.out`

### Recovery Procedures

**If training fails**: Fix issue and restart from training step
```bash
python orchestrate_experiment.py configs/my_config.yaml --step training
```

**If evaluation fails**: Restart from evaluation (model already uploaded)
```bash
python orchestrate_experiment.py configs/my_config.yaml --step evaluation
```

**If processing fails**: Restart processing only
```bash
python orchestrate_experiment.py configs/my_config.yaml --step processing
```

## 🔧 Advanced Usage

### Multi-Experiment Studies

Run multiple experiments with different configurations:

```bash
# Create variations of base config
for lr in 1e-5 2e-5 5e-5; do
    sed "s/learning_rate: 1e-5/learning_rate: $lr/" \
        configs/experiment_config.yaml > configs/lr_${lr}_config.yaml
    python orchestrate_experiment.py configs/lr_${lr}_config.yaml
done
```

### Custom Scripts

The orchestration system generates SLURM scripts that you can customize:

1. Run with `--dry-run` to generate scripts
2. Modify generated scripts in `scripts/` directory
3. Submit manually with `sbatch scripts/script_name.slurm`

### Integration with External Tools

- **W&B**: Automatic logging during training
- **HuggingFace Hub**: Automatic model and dataset uploads
- **Custom Analysis**: Process results with your own analysis scripts

## 📝 Configuration Templates

### Minimal Test Configuration
```yaml
experiment:
  name: "quick_test"

training:
  model_name: "Qwen/Qwen2.5-1.5B-Instruct"
  epochs: 1
  batch_size: 8

evaluation:
  n: 32
  dataset_end: 100
```

### Production Configuration
```yaml
experiment:
  name: "production_experiment"

training:
  model_name: "Qwen/Qwen2.5-1.5B-Instruct"
  epochs: 5
  batch_size: 16
  topk_k: 128

evaluation:
  n: 128
  dataset_end: 500
```

### Large Scale Configuration
```yaml
experiment:
  name: "large_scale_study"

training:
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  epochs: 3
  batch_size: 32
  topk_k: 256

evaluation:
  n: 256
  dataset_end: 500
```

## 🎯 Best Practices

1. **Start Small**: Use `quick_test_config.yaml` to verify the pipeline works
2. **Monitor Resources**: Check SLURM allocation and adjust as needed
3. **Version Control**: Keep configuration files in git for reproducibility
4. **Naming Convention**: Use descriptive experiment names
5. **Clean Up**: Regularly clean old experiments to save storage
6. **Documentation**: Add experiment descriptions in config files

## 🤝 Contributing

To add new features or modify the orchestration system:

1. Test changes with `quick_test_config.yaml`
2. Update documentation in this README
3. Add example configurations for new features
4. Ensure backward compatibility with existing configs

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review log files for error messages
3. Use `manage_experiments.py` to diagnose issues
4. Check SLURM documentation for resource-related problems

This orchestration system is designed to make running DAFT experiments efficient and reproducible. Happy experimenting! 🚀
