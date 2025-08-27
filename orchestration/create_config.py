#!/usr/bin/env python3
"""
DAFT Configuration Generator

This script helps create new experiment configurations based on templates
and parameter variations.

Usage:
    python create_config.py --template quick_test --name my_experiment
    python create_config.py --template experiment --model Qwen/Qwen2.5-7B-Instruct --lr 2e-5
    python create_config.py --batch-study --base-config experiment_config.yaml
"""

import argparse
import yaml
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import copy

class ConfigGenerator:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.configs_dir = self.base_dir / "configs"
        
        # Template configurations
        self.templates = {
            'quick_test': 'quick_test_config.yaml',
            'experiment': 'experiment_config.yaml', 
            'large_scale': 'large_scale_config.yaml'
        }
    
    def load_template(self, template_name: str) -> Dict[str, Any]:
        """Load a template configuration."""
        if template_name in self.templates:
            template_file = self.configs_dir / self.templates[template_name]
        else:
            template_file = Path(template_name)
        
        if not template_file.exists():
            raise ValueError(f"Template file not found: {template_file}")
        
        with open(template_file, 'r') as f:
            return yaml.safe_load(f)
    
    def create_single_config(self, template_name: str, output_name: str, 
                           modifications: Dict[str, Any]) -> str:
        """Create a single configuration file with modifications."""
        config = self.load_template(template_name)
        
        # Apply modifications
        config = self.apply_modifications(config, modifications)
        
        # Update experiment name if provided
        if 'name' in modifications:
            config['experiment']['name'] = modifications['name']
        
        # Save configuration
        output_path = self.configs_dir / f"{output_name}.yaml"
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        return str(output_path)
    
    def apply_modifications(self, config: Dict[str, Any], 
                          modifications: Dict[str, Any]) -> Dict[str, Any]:
        """Apply modifications to a configuration."""
        result = copy.deepcopy(config)
        
        # Handle nested modifications using dot notation
        for key, value in modifications.items():
            if '.' in key:
                # Handle nested keys like 'training.learning_rate'
                parts = key.split('.')
                current = result
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                # Handle simple modifications
                if key == 'model' or key == 'model_name':
                    result['training']['model_name'] = value
                elif key == 'lr' or key == 'learning_rate':
                    result['training']['learning_rate'] = float(value)
                elif key == 'epochs':
                    result['training']['epochs'] = int(value)
                elif key == 'batch_size':
                    result['training']['batch_size'] = int(value)
                elif key == 'topk_k':
                    result['training']['topk_k'] = int(value)
                elif key == 'eval_n' or key == 'n':
                    result['evaluation']['n'] = int(value)
                elif key == 'dataset_end':
                    result['evaluation']['dataset_end'] = int(value)
                elif key == 'wandb_project':
                    result['training']['wandb_project'] = value
                elif key == 'hub_model_id':
                    result['training']['hub_model_id'] = value
                elif key == 'hub_dataset_id':
                    result['evaluation']['hub_dataset_id'] = value
                elif key == 'name':
                    # Will be handled separately
                    pass
                else:
                    print(f"Warning: Unknown modification key: {key}")
        
        return result
    
    def create_batch_study(self, base_config: str, variations: Dict[str, List[Any]],
                          study_name: str) -> List[str]:
        """Create multiple configurations for a parameter study."""
        configs = []
        
        # Generate all combinations
        import itertools
        
        keys = list(variations.keys())
        values = list(variations.values())
        
        for combination in itertools.product(*values):
            modifications = dict(zip(keys, combination))
            
            # Create descriptive name
            name_parts = [study_name]
            for key, value in modifications.items():
                if key == 'learning_rate' or key == 'lr':
                    name_parts.append(f"lr{value}")
                elif key == 'batch_size':
                    name_parts.append(f"bs{value}")
                elif key == 'epochs':
                    name_parts.append(f"ep{value}")
                elif key == 'topk_k':
                    name_parts.append(f"k{value}")
                elif key == 'model_name' or key == 'model':
                    model_short = value.split('/')[-1].replace('-', '').lower()
                    name_parts.append(model_short)
            
            config_name = "_".join(name_parts)
            modifications['name'] = config_name
            
            output_path = self.create_single_config(base_config, config_name, modifications)
            configs.append(output_path)
            
            print(f"✅ Created: {config_name}")
        
        return configs
    
    def create_model_comparison(self, models: List[str], base_template: str = 'experiment') -> List[str]:
        """Create configurations for comparing different models."""
        configs = []
        
        for model in models:
            model_short = model.split('/')[-1].replace('-', '_').lower()
            modifications = {
                'model_name': model,
                'name': f"model_comparison_{model_short}",
                'wandb_project': f"model_comparison_{model_short}",
                'hub_model_id': f"kheuton/{model_short}_s1_custom",
                'hub_dataset_id': f"kheuton/{model_short}_s1_bon_completions"
            }
            
            # Adjust parameters based on model size
            if '7B' in model:
                modifications.update({
                    'learning_rate': 5e-6,  # Lower LR for larger models
                    'batch_size': 8,
                    'epochs': 3,
                    'topk_k': 256
                })
            elif '32B' in model:
                modifications.update({
                    'learning_rate': 2e-6,
                    'batch_size': 4,
                    'epochs': 2,
                    'topk_k': 512
                })
            
            config_name = f"model_comparison_{model_short}"
            output_path = self.create_single_config(base_template, config_name, modifications)
            configs.append(output_path)
            
            print(f"✅ Created: {config_name}")
        
        return configs
    
    def interactive_config_builder(self):
        """Interactive configuration builder."""
        print("🔧 Interactive Configuration Builder")
        print("==================================")
        
        # Get template
        print("\nAvailable templates:")
        for name, file in self.templates.items():
            print(f"  {name}: {file}")
        
        template = input("Choose template (or path to custom template): ").strip()
        if not template:
            template = 'experiment'
        
        # Get experiment name
        name = input("Experiment name: ").strip()
        if not name:
            name = "interactive_experiment"
        
        modifications = {'name': name}
        
        # Get modifications
        print("\nConfiguration options (press Enter to skip):")
        
        model = input("Model name [Qwen/Qwen2.5-1.5B-Instruct]: ").strip()
        if model:
            modifications['model_name'] = model
        
        lr = input("Learning rate [1e-5]: ").strip()
        if lr:
            modifications['learning_rate'] = float(lr)
        
        epochs = input("Epochs [5]: ").strip()
        if epochs:
            modifications['epochs'] = int(epochs)
        
        batch_size = input("Batch size [16]: ").strip()
        if batch_size:
            modifications['batch_size'] = int(batch_size)
        
        topk_k = input("Top-k parameter [128]: ").strip()
        if topk_k:
            modifications['topk_k'] = int(topk_k)
        
        eval_n = input("Evaluation n [128]: ").strip()
        if eval_n:
            modifications['n'] = int(eval_n)
        
        wandb_project = input("W&B project name: ").strip()
        if wandb_project:
            modifications['wandb_project'] = wandb_project
        
        # Create configuration
        try:
            output_path = self.create_single_config(template, name, modifications)
            print(f"\n✅ Configuration created: {output_path}")
            
            # Ask if user wants to run it
            run_now = input("\nRun experiment now? [y/N]: ").strip().lower()
            if run_now == 'y':
                dry_run = input("Dry run first? [Y/n]: ").strip().lower()
                dry_flag = "" if dry_run == 'n' else "--dry-run"
                
                cmd = f"python {self.base_dir}/orchestrate_experiment.py {output_path} {dry_flag}"
                print(f"Running: {cmd}")
                os.system(cmd)
                
        except Exception as e:
            print(f"❌ Error creating configuration: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate DAFT experiment configurations")
    parser.add_argument('--template', choices=['quick_test', 'experiment', 'large_scale'],
                       default='experiment', help='Base template to use')
    parser.add_argument('--name', help='Experiment name')
    parser.add_argument('--model', help='Model name')
    parser.add_argument('--lr', '--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--topk-k', type=int, help='Top-k parameter')
    parser.add_argument('--eval-n', type=int, help='Evaluation n parameter')
    parser.add_argument('--wandb-project', help='W&B project name')
    parser.add_argument('--hub-model-id', help='HuggingFace model ID')
    parser.add_argument('--hub-dataset-id', help='HuggingFace dataset ID')
    
    # Study options
    parser.add_argument('--batch-study', help='Create batch study from base config')
    parser.add_argument('--lr-study', help='Learning rate study', 
                       metavar='1e-6,5e-6,1e-5,2e-5')
    parser.add_argument('--model-comparison', help='Compare multiple models',
                       metavar='model1,model2,model3')
    
    # Interactive mode
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive configuration builder')
    
    args = parser.parse_args()
    
    generator = ConfigGenerator()
    
    if args.interactive:
        generator.interactive_config_builder()
        return
    
    if args.model_comparison:
        models = args.model_comparison.split(',')
        configs = generator.create_model_comparison(models, args.template)
        print(f"\n✅ Created {len(configs)} model comparison configurations")
        return
    
    if args.lr_study:
        learning_rates = [float(lr) for lr in args.lr_study.split(',')]
        variations = {'learning_rate': learning_rates}
        study_name = args.name or 'lr_study'
        configs = generator.create_batch_study(args.template, variations, study_name)
        print(f"\n✅ Created {len(configs)} learning rate study configurations")
        return
    
    if args.batch_study:
        # Example batch study - you can customize this
        variations = {
            'learning_rate': [1e-5, 2e-5, 5e-5],
            'batch_size': [8, 16, 32]
        }
        study_name = args.name or 'batch_study'
        configs = generator.create_batch_study(args.batch_study, variations, study_name)
        print(f"\n✅ Created {len(configs)} batch study configurations")
        return
    
    # Single configuration
    if not args.name:
        print("❌ Error: --name is required for single configurations")
        sys.exit(1)
    
    modifications = {}
    for arg_name, config_key in [
        ('model', 'model_name'),
        ('lr', 'learning_rate'),
        ('epochs', 'epochs'),
        ('batch_size', 'batch_size'),
        ('topk_k', 'topk_k'),
        ('eval_n', 'n'),
        ('wandb_project', 'wandb_project'),
        ('hub_model_id', 'hub_model_id'),
        ('hub_dataset_id', 'hub_dataset_id')
    ]:
        value = getattr(args, arg_name.replace('-', '_'))
        if value is not None:
            modifications[config_key] = value
    
    modifications['name'] = args.name
    
    try:
        output_path = generator.create_single_config(args.template, args.name, modifications)
        print(f"✅ Configuration created: {output_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
