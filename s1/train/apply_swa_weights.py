#!/usr/bin/env python3
"""
Script to apply SWA averaged weights to a trained model.

This script loads the SWA averaged weights and creates a new model checkpoint
with the averaged weights applied. This is especially useful for distributed
training where loading weights during training can cause issues.

Usage:
    python apply_swa_weights.py --model_dir /path/to/model --swa_checkpoint /path/to/swa_weights.pt
"""

import argparse
import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def apply_swa_weights(model_dir, swa_checkpoint_path, output_dir=None):
    """
    Apply SWA averaged weights to a model and save the result.
    
    Args:
        model_dir: Path to the trained model directory
        swa_checkpoint_path: Path to the SWA averaged weights checkpoint
        output_dir: Output directory for the SWA model (default: model_dir + "_swa")
    """
    
    if output_dir is None:
        output_dir = model_dir + "_swa"
    
    logging.info(f"Loading model from: {model_dir}")
    logging.info(f"Loading SWA weights from: {swa_checkpoint_path}")
    logging.info(f"Output directory: {output_dir}")
    
    # Load the original model
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        logging.info("Model and tokenizer loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return False
    
    # Load SWA weights
    try:
        swa_state_dict = torch.load(swa_checkpoint_path, map_location='cpu')
        logging.info(f"SWA weights loaded: {len(swa_state_dict)} parameters")
    except Exception as e:
        logging.error(f"Failed to load SWA weights: {e}")
        return False
    
    # Apply SWA weights to model
    loaded_count = 0
    skipped_count = 0
    missing_count = 0
    
    model_state_dict = model.state_dict()
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in swa_state_dict:
                swa_param = swa_state_dict[name]
                
                # Check shapes match
                if param.shape != swa_param.shape:
                    logging.warning(f"Shape mismatch for {name}: model={param.shape}, swa={swa_param.shape}")
                    skipped_count += 1
                    continue
                
                try:
                    param.data.copy_(swa_param.to(param.device))
                    loaded_count += 1
                except Exception as e:
                    logging.warning(f"Failed to load parameter {name}: {e}")
                    skipped_count += 1
            else:
                missing_count += 1
    
    logging.info(f"Weight loading summary:")
    logging.info(f"  - Loaded: {loaded_count} parameters")
    logging.info(f"  - Skipped (shape mismatch): {skipped_count} parameters")
    logging.info(f"  - Missing in SWA: {missing_count} parameters")
    
    if loaded_count == 0:
        logging.error("No weights were loaded! Check that the SWA checkpoint matches the model.")
        return False
    
    # Save the model with SWA weights
    try:
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save a summary file
        summary_path = os.path.join(output_dir, "swa_application_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("SWA Weight Application Summary\n")
            f.write("=============================\n")
            f.write(f"Original model: {model_dir}\n")
            f.write(f"SWA checkpoint: {swa_checkpoint_path}\n")
            f.write(f"Parameters loaded: {loaded_count}\n")
            f.write(f"Parameters skipped: {skipped_count}\n")
            f.write(f"Parameters missing: {missing_count}\n")
        
        logging.info(f"SWA model saved to: {output_dir}")
        logging.info(f"Summary saved to: {summary_path}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to save SWA model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Apply SWA averaged weights to a trained model")
    parser.add_argument("--model_dir", required=True, help="Path to the trained model directory")
    parser.add_argument("--swa_checkpoint", required=True, help="Path to the SWA averaged weights checkpoint (.pt file)")
    parser.add_argument("--output_dir", help="Output directory for the SWA model (default: model_dir + '_swa')")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_dir):
        logging.error(f"Model directory not found: {args.model_dir}")
        return 1
    
    if not os.path.exists(args.swa_checkpoint):
        logging.error(f"SWA checkpoint not found: {args.swa_checkpoint}")
        return 1
    
    # Apply SWA weights
    success = apply_swa_weights(args.model_dir, args.swa_checkpoint, args.output_dir)
    
    if success:
        logging.info("SWA weights applied successfully!")
        return 0
    else:
        logging.error("Failed to apply SWA weights")
        return 1

if __name__ == "__main__":
    exit(main())
