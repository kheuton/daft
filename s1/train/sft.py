import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset, concatenate_datasets, DatasetDict
import transformers
from transformers import TrainerCallback, TrainerState, TrainerControl
import trl
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math


class SimplePredictionWrapper(nn.Module):
    """
    Simplified wrapper around a pretrained model that only implements top_k and top_p filtering.
    This is mainly kept for backward compatibility, but the preferred approach is to use
    custom loss during training and set sampling parameters in VLLM for evaluation.
    """
    
    def __init__(self, base_model, prediction_config=None):
        super().__init__()
        self.base_model = base_model
        self.prediction_config = prediction_config or {}
        
        # Copy important attributes from the base model
        self.config = base_model.config
        self.generation_config = getattr(base_model, 'generation_config', None)
        
    def forward(self, *args, **kwargs):
        # Get outputs from the base model
        outputs = self.base_model(*args, **kwargs)
        
        # If we have logits, modify them according to our prediction strategy
        if hasattr(outputs, 'logits') and outputs.logits is not None:
            modified_logits = self._modify_logits(outputs.logits)
            # Create new outputs with modified logits
            if hasattr(outputs, '_replace'):  # For NamedTuple-like outputs
                outputs = outputs._replace(logits=modified_logits)
            else:  # For dict-like outputs
                outputs.logits = modified_logits
                
        return outputs
    
    def _modify_logits(self, logits):
        """
        Apply only top-k and top-p filtering to logits.
        """
        # Top-k filtering (set non-top-k logits to very negative values)
        if 'top_k' in self.prediction_config:
            k = self.prediction_config['top_k']
            if k > 0 and k < logits.size(-1):
                # Get top-k values and indices
                topk_values, topk_indices = torch.topk(logits, k=k, dim=-1)
                # Create mask for top-k elements
                topk_mask = torch.zeros_like(logits, dtype=torch.bool)
                topk_mask.scatter_(-1, topk_indices, True)
                # Set non-top-k logits to very negative values
                logits = logits.masked_fill(~topk_mask, -1e9)
                
        # Top-p (nucleus) filtering
        if 'top_p' in self.prediction_config:
            top_p = self.prediction_config['top_p']
            if 0 < top_p < 1:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Create the mask in original order
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, -1e9)
                
        return logits
    
    def generate(self, *args, **kwargs):
        """Forward generate calls to the base model"""
        return self.base_model.generate(*args, **kwargs)
    
    def __getattr__(self, name):
        """Forward attribute access to the base model"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


class SWACallback(TrainerCallback):
    """
    Memory-efficient Stochastic Weight Averaging (SWA) callback for large language models.
    
    This implementation is optimized for billion-parameter models by using:
    1. Checkpoint-based weight averaging (saves to disk, loads for averaging)
    2. Streaming average computation to minimize memory usage
    3. Optional batch normalization momentum updates (if applicable)
    
    Memory usage: Only requires storing one additional copy of weights temporarily during averaging.
    """
    
    def __init__(self, swa_config, output_dir=None):
        self.swa_config = swa_config
        self.swa_started = False
        self.swa_n = 0  # Number of models averaged so far
        self.original_scheduler = None
        self.swa_lr_scheduler = None
        self.last_epoch_averaged = -1
        self.output_dir = output_dir
        self.swa_checkpoint_dir = None
        self.swa_model_path = None
        self._debug_param_groups_logged = False  # For debugging parameter groups
        self._current_epoch = 0  # Track current epoch for scheduler
        self._trainer = None  # Store trainer reference
        
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Initialize SWA when training begins."""
        if not self.swa_config.use_swa:
            return
            
        # Check if we're in distributed training
        import torch.distributed as dist
        is_distributed = dist.is_available() and dist.is_initialized()
        is_main_process = not is_distributed or dist.get_rank() == 0
        
        self.output_dir = args.output_dir
        self.swa_checkpoint_dir = os.path.join(self.output_dir, "swa_checkpoints")
        self.swa_model_path = os.path.join(self.swa_checkpoint_dir, "swa_averaged_weights.pt")
        
        # Only main process creates directories to avoid race conditions
        if is_main_process:
            os.makedirs(self.swa_checkpoint_dir, exist_ok=True)
        
        # Synchronize all processes if distributed
        if is_distributed:
            dist.barrier()
        
        if is_main_process:
            logging.info(f"Memory-efficient SWA enabled: will start at epoch {self.swa_config.swa_start_epoch}")
            logging.info(f"SWA LR: {self.swa_config.swa_lr}, Schedule: {self.swa_config.swa_schedule_type}")
            logging.info(f"SWA frequency: every {self.swa_config.swa_freq} epoch(s)")
            logging.info(f"SWA checkpoints will be saved to: {self.swa_checkpoint_dir}")
            logging.info("Using checkpoint-based averaging for memory efficiency")
            if is_distributed:
                logging.info(f"Distributed training detected: using rank 0 for checkpoint operations")
    
    def on_epoch_end(self, args, state, control, model=None, optimizer=None, lr_scheduler=None, **kwargs):
        """Handle SWA logic at the end of each epoch."""
        if not self.swa_config.use_swa:
            return
            
        current_epoch = int(state.epoch)
        
        # Start SWA if we've reached the start epoch
        if current_epoch >= self.swa_config.swa_start_epoch and not self.swa_started:
            logging.info(f"Starting SWA at epoch {current_epoch}")
            self.swa_started = True
            
        # If SWA is active, handle weight averaging
        if self.swa_started:
            # Log current learning rate from the trainer's scheduler (informational only)
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Epoch {current_epoch}: Current LR: {current_lr:.2e}")
            
            # Check if we should average weights this epoch
            epochs_since_start = current_epoch - self.swa_config.swa_start_epoch
            if epochs_since_start % self.swa_config.swa_freq == 0:
                self._update_swa_checkpoint_based(model)
                self.last_epoch_averaged = current_epoch
                logging.info(f"Updated SWA model at epoch {current_epoch} (SWA model #{self.swa_n})")
                
    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Finalize SWA by preparing the averaged weights."""
        if not self.swa_config.use_swa or not self.swa_started or self.swa_n == 0:
            return
            
        # Check if we're in distributed training
        import torch.distributed as dist
        is_distributed = dist.is_available() and dist.is_initialized()
        is_main_process = not is_distributed or dist.get_rank() == 0
        
        if is_main_process:
            logging.info(f"Finalizing SWA: averaged {self.swa_n} models")
        
        # For distributed training, we avoid loading weights back into the model
        # during training end, as this can cause issues with FSDP and other
        # distributed strategies. Instead, we just ensure the SWA checkpoint
        # is available for manual loading later.
        
        if is_distributed:
            if is_main_process:
                logging.info("SWA: In distributed mode, averaged weights saved to checkpoint.")
                logging.info(f"SWA: To use averaged weights, manually load from: {self.swa_model_path}")
                logging.info("SWA: The final model will contain the last training weights, not averaged weights.")
            dist.barrier()
        else:
            # In single-GPU mode, we can safely load the averaged weights
            if os.path.exists(self.swa_model_path):
                self._load_swa_weights_to_model(model)
            else:
                logging.warning("SWA: No averaged weights found to load")
        
        # Save information about SWA for later use
        if is_main_process:
            swa_info_path = os.path.join(self.output_dir, "swa_info.txt")
            with open(swa_info_path, 'w') as f:
                f.write(f"SWA Training Summary\n")
                f.write(f"==================\n")
                f.write(f"Models averaged: {self.swa_n}\n")
                f.write(f"SWA checkpoint: {self.swa_model_path}\n")
                f.write(f"Last epoch averaged: {self.last_epoch_averaged}\n")
                f.write(f"SWA config: {self.swa_config}\n")
                f.write(f"\nTo load averaged weights:\n")
                f.write(f"  swa_weights = torch.load('{self.swa_model_path}')\n")
                f.write(f"  model.load_state_dict(swa_weights, strict=False)\n")
            logging.info(f"SWA info saved to: {swa_info_path}")
        
        # Synchronize all processes if distributed
        if is_distributed:
            dist.barrier()
    '''    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Ensure WandB logs the correct learning rate during SWA."""
        if not self.swa_config.use_swa or not self.swa_started or logs is None:
            return
            
        # Check if we have optimizer info available in kwargs
        optimizer = kwargs.get('optimizer', None)
        if optimizer is not None:
            # Get current learning rate from optimizer
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update logs to ensure WandB gets the correct learning rate
            logs['learning_rate'] = current_lr
            logs['train/learning_rate'] = current_lr  # Some versions use this key
            logs['swa_learning_rate'] = current_lr  # Our custom metric
            
            # Log SWA-specific info
            logs['swa_n_averaged'] = self.swa_n
            logs['swa_started'] = 1.0 if self.swa_started else 0.0'''
        
                
    def _update_swa_checkpoint_based(self, model):
        """
        Update SWA model using checkpoint-based averaging for memory efficiency.
        This approach handles distributed training by only allowing rank 0 to update weights.
        """
        # Check if we're in distributed training
        import torch.distributed as dist
        is_distributed = dist.is_available() and dist.is_initialized()
        is_main_process = not is_distributed or dist.get_rank() == 0
        
        # Handle wrapped models
        if hasattr(model, 'module'):
            base_model = model.module
        else:
            base_model = model
            
        self.swa_n += 1
        
        # Only the main process (rank 0) handles checkpoint operations to avoid corruption
        if is_main_process:
            # Extract current model state dict with FSDP-compatible approach
            current_state_dict = {}
            
            # Check if we're using FSDP
            try:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                from torch.distributed.fsdp import StateDictType, FullStateDictConfig
                is_fsdp = isinstance(base_model, FSDP)
            except ImportError:
                is_fsdp = False
            
            if is_fsdp:
                # FSDP path: Use proper state dict gathering
                logging.info("SWA: Using FSDP-compatible state dict extraction")
                with FSDP.state_dict_type(base_model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
                    full_state_dict = base_model.state_dict()
                
                # Filter for trainable parameters only
                for name, param in base_model.named_parameters():
                    if param.requires_grad and name in full_state_dict:
                        current_state_dict[name] = full_state_dict[name].clone()
                        
                logging.info(f"SWA: Extracted {len(current_state_dict)} FSDP parameters")
            else:
                # Non-FSDP path: Use regular parameter extraction
                logging.info("SWA: Using regular state dict extraction")
                for name, param in base_model.named_parameters():
                    if param.requires_grad:
                        current_state_dict[name] = param.data.cpu().clone()
            
            if self.swa_n == 1:
                # First model: just save current weights as the SWA weights
                torch.save(current_state_dict, self.swa_model_path)
                logging.info("SWA: Saved first model weights")
            else:
                # Load existing SWA weights with retry logic for robustness
                max_retries = 3
                swa_state_dict = None
                
                for attempt in range(max_retries):
                    try:
                        if os.path.exists(self.swa_model_path):
                            swa_state_dict = torch.load(self.swa_model_path, map_location='cpu')
                            break
                    except (RuntimeError, EOFError, IOError) as e:
                        logging.warning(f"SWA: Attempt {attempt + 1} failed to load checkpoint: {e}")
                        if attempt < max_retries - 1:
                            import time
                            time.sleep(0.5)  # Brief delay before retry
                        else:
                            logging.error("SWA: Failed to load checkpoint after retries, reinitializing")
                            swa_state_dict = None
                
                if swa_state_dict is not None:
                    # Update running average: new_avg = (old_avg * (n-1) + current) / n
                    alpha = 1.0 / self.swa_n  # Weight for current model
                    
                    for name in current_state_dict:
                        if name in swa_state_dict:
                            swa_state_dict[name] = (1 - alpha) * swa_state_dict[name] + alpha * current_state_dict[name]
                        else:
                            swa_state_dict[name] = current_state_dict[name]
                    
                    # Save updated average with atomic write (write to temp file, then rename)
                    temp_path = self.swa_model_path + '.tmp'
                    try:
                        torch.save(swa_state_dict, temp_path)
                        os.rename(temp_path, self.swa_model_path)  # Atomic operation on most filesystems
                        logging.info(f"SWA: Updated averaged weights (n={self.swa_n})")
                    except Exception as e:
                        logging.error(f"SWA: Failed to save checkpoint: {e}")
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    
                    del swa_state_dict  # Free memory
                else:
                    # Fallback: save current as first
                    torch.save(current_state_dict, self.swa_model_path)
                    logging.warning("SWA: Previous weights corrupted/missing, saving current as first")
            
            del current_state_dict  # Free memory
        
        # Synchronize all processes if distributed
        if is_distributed:
            dist.barrier()
            logging.info(f"SWA: Rank {dist.get_rank()} synchronized after checkpoint update")
        
    def _load_swa_weights_to_model(self, model):
        """Load the averaged SWA weights back into the model."""
        if not os.path.exists(self.swa_model_path):
            logging.warning("SWA weights file not found, skipping weight loading")
            return
            
        # Check if we're in distributed training
        import torch.distributed as dist
        is_distributed = dist.is_available() and dist.is_initialized()
        is_main_process = not is_distributed or dist.get_rank() == 0
        
        # Only load weights on main process to avoid distributed training issues
        if not is_main_process:
            if is_distributed:
                dist.barrier()  # Wait for main process to finish
            return
        
        # Handle wrapped models
        if hasattr(model, 'module'):
            base_model = model.module
        else:
            base_model = model
            
        # Load SWA weights with retry logic
        swa_state_dict = None
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                swa_state_dict = torch.load(self.swa_model_path, map_location='cpu')
                break
            except (RuntimeError, EOFError, IOError) as e:
                logging.warning(f"SWA: Attempt {attempt + 1} failed to load final weights: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(0.5)
                else:
                    logging.error("SWA: Failed to load final weights after retries")
                    if is_distributed:
                        dist.barrier()
                    return
        
        if swa_state_dict is None:
            logging.error("SWA: Could not load averaged weights")
            if is_distributed:
                dist.barrier()
            return
        
        # Copy weights back to model with careful shape checking
        loaded_count = 0
        skipped_count = 0
        
        with torch.no_grad():
            for name, param in base_model.named_parameters():
                if param.requires_grad and name in swa_state_dict:
                    swa_param = swa_state_dict[name]
                    
                    # Check shapes match
                    if param.shape != swa_param.shape:
                        logging.warning(f"SWA: Shape mismatch for {name}: model={param.shape}, swa={swa_param.shape}, skipping")
                        skipped_count += 1
                        continue
                    
                    try:
                        param.data.copy_(swa_param.to(param.device))
                        loaded_count += 1
                    except Exception as e:
                        logging.warning(f"SWA: Failed to load parameter {name}: {e}")
                        skipped_count += 1
                        continue
                        
        del swa_state_dict  # Free memory
        
        logging.info(f"SWA averaged weights loaded into model: {loaded_count} parameters loaded, {skipped_count} skipped")
        
        # Synchronize all processes if distributed
        if is_distributed:
            dist.barrier()
                    
    def get_swa_info(self):
        """Return information about the current SWA state."""
        return {
            'swa_started': self.swa_started,
            'swa_n': self.swa_n,
            'last_epoch_averaged': self.last_epoch_averaged,
            'swa_config': self.swa_config,
            'checkpoint_dir': self.swa_checkpoint_dir,
            'swa_model_path': self.swa_model_path,
            'memory_efficient': True,
            'method': 'checkpoint_based'
        }


@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-32B-Instruct")
    block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default="Qwen2.5-1.5B-Instruct-s1-top128")
    wandb_entity: Optional[str] = field(default="wandb_kheuton")
    train_file_path: Optional[str] = field(default='simplescaling/s1K_tokenized')
    dagger: bool = field(default=False)
    
    # Custom loss configuration
    use_custom_loss: bool = field(default=False)
    loss_type: str = field(default="cross_entropy")  # Options: "cross_entropy", "focal", "label_smoothing", "topk_cross_entropy", etc.
    focal_alpha: float = field(default=1.0)
    focal_gamma: float = field(default=2.0)
    label_smoothing: float = field(default=0.1)
    
    # Top-k parameters for custom loss
    topk_k: int = field(default=50)  # Number of top predictions to keep
    topk_temperature: float = field(default=1.0)  # Temperature for softmax before top-k filtering
    
    # Simplified prediction behavior configuration (legacy support)
    # Note: Prefer using custom loss instead of prediction wrappers
    use_simple_prediction_wrapper: bool = field(default=False)
    prediction_top_k: int = field(default=0)  # Top-k filtering (0 = disabled)
    prediction_top_p: float = field(default=1.0)  # Top-p filtering (1.0 = disabled)
    
    # Stochastic Weight Averaging (SWA) configuration
    use_swa: bool = field(default=False)
    swa_start_epoch: int = field(default=2)  # Epoch to start SWA (0-indexed)
    swa_lr: float = field(default=1e-5)  # SWA learning rate (typically lower than base LR)
    swa_freq: int = field(default=1)  # Frequency of model averaging (in epochs)
    swa_schedule_type: str = field(default="cyclic")  # "cyclic" or "constant"
    swa_cycle_length: int = field(default=2)  # Length of LR cycle for cyclic schedule (in epochs)
    swa_lr_min_ratio: float = field(default=0.1)  # Min LR as ratio of swa_lr for cyclic schedule

    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project
        os.environ['WANDB_ENTITY'] = self.wandb_entity


class CustomSFTTrainer(trl.SFTTrainer):
    """Custom SFT Trainer with configurable loss functions and SWA-aware learning rate scheduling"""
    
    def __init__(self, loss_config, swa_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_config = loss_config
        self.swa_config = swa_config
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        Based on the original SFTTrainer compute_loss method.
        """
        # Original logic: Handle labels and compute_loss_func
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
            
        # Original logic: Handle model loss kwargs
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
            
        # Original logic: Get model outputs
        outputs = model(**inputs)
        
        # Original logic: Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            
            # Check for PEFT model (from original)
            try:
                from trl.trainer.sft_trainer import _is_peft_model
                is_peft = _is_peft_model(unwrapped_model)
            except ImportError:
                # Fallback for different TRL versions
                is_peft = hasattr(unwrapped_model, 'base_model')
                
            if is_peft:
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
                
            # CUSTOM MODIFICATION: Check if we should use custom loss
            if self.loss_config.use_custom_loss:
                loss = self._compute_custom_loss(outputs, labels, num_items_in_batch)
            # Original logic: User-defined compute_loss function
            elif self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            # Original logic: Use appropriate loss based on model type
            else:
                try:
                    from transformers import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
                except ImportError:
                    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = {}
                    
                if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    loss = self.label_smoother(outputs, labels)
        else:
            # Original logic: Handle case where no labels
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]



        return (loss, outputs) if return_outputs else loss
    
    def _compute_custom_loss(self, outputs, labels, num_items_in_batch=None):
        """
        Compute custom loss based on configuration.
        This mimics what the label_smoother would do but with custom loss functions.
        """
        try:
            # Extract logits from outputs
            if isinstance(outputs, dict):
                logits = outputs.get("logits")
            else:
                logits = outputs[0] if hasattr(outputs, '__getitem__') else outputs.logits
            
            if logits is None:
                raise ValueError("Could not extract logits from model outputs")
            
            # Shift labels and logits for causal language modeling (shift_labels=True behavior)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            # Only compute loss on non-ignored tokens (labels != -100)
            valid_mask = shift_labels != -100
            
            if not valid_mask.any():
                logging.warning("No valid tokens found! Returning zero loss.")
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
            # Filter to only valid tokens
            valid_logits = shift_logits[valid_mask]
            valid_labels = shift_labels[valid_mask]
            
            # Apply the specific loss function
            if self.loss_config.loss_type == "focal":
                return self._focal_loss(valid_logits, valid_labels)
            elif self.loss_config.loss_type == "label_smoothing":
                return self._label_smoothing_loss(valid_logits, valid_labels)
            elif self.loss_config.loss_type == "topk_cross_entropy":
                return self._topk_cross_entropy_loss(valid_logits, valid_labels)
            else:
                # Default cross entropy
                return F.cross_entropy(valid_logits, valid_labels)
                
        except Exception as e:
            logging.error(f"Error in custom loss computation: {e}")
            # Fallback to default loss computation using label_smoother
            return self.label_smoother(outputs, labels, shift_labels=True)
    
    def _focal_loss(self, logits, labels):
        """Focal Loss implementation for handling class imbalance"""
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.loss_config.focal_alpha * (1 - pt) ** self.loss_config.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def _label_smoothing_loss(self, logits, labels):
        """Label smoothing loss implementation"""
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = F.nll_loss(log_probs, labels, reduction='none')
        
        # Apply label smoothing
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.loss_config.label_smoothing) * nll_loss + self.loss_config.label_smoothing * smooth_loss
        return loss.mean()
    
    def _topk_cross_entropy_loss(self, logits, labels):
        """
        Top-k cross-entropy loss: rescale probabilities so only top-k predictions have mass
        """
        try:
            # Apply temperature scaling if specified
            scaled_logits = logits / self.loss_config.topk_temperature
            
            # Ensure k doesn't exceed vocabulary size
            vocab_size = scaled_logits.size(-1)
            k = min(self.loss_config.topk_k, vocab_size)
            
            # Get top-k values and indices
            topk_values, topk_indices = torch.topk(scaled_logits, k=k, dim=-1)
            
            # Create a mask for top-k elements
            topk_mask = torch.zeros_like(scaled_logits, dtype=torch.bool)
            topk_mask.scatter_(-1, topk_indices, True)
            
            # Set non-top-k logits to very negative values (effectively zero probability)
            masked_logits = scaled_logits.clone()
            masked_logits[~topk_mask] = -1e9  # Use -1e9 instead of -inf for numerical stability
            
            # Compute cross-entropy with the masked logits
            loss = F.cross_entropy(masked_logits, labels)
            
            return loss
            
        except Exception as e:
            logging.error(f"Error in top-k loss: {e}, falling back to regular cross-entropy")
            return F.cross_entropy(logits, labels)
    
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """
        Create a custom learning rate scheduler that handles SWA scheduling.
        
        This method is called by the Trainer to create the learning rate scheduler.
        For SWA, we create a scheduler that implements:
        1. Normal LR schedule before SWA starts
        2. SWA-specific schedule during the SWA phase
        """
        from torch.optim.lr_scheduler import LambdaLR
        import math
        
        if not self.swa_config or not self.swa_config.use_swa:
            # No SWA, use default scheduler
            return super().create_scheduler(num_training_steps, optimizer)
        
        # Calculate when SWA starts
        num_epochs = self.args.num_train_epochs
        steps_per_epoch = num_training_steps // num_epochs
        swa_start_step = self.swa_config.swa_start_epoch * steps_per_epoch
        
        logging.info(f"Creating SWA-aware scheduler:")
        logging.info(f"  Total training steps: {num_training_steps}")
        logging.info(f"  Steps per epoch: {steps_per_epoch}")
        logging.info(f"  SWA starts at step: {swa_start_step} (epoch {self.swa_config.swa_start_epoch})")
        logging.info(f"  SWA LR: {self.swa_config.swa_lr}")
        logging.info(f"  SWA schedule type: {self.swa_config.swa_schedule_type}")
        
        # Get the base learning rate from training args
        base_lr = self.args.learning_rate
        
        def lr_lambda(current_step):
            """Learning rate schedule function"""
            if current_step < swa_start_step:
                # Before SWA: use normal cosine annealing or other schedule
                if self.args.lr_scheduler_type == "cosine":
                    # Cosine annealing from base_lr to swa_lr over the pre-SWA period
                    progress = current_step / swa_start_step
                    return 0.5 * (1 + math.cos(math.pi * progress)) * (1 - self.swa_config.swa_lr / base_lr) + self.swa_config.swa_lr / base_lr
                elif self.args.lr_scheduler_type == "linear":
                    # Linear decay from base_lr to swa_lr
                    progress = current_step / swa_start_step
                    return (1 - progress) + progress * (self.swa_config.swa_lr / base_lr)
                else:
                    # Default: linear decay to SWA LR
                    progress = current_step / swa_start_step
                    return (1 - progress) + progress * (self.swa_config.swa_lr / base_lr)
            else:
                # During SWA: use SWA-specific schedule
                steps_since_swa = current_step - swa_start_step
                
                if self.swa_config.swa_schedule_type == "cyclic":
                    # Cyclic schedule within SWA
                    cycle_length_steps = self.swa_config.swa_cycle_length * steps_per_epoch
                    cycle_position = (steps_since_swa % cycle_length_steps) / cycle_length_steps
                    
                    # Cosine annealing within cycle
                    min_lr = self.swa_config.swa_lr * self.swa_config.swa_lr_min_ratio
                    max_lr = self.swa_config.swa_lr
                    target_lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * cycle_position))
                    
                    return target_lr / base_lr
                else:
                    # Constant SWA learning rate
                    return self.swa_config.swa_lr / base_lr
        
        # Create the scheduler
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        # Set the required attributes for HuggingFace Trainer
        self.lr_scheduler = scheduler
        self._created_lr_scheduler = True
        
        return self.lr_scheduler





def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")
    
    # Check TRL version for compatibility
    try:
        trl_version = trl.__version__
        logging.info(f"Using TRL version: {trl_version}")
    except:
        logging.warning("Could not determine TRL version")

    # loading model
    kwargs = {}
    if "70B" in config.model_name:
        # Removed "low_cpu_mem_usage": True, for 70B, since by default we are in FSDP,
        # it's more efficient to do  "cpu_ram_efficient_loading": true, in fsdp_config.json
        kwargs = {"device_map": "auto", "torch_dtype": "auto",
                  "attn_implementation": "flash_attention_2", "use_cache": False}
        base_model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    else:
        base_model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name)
    
    # Optionally wrap model with simple prediction behavior (legacy support)
    # Note: The preferred approach is to use custom loss during training and 
    # set sampling parameters in VLLM during evaluation
    if config.use_simple_prediction_wrapper:
        prediction_config = {}
        if config.prediction_top_k > 0:
            prediction_config['top_k'] = config.prediction_top_k
        if config.prediction_top_p < 1.0:
            prediction_config['top_p'] = config.prediction_top_p
        
        if prediction_config:
            model = SimplePredictionWrapper(base_model, prediction_config)
            logging.info(f"Using simple prediction wrapper with config: {prediction_config}")
        else:
            model = base_model
            logging.info("Simple prediction wrapper enabled but no parameters set, using base model")
    else:
        model = base_model

    # Load dataset with error handling
    try:
        dataset = load_dataset(config.train_file_path)
        logging.info(f"Loaded dataset with keys: {dataset.keys()}")
        logging.info(f"Train dataset size: {len(dataset['train'])}")
        if 'test' in dataset:
            logging.info(f"Test dataset size: {len(dataset['test'])}")
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in config.model_name:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|fim_pad|>"

    # Only compute loss over assistant responses
    # Verified that it precisely starts where the thinking tokens start and ends with the first pad token
    # via labels being set to -100
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Set training arguments (avoid duplicate assignments)
    if not hasattr(args, 'dataset_text_field') or args.dataset_text_field is None:
        args.dataset_text_field = 'text'
    if not hasattr(args, 'max_seq_length') or args.max_seq_length is None:
        args.max_seq_length = config.block_size
    
    # Create SWA callback if enabled
    callbacks = []
    if config.use_swa:
        swa_callback = SWACallback(config, output_dir=args.output_dir)
        callbacks.append(swa_callback)
        logging.info("Memory-efficient SWA callback added to training")
    
    # Create trainer - use CustomSFTTrainer if custom loss OR SWA is enabled
    if config.use_custom_loss or config.use_swa:
        trainer = CustomSFTTrainer(
            loss_config=config,
            swa_config=config if config.use_swa else None,
            model=model,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'] if 'test' in dataset  else dataset['train'],
            args=args,
            data_collator=collator,
            callbacks=callbacks,
        )
        logging.info(f"Using CustomSFTTrainer with custom_loss={config.use_custom_loss}, swa={config.use_swa}")
    else:
        trainer = trl.SFTTrainer(
            model,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
            args=args,
            data_collator=collator,
            callbacks=callbacks,
        )

    # Set trainer reference on SWA callback if enabled
    if config.use_swa and callbacks:
        swa_callback = callbacks[0]
    trainer.train()
    
    # If SWA was used, log final information
    if config.use_swa and callbacks:
        swa_info = callbacks[0].get_swa_info()
        logging.info(f"SWA training completed. Final SWA info: {swa_info}")
    
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
