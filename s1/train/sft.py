import os
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset, concatenate_datasets, DatasetDict
import transformers
import trl
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomPredictionWrapper(nn.Module):
    """
    Wrapper around a pretrained model that modifies prediction behavior.
    This allows you to customize how the model computes next token probabilities
    without modifying the underlying pretrained weights.
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
        Modify the logits before they're converted to probabilities.
        Override this method or configure via prediction_config to implement
        different prediction strategies.
        """
        # Example modifications (you can customize these):
        
        # 1. Temperature scaling
        if 'temperature' in self.prediction_config:
            temperature = self.prediction_config['temperature']
            logits = logits / temperature
            
        # 2. Top-k filtering (set non-top-k logits to very negative values)
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
                
        # 3. Top-p (nucleus) filtering
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
                
        # 4. Repetition penalty (penalize recently generated tokens)
        if 'repetition_penalty' in self.prediction_config and 'input_ids' in self.prediction_config:
            penalty = self.prediction_config['repetition_penalty']
            input_ids = self.prediction_config['input_ids']
            if penalty != 1.0:
                # Apply repetition penalty to tokens that appear in input
                for batch_idx in range(logits.size(0)):
                    for token_id in input_ids[batch_idx].unique():
                        if token_id >= 0:  # Valid token
                            if logits[batch_idx, -1, token_id] < 0:
                                logits[batch_idx, -1, token_id] *= penalty
                            else:
                                logits[batch_idx, -1, token_id] /= penalty
                                
        # 5. Custom probability redistribution
        if 'probability_redistribution' in self.prediction_config:
            redistribution_fn = self.prediction_config['probability_redistribution']
            if callable(redistribution_fn):
                logits = redistribution_fn(logits)
                
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


@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-32B-Instruct")
    block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default="Qwen2.5-1.5B-Instruct-s1-top128")
    wandb_entity: Optional[str] = field(default="wandb_kheuton")
    train_file_path: Optional[str] = field(default='simplescaling/s1K_tokenized')
    dagger: bool = field(default=False)
    # Add custom loss configuration
    use_custom_loss: bool = field(default=False)
    loss_type: str = field(default="cross_entropy")  # Options: "cross_entropy", "focal", "label_smoothing", "topk_cross_entropy", etc.
    focal_alpha: float = field(default=1.0)
    focal_gamma: float = field(default=2.0)
    label_smoothing: float = field(default=0.1)
    # Top-k parameters
    topk_k: int = field(default=50)  # Number of top predictions to keep
    topk_temperature: float = field(default=1.0)  # Temperature for softmax before top-k filtering
    
    # Custom prediction behavior configuration
    use_custom_prediction: bool = field(default=False)
    prediction_wrapper_type: str = field(default="basic")  # Options: "basic", "advanced", "contrastive"
    prediction_temperature: float = field(default=1.0)  # Temperature scaling for predictions
    prediction_top_k: int = field(default=0)  # Top-k filtering (0 = disabled)
    prediction_top_p: float = field(default=1.0)  # Top-p filtering (1.0 = disabled) 
    prediction_repetition_penalty: float = field(default=1.0)  # Repetition penalty (1.0 = disabled)
    
    # Advanced prediction options
    use_learnable_bias: bool = field(default=False)  # Add learnable bias to token predictions
    use_attention_reweighting: bool = field(default=False)  # Context-dependent reweighting
    
    # Contrastive prediction options (provide as JSON strings or configure programmatically)
    boost_token_ids: Optional[str] = field(default=None)  # JSON list of token IDs to boost
    suppress_token_ids: Optional[str] = field(default=None)  # JSON list of token IDs to suppress

    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project
        os.environ['WANDB_ENTITY'] = self.wandb_entity


class CustomSFTTrainer(trl.SFTTrainer):
    """Custom SFT Trainer with configurable loss functions, based on the original compute_loss method"""
    
    def __init__(self, loss_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_config = loss_config
    
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


class AdvancedPredictionWrapper(CustomPredictionWrapper):
    """
    Advanced wrapper with more sophisticated prediction modification techniques.
    """
    
    def __init__(self, base_model, prediction_config=None):
        super().__init__(base_model, prediction_config)
        
        # Initialize learnable parameters if needed
        if prediction_config and 'learnable_bias' in prediction_config:
            vocab_size = base_model.config.vocab_size
            self.prediction_bias = nn.Parameter(torch.zeros(vocab_size))
        else:
            self.prediction_bias = None
            
        if prediction_config and 'attention_weights' in prediction_config:
            # Add a small MLP to reweight attention to different tokens
            hidden_size = base_model.config.hidden_size
            self.attention_reweighter = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, base_model.config.vocab_size)
            )
        else:
            self.attention_reweighter = None
    
    def _modify_logits(self, logits):
        """Advanced logit modifications"""
        
        # Apply base modifications first
        logits = super()._modify_logits(logits)
        
        # 1. Add learnable bias to specific tokens
        if self.prediction_bias is not None:
            logits = logits + self.prediction_bias.unsqueeze(0).unsqueeze(0)
            
        # 2. Apply context-dependent reweighting
        if self.attention_reweighter is not None and hasattr(self, '_last_hidden_state'):
            # Use the last hidden state to compute context-dependent weights
            context_weights = self.attention_reweighter(self._last_hidden_state[:, -1:, :])
            logits = logits + context_weights
            
        # 3. Apply frequency-based penalties
        if 'frequency_penalty' in self.prediction_config:
            penalty_strength = self.prediction_config['frequency_penalty']
            # This would require maintaining token frequency statistics
            # Implementation depends on your specific needs
            
        # 4. Apply semantic constraints
        if 'semantic_constraints' in self.prediction_config:
            constraints = self.prediction_config['semantic_constraints']
            # Example: boost probability of certain semantic categories
            if 'boost_tokens' in constraints:
                boost_tokens = constraints['boost_tokens']
                boost_factor = constraints.get('boost_factor', 1.5)
                for token_id in boost_tokens:
                    if token_id < logits.size(-1):
                        logits[..., token_id] *= boost_factor
                        
            # Example: suppress probability of certain token types  
            if 'suppress_tokens' in constraints:
                suppress_tokens = constraints['suppress_tokens']
                suppress_factor = constraints.get('suppress_factor', 0.5)
                for token_id in suppress_tokens:
                    if token_id < logits.size(-1):
                        logits[..., token_id] *= suppress_factor
        
        return logits
    
    def forward(self, *args, **kwargs):
        # Store hidden states for context-dependent modifications
        outputs = self.base_model(*args, **kwargs)
        
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            self._last_hidden_state = outputs.hidden_states[-1]  # Last layer
        elif hasattr(outputs, 'last_hidden_state'):
            self._last_hidden_state = outputs.last_hidden_state
            
        # Apply logit modifications
        if hasattr(outputs, 'logits') and outputs.logits is not None:
            modified_logits = self._modify_logits(outputs.logits)
            if hasattr(outputs, '_replace'):
                outputs = outputs._replace(logits=modified_logits)
            else:
                outputs.logits = modified_logits
                
        return outputs


class ContrastivePredictionWrapper(CustomPredictionWrapper):
    """
    Wrapper that modifies predictions using contrastive learning principles.
    This can help the model avoid generating certain types of content or 
    push it towards generating more desirable content.
    """
    
    def __init__(self, base_model, prediction_config=None):
        super().__init__(base_model, prediction_config)
        
    def _modify_logits(self, logits):
        """Apply contrastive modifications to logits"""
        
        # Apply base modifications
        logits = super()._modify_logits(logits)
        
        # Contrastive prediction: modify based on what we want to avoid/prefer
        if 'contrastive_targets' in self.prediction_config:
            targets = self.prediction_config['contrastive_targets']
            
            # Increase probability of positive targets
            if 'positive_tokens' in targets:
                positive_tokens = targets['positive_tokens']
                boost_factor = targets.get('positive_boost', 1.2)
                for token_id in positive_tokens:
                    if token_id < logits.size(-1):
                        logits[..., token_id] = logits[..., token_id] + torch.log(torch.tensor(boost_factor))
            
            # Decrease probability of negative targets  
            if 'negative_tokens' in targets:
                negative_tokens = targets['negative_tokens']
                suppress_factor = targets.get('negative_suppress', 0.8)
                for token_id in negative_tokens:
                    if token_id < logits.size(-1):
                        logits[..., token_id] = logits[..., token_id] + torch.log(torch.tensor(suppress_factor))
        
        return logits


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
    
    # Wrap model with custom prediction behavior if enabled
    if config.use_custom_prediction:
        import json
        prediction_config = {
            'temperature': config.prediction_temperature,
            'top_k': config.prediction_top_k if config.prediction_top_k > 0 else None,
            'top_p': config.prediction_top_p if config.prediction_top_p < 1.0 else None,
            'repetition_penalty': config.prediction_repetition_penalty if config.prediction_repetition_penalty != 1.0 else None,
        }
        
        # Add advanced options
        if config.prediction_wrapper_type == "advanced":
            if config.use_learnable_bias:
                prediction_config['learnable_bias'] = True
            if config.use_attention_reweighting:
                prediction_config['attention_weights'] = True
                
        # Add contrastive options
        elif config.prediction_wrapper_type == "contrastive":
            contrastive_targets = {}
            if config.boost_token_ids:
                try:
                    boost_tokens = json.loads(config.boost_token_ids)
                    contrastive_targets['positive_tokens'] = boost_tokens
                    contrastive_targets['positive_boost'] = 1.2  # Default boost factor
                except json.JSONDecodeError:
                    logging.warning("Invalid JSON for boost_token_ids, ignoring")
                    
            if config.suppress_token_ids:
                try:
                    suppress_tokens = json.loads(config.suppress_token_ids)
                    contrastive_targets['negative_tokens'] = suppress_tokens  
                    contrastive_targets['negative_suppress'] = 0.8  # Default suppress factor
                except json.JSONDecodeError:
                    logging.warning("Invalid JSON for suppress_token_ids, ignoring")
                    
            if contrastive_targets:
                prediction_config['contrastive_targets'] = contrastive_targets
        
        # Remove None values
        prediction_config = {k: v for k, v in prediction_config.items() if v is not None}
        
        # Choose appropriate wrapper
        if config.prediction_wrapper_type == "advanced":
            model = AdvancedPredictionWrapper(base_model, prediction_config)
            logging.info(f"Using advanced prediction wrapper with config: {prediction_config}")
        elif config.prediction_wrapper_type == "contrastive":
            model = ContrastivePredictionWrapper(base_model, prediction_config)
            logging.info(f"Using contrastive prediction wrapper with config: {prediction_config}")
        else:  # basic
            model = CustomPredictionWrapper(base_model, prediction_config)
            logging.info(f"Using basic prediction wrapper with config: {prediction_config}")
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
    
    # Create trainer - use CustomSFTTrainer if custom loss is enabled
    if config.use_custom_loss:
        trainer = CustomSFTTrainer(
            loss_config=config,
            model=model,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'] if 'test' in dataset  else dataset['train'],
            args=args,
            data_collator=collator,
        )
    else:
        trainer = trl.SFTTrainer(
            model,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
            args=args,
            data_collator=collator,
        )

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
