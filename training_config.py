"""
Training configuration for MoE model training.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import json

from configs.model_config import MoEModelConfig


@dataclass
class TrainingConfig:
    """Configuration for training MoE models."""
    
    # Model configuration
    model_config: MoEModelConfig = field(default_factory=lambda: MoEModelConfig())
    
    # Data configuration
    train_data_dir: str = "./data/train"
    eval_data_dir: Optional[str] = "./data/eval"
    tokenizer_path: str = "./tokenizer"
    vocab_size: int = 32000
    max_seq_len: int = 2048
    
    # Training parameters
    num_epochs: int = 3
    max_steps: int = 100000
    batch_size: int = 32
    eval_batch_size: Optional[int] = None
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Learning rate scheduling
    warmup_steps: int = 1000
    lr_scheduler_type: str = "cosine_warmup"  # cosine_warmup, polynomial, linear
    min_lr_ratio: float = 0.1
    
    # Multi-token prediction
    multi_token_prediction: int = 1  # Number of future tokens to predict
    
    # Evaluation and logging
    eval_steps: int = 1000
    logging_steps: int = 100
    save_steps: int = 1000
    logging_first_step: bool = True
    
    # Checkpointing
    output_dir: str = "./checkpoints"
    save_total_limit: int = 5
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Optimization
    optimizer_type: str = "adamw"  # adamw, layerwise_adamw
    use_expert_specific_opt: bool = True
    gate_regularization: float = 0.01
    layer_wise_decay: bool = False
    layer_decay_rate: float = 0.65
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    
    # Expert load balancing
    expert_load_balancing: bool = True
    load_balance_weight: float = 0.01
    
    # Gate temperature scheduling
    gate_temperature_schedule: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        'type': 'linear',
        'start': 1.0,
        'end': 0.5
    })
    
    # Mixed precision and memory
    fp16: bool = True
    bf16: bool = False
    fp8: bool = False
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # DeepSpeed configuration
    deepspeed_config_path: Optional[str] = "./configs/ds_moe_config.json"
    zero_stage: int = 3
    cpu_offload: bool = True
    nvme_offload: bool = False
    nvme_path: str = "/tmp/nvme_offload"
    
    # Distributed training
    local_rank: int = -1
    distributed: bool = False
    
    # Data preprocessing
    shuffle_data: bool = True
    data_seed: int = 42
    preprocessing_num_workers: int = 4
    
    # Monitoring and experiment tracking
    experiment_name: str = "moe_training"
    use_tensorboard: bool = True
    use_mlflow: bool = False
    mlflow_tracking_uri: Optional[str] = None
    log_dir: str = "./logs"
    
    # Validation and safety
    ignore_data_skip: bool = False
    skip_memory_metrics: bool = False
    
    def __post_init__(self):
        """Post-initialization validation and defaults."""
        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size
        
        # Ensure output directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Validate mixed precision settings
        precision_count = sum([self.fp16, self.bf16, self.fp8])
        if precision_count > 1:
            raise ValueError("Only one of fp16, bf16, fp8 can be enabled")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        # Handle model config separately
        if 'model_config' in config_dict:
            model_config_dict = config_dict.pop('model_config')
            if isinstance(model_config_dict, dict):
                model_config = MoEModelConfig.from_dict(model_config_dict)
            else:
                model_config = model_config_dict
            config_dict['model_config'] = model_config
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, MoEModelConfig):
                config_dict[key] = value.to_dict()
            else:
                config_dict[key] = value
        return config_dict
    
    def save(self, filepath: str):
        """Save config to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'TrainingConfig':
        """Load config from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def get_effective_batch_size(self) -> int:
        """Get effective batch size including gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps
    
    def get_total_steps(self) -> int:
        """Get total training steps."""
        return min(self.max_steps, self.num_epochs * 1000000)  # Rough estimate
    
    def get_warmup_ratio(self) -> float:
        """Get warmup ratio."""
        total_steps = self.get_total_steps()
        return self.warmup_steps / total_steps if total_steps > 0 else 0.0
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Basic validation
        if self.batch_size <= 0:
            issues.append("batch_size must be positive")
        
        if self.learning_rate <= 0:
            issues.append("learning_rate must be positive")
        
        if not 0 <= self.weight_decay <= 1:
            issues.append("weight_decay must be between 0 and 1")
        
        if self.gradient_accumulation_steps <= 0:
            issues.append("gradient_accumulation_steps must be positive")
        
        if self.max_grad_norm <= 0:
            issues.append("max_grad_norm must be positive")
        
        # Learning rate validation
        if not 0 <= self.min_lr_ratio <= 1:
            issues.append("min_lr_ratio must be between 0 and 1")
        
        if self.warmup_steps < 0:
            issues.append("warmup_steps must be non-negative")
        
        # Multi-token prediction
        if self.multi_token_prediction <= 0:
            issues.append("multi_token_prediction must be positive")
        
        # Evaluation steps
        if self.eval_steps <= 0:
            issues.append("eval_steps must be positive")
        
        if self.logging_steps <= 0:
            issues.append("logging_steps must be positive")
        
        if self.save_steps <= 0:
            issues.append("save_steps must be positive")
        
        # Checkpointing
        if self.save_total_limit <= 0:
            issues.append("save_total_limit must be positive")
        
        # Data paths
        if not os.path.exists(self.train_data_dir):
            issues.append(f"train_data_dir does not exist: {self.train_data_dir}")
        
        # DeepSpeed
        if self.deepspeed_config_path and not os.path.exists(self.deepspeed_config_path):
            issues.append(f"deepspeed_config_path does not exist: {self.deepspeed_config_path}")
        
        # Memory warnings
        effective_batch_size = self.get_effective_batch_size()
        if effective_batch_size > 512:
            issues.append(f"Warning: Large effective batch size ({effective_batch_size})")
        
        # Model config validation
        model_issues = self.model_config.validate() if hasattr(self.model_config, 'validate') else []
        issues.extend([f"Model config: {issue}" for issue in model_issues])
        
        return issues


# Predefined training configurations
TRAINING_CONFIGS = {
    "debug": TrainingConfig(
        max_steps=100,
        batch_size=4,
        gradient_accumulation_steps=1,
        eval_steps=50,
        logging_steps=10,
        save_steps=50,
        warmup_steps=10,
    ),
    
    "small": TrainingConfig(
        max_steps=10000,
        batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        eval_steps=500,
        logging_steps=50,
        save_steps=1000,
        warmup_steps=500,
    ),
    
    "base": TrainingConfig(
        max_steps=50000,
        batch_size=32,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        eval_steps=1000,
        logging_steps=100,
        save_steps=2000,
        warmup_steps=1000,
    ),
    
    "large": TrainingConfig(
        max_steps=100000,
        batch_size=64,
        gradient_accumulation_steps=8,
        learning_rate=8e-5,
        eval_steps=2000,
        logging_steps=200,
        save_steps=5000,
        warmup_steps=2000,
        cpu_offload=True,
        nvme_offload=False,
    ),
    
    "xl": TrainingConfig(
        max_steps=200000,
        batch_size=128,
        gradient_accumulation_steps=16,
        learning_rate=5e-5,
        eval_steps=5000,
        logging_steps=500,
        save_steps=10000,
        warmup_steps=5000,
        cpu_offload=True,
        nvme_offload=True,
        fp16=False,
        bf16=True,
    ),
}


def get_training_config(config_name: str) -> TrainingConfig:
    """Get a predefined training configuration."""
    if config_name not in TRAINING_CONFIGS:
        raise ValueError(f"Unknown training config: {config_name}. Available: {list(TRAINING_CONFIGS.keys())}")
    
    return TRAINING_CONFIGS[config_name]


def create_custom_training_config(
    base_config: str = "base",
    model_size: str = "base",
    **overrides
) -> TrainingConfig:
    """Create a custom training configuration."""
    from configs.model_config import get_model_config
    
    # Get base training config
    training_config = get_training_config(base_config)
    
    # Get model config
    model_config = get_model_config(model_size)
    training_config.model_config = model_config
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(training_config, key):
            setattr(training_config, key, value)
        else:
            raise ValueError(f"Unknown training config parameter: {key}")
    
    return training_config


def print_training_summary(config: TrainingConfig):
    """Print a summary of the training configuration."""
    print("="*60)
    print("Training Configuration Summary")
    print("="*60)
    
    print(f"Data:")
    print(f"  Train data dir: {config.train_data_dir}")
    print(f"  Eval data dir: {config.eval_data_dir}")
    print(f"  Tokenizer path: {config.tokenizer_path}")
    print(f"  Max sequence length: {config.max_seq_len}")
    
    print(f"\nTraining:")
    print(f"  Max steps: {config.max_steps:,}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.get_effective_batch_size()}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Weight decay: {config.weight_decay}")
    print(f"  Max grad norm: {config.max_grad_norm}")
    
    print(f"\nScheduling:")
    print(f"  Warmup steps: {config.warmup_steps}")
    print(f"  LR scheduler: {config.lr_scheduler_type}")
    print(f"  Min LR ratio: {config.min_lr_ratio}")
    
    print(f"\nExpert Configuration:")
    print(f"  Load balancing: {config.expert_load_balancing}")
    print(f"  Load balance weight: {config.load_balance_weight}")
    print(f"  Gate regularization: {config.gate_regularization}")
    
    print(f"\nPrecision:")
    if config.fp16:
        precision = "FP16"
    elif config.bf16:
        precision = "BF16"
    elif config.fp8:
        precision = "FP8"
    else:
        precision = "FP32"
    print(f"  Mixed precision: {precision}")
    
    print(f"\nDeepSpeed:")
    print(f"  Config path: {config.deepspeed_config_path}")
    print(f"  ZeRO stage: {config.zero_stage}")
    print(f"  CPU offload: {config.cpu_offload}")
    print(f"  NVMe offload: {config.nvme_offload}")
    
    print(f"\nLogging & Evaluation:")
    print(f"  Output dir: {config.output_dir}")
    print(f"  Log dir: {config.log_dir}")
    print(f"  Eval steps: {config.eval_steps}")
    print(f"  Logging steps: {config.logging_steps}")
    print(f"  Save steps: {config.save_steps}")
    print(f"  Use TensorBoard: {config.use_tensorboard}")
    print(f"  Use MLflow: {config.use_mlflow}")
    
    # Validation
    issues = config.validate()
    if issues:
        print(f"\nConfiguration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\nConfiguration: ✓ Valid")
    
    print("="*60)


def estimate_training_time(
    config: TrainingConfig,
    tokens_per_second: float = 1000,
    num_gpus: int = 1,
) -> Dict[str, float]:
    """
    Estimate training time based on configuration.
    
    Args:
        config: Training configuration
        tokens_per_second: Estimated tokens processed per second per GPU
        num_gpus: Number of GPUs
        
    Returns:
        Dictionary with time estimates
    """
    effective_batch_size = config.get_effective_batch_size()
    total_tokens_per_step = effective_batch_size * config.max_seq_len
    
    # Adjust for distributed training
    if num_gpus > 1:
        tokens_per_second *= num_gpus
    
    steps_per_second = tokens_per_second / total_tokens_per_step
    
    total_steps = config.get_total_steps()
    training_time_seconds = total_steps / steps_per_second
    
    return {
        "training_time_hours": training_time_seconds / 3600,
        "training_time_days": training_time_seconds / (3600 * 24),
        "steps_per_second": steps_per_second,
        "tokens_per_second": tokens_per_second,
        "total_tokens": total_steps * total_tokens_per_step,
    }


if __name__ == "__main__":
    # Example usage
    config = get_training_config("base")
    print_training_summary(config)
    
    # Time estimation
    time_est = estimate_training_time(config, tokens_per_second=2000, num_gpus=8)
    print(f"\nEstimated training time: {time_est['training_time_hours']:.1f} hours")
