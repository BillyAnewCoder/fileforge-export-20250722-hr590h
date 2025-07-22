"""
DeepSpeed configuration generator for MoE training.
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path


class DeepSpeedConfigGenerator:
    """
    Generate DeepSpeed configurations optimized for MoE training.
    """
    
    @staticmethod
    def create_zero_stage3_config(
        train_batch_size: int = 32,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        fp16_enabled: bool = True,
        bf16_enabled: bool = False,
        cpu_offload: bool = True,
        nvme_offload: bool = False,
        nvme_path: str = "/tmp/nvme_offload",
        overlap_comm: bool = True,
        contiguous_gradients: bool = True,
        allgather_bucket_size: int = 200000000,
        reduce_bucket_size: int = 200000000,
        stage3_prefetch_bucket_size: int = 100000000,
        stage3_param_persistence_threshold: int = 100000,
        stage3_max_live_parameters: int = 1000000000,
        stage3_max_reuse_distance: int = 1000000000,
    ) -> Dict[str, Any]:
        """
        Create ZeRO Stage 3 configuration for maximum memory efficiency.
        
        Args:
            train_batch_size: Global training batch size
            gradient_accumulation_steps: Gradient accumulation steps
            learning_rate: Learning rate
            weight_decay: Weight decay
            max_grad_norm: Maximum gradient norm for clipping
            fp16_enabled: Enable FP16 mixed precision
            bf16_enabled: Enable BF16 mixed precision
            cpu_offload: Enable CPU offload
            nvme_offload: Enable NVMe offload
            nvme_path: Path for NVMe offload
            overlap_comm: Overlap communication and computation
            contiguous_gradients: Use contiguous gradients
            allgather_bucket_size: AllGather bucket size
            reduce_bucket_size: Reduce bucket size
            stage3_prefetch_bucket_size: Stage 3 prefetch bucket size
            stage3_param_persistence_threshold: Parameter persistence threshold
            stage3_max_live_parameters: Maximum live parameters
            stage3_max_reuse_distance: Maximum reuse distance
            
        Returns:
            DeepSpeed configuration dictionary
        """
        
        config = {
            "train_batch_size": train_batch_size,
            "train_micro_batch_size_per_gpu": train_batch_size // gradient_accumulation_steps,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_clipping": max_grad_norm,
            "steps_per_print": 100,
            "wall_clock_breakdown": False,
            "dump_state": False,
            
            # Optimizer configuration
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": learning_rate,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": weight_decay,
                }
            },
            
            # Learning rate scheduler
            "scheduler": {
                "type": "WarmupCosineLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": learning_rate,
                    "warmup_num_steps": 1000,
                    "total_num_steps": 100000,
                }
            },
            
            # ZeRO Stage 3 configuration
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu" if cpu_offload else "none",
                    "pin_memory": True,
                },
                "offload_param": {
                    "device": "cpu" if cpu_offload else "none",
                    "pin_memory": True,
                    "buffer_count": 5,
                    "buffer_size": 1e8,
                    "max_in_cpu": 1e9,
                },
                "overlap_comm": overlap_comm,
                "contiguous_gradients": contiguous_gradients,
                "sub_group_size": 1e9,
                "reduce_bucket_size": reduce_bucket_size,
                "stage3_prefetch_bucket_size": stage3_prefetch_bucket_size,
                "stage3_param_persistence_threshold": stage3_param_persistence_threshold,
                "stage3_max_live_parameters": stage3_max_live_parameters,
                "stage3_max_reuse_distance": stage3_max_reuse_distance,
                "stage3_gather_16bit_weights_on_model_save": True,
                "allgather_partitions": True,
                "allgather_bucket_size": allgather_bucket_size,
                "reduce_scatter": True,
            },
            
            # Communication configuration
            "communication_data_type": "fp16" if fp16_enabled else "fp32",
            "elastic_checkpoint": True,
        }
        
        # Mixed precision configuration
        if fp16_enabled:
            config["fp16"] = {
                "enabled": True,
                "auto_cast": False,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "consecutive_hysteresis": False,
                "min_loss_scale": 1,
            }
        elif bf16_enabled:
            config["bf16"] = {
                "enabled": True,
                "auto_cast": False,
            }
        
        # NVMe offload configuration
        if nvme_offload:
            config["aio"] = {
                "block_size": 1048576,
                "queue_depth": 8,
                "thread_count": 1,
                "single_submit": False,
                "overlap_events": True,
            }
            
            config["zero_optimization"]["offload_optimizer"]["nvme_path"] = nvme_path
            config["zero_optimization"]["offload_param"]["nvme_path"] = nvme_path
        
        return config
    
    @staticmethod
    def create_moe_config(
        num_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        eval_capacity_factor: float = 2.0,
        min_capacity: int = 4,
        use_residual: bool = False,
        **base_config_kwargs
    ) -> Dict[str, Any]:
        """
        Create MoE-specific DeepSpeed configuration.
        
        Args:
            num_experts: Number of experts
            top_k: Top-k experts to route to
            capacity_factor: Capacity factor for training
            eval_capacity_factor: Capacity factor for evaluation
            min_capacity: Minimum capacity per expert
            use_residual: Use residual connections in MoE
            **base_config_kwargs: Base configuration arguments
            
        Returns:
            DeepSpeed configuration with MoE settings
        """
        # Start with base ZeRO Stage 3 config
        config = DeepSpeedConfigGenerator.create_zero_stage3_config(**base_config_kwargs)
        
        # Add MoE-specific configuration
        config["moe"] = {
            "enabled": True,
            "num_experts": num_experts,
            "top_k": top_k,
            "capacity_factor": capacity_factor,
            "eval_capacity_factor": eval_capacity_factor,
            "min_capacity": min_capacity,
            "use_residual": use_residual,
            "expert_parallel_size": 1,  # Can be adjusted based on number of GPUs
            "enable_expert_tensor_parallelism": False,
            "use_tutel": False,  # Can enable Tutel for better performance
        }
        
        # Adjust batch sizes for MoE (typically need larger batches)
        original_batch_size = config["train_batch_size"]
        config["train_batch_size"] = max(original_batch_size, num_experts * 2)
        config["train_micro_batch_size_per_gpu"] = config["train_batch_size"] // config["gradient_accumulation_steps"]
        
        return config
    
    @staticmethod
    def create_inference_config(
        fp16_enabled: bool = True,
        tensor_parallel_size: int = 1,
        replace_method: str = "auto",
        replace_with_kernel_inject: bool = True,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """
        Create configuration optimized for inference.
        
        Args:
            fp16_enabled: Enable FP16 for inference
            tensor_parallel_size: Tensor parallel size
            replace_method: Method for kernel replacement
            replace_with_kernel_inject: Use kernel injection
            max_tokens: Maximum tokens for inference
            
        Returns:
            DeepSpeed inference configuration
        """
        config = {
            "train_batch_size": 1,  # Not used in inference
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            
            # Inference-specific settings
            "fp16": {
                "enabled": fp16_enabled,
            },
            
            # Tensor parallelism for inference
            "tensor_parallel": {
                "tp_size": tensor_parallel_size,
            },
            
            # Kernel injection for faster inference
            "replace_method": replace_method,
            "replace_with_kernel_inject": replace_with_kernel_inject,
            
            # Memory optimization for inference
            "checkpoint_activations": False,
            "checkpoint_num_layers": 0,
            
            # Communication settings
            "comms_logger": {
                "enabled": False,
            },
        }
        
        return config
    
    @staticmethod
    def save_config(config: Dict[str, Any], filepath: str):
        """
        Save DeepSpeed configuration to JSON file.
        
        Args:
            config: Configuration dictionary
            filepath: Path to save configuration
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"DeepSpeed config saved to {filepath}")
    
    @staticmethod
    def load_config(filepath: str) -> Dict[str, Any]:
        """
        Load DeepSpeed configuration from JSON file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        return config
    
    @staticmethod
    def create_multi_node_config(
        num_nodes: int = 2,
        num_gpus_per_node: int = 8,
        **base_config_kwargs
    ) -> Dict[str, Any]:
        """
        Create configuration optimized for multi-node training.
        
        Args:
            num_nodes: Number of nodes
            num_gpus_per_node: GPUs per node
            **base_config_kwargs: Base configuration arguments
            
        Returns:
            Multi-node DeepSpeed configuration
        """
        total_gpus = num_nodes * num_gpus_per_node
        
        # Adjust batch size for multi-node setup
        if 'train_batch_size' not in base_config_kwargs:
            base_config_kwargs['train_batch_size'] = 32 * total_gpus
        
        config = DeepSpeedConfigGenerator.create_zero_stage3_config(**base_config_kwargs)
        
        # Multi-node specific settings
        config.update({
            "distributed_backend": "nccl",
            "prescale_gradients": False,
            "sparse_gradients": False,
            
            # Communication optimizations
            "zero_optimization": {
                **config["zero_optimization"],
                "allgather_partitions": True,
                "allgather_bucket_size": 500000000,  # Larger bucket for multi-node
                "reduce_bucket_size": 500000000,
                "overlap_comm": True,
                "contiguous_gradients": True,
            },
        })
        
        return config
    
    @staticmethod
    def create_experimental_config(
        use_cpu_adam: bool = False,
        use_fp8: bool = False,
        activation_checkpointing: bool = True,
        partition_activations: bool = False,
        **base_config_kwargs
    ) -> Dict[str, Any]:
        """
        Create configuration with experimental features.
        
        Args:
            use_cpu_adam: Use CPU Adam optimizer
            use_fp8: Use FP8 precision (experimental)
            activation_checkpointing: Enable activation checkpointing
            partition_activations: Partition activations across GPUs
            **base_config_kwargs: Base configuration arguments
            
        Returns:
            Experimental DeepSpeed configuration
        """
        config = DeepSpeedConfigGenerator.create_zero_stage3_config(**base_config_kwargs)
        
        # CPU Adam
        if use_cpu_adam:
            config["optimizer"] = {
                "type": "CPUAdam",
                "params": {
                    "lr": base_config_kwargs.get('learning_rate', 1e-4),
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": base_config_kwargs.get('weight_decay', 0.01),
                }
            }
        
        # FP8 support (experimental)
        if use_fp8:
            config["fp8"] = {
                "enabled": True,
                "margin": 0,
                "interval": 1,
                "amax_history_len": 1024,
                "amax_compute_algo": "most_recent",
            }
            # Remove FP16/BF16 if FP8 is enabled
            config.pop("fp16", None)
            config.pop("bf16", None)
        
        # Activation checkpointing
        if activation_checkpointing:
            config["activation_checkpointing"] = {
                "partition_activations": partition_activations,
                "cpu_checkpointing": False,
                "contiguous_memory_optimization": False,
                "number_checkpoints": None,
                "synchronize_checkpoint_boundary": False,
                "profile": False,
            }
        
        return config


# Predefined configurations for common use cases
PREDEFINED_CONFIGS = {
    "small_model": {
        "train_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "fp16_enabled": True,
        "cpu_offload": False,
    },
    
    "large_model": {
        "train_batch_size": 64,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "fp16_enabled": True,
        "cpu_offload": True,
        "nvme_offload": False,
    },
    
    "huge_model": {
        "train_batch_size": 128,
        "gradient_accumulation_steps": 8,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "fp16_enabled": True,
        "cpu_offload": True,
        "nvme_offload": True,
    },
    
    "moe_8_experts": {
        "num_experts": 8,
        "top_k": 2,
        "capacity_factor": 1.25,
        "train_batch_size": 64,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1e-4,
        "fp16_enabled": True,
        "cpu_offload": True,
    },
    
    "moe_64_experts": {
        "num_experts": 64,
        "top_k": 2,
        "capacity_factor": 1.0,
        "train_batch_size": 256,
        "gradient_accumulation_steps": 4,
        "learning_rate": 8e-5,
        "fp16_enabled": True,
        "cpu_offload": True,
        "nvme_offload": True,
    },
}


def get_config(config_name: str, **overrides) -> Dict[str, Any]:
    """
    Get a predefined configuration with optional overrides.
    
    Args:
        config_name: Name of predefined configuration
        **overrides: Configuration overrides
        
    Returns:
        DeepSpeed configuration
    """
    if config_name not in PREDEFINED_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(PREDEFINED_CONFIGS.keys())}")
    
    base_config = PREDEFINED_CONFIGS[config_name].copy()
    base_config.update(overrides)
    
    if config_name.startswith("moe_"):
        return DeepSpeedConfigGenerator.create_moe_config(**base_config)
    else:
        return DeepSpeedConfigGenerator.create_zero_stage3_config(**base_config)
