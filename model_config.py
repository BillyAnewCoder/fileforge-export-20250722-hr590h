"""
Model configuration for MoE Transformer with Multi-Head Latent Attention.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import json


@dataclass
class MoEModelConfig:
    """Configuration for MoE Transformer model."""
    
    # Model architecture
    vocab_size: int = 32000
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    max_seq_len: int = 2048
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    
    # MoE configuration
    n_experts: int = 8
    expert_capacity: int = 4
    top_k: int = 2
    gate_temperature: float = 1.0
    load_balance_weight: float = 0.01
    
    # Multi-Head Latent Attention
    mla_interval: int = 4  # Add MLA every N layers
    mla_latent_dim: int = 256
    mla_n_latents: int = 64
    mla_use_rope: bool = True
    
    # Expert architecture
    expert_d_ff: Optional[int] = None  # Defaults to 4 * d_model
    expert_activation: str = "gelu"
    
    # Gate temperature scheduling
    gate_temperature_schedule: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        'type': 'linear',
        'start': 1.0,
        'end': 0.5,
        'decay_rate': 0.95
    })
    
    def __post_init__(self):
        """Post-initialization validation and defaults."""
        if self.expert_d_ff is None:
            self.expert_d_ff = 4 * self.d_model
        
        # Validate MLA configuration
        if self.mla_latent_dim % self.n_heads != 0:
            raise ValueError("mla_latent_dim must be divisible by n_heads")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MoEModelConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'n_experts': self.n_experts,
            'expert_capacity': self.expert_capacity,
            'top_k': self.top_k,
            'mla_interval': self.mla_interval,
            'mla_latent_dim': self.mla_latent_dim,
            'dropout': self.dropout,
            'max_seq_len': self.max_seq_len,
            'gate_temperature': self.gate_temperature,
            'load_balance_weight': self.load_balance_weight,
        }
    
    def save(self, filepath: str):
        """Save config to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'MoEModelConfig':
        """Load config from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Predefined model configurations
MODEL_CONFIGS = {
    "small": MoEModelConfig(
        vocab_size=32000,
        d_model=512,
        n_heads=8,
        n_layers=8,
        n_experts=4,
        expert_capacity=4,
        top_k=2,
        mla_interval=4,
        mla_latent_dim=128,
        max_seq_len=1024,
    ),
    
    "base": MoEModelConfig(
        vocab_size=32000,
        d_model=768,
        n_heads=12,
        n_layers=12,
        n_experts=8,
        expert_capacity=4,
        top_k=2,
        mla_interval=4,
        mla_latent_dim=256,
        max_seq_len=2048,
    ),
    
    "large": MoEModelConfig(
        vocab_size=32000,
        d_model=1024,
        n_heads=16,
        n_layers=24,
        n_experts=16,
        expert_capacity=4,
        top_k=2,
        mla_interval=4,
        mla_latent_dim=256,
        max_seq_len=2048,
    ),
    
    "xl": MoEModelConfig(
        vocab_size=32000,
        d_model=1536,
        n_heads=24,
        n_layers=24,
        n_experts=32,
        expert_capacity=4,
        top_k=2,
        mla_interval=4,
        mla_latent_dim=384,
        max_seq_len=4096,
    ),
    
    "xxl": MoEModelConfig(
        vocab_size=32000,
        d_model=2048,
        n_heads=32,
        n_layers=32,
        n_experts=64,
        expert_capacity=4,
        top_k=2,
        mla_interval=6,
        mla_latent_dim=512,
        max_seq_len=4096,
    ),
}


def get_model_config(config_name: str) -> MoEModelConfig:
    """Get a predefined model configuration."""
    if config_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model config: {config_name}. Available: {list(MODEL_CONFIGS.keys())}")
    
    return MODEL_CONFIGS[config_name]


def create_custom_config(
    base_config: str = "base",
    **overrides
) -> MoEModelConfig:
    """Create a custom configuration based on a base config with overrides."""
    base = get_model_config(base_config)
    
    # Convert to dict, apply overrides, and create new config
    config_dict = base.to_dict()
    config_dict.update(overrides)
    
    return MoEModelConfig.from_dict(config_dict)


# Configuration validation
def validate_model_config(config: MoEModelConfig) -> List[str]:
    """Validate model configuration and return list of issues."""
    issues = []
    
    # Basic validation
    if config.vocab_size <= 0:
        issues.append("vocab_size must be positive")
    
    if config.d_model % config.n_heads != 0:
        issues.append("d_model must be divisible by n_heads")
    
    if config.n_experts < config.top_k:
        issues.append("n_experts must be >= top_k")
    
    if config.expert_capacity < 1:
        issues.append("expert_capacity must be >= 1")
    
    if not 0 < config.dropout < 1:
        issues.append("dropout must be between 0 and 1")
    
    if config.gate_temperature <= 0:
        issues.append("gate_temperature must be positive")
    
    # MLA validation
    if config.mla_latent_dim <= 0:
        issues.append("mla_latent_dim must be positive")
    
    if config.mla_n_latents <= 0:
        issues.append("mla_n_latents must be positive")
    
    if config.mla_interval <= 0:
        issues.append("mla_interval must be positive")
    
    if config.mla_interval >= config.n_layers:
        issues.append("mla_interval should be less than n_layers")
    
    # Memory estimation warnings
    estimated_params = estimate_model_parameters(config)
    if estimated_params > 100e9:  # 100B parameters
        issues.append(f"Warning: Large model with ~{estimated_params/1e9:.1f}B parameters")
    
    return issues


def estimate_model_parameters(config: MoEModelConfig) -> int:
    """Estimate total number of parameters in the model."""
    
    # Token embeddings
    token_embed_params = config.vocab_size * config.d_model
    
    # Position embeddings
    pos_embed_params = config.max_seq_len * config.d_model
    
    # Transformer layers
    layer_params = 0
    for layer_idx in range(config.n_layers):
        # Self-attention
        attn_params = 4 * config.d_model * config.d_model  # Q, K, V, O projections
        
        # Layer norms
        norm_params = 2 * config.d_model  # Pre-attention and pre-MLP norms
        
        # MLA (if applicable)
        mla_params = 0
        if (layer_idx + 1) % config.mla_interval == 0 and layer_idx > 0:
            mla_params = (
                config.mla_n_latents * config.mla_latent_dim +  # Latent vectors
                config.d_model * config.mla_latent_dim +        # Token to latent
                config.mla_latent_dim * config.d_model +        # Latent to token
                3 * config.mla_latent_dim * config.mla_latent_dim +  # Q, K, V for latents
                3 * config.mla_latent_dim * config.d_model      # Cross-attention projections
            )
            norm_params += config.mla_latent_dim + config.d_model  # Additional norms
        
        # MoE expert layer
        expert_params = 0
        if config.n_experts > 0:
            # Experts (2-layer MLPs)
            expert_mlp_params = 2 * config.d_model * config.expert_d_ff
            expert_params = config.n_experts * expert_mlp_params
            
            # Gating network
            gate_params = config.d_model * config.n_experts
            expert_params += gate_params
        else:
            # Standard FFN
            expert_params = config.d_model * config.expert_d_ff + config.expert_d_ff * config.d_model
        
        layer_params += attn_params + norm_params + mla_params + expert_params
    
    # Output projection
    output_params = config.d_model * config.vocab_size
    
    total_params = token_embed_params + pos_embed_params + layer_params + output_params
    
    return total_params


def print_model_summary(config: MoEModelConfig):
    """Print a summary of the model configuration."""
    print("="*60)
    print("MoE Model Configuration Summary")
    print("="*60)
    
    print(f"Architecture:")
    print(f"  Vocabulary size: {config.vocab_size:,}")
    print(f"  Model dimension: {config.d_model}")
    print(f"  Attention heads: {config.n_heads}")
    print(f"  Layers: {config.n_layers}")
    print(f"  Max sequence length: {config.max_seq_len}")
    print(f"  Dropout: {config.dropout}")
    
    print(f"\nMixture of Experts:")
    print(f"  Number of experts: {config.n_experts}")
    print(f"  Expert capacity: {config.expert_capacity}")
    print(f"  Top-k routing: {config.top_k}")
    print(f"  Gate temperature: {config.gate_temperature}")
    print(f"  Load balance weight: {config.load_balance_weight}")
    
    print(f"\nMulti-Head Latent Attention:")
    print(f"  MLA interval: {config.mla_interval}")
    print(f"  Latent dimension: {config.mla_latent_dim}")
    print(f"  Number of latents: {config.mla_n_latents}")
    print(f"  Use RoPE: {config.mla_use_rope}")
    
    # Count MLA layers
    mla_layers = sum(1 for i in range(config.n_layers) if (i + 1) % config.mla_interval == 0 and i > 0)
    print(f"  MLA layers: {mla_layers}/{config.n_layers}")
    
    # Parameter estimation
    total_params = estimate_model_parameters(config)
    if total_params > 1e9:
        param_str = f"{total_params/1e9:.1f}B"
    elif total_params > 1e6:
        param_str = f"{total_params/1e6:.1f}M"
    else:
        param_str = f"{total_params/1e3:.1f}K"
    
    print(f"\nEstimated Parameters: {param_str}")
    
    # Validation
    issues = validate_model_config(config)
    if issues:
        print(f"\nConfiguration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\nConfiguration: ✓ Valid")
    
    print("="*60)


if __name__ == "__main__":
    # Example usage
    config = get_model_config("base")
    print_model_summary(config)
    
    # Create custom configuration
    custom_config = create_custom_config(
        base_config="base",
        n_experts=16,
        mla_latent_dim=384,
    )
    
    print("\nCustom Configuration:")
    print_model_summary(custom_config)
