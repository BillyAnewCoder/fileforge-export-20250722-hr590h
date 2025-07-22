"""
Main Mixture-of-Experts model with Multi-Head Latent Attention integration.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any

from .transformer import TransformerBlock
from .mla import MultiHeadLatentAttention
from .expert import ExpertLayer
from .gating import GatingNetwork


class MoETransformerModel(nn.Module):
    """
    Mixture-of-Experts Transformer model with Multi-Head Latent Attention.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        n_experts: int = 8,
        expert_capacity: int = 4,
        top_k: int = 2,
        mla_interval: int = 4,
        mla_latent_dim: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        gate_temperature: float = 1.0,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.expert_capacity = expert_capacity
        self.top_k = top_k
        self.mla_interval = mla_interval
        self.mla_latent_dim = mla_latent_dim
        self.max_seq_len = max_seq_len
        self.gate_temperature = gate_temperature
        self.load_balance_weight = load_balance_weight
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer layers with MoE and MLA
        self.layers = nn.ModuleList()
        for layer_idx in range(n_layers):
            # Add MLA layer at regular intervals
            use_mla = (layer_idx + 1) % mla_interval == 0 and layer_idx > 0
            
            layer = TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_experts=n_experts,
                expert_capacity=expert_capacity,
                top_k=top_k,
                dropout=dropout,
                use_mla=use_mla,
                mla_latent_dim=mla_latent_dim if use_mla else None,
                gate_temperature=gate_temperature,
            )
            self.layers.append(layer)
        
        # Output layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)
    
    def get_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate position IDs for input tokens."""
        batch_size, seq_len = input_ids.shape
        return torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_expert_metrics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the MoE model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target labels for language modeling [batch_size, seq_len]
            return_expert_metrics: Whether to return expert utilization metrics
            
        Returns:
            Dictionary containing logits, loss, and optional expert metrics
        """
        batch_size, seq_len = input_ids.shape
        
        # Generate attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Token and position embeddings
        position_ids = self.get_position_ids(input_ids)
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        
        hidden_states = self.dropout(token_embeds + position_embeds)
        
        # Track expert metrics
        all_expert_metrics = []
        total_load_balance_loss = 0.0
        
        # Pass through transformer layers
        for layer in self.layers:
            layer_output = layer(hidden_states, attention_mask)
            hidden_states = layer_output["hidden_states"]
            
            if "expert_metrics" in layer_output:
                all_expert_metrics.append(layer_output["expert_metrics"])
            
            if "load_balance_loss" in layer_output:
                total_load_balance_loss += layer_output["load_balance_loss"]
        
        # Final layer norm and output projection
        hidden_states = self.layer_norm(hidden_states)
        logits = self.output_projection(hidden_states)
        
        # Prepare output
        outputs = {"logits": logits}
        
        # Calculate language modeling loss if labels provided
        if labels is not None:
            # Shift labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross-entropy loss
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Add load balance loss
            total_loss = lm_loss + self.load_balance_weight * total_load_balance_loss
            
            outputs["loss"] = total_loss
            outputs["lm_loss"] = lm_loss
            outputs["load_balance_loss"] = total_load_balance_loss
        
        # Add expert metrics if requested
        if return_expert_metrics and all_expert_metrics:
            outputs["expert_metrics"] = self._aggregate_expert_metrics(all_expert_metrics)
        
        return outputs
    
    def _aggregate_expert_metrics(self, metrics_list: list) -> Dict[str, torch.Tensor]:
        """Aggregate expert metrics across all layers."""
        if not metrics_list:
            return {}
        
        # Average utilization across layers
        total_utilization = torch.stack([m["expert_utilization"] for m in metrics_list])
        avg_utilization = total_utilization.mean(dim=0)
        
        # Sum routing probabilities
        total_routing_probs = torch.stack([m["routing_probabilities"] for m in metrics_list])
        sum_routing_probs = total_routing_probs.sum(dim=0)
        
        return {
            "expert_utilization": avg_utilization,
            "routing_probabilities": sum_routing_probs,
            "num_layers_with_experts": len(metrics_list),
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text using the trained model.
        
        Args:
            input_ids: Initial token IDs [batch_size, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            do_sample: Whether to use sampling or greedy decoding
            eos_token_id: End-of-sequence token ID
            
        Returns:
            Generated token IDs [batch_size, generated_len]
        """
        self.eval()
        batch_size = input_ids.size(0)
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                outputs = self(generated)
                logits = outputs["logits"]
                
                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature
                
                if do_sample:
                    # Top-p sampling
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits.scatter_(1, indices_to_remove.unsqueeze(1), float('-inf'))
                    
                    # Sample from filtered distribution
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_tokens], dim=-1)
                
                # Check for end of sequence
                if eos_token_id is not None and (next_tokens == eos_token_id).all():
                    break
        
        return generated
    
    def get_expert_stats(self) -> Dict[str, Any]:
        """Get statistics about expert usage across the model."""
        stats = {
            "total_experts": 0,
            "experts_per_layer": [],
            "mla_layers": [],
        }
        
        for layer_idx, layer in enumerate(self.layers):
            if hasattr(layer, 'expert_layer') and layer.expert_layer is not None:
                stats["total_experts"] += layer.expert_layer.n_experts
                stats["experts_per_layer"].append(layer.expert_layer.n_experts)
            else:
                stats["experts_per_layer"].append(0)
            
            if hasattr(layer, 'mla') and layer.mla is not None:
                stats["mla_layers"].append(layer_idx)
        
        return stats
