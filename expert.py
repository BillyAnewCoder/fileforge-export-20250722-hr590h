"""
Expert module implementation for Mixture-of-Experts architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class Expert(nn.Module):
    """
    Individual expert module - a 2-layer MLP with GeLU activation.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Two-layer MLP
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through expert.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # First linear layer + activation + dropout
        hidden = self.linear1(x)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        
        # Second linear layer
        output = self.linear2(hidden)
        
        return output


class ExpertLayer(nn.Module):
    """
    Layer containing multiple experts with sparse routing.
    """
    
    def __init__(
        self,
        d_model: int,
        n_experts: int = 8,
        expert_capacity: int = 4,
        top_k: int = 2,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        gate_temperature: float = 1.0,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_experts = n_experts
        self.expert_capacity = expert_capacity
        self.top_k = top_k
        self.gate_temperature = gate_temperature
        self.load_balance_weight = load_balance_weight
        
        # Create experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(n_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        
        # Initialize gate weights to small values for stable training
        with torch.no_grad():
            self.gate.weight.normal_(0, 0.1)
    
    def _compute_auxiliary_loss(self, gate_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute auxiliary loss for load balancing.
        
        Args:
            gate_probs: Gate probabilities [batch_size, seq_len, n_experts]
            
        Returns:
            Auxiliary loss scalar
        """
        # Average probability of selecting each expert
        prob_mean = gate_probs.mean(dim=[0, 1])  # [n_experts]
        
        # Fraction of tokens routed to each expert
        tokens_per_expert = (gate_probs > 0).float().mean(dim=[0, 1])  # [n_experts]
        
        # Load balance loss encourages uniform distribution
        aux_loss = torch.sum(prob_mean * tokens_per_expert) * self.n_experts
        
        return aux_loss
    
    def forward(
        self,
        x: torch.Tensor,
        return_metrics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through expert layer with sparse routing.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            return_metrics: Whether to return expert utilization metrics
            
        Returns:
            Dictionary containing output and optional metrics
        """
        batch_size, seq_len, d_model = x.shape
        
        # Reshape for easier processing
        x_flat = x.view(-1, d_model)  # [batch_size * seq_len, d_model]
        
        # Compute gating scores
        gate_logits = self.gate(x_flat)  # [batch_size * seq_len, n_experts]
        gate_logits = gate_logits / self.gate_temperature
        
        # Apply softmax to get probabilities
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Select top-k experts for each token
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        
        # Renormalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Route tokens to selected experts
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]  # [batch_size * seq_len]
            expert_weights = top_k_probs[:, k]    # [batch_size * seq_len]
            
            # Process tokens for each expert
            for expert_id in range(self.n_experts):
                # Find tokens assigned to this expert
                expert_mask = (expert_indices == expert_id)
                
                if expert_mask.any():
                    # Get tokens for this expert
                    expert_tokens = x_flat[expert_mask]  # [n_tokens, d_model]
                    expert_weights_masked = expert_weights[expert_mask]  # [n_tokens]
                    
                    # Process through expert
                    expert_output = self.experts[expert_id](expert_tokens)
                    
                    # Weight by gating probability
                    weighted_output = expert_output * expert_weights_masked.unsqueeze(-1)
                    
                    # Add to output
                    output[expert_mask] += weighted_output
        
        # Reshape back to original shape
        output = output.view(batch_size, seq_len, d_model)
        
        # Prepare return dictionary
        result = {"output": output}
        
        # Compute auxiliary loss for load balancing
        aux_loss = self._compute_auxiliary_loss(gate_probs.view(batch_size, seq_len, -1))
        result["load_balance_loss"] = aux_loss * self.load_balance_weight
        
        # Add metrics if requested
        if return_metrics:
            # Expert utilization (fraction of tokens routed to each expert)
            expert_utilization = torch.zeros(self.n_experts, device=x.device)
            for expert_id in range(self.n_experts):
                expert_mask = (top_k_indices == expert_id).any(dim=-1)
                expert_utilization[expert_id] = expert_mask.float().mean()
            
            result["expert_metrics"] = {
                "expert_utilization": expert_utilization,
                "routing_probabilities": gate_probs.view(batch_size, seq_len, -1),
                "top_k_indices": top_k_indices.view(batch_size, seq_len, -1),
                "top_k_probs": top_k_probs.view(batch_size, seq_len, -1),
            }
        
        return result
    
    def get_expert_weights(self) -> Dict[str, torch.Tensor]:
        """Get current expert gating weights for analysis."""
        return {
            "gate_weights": self.gate.weight.data.clone(),
        }
    
    def set_gate_temperature(self, temperature: float):
        """Update gate temperature for temperature scheduling."""
        self.gate_temperature = temperature
