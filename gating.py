"""
Gating network for routing tokens to experts in Mixture-of-Experts architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class GatingNetwork(nn.Module):
    """
    Advanced gating network with load balancing and noise for MoE routing.
    """
    
    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        gate_noise: float = 1e-2,
        temperature: float = 1.0,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.gate_noise = gate_noise
        self.temperature = temperature
        self.load_balance_weight = load_balance_weight
        
        # Main gating linear layer
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        
        # Noise layer for exploration during training
        self.noise_gate = nn.Linear(d_model, n_experts, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize gating weights with small uniform distribution."""
        nn.init.uniform_(self.gate.weight, -0.1, 0.1)
        nn.init.uniform_(self.noise_gate.weight, -0.1, 0.1)
    
    def _add_noise(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Add noise to gating logits during training for exploration.
        
        Args:
            logits: Gating logits [batch_size, seq_len, n_experts]
            
        Returns:
            Noisy logits
        """
        if not self.training or self.gate_noise == 0:
            return logits
        
        # Generate noise
        noise = torch.randn_like(logits) * self.gate_noise
        return logits + noise
    
    def _compute_capacity(self, batch_size: int, seq_len: int) -> int:
        """
        Compute expert capacity based on capacity factor.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Expert capacity (max tokens per expert)
        """
        total_tokens = batch_size * seq_len
        tokens_per_expert = total_tokens / self.n_experts
        capacity = int(tokens_per_expert * self.capacity_factor)
        return max(capacity, 1)  # Ensure at least 1 token per expert
    
    def _compute_load_balance_loss(
        self,
        gate_probs: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute load balance loss to encourage uniform expert utilization.
        
        Args:
            gate_probs: Gate probabilities [batch_size, seq_len, n_experts]
            expert_indices: Selected expert indices [batch_size, seq_len, top_k]
            
        Returns:
            Load balance loss scalar
        """
        batch_size, seq_len, _ = gate_probs.shape
        
        # Average gate probability for each expert
        gate_mean = gate_probs.mean(dim=[0, 1])  # [n_experts]
        
        # Fraction of tokens routed to each expert
        expert_mask = F.one_hot(expert_indices, num_classes=self.n_experts).float()  # [batch, seq, top_k, n_experts]
        expert_fraction = expert_mask.sum(dim=[0, 1, 2]) / (batch_size * seq_len * self.top_k)  # [n_experts]
        
        # Load balance loss (encourage uniform distribution)
        load_balance_loss = torch.sum(gate_mean * expert_fraction) * self.n_experts
        
        return load_balance_loss * self.load_balance_weight
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_capacity: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through gating network.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, d_model]
            expert_capacity: Maximum tokens per expert (if None, computed automatically)
            
        Returns:
            Dictionary containing:
            - expert_indices: Selected expert indices [batch_size, seq_len, top_k]
            - expert_weights: Corresponding weights [batch_size, seq_len, top_k]
            - gate_probs: All gate probabilities [batch_size, seq_len, n_experts]
            - load_balance_loss: Load balancing loss scalar
            - capacity_mask: Mask for capacity constraints [batch_size, seq_len, top_k]
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Compute gate logits
        gate_logits = self.gate(hidden_states)  # [batch_size, seq_len, n_experts]
        
        # Add noise for exploration during training
        if self.training:
            noise_logits = self.noise_gate(hidden_states)
            gate_logits = self._add_noise(gate_logits + noise_logits)
        
        # Apply temperature scaling
        gate_logits = gate_logits / self.temperature
        
        # Convert to probabilities
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        
        # Compute expert capacity
        if expert_capacity is None:
            expert_capacity = self._compute_capacity(batch_size, seq_len)
        
        # Apply capacity constraints
        capacity_mask = self._apply_capacity_constraints(
            top_k_indices, expert_capacity, batch_size, seq_len
        )
        
        # Mask probabilities based on capacity
        masked_probs = top_k_probs * capacity_mask
        
        # Renormalize probabilities
        prob_sum = masked_probs.sum(dim=-1, keepdim=True)
        prob_sum = torch.clamp(prob_sum, min=1e-8)  # Avoid division by zero
        normalized_probs = masked_probs / prob_sum
        
        # Compute load balance loss
        load_balance_loss = self._compute_load_balance_loss(gate_probs, top_k_indices)
        
        return {
            "expert_indices": top_k_indices,
            "expert_weights": normalized_probs,
            "gate_probs": gate_probs,
            "load_balance_loss": load_balance_loss,
            "capacity_mask": capacity_mask,
            "expert_capacity": expert_capacity,
        }
    
    def _apply_capacity_constraints(
        self,
        expert_indices: torch.Tensor,
        capacity: int,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Apply capacity constraints to limit tokens per expert.
        
        Args:
            expert_indices: Selected expert indices [batch_size, seq_len, top_k]
            capacity: Maximum tokens per expert
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Capacity mask [batch_size, seq_len, top_k]
        """
        device = expert_indices.device
        capacity_mask = torch.ones_like(expert_indices, dtype=torch.float, device=device)
        
        # Track token count for each expert
        expert_counts = torch.zeros(self.n_experts, dtype=torch.long, device=device)
        
        # Flatten indices for easier processing
        flat_indices = expert_indices.view(-1, self.top_k)  # [batch_size * seq_len, top_k]
        flat_mask = capacity_mask.view(-1, self.top_k)     # [batch_size * seq_len, top_k]
        
        # Process each token position
        for pos in range(flat_indices.shape[0]):
            for k in range(self.top_k):
                expert_id = flat_indices[pos, k].item()
                
                # Check if expert has capacity
                if expert_counts[expert_id] < capacity:
                    expert_counts[expert_id] += 1
                else:
                    # Expert at capacity, mask this assignment
                    flat_mask[pos, k] = 0.0
        
        return flat_mask.view(batch_size, seq_len, self.top_k)
    
    def update_temperature(self, temperature: float):
        """Update gating temperature for temperature scheduling."""
        self.temperature = temperature
    
    def get_routing_statistics(self) -> Dict[str, float]:
        """Get current routing statistics for monitoring."""
        with torch.no_grad():
            # Get gate weight statistics
            gate_weights = self.gate.weight.data
            
            stats = {
                "gate_weight_mean": gate_weights.mean().item(),
                "gate_weight_std": gate_weights.std().item(),
                "gate_weight_min": gate_weights.min().item(),
                "gate_weight_max": gate_weights.max().item(),
                "current_temperature": self.temperature,
                "gate_noise": self.gate_noise,
            }
            
            return stats


class SwitchGating(GatingNetwork):
    """
    Switch Transformer style gating (top-1 routing with capacity).
    """
    
    def __init__(self, *args, **kwargs):
        kwargs['top_k'] = 1  # Force top-1 routing
        super().__init__(*args, **kwargs)
    
    def forward(self, hidden_states: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass with Switch-style routing."""
        result = super().forward(hidden_states, **kwargs)
        
        # For Switch, we use top-1 so no need to normalize across k
        # But we still apply the same logic
        return result


class GLaM_Gating(GatingNetwork):
    """
    GLaM style gating with specific capacity and load balancing strategies.
    """
    
    def __init__(self, *args, **kwargs):
        # GLaM typically uses higher capacity factor
        kwargs.setdefault('capacity_factor', 2.0)
        kwargs.setdefault('top_k', 2)
        super().__init__(*args, **kwargs)
    
    def _compute_load_balance_loss(self, gate_probs: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """GLaM-specific load balance loss computation."""
        # Use coefficient of variation for load balancing
        batch_size, seq_len, _ = gate_probs.shape
        
        # Count tokens per expert
        expert_counts = torch.zeros(self.n_experts, device=gate_probs.device)
        expert_mask = F.one_hot(expert_indices, num_classes=self.n_experts).float()
        expert_counts = expert_mask.sum(dim=[0, 1, 2])
        
        # Compute coefficient of variation
        mean_count = expert_counts.mean()
        std_count = expert_counts.std()
        cv = std_count / (mean_count + 1e-8)
        
        return cv * self.load_balance_weight
