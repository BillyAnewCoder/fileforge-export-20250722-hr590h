"""
Transformer block with integrated Mixture-of-Experts and Multi-Head Latent Attention.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .expert import ExpertLayer
from .mla import MultiHeadLatentAttention


class TransformerBlock(nn.Module):
    """
    Transformer block that integrates MoE and MLA capabilities.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 12,
        n_experts: Optional[int] = None,
        expert_capacity: int = 4,
        top_k: int = 2,
        dropout: float = 0.1,
        use_mla: bool = False,
        mla_latent_dim: Optional[int] = None,
        gate_temperature: float = 1.0,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_mla = use_mla
        self.use_moe = n_experts is not None
        
        # Self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Multi-Head Latent Attention (optional)
        if use_mla:
            assert mla_latent_dim is not None, "mla_latent_dim must be provided when use_mla=True"
            self.mla = MultiHeadLatentAttention(
                d_model=d_model,
                n_heads=n_heads,
                latent_dim=mla_latent_dim,
                dropout=dropout,
            )
        else:
            self.mla = None
        
        # Expert layer (MoE) or standard FFN
        if self.use_moe:
            self.expert_layer = ExpertLayer(
                d_model=d_model,
                n_experts=n_experts,
                expert_capacity=expert_capacity,
                top_k=top_k,
                dropout=dropout,
                gate_temperature=gate_temperature,
            )
            self.ffn = None
        else:
            # Standard feed-forward network
            d_ff = 4 * d_model
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )
            self.expert_layer = None
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        if use_mla:
            self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_expert_metrics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through transformer block.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, d_model]
            attention_mask: Attention mask [batch_size, seq_len]
            return_expert_metrics: Whether to return expert utilization metrics
            
        Returns:
            Dictionary containing hidden_states and optional metrics
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Prepare attention mask for PyTorch's MultiheadAttention
        if attention_mask is not None:
            # Convert from [batch, seq_len] to [batch*n_heads, seq_len, seq_len]
            attn_mask = self._prepare_attention_mask(attention_mask, seq_len)
        else:
            attn_mask = None
        
        # 1. Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        
        attn_output, attn_weights = self.self_attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=~attention_mask.bool() if attention_mask is not None else None,
            need_weights=False,
        )
        
        hidden_states = residual + self.dropout(attn_output)
        
        # 2. Multi-Head Latent Attention (optional)
        if self.mla is not None:
            residual = hidden_states
            hidden_states = self.norm3(hidden_states)
            
            mla_output = self.mla(hidden_states, attention_mask)
            hidden_states = residual + self.dropout(mla_output["hidden_states"])
        
        # 3. Expert layer (MoE) or standard FFN
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        
        result = {"hidden_states": hidden_states}
        
        if self.use_moe:
            # Process through expert layer
            expert_output = self.expert_layer(hidden_states, return_metrics=return_expert_metrics)
            hidden_states = residual + self.dropout(expert_output["output"])
            
            # Add expert-specific outputs
            if "load_balance_loss" in expert_output:
                result["load_balance_loss"] = expert_output["load_balance_loss"]
            
            if "expert_metrics" in expert_output:
                result["expert_metrics"] = expert_output["expert_metrics"]
        else:
            # Standard FFN
            ffn_output = self.ffn(hidden_states)
            hidden_states = residual + ffn_output
        
        result["hidden_states"] = hidden_states
        return result
    
    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Prepare attention mask for PyTorch's MultiheadAttention.
        
        Args:
            attention_mask: Input mask [batch_size, seq_len]
            seq_len: Sequence length
            
        Returns:
            Prepared mask for attention computation
        """
        batch_size = attention_mask.size(0)
        
        # Create causal mask for autoregressive generation
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=attention_mask.device),
            diagonal=1
        ).bool()
        
        # Expand attention mask to [batch, seq_len, seq_len]
        expanded_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine with causal mask
        combined_mask = expanded_mask.unsqueeze(1) & (~causal_mask.unsqueeze(0))
        
        return combined_mask
    
    def set_gate_temperature(self, temperature: float):
        """Update expert gate temperature for scheduling."""
        if self.expert_layer is not None:
            self.expert_layer.set_gate_temperature(temperature)
    
    def get_expert_stats(self) -> Dict:
        """Get expert statistics for monitoring."""
        if self.expert_layer is not None:
            return {
                "has_experts": True,
                "n_experts": self.expert_layer.n_experts,
                "top_k": self.expert_layer.top_k,
                "expert_capacity": self.expert_layer.expert_capacity,
            }
        else:
            return {"has_experts": False}


class DecoderOnlyTransformerBlock(TransformerBlock):
    """
    Decoder-only transformer block optimized for language modeling.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Override self-attention with causal mask
        self.self_attention = CausalMultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            dropout=kwargs.get('dropout', 0.1),
            batch_first=True,
        )


class CausalMultiheadAttention(nn.Module):
    """
    Causal multi-head attention for decoder-only models.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        batch_first: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Cache for causal mask
        self._causal_mask = None
        self._causal_mask_size = 0
    
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or create causal mask for the given sequence length."""
        if self._causal_mask is None or self._causal_mask_size < seq_len:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device),
                diagonal=1
            ).bool()
            self._causal_mask = mask
            self._causal_mask_size = seq_len
        
        return self._causal_mask[:seq_len, :seq_len]
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> tuple:
        """
        Forward pass with causal masking.
        
        Args:
            query: Query tensor [batch_size, seq_len, embed_dim]
            key: Key tensor [batch_size, seq_len, embed_dim]
            value: Value tensor [batch_size, seq_len, embed_dim]
            key_padding_mask: Padding mask [batch_size, seq_len]
            need_weights: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, embed_dim = query.shape
        
        # Project to Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply causal mask
        causal_mask = self._get_causal_mask(seq_len, query.device)
        scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            scores.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_proj(output)
        output = self.proj_dropout(output)
        
        if need_weights:
            return output, attn_weights.mean(dim=1)  # Average over heads
        else:
            return output, None
