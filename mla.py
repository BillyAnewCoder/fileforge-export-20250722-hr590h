"""
Multi-Head Latent Attention (MLA) implementation.
Projects tokens to latent space before expert routing, then back to token space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention layer that projects tokens to a smaller latent space
    before processing and then projects back to the original token space.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        latent_dim: int = 256,
        n_latents: int = 64,
        dropout: float = 0.1,
        use_rope: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.latent_dim = latent_dim
        self.n_latents = n_latents
        self.dropout = dropout
        self.use_rope = use_rope
        
        assert latent_dim % n_heads == 0, "latent_dim must be divisible by n_heads"
        self.head_dim = latent_dim // n_heads
        
        # Learnable latent vectors
        self.latent_vectors = nn.Parameter(torch.randn(n_latents, latent_dim))
        
        # Projection layers
        self.token_to_latent = nn.Linear(d_model, latent_dim)
        self.latent_to_token = nn.Linear(latent_dim, d_model)
        
        # Attention components for latent space
        self.latent_q_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.latent_k_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.latent_v_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        
        # Cross-attention: tokens attend to latents
        self.cross_q_proj = nn.Linear(d_model, latent_dim, bias=False)
        self.cross_k_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.cross_v_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(latent_dim, d_model)
        
        # Layer normalization
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.output_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # RoPE (Rotary Position Embedding) for latent attention
        if use_rope:
            self.rope = RotaryPositionalEmbedding(self.head_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following standard transformer initialization."""
        for module in [self.latent_q_proj, self.latent_k_proj, self.latent_v_proj,
                      self.cross_q_proj, self.cross_k_proj, self.cross_v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
        
        # Initialize latent vectors with small random values
        nn.init.normal_(self.latent_vectors, std=0.02)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Multi-Head Latent Attention.
        
        Args:
            hidden_states: Input token representations [batch_size, seq_len, d_model]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary containing:
            - hidden_states: Updated token representations
            - latent_representations: Latent space representations
            - attention_weights: Cross-attention weights
        """
        batch_size, seq_len, d_model = hidden_states.shape
        device = hidden_states.device
        
        # Expand latent vectors for batch processing
        latents = self.latent_vectors.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, n_latents, latent_dim]
        
        # Step 1: Self-attention in latent space
        latent_attended = self._latent_self_attention(latents, attention_mask)
        
        # Step 2: Cross-attention - tokens attend to updated latents
        cross_attended = self._cross_attention(hidden_states, latent_attended, attention_mask)
        
        # Step 3: Project back to token space
        output = self.out_proj(cross_attended)
        output = self.proj_dropout(output)
        
        # Residual connection and layer norm
        output = self.output_norm(output + hidden_states)
        
        return {
            "hidden_states": output,
            "latent_representations": latent_attended,
        }
    
    def _latent_self_attention(
        self,
        latents: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Self-attention within latent space.
        
        Args:
            latents: Latent vectors [batch_size, n_latents, latent_dim]
            attention_mask: Optional attention mask
            
        Returns:
            Updated latent vectors [batch_size, n_latents, latent_dim]
        """
        batch_size, n_latents, latent_dim = latents.shape
        
        # Project to Q, K, V
        q = self.latent_q_proj(latents)  # [batch, n_latents, latent_dim]
        k = self.latent_k_proj(latents)  # [batch, n_latents, latent_dim]
        v = self.latent_v_proj(latents)  # [batch, n_latents, latent_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, n_latents, self.n_heads, self.head_dim).transpose(1, 2)  # [batch, heads, n_latents, head_dim]
        k = k.view(batch_size, n_latents, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, n_latents, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE if enabled
        if self.use_rope:
            q, k = self.rope(q, k)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, v)  # [batch, heads, n_latents, head_dim]
        
        # Reshape back
        attended = attended.transpose(1, 2).contiguous()  # [batch, n_latents, heads, head_dim]
        attended = attended.view(batch_size, n_latents, latent_dim)  # [batch, n_latents, latent_dim]
        
        # Residual connection and layer norm
        attended = self.latent_norm(attended + latents)
        
        return attended
    
    def _cross_attention(
        self,
        tokens: torch.Tensor,
        latents: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-attention where tokens attend to latents.
        
        Args:
            tokens: Token representations [batch_size, seq_len, d_model]
            latents: Latent representations [batch_size, n_latents, latent_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Updated token representations in latent space [batch_size, seq_len, latent_dim]
        """
        batch_size, seq_len, d_model = tokens.shape
        n_latents = latents.size(1)
        
        # Queries from tokens, Keys and Values from latents
        q = self.cross_q_proj(tokens)    # [batch, seq_len, latent_dim]
        k = self.cross_k_proj(latents)   # [batch, n_latents, latent_dim]
        v = self.cross_v_proj(latents)   # [batch, n_latents, latent_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)      # [batch, heads, seq_len, head_dim]
        k = k.view(batch_size, n_latents, self.n_heads, self.head_dim).transpose(1, 2)   # [batch, heads, n_latents, head_dim]
        v = v.view(batch_size, n_latents, self.n_heads, self.head_dim).transpose(1, 2)   # [batch, heads, n_latents, head_dim]
        
        # Cross-attention scores: tokens attend to latents
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch, heads, seq_len, n_latents]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for cross-attention
            mask = attention_mask.unsqueeze(1).unsqueeze(-1)  # [batch, 1, seq_len, 1]
            mask = mask.expand(-1, self.n_heads, -1, n_latents)  # [batch, heads, seq_len, n_latents]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to latent values
        attended = torch.matmul(attn_weights, v)  # [batch, heads, seq_len, head_dim]
        
        # Reshape back
        attended = attended.transpose(1, 2).contiguous()  # [batch, seq_len, heads, head_dim]
        attended = attended.view(batch_size, seq_len, self.latent_dim)  # [batch, seq_len, latent_dim]
        
        return attended


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for improved positional encoding.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute rotary embeddings
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache for precomputed sin and cos values
        self._cached_seq_len = 0
        self._cached_cos = None
        self._cached_sin = None
    
    def _compute_cos_sin(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos and sin values for given sequence length."""
        if seq_len > self._cached_seq_len or self._cached_cos is None:
            # Extend cache
            positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(positions, self.inv_freq)
            
            # Compute sin and cos
            cos = freqs.cos()
            sin = freqs.sin()
            
            # Cache results
            self._cached_seq_len = seq_len
            self._cached_cos = cos
            self._cached_sin = sin
        
        return self._cached_cos[:seq_len], self._cached_sin[:seq_len]
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding to query and key tensors.
        
        Args:
            q: Query tensor [batch, heads, seq_len, head_dim]
            k: Key tensor [batch, heads, seq_len, head_dim]
            
        Returns:
            Tuple of rotated (q, k) tensors
        """
        seq_len = q.size(-2)
        device = q.device
        
        # Get cos and sin values
        cos, sin = self._compute_cos_sin(seq_len, device)
        
        # Apply rotary embedding
        def rotate_half(x):
            """Rotate half the hidden dims of the input."""
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)
        
        # Expand cos and sin for broadcasting
        cos = cos[None, None, :, :]  # [1, 1, seq_len, dim//2]
        sin = sin[None, None, :, :]  # [1, 1, seq_len, dim//2]
        
        # Apply rotation
        q_rotated = q * cos + rotate_half(q) * sin
        k_rotated = k * cos + rotate_half(k) * sin
        
        return q_rotated, k_rotated


class AdaptiveMLA(MultiHeadLatentAttention):
    """
    Adaptive Multi-Head Latent Attention that can dynamically adjust the number of latents.
    """
    
    def __init__(self, *args, adaptive_latents: bool = True, min_latents: int = 16, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.adaptive_latents = adaptive_latents
        self.min_latents = min_latents
        
        if adaptive_latents:
            # Gating mechanism to select active latents
            self.latent_gate = nn.Linear(self.latent_dim, 1)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with adaptive latent selection."""
        if self.adaptive_latents and self.training:
            # Select active latents based on gating
            batch_size = hidden_states.size(0)
            latents = self.latent_vectors.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Compute gate scores
            gate_scores = self.latent_gate(latents).squeeze(-1)  # [batch, n_latents]
            gate_probs = torch.sigmoid(gate_scores)
            
            # Sample latents (during training) or select top-k (during inference)
            if self.training:
                # Gumbel-Softmax for differentiable sampling
                selected_mask = torch.bernoulli(gate_probs)
                # Ensure minimum number of latents
                n_selected = selected_mask.sum(dim=-1, keepdim=True)
                mask_adjustment = (n_selected < self.min_latents).float()
                # Add random latents if below minimum
                if mask_adjustment.any():
                    additional_needed = self.min_latents - n_selected
                    random_indices = torch.randint(0, self.n_latents, (batch_size, self.min_latents), device=hidden_states.device)
                    additional_mask = torch.zeros_like(selected_mask)
                    additional_mask.scatter_(1, random_indices, 1)
                    selected_mask = torch.clamp(selected_mask + additional_mask * mask_adjustment, 0, 1)
            else:
                # Select top latents during inference
                _, top_indices = torch.topk(gate_probs, k=max(self.min_latents, self.n_latents // 2), dim=-1)
                selected_mask = torch.zeros_like(gate_probs)
                selected_mask.scatter_(1, top_indices, 1)
            
            # Apply mask to latents
            latents = latents * selected_mask.unsqueeze(-1)
            
            # Continue with normal MLA forward pass
            result = super().forward(hidden_states, attention_mask)
            result["active_latents"] = selected_mask.sum(dim=-1).float().mean()
            return result
        else:
            return super().forward(hidden_states, attention_mask)
