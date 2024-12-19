import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, context_length, emdg_dim, dropout_rate, qkv_bias=False):
        super().__init__()
        assert(embdg_dim % num_heads == 0), "embedding dimension must be divisible by the number of heads"
        # Initialize multi-head attention
        self.num_heads = num_heads
        self.context_length = context_length
        self.embdg_dim = emdg_dim
        self.qkv_bias = qkv_bias
        self.head_dim = embdg_dim//num_heads # Dimension of each head
        # Initialize weight vectors for q,k,v
        self.query_w = nn.Linear(embdg_dim, embdg_dim, bias=qkv_bias)
        self.key_w = nn.Linear(embdg_dim, embdg_dim, bias=qkv_bias)
        self.value_w = nn.Linear(embdg_dim, embdg_dim, bias=qkv_bias)
        # Initialize mask
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
        # Initialize dropout
        self.dropout = nn.Dropout(dropout_rate)
        # Output projection
        self.out_proj = nn.Linear(embdg_dim, embdg_dim)
    
    def forward(self, x):
        # Get dimensions
        num_batches, num_tokens, _ = x.shape
        # Get query, key, value vectors from weight vectors
        query, key, value = self.query_w(x), self.key_w(x), self.value_w(x) # Dim: [num_batches, num_tokens, embdg_dim]
        # Transform vectors from [num_batches, num_tokens, embdg_dim] -> [num_batches, num_tokens, num_heads, head_dim]
        # then -> [num_batches, num_heads, num_tokens, head_dim] to keep num_tokens and head_dim next to each other
        query = query.view(num_batches, num_tokens, self.num_heads, self.head_dim)
        query = query.transpose(1,2)
        key = key.view(num_batches, num_tokens, self.num_heads, self.head_dim)
        key = key.transpose(1,2)
        value = value.view(num_batches, num_tokens, self.num_heads, self.head_dim)
        value = value.transpose(1,2)
        # Calculate attention scores
        attn_scores = query @ key.transpose(2, 3)
        # Apply mask
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens][:num_tokens], -torch.inf)
        # Transform attention scores to weights
        attn_weights = nn.Softmax(attn_scores/key.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # Create context vector
        context_vec = attn_weight @ value
        # Combine context vectors from each head
        context_vec = context_vec.transpose(1, 2)
        context_vec = context_vec.contiguous().view(num_batches, num_tokens, self.embdg_dim)
        return self.out_proj(context_vec)