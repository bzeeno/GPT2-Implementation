import torch
import torch.nn as nn

from LayerNormalization import LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.first_layerNorm = LayerNorm(config.embdg_dim)
        self.attn_heads = MultiHeadAttention(config)
        self.second_layerNorm = LayerNorm(config.embdg_dim)
        self.MLP = MLP(config.embdg_dim)
        self.shortcut_dropout = nn.Dropout(config.drop_rate)

    def forward(self, x):
        # LayerNorm -> MultiHead Attention -> dropout + shortcut
        shortcut = x # shortcut connection for multi-head attention
        x = self.first_layerNorm(x)
        x = self.MultiHeadAttention(x)
        x = self.shortcut_dropout(x)
        x = x + shortcut

        # LayerNorm -> MLP -> dropout + shortcut
        shortcut = x # Shortcut for MLP
        x = self.second_layerNorm(x)
        x = self.MLP(x)
        x = self.shortcut_dropout(x)
        x = x + shortcut

        return x

class MLP(nn.module):
    def __init__(self, embdg_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embdg_dim, 4 * embdg_dim),
            nn.GELU(),
            nn.Linear(4 * embdg_dim, embdg_dim)
        )

    def forward(self, x):
        return self.layers(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert(config.embdg_dim % config.num_heads == 0), "embedding dimension must be divisible by the number of heads"
        # Initialize multi-head attention
        self.num_heads = config.num_heads
        self.context_length = config.context_length
        self.embdg_dim = config.embdg_dim
        self.qkv_bias = config.qkv_bias
        self.head_dim = config.embdg_dim//config.num_heads # Dimension of each head
        # Initialize weight vectors for q,k,v
        self.query_w = nn.Linear(config.embdg_dim, config.embdg_dim, bias=qkv_bias)
        self.key_w = nn.Linear(config.embdg_dim, config.embdg_dim, bias=qkv_bias)
        self.value_w = nn.Linear(config.embdg_dim, config.embdg_dim, bias=qkv_bias)
        # Initialize mask
        self.register_buffer("mask", torch.triu(torch.ones(config.context_length, config.context_length), diagonal=1))
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