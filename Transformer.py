import torch
import torch.nn as nn

class TransformerLayer(nn.Module):
    def __init__():
        super().__init__()


class CausalAttention(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, context_length, qkv_bias=False):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        # Initialize query, key, value weight matrices
        self.query_w = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.key_w = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.value_w = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        # Initialize dropout
        self.dropout = nn.Dropout(dropout)
        # Initialize mask
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length)), diagonal=1)

    def forward(self, x):
        # Decompose shape of input
        batch_size, num_tokens, in_dim = x.shape
        # Get query, key, value vectors for the inputs
        q, k, v = self.query_w(x), self.key_w(x), self.value_w(x)
        # Calculate attention scores and create mask
        attention_scores = q @ k.transpose(1, 2) # transpose input sequence, but keep batches as they are.
        attention_scores.masked_fill_(self.mask.bool()[:num_tokens][:num_tokens], -torch.inf) # create mask in-line only including up to the number of tokens, where all masked values are set to -infinity
        # Calculate attention weights
        attention_weights = nn.Softmax(attention_scores/k.shape[-1]**0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)
        # Create context vectors
        context_vec = attention_weights @ v
        return context_vec

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, in_dim, out_dim, dropout, context_length, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([CausalAttention(in_dim, out_dim, dropout, context_length, qkv_bias) for _ in range(num_heads)])
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
