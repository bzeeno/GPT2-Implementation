import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, embdg_dim):
        super().__init__()
        self.epsilon = 1e-5 # epsilon is a small constant added to the variance to prevent division by 0
        # Learnable parameters to adjust normalization if the model thinks it will help
        self.scale = nn.Parameter(torch.ones(embdg_dim)) # Initialize to ones for scaling
        self.shift = nn.Parameter(torch.zeros(embdg_dim)) # Initialize to zeros for shifting

    def forward(self, x):
        # Get mean and variance
        x_mean = x.mean(dim=-1, keepdim=True)
        x_var = x.var(dim=-1, keepdim=True)
        # Normalize x
        x_norm = (x - x_mean)/torch.sqrt(var + self.epsilon)
        # Add trainable paramters
        return self.scale * x_norm + self.shift

