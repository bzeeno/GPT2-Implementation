import torch
import torch.nn as nn

from gpt2_config import GPT2_CONFIG_124M
from Transformer import TransformerLayer

class NormLayer(nn.Module):
    def __init__():
        super().__init__()

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Inputs to transformer layers
        self.token_embdg_layer = nn.Embedding(cfg["vocab_size"], cfg["embedding_dimension"])
        self.pos_embdg_layer = nn.Embedding(cfg["context_length"], cfg["embedding_dimension"])

        # Transformer layers
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_transformer_blocks"])])
        
        # Output layers
        self.final_layer_norm = NormLayer(cfg["embedding_dimension"])
        self.out_head = nn.Linear(cfg["embedding_dimension"], cfg["vocab_size"])

    def forward(self, x):
        return