import torch
import torch.nn as nn

from util.gpt2_config import GPT2_CONFIG_124M
from gpt.transformer import TransformerBlock
from gpt.layer_normalization import LayerNorm

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Inputs to transformer layers
        self.token_embdg_layer = nn.Embedding(config.vocab_size, config.embdg_dim)
        self.pos_embdg_layer = nn.Embedding(config.context_length, config.embdg_dim)
        self.embdg_dropout = nn.Dropout(config.drop_rate)

        # Transformer layers
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_transformer_blocks)])
        
        # Output layers
        self.final_layer_norm = LayerNorm(config.embdg_dim)
        self.out_head = nn.Linear(config.embdg_dim, config.vocab_size, bias=False)

        # Weight tying
        self.token_embdg_layer.weight = self.out_head.weight

        # Initialize model
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_seq):
        # Initialize size and embeddings
        num_batches, seq_length = token_seq.shape
        token_embdgs = self.token_embdg_layer(token_seq)
        pos_embdgs = self.pos_embdg_layer(torch.arange(0, seq_length, device=token_seq.device))
        # Get embeddings and dropout
        embdgs = token_embdgs + pos_embdgs
        self.embdg_dropout(embdgs)
        # Pass embeddings into transformer blocks
        x = self.transformer_blocks(embdgs)
        # Output from transformer blocks to final layer norm
        x = self.final_layer_norm(x)
        # Get final logits
        logits = self.out_head(x)
        return logits
