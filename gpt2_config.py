from dataclasses import dataclass

@dataclass
class GPT2_CONFIG_124M:
    vocab_size: int = 50257          # Vocabulary size
    context_length: int = 1024       # Context length
    embdg_dim: int = 768             # Embedding dimension
    n_heads: int = 12                # Number of attention heads
    n_transformer_blocks: int = 12   # Number of layers
    drop_rate: float = 0.1           # Dropout rate
    qkv_bias: bool = False           # Query-Key-Value bias    
