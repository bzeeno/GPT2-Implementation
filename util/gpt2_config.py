from dataclasses import dataclass

@dataclass
class GPT2_CONFIG_124M:
    vocab_size: int = 50257          # Vocabulary size
    context_length: int = 256       # Context length 1024
    embdg_dim: int = 768             # Embedding dimension
    n_heads: int = 12                # Number of attention heads
    n_transformer_blocks: int = 12   # Number of layers
    drop_rate: float = 0.1           # Dropout rate
    qkv_bias: bool = False           # Query-Key-Value bias

@dataclass
class TRAIN_SETTINGS:
    learning_rate: float = 5e-4
    train_ratio: float = 0.9
    num_epochs: int = 1
    batch_size: int = 2    
    weight_decay: float = 0.1