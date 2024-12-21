import torch
import torch.nn as nn
import tiktoken

from gpt2_config import GPT2_CONFIG_124M
from Transformer import TransformerBlock
from GPT import GPTModel

def main():
    torch.manual_seed(24)
    model = GPTModel(GPT2_CONFIG_124M)
    model.eval()
    starting_context = "Hello world!"

    # Tokenize Input
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(starting_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    # Testing
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    # Run inference on model

    # Decode outputs

main()