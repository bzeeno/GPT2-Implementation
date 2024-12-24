import torch
import torch.nn as nn
import tiktoken

from gpt2_config import GPT2_CONFIG_124M
from Transformer import TransformerBlock
from GPT import GPTModel

def gen_text_sample(model, token_seq, max_new_tokens, context_length):
    for _ in range(max_new_tokens):
        in_seq = token_seq[:, -context_length:]
        with torch.no_grad():
            logits = model(in_seq)
        logits = logits[:, -1, :] # focus on last embedding
        prob = torch.softmax(logits, dim=-1)
        next_tok = torch.argmax(prob, dim=-1, keepdim=True)
        token_seq = torch.cat((token_seq, next_tok), dim=1)
    return token_seq

def main():
    torch.manual_seed(24)
    model = GPTModel(GPT2_CONFIG_124M)
    model.eval()
    starting_context = "Hello world!"

    # Tokenize Input
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(starting_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    # Print num of parameters and memory requirements
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"Total size of the model: {total_size_mb:.2f} MB")

    print("input:", encoded_tensor)

    # Run inference on model
    out = gen_text_sample(
        model=model,
        token_seq=encoded_tensor,
        max_new_tokens=6,
        context_length=GPT2_CONFIG_124M.context_length
    )
    print("Output:", out)
    print("Output length:", len(out[0]))
    # Decode outputs
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)

main()