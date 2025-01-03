import torch
import torch.nn as nn
import tiktoken

from util.gpt2_config import GPT2_CONFIG_124M
from gpt.gpt_implementation import GPTModel

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

def encode_text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def decode_token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

def get_num_paramters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def get_mem_requirements(total_params):
    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    return total_size_mb

if __name__ == "__main__":
    torch.manual_seed(24)
    model = GPTModel(GPT2_CONFIG_124M)
    model.eval()
    starting_context = "Hello world!"

    # Tokenize Input
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded_tensor = encode_text_to_token_ids(starting_context, tokenizer)

    print("input:", encoded_tensor)

    # Print num of parameters and memory requirements
    total_params = get_num_paramters(model)
    print(f"Total number of parameters: {total_params:,}")
    total_size_mb = get_mem_requirements(total_params)
    print(f"Total size of the model: {total_size_mb:.2f} MB")

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
    print(decode_token_ids_to_text(out, tokenizer))