import os
import urllib.request
import tiktoken

import torch
import torch.nn as nn

import seaborn as sns
import matplotlib.pyplot as plt

from util.gpt2_config import GPT2_CONFIG_124M, TRAIN_SETTINGS
from gpt.gpt_implementation import GPTModel
from util.gpt_dataloader import GPTDataset, create_gpt_dataloader
from sample import encode_text_to_token_ids, gen_text_sample, decode_token_ids_to_text

def calc_loss(model, device, x_batch, y_batch):
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    logits = model(x_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y_batch.flatten())
    return loss

def calc_loss_dataloader(model, device, dataloader, num_batches=None):
    total_loss = 0. # running sum of loss
    if len(dataloader) == 0:
        return float("nan")
    elif num_batches == None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))
    
    for i, (x_batch, y_batch) in enumerate(dataloader):
        if i < num_batches:
            curr_loss = calc_loss(model, device, x_batch, y_batch)
            total_loss += curr_loss.item()
        else:
            break
    return total_loss / num_batches

def eval_model(model, device, train_dataloader, val_dataloader, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_dataloader(model, device, train_dataloader, num_batches=eval_iter)
        val_loss = calc_loss_dataloader(model, device, val_dataloader, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def gen_and_print_sample(model, device, tokenizer, start_context):
    model.eval()
    context_size = model.pos_embdg_layer.weight.shape[0]
    encoded_tensor = encode_text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = gen_text_sample(model, encoded_tensor, max_new_tokens=50, context_length=context_size)
        text = decode_token_ids_to_text(token_ids, tokenizer)
        print(text.replace("\n", " "))
    model.train()

def training_loop(model, device, tokenizer, optimizer, num_epochs, start_context, train_dataloader, val_dataloader, eval_freq, eval_iter):
    train_losses, val_losses, tokens_seen = [], [], []
    num_tokens_seen = 0
    num_batches_processed = -1

    for epoch in range(num_epochs):
        model.train() # Set to train mode

        for x_batch, y_batch in train_dataloader:
            optimizer.zero_grad() # Reset gradients to 0
            loss = calc_loss(model, device, x_batch, y_batch)
            loss.backward() # calculate gradients
            optimizer.step() # update weights
            num_tokens_seen += torch.numel(x_batch)
            num_batches_processed += 1

            # Evaluation
            if num_batches_processed % eval_freq == 0:
                train_loss, val_loss = eval_model(model, device, train_dataloader, val_dataloader, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                tokens_seen.append(num_tokens_seen)
                print(
                    f"epoch: {epoch}\n"
                    f"num_batches_processed: {num_batches_processed}\n"
                    f"train_loss: {train_loss}\n"
                    f"val_loss: {val_loss}\n"
                    f"\n"
                )
            
        # Sample from model and print outputs
        print("Model output:\n")
        gen_and_print_sample(model, device, tokenizer, start_context)
        print("\n-----------------------------------------------\n\n")

    return train_losses, val_losses, tokens_seen

def train():
    torch.manual_seed(24)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get text data
    data_dir = "data/"
    file_path = data_dir + "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    # Get model and optimizer
    model = GPTModel(GPT2_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_SETTINGS.learning_rate, weight_decay=TRAIN_SETTINGS.weight_decay)
    tokenizer = tiktoken.get_encoding("gpt2")

    # Set up dataloaders
    split_idx = int(TRAIN_SETTINGS.train_ratio * len(text_data))

    train_dataloader = create_gpt_dataloader(
        text = text_data[:split_idx], 
        tokenizer = tokenizer, 
        batch_size = TRAIN_SETTINGS.batch_size, 
        window_size = GPT2_CONFIG_124M.context_length,
        shuffle = True,
        drop_last = True,
        num_workers = 0
    )

    val_dataloader = create_gpt_dataloader(
        text = text_data[split_idx:],
        tokenizer = tokenizer,
        batch_size = TRAIN_SETTINGS.batch_size,
        window_size = GPT2_CONFIG_124M.context_length,
        shuffle = False,
        drop_last = False,
        num_workers = 0
    )

    # Train model
    train_losses, val_losses, num_tokens_seen = training_loop(
        model = model,
        device = device,
        tokenizer = tokenizer,
        optimizer = optimizer,
        num_epochs=TRAIN_SETTINGS.num_epochs,
        start_context="Hello World!",
        train_dataloader = train_dataloader,
        val_dataloader = val_dataloader,
        eval_freq=5,
        eval_iter=1
    )

    # Put model back to eval
    model.eval()

    return train_losses, val_losses, num_tokens_seen, model

def plot_losses(train_losses, val_losses, num_epochs, num_tokens_seen):
    sns.set_theme()
    sns.set_style("white")

    fig, ax1 = plt.subplots()
    ax2 = ax1.twiny()
    
    # Plot losses
    sns.lineplot(x=num_epochs, y=train_losses, ax=ax1, color="teal", label="Training Loss")
    sns.lineplot(x=num_epochs, y=val_losses, ax=ax1, linestyle="--", color="mediumvioletred", label="Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    # Add token count to x axis
    sns.lineplot(x=num_tokens_seen, y=train_losses, ax=ax2)
    ax2.set_xlabel("Num of Tokens Seen")

    ax1.legend(loc="upper right")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Train
    train_losses, val_losses, num_tokens_seen, model = train()
    # Plot
    epochs_tensor = torch.linspace(0, TRAIN_SETTINGS.num_epochs, len(train_losses))
    plot_losses(
        train_losses = train_losses,
        val_losses = val_losses,
        num_epochs = epochs_tensor,
        num_tokens_seen = num_tokens_seen
    )
    plt.savefig("losses.pdf")
    # Save Model
    torch.save(model.state_dict(), "model.pth")