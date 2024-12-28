import torch
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, window_size):
        self.x, self.y = [], [] # Initialize input and target lists (holds token ids)
        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - window_size, window_size):
            x_window = token_ids[i : i + window_size] # get current window of inputs
            y_window = token_ids[i + 1 : i + window_size + 1] # get current window of targets (inputs shifted by 1)
            self.x.append(torch.tensor(x_window))
            self.y.append(torch.tensor(y_window))
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def create_gpt_dataloader(text, tokenizer, batch_size = 4, window_size = 256, shuffle = True, drop_last = True, num_workers = 0):
    dataset = GPTDataset(text, tokenizer, window_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)