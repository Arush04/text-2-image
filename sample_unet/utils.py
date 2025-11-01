import os
import torch
import torch.nn as nn
import pickle
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

class DDPM_Scheduler(nn.Module):
    def __init__(self, num_time_steps: int=1000, device: torch.device = None):
        super().__init__()
        beta = torch.linspace(1e-4, 0.02, num_time_steps, device=device)
        alpha = 1 - beta
        alpha_cumprod = torch.cumprod(alpha, dim=0)
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha_cumprod)

    def forward(self, t):
        return self.beta[t], self.alpha[t]

    def to(self, device):
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        return self

class CIFAR10Dataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform

        if train:
            batch_files = [f'data_batch_{i}' for i in range(1, 6)]
        else:
            batch_files = ['test_batch']

        for batch_file in batch_files:
            with open(os.path.join(data_dir, batch_file), 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.labels += entry['labels']

        self.data = np.concatenate(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose(0, 2, 3, 1)  # Convert to HWC (32,32,3)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.fromarray(self.data[idx])
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

def main():
    scheduler = DDPM_Scheduler(num_time_steps=1000)
    print(scheduler(999))

if __name__ == '__main__':
    main()
