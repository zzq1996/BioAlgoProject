from torch.utils.data import Dataset
import torch


class MyDataset(Dataset):
    def __init__(self, split='train'):
        self.data = torch.randn(100, 10)
        self.labels = torch.randn(100, 1)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
