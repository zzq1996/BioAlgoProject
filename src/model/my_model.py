import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        return self.net(x)
