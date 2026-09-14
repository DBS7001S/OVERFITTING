import torch
import torch.nn as nn

class OverfittedNet(nn.Module):
    """Mô hình quá rộng và sâu so với 18 mẫu dữ liệu train -> Rất dễ Overfit"""
    def __init__(self):
        super(OverfittedNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.net(x)
