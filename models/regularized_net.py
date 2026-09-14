import torch
import torch.nn as nn

class RegularizedNet(nn.Module):
    """Mô hình đã thêm Dropout để chống Overfit"""
    def __init__(self, dropout_rate=0.2):
        super(RegularizedNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.net(x)
