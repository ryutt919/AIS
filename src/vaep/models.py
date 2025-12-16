import torch
import torch.nn as nn

# 버전 명시: v1.0
__version__ = "v1.0"

class VaepMLP(nn.Module):
    """
    Atomic SPADL 기반 액션 가치(Scores/Concedes 확률) 추정 MLP 모델
    """
    def __init__(self, input_dim: int = 8, hidden_dims: list = [64, 32], dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)
