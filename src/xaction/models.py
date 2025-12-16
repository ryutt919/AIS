import torch
import torch.nn as nn

# 버전 명시: v1.0
__version__ = "v1.0"

class MultiTaskActionPredictor(nn.Module):
    """
    Next Action 예측을 위한 다중 과제 신경망 (GRU, LSTM, Transformer 지원)
    출력: Next Action Type (35개 클래스), Next Action Category (8개 클래스)
    """
    def __init__(self, model_type: str = "transformer", input_dim: int = 6, 
                 hidden_dim: int = 64, num_layers: int = 2, 
                 num_types: int = 35, num_cats: int = 8):
        super().__init__()
        self.model_type = model_type.lower()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        if self.model_type == "gru":
            self.backbone = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        elif self.model_type == "lstm":
            self.backbone = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        else: # transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*2, 
                dropout=0.1, batch_first=True
            )
            self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
        # Multi-task heads
        self.head_type = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_types)
        )
        self.head_cat = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_cats)
        )

    def forward(self, x):
        h = self.input_proj(x)
        if self.model_type in ["gru", "lstm"]:
            out, _ = self.backbone(h)
            last_feat = out[:, -1, :]
        else:
            out = self.backbone(h)
            last_feat = out[:, -1, :]
            
        logits_type = self.head_type(last_feat)
        logits_cat = self.head_cat(last_feat)
        return logits_type, logits_cat
