import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

# 버전 명시: v1.0
__version__ = "v1.0"

class NextActionSequenceDataset(Dataset):
    """
    시계열 액션 로그에서 seq_len 만큼의 슬라이딩 윈도우를 구성하고
    다음 액션의 type_id 및 category_id를 타깃으로 반환하는 PyTorch Dataset.
    """
    def __init__(self, df: pd.DataFrame, seq_len: int = 10, type_map: dict = None):
        self.seq_len = seq_len
        self.samples = []
        
        # 타입 및 범주 매핑
        df = df.sort_values(by=['game_id', 'period_id', 'time_seconds', 'action_id']).reset_index(drop=True)
        
        # 연속형 피처: x, y, dx, dy (정규화: 105m, 68m 기준)
        norm_x = (df['x'].values / 105.0).astype(np.float32)
        norm_y = (df['y'].values / 68.0).astype(np.float32)
        norm_dx = (df['dx'].values / 105.0).astype(np.float32)
        norm_dy = (df['dy'].values / 68.0).astype(np.float32)
        types = df['type_id'].values.astype(np.int64)
        bodyparts = df['bodypart_id'].values.astype(np.int64)
        
        # 임의의 카테고리 매핑 (type_id // 5)
        categories = (types // 5).astype(np.int64)
        
        game_ids = df['game_id'].values
        player_ids = df['player_id'].values
        
        # 경기 단위 슬라이딩 윈도우 생성
        for g in np.unique(game_ids):
            idx = np.where(game_ids == g)[0]
            if len(idx) <= seq_len:
                continue
            for i in range(len(idx) - seq_len):
                seq_indices = idx[i : i + seq_len]
                target_idx = idx[i + seq_len]
                
                feat = np.stack([
                    norm_x[seq_indices],
                    norm_y[seq_indices],
                    norm_dx[seq_indices],
                    norm_dy[seq_indices],
                    types[seq_indices] / 35.0,
                    bodyparts[seq_indices] / 5.0
                ], axis=1) # (seq_len, 6)
                
                self.samples.append({
                    'features': torch.tensor(feat, dtype=torch.float32),
                    'target_type': torch.tensor(types[target_idx], dtype=torch.long),
                    'target_cat': torch.tensor(categories[target_idx], dtype=torch.long),
                    'game_id': game_ids[target_idx],
                    'player_id': player_ids[target_idx]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
