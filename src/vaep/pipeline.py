import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import json
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from .labels import create_vaep_labels
from .models import VaepMLP

# 버전 명시: v1.0
__version__ = "v1.0"

def train_and_evaluate_vaep(cfg: dict, df: pd.DataFrame) -> dict:
    """
    VAEP(Scores & Concedes) 모델 학습 및 가치 계산, 선수/경기 단위 집계 생성
    """
    print("\n=== [Track 3] Valuing Actions by Estimating Probabilities (VAEP) Pipeline ===")
    v_cfg = cfg.get("vaep", {})
    horizon = v_cfg.get("horizon", 5)
    
    labeled_df = create_vaep_labels(df, horizon=horizon)
    print(f"Generated VAEP labels (horizon={horizon}). "
          f"Scores rate: {labeled_df['scores'].mean():.4f}, Concedes rate: {labeled_df['concedes'].mean():.4f}")
          
    # 입력 피처 구성 (x, y, dx, dy, dist, angle, type, bodypart)
    dx = labeled_df['dx'].fillna(0.0).values
    dy = labeled_df['dy'].fillna(0.0).values
    dist = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    
    features = np.column_stack([
        labeled_df['x'].values / 105.0,
        labeled_df['y'].values / 68.0,
        dx / 105.0,
        dy / 68.0,
        dist / 50.0,
        angle / np.pi,
        labeled_df['type_id'].values / 35.0,
        labeled_df['bodypart_id'].values / 5.0
    ]).astype(np.float32)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Scores & Concedes 두 개의 모델 학습
    metrics = {}
    prob_dict = {}
    
    for target_name in ['scores', 'concedes']:
        print(f"--- Training VAEP Model for [{target_name}] ---")
        y = labeled_df[target_name].values.astype(np.float32)
        
        n_train = int(len(features) * 0.7)
        X_train, y_train = features[:n_train], y[:n_train]
        X_val, y_val = features[n_train:], y[n_train:]
        
        train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        train_loader = DataLoader(train_ds, batch_size=v_cfg.get("batch_size", 128), shuffle=True)
        
        model = VaepMLP(input_dim=features.shape[1], hidden_dims=v_cfg.get("hidden_dims", [64, 32])).to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=v_cfg.get("learning_rate", 0.001))
        
        epochs = v_cfg.get("epochs", 3)
        for ep in range(epochs):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                
        # 평가
        model.eval()
        with torch.no_grad():
            preds_val = model(torch.tensor(X_val).to(device)).cpu().numpy()
            full_preds = model(torch.tensor(features).to(device)).cpu().numpy()
            
        prob_dict[target_name] = full_preds
        
        try:
            auc = float(roc_auc_score(y_val, preds_val))
        except Exception:
            auc = 0.5
        loss_val = float(log_loss(y_val, preds_val))
        brier = float(brier_score_loss(y_val, preds_val))
        
        metrics[target_name] = {
            "roc_auc": auc,
            "log_loss": loss_val,
            "brier": brier
        }
        print(f"[{target_name}] Val AUC: {auc:.4f}, LogLoss: {loss_val:.4f}, Brier: {brier:.4f}")
        
    metrics_path = cfg["paths"]["metrics_dir"] / "vaep_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    # 가치 계산: vaep = scores_prob - concedes_prob
    labeled_df['scores_prob'] = prob_dict['scores']
    labeled_df['concedes_prob'] = prob_dict['concedes']
    labeled_df['vaep'] = labeled_df['scores_prob'] - labeled_df['concedes_prob']
    
    # 경기 단위 선수 집계 (player_game_vaep_atomic.csv)
    game_vaep = labeled_df.groupby(['player_id', 'game_id']).agg(
        num_actions=('action_id', 'count'),
        total_scores_prob=('scores_prob', 'sum'),
        total_concedes_prob=('concedes_prob', 'sum'),
        vaep_total=('vaep', 'sum'),
        vaep_mean=('vaep', 'mean')
    ).reset_index()
    
    out_game_csv = cfg["paths"]["refined_dir"] / "player_game_vaep_atomic.csv"
    game_vaep.to_csv(out_game_csv, index=False)
    print(f"Saved player-game VAEP summary to {out_game_csv} ({len(game_vaep)} rows)")
    
    # 시즌 단위 선수 집계 (player_season_vaep_atomic.csv)
    season_vaep = labeled_df.groupby('player_id').agg(
        num_games=('game_id', 'nunique'),
        total_actions=('action_id', 'count'),
        season_vaep_total=('vaep', 'sum'),
        avg_vaep_per_game=('vaep', lambda x: x.sum() / max(1, labeled_df.loc[x.index, 'game_id'].nunique()))
    ).reset_index()
    season_vaep['vaep_per90'] = season_vaep['avg_vaep_per_game']
    
    out_season_csv = cfg["paths"]["refined_dir"] / "player_season_vaep_atomic.csv"
    season_vaep.to_csv(out_season_csv, index=False)
    print(f"Saved player-season VAEP summary to {out_season_csv} ({len(season_vaep)} rows)")
    
    return metrics
