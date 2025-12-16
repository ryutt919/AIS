import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import json
from sklearn.metrics import f1_score, accuracy_score
from .dataset import NextActionSequenceDataset
from .models import MultiTaskActionPredictor

# 버전 명시: v1.0
__version__ = "v1.0"

def train_and_evaluate_xaction(cfg: dict, df: pd.DataFrame) -> dict:
    """
    xAction 모델 학습 및 평가, 잔차 분석 요약 파일 생성
    """
    print("\n=== [Track 1] Next Action Prediction (xAction) Pipeline ===")
    x_cfg = cfg["xaction"]
    device = torch.device("cuda" if torch.cuda.is_available() and x_cfg.get("device") != "cpu" else "cpu")
    print(f"Using device: {device}")
    
    # 데이터셋 구성
    dataset = NextActionSequenceDataset(df, seq_len=x_cfg.get("seq_len", 10))
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Ensure enough sequential actions are present.")
        
    train_size = int(len(dataset) * 0.7)
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=x_cfg.get("batch_size", 64), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=x_cfg.get("batch_size", 64), shuffle=False)
    
    model = MultiTaskActionPredictor(
        model_type=x_cfg.get("model_type", "transformer"),
        hidden_dim=x_cfg.get("hidden_dim", 64),
        num_layers=x_cfg.get("num_layers", 2)
    ).to(device)
    
    criterion_type = nn.CrossEntropyLoss()
    criterion_cat = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=x_cfg.get("learning_rate", 0.001))
    
    loss_t_w = x_cfg.get("loss_type_weight", 1.0)
    loss_c_w = x_cfg.get("loss_cat_weight", 0.5)
    
    epochs = x_cfg.get("epochs", 3)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            feats = batch["features"].to(device)
            target_t = batch["target_type"].to(device)
            target_c = batch["target_cat"].to(device)
            
            optimizer.zero_grad()
            pred_t, pred_c = model(feats)
            
            loss = loss_t_w * criterion_type(pred_t, target_t) + loss_c_w * criterion_cat(pred_c, target_c)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {total_loss / len(train_loader):.4f}")
        
    # Validation & Residual Collection
    model.eval()
    all_true_t, all_pred_t = [], []
    records = []
    
    with torch.no_grad():
        for batch in val_loader:
            feats = batch["features"].to(device)
            target_t = batch["target_type"].to(device)
            pred_t, _ = model(feats)
            
            preds = torch.argmax(pred_t, dim=1).cpu().numpy()
            trues = target_t.cpu().numpy()
            
            all_true_t.extend(trues)
            all_pred_t.extend(preds)
            
            g_ids = batch["game_id"].numpy()
            p_ids = batch["player_id"].numpy()
            
            for g, p, t, pr in zip(g_ids, p_ids, trues, preds):
                records.append({
                    "game_id": int(g),
                    "player_id": int(p),
                    "true_type": int(t),
                    "pred_type": int(pr),
                    "matched": int(t == pr)
                })
                
    macro_f1 = f1_score(all_true_t, all_pred_t, average="macro", zero_division=0)
    acc = accuracy_score(all_true_t, all_pred_t)
    print(f"Validation Macro F1: {macro_f1:.4f}, Accuracy: {acc:.4f}")
    
    # Save Metrics
    metrics = {
        "model_type": x_cfg.get("model_type", "transformer"),
        "val_type_macro_f1": float(macro_f1),
        "val_type_accuracy": float(acc),
        "total_eval_samples": len(all_true_t)
    }
    metrics_path = cfg["paths"]["metrics_dir"] / "xaction_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    # 선수 및 경기 단위 Residual (match_rate) 요약 생성
    res_df = pd.DataFrame(records)
    player_summary = res_df.groupby(["player_id", "game_id"]).agg(
        num_actions=("matched", "count"),
        match_count=("matched", "sum")
    ).reset_index()
    player_summary["match_rate"] = player_summary["match_count"] / player_summary["num_actions"]
    
    out_csv = cfg["paths"]["refined_dir"] / "per_match_player_summary.csv"
    player_summary.to_csv(out_csv, index=False)
    print(f"Saved match-level player mismatch summary to {out_csv} ({len(player_summary)} rows)")
    
    return metrics
