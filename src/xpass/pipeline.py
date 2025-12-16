import pandas as pd
import numpy as np
import json
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from .features import extract_xpass_features

# 버전 명시: v1.0
__version__ = "v1.0"

def train_and_evaluate_xpass(cfg: dict, df: pd.DataFrame) -> dict:
    """
    xPass 모델 학습 및 평가, 예측 확률 및 요약 생성
    """
    print("\n=== [Track 2] Expected Pass (xPass) Pipeline ===")
    pass_df = extract_xpass_features(df)
    print(f"Extracted {len(pass_df)} pass events. Positive rate: {pass_df['success'].mean():.4f}")
    
    if len(pass_df) < 20:
        raise ValueError("Not enough pass events to train xPass.")
        
    feature_cols = ['start_x', 'start_y', 'end_x', 'end_y', 'dx', 'dy', 'dist', 'angle', 'dist_to_goal', 'bodypart_id']
    X = pass_df[feature_cols]
    y = pass_df['success']
    
    # Train / Val Split (경기 단위)
    unique_games = pass_df['game_id'].unique()
    train_games = unique_games[:max(1, int(len(unique_games)*0.7))]
    val_games = unique_games[max(1, int(len(unique_games)*0.7)):]
    if len(val_games) == 0:
        val_games = train_games
        
    train_mask = pass_df['game_id'].isin(train_games)
    val_mask = pass_df['game_id'].isin(val_games)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    
    xp_cfg = cfg.get("xpass", {})
    clf = xgb.XGBClassifier(
        n_estimators=xp_cfg.get("n_estimators", 50),
        max_depth=xp_cfg.get("max_depth", 4),
        learning_rate=xp_cfg.get("learning_rate", 0.1),
        subsample=xp_cfg.get("subsample", 0.8),
        eval_metric="logloss",
        random_state=42
    )
    
    clf.fit(X_train, y_train)
    
    preds_val_prob = clf.predict_proba(X_val)[:, 1]
    
    # 평가 지표
    try:
        auc = float(roc_auc_score(y_val, preds_val_prob))
    except Exception:
        auc = 0.5
    loss = float(log_loss(y_val, preds_val_prob))
    brier = float(brier_score_loss(y_val, preds_val_prob))
    
    print(f"Validation Metrics -> AUC-ROC: {auc:.4f}, LogLoss: {loss:.4f}, Brier: {brier:.4f}")
    
    metrics = {
        "auc_roc": auc,
        "log_loss": loss,
        "brier_score": brier,
        "total_pass_samples": len(pass_df),
        "pos_rate": float(pass_df['success'].mean())
    }
    
    metrics_path = cfg["paths"]["metrics_dir"] / "xpass_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    # 전체 패스에 대한 xPass 추론값 할당 및 선수/시즌 집계
    pass_df['xpass'] = clf.predict_proba(X)[:, 1]
    pass_df['xpass_residual'] = pass_df['success'] - pass_df['xpass']
    
    player_xpass = pass_df.groupby("player_id").agg(
        total_passes=("success", "count"),
        completed_passes=("success", "sum"),
        expected_passes=("xpass", "sum"),
        avg_xpass=("xpass", "mean"),
        xpass_added=("xpass_residual", "sum")
    ).reset_index()
    
    out_csv = cfg["paths"]["refined_dir"] / "player_season_xpass.csv"
    player_xpass.to_csv(out_csv, index=False)
    print(f"Saved player xPass summary to {out_csv} ({len(player_xpass)} players)")
    
    return metrics
