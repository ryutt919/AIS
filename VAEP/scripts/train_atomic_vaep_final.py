"""
Atomic SPADL 기반 VAEP 모델 학습 스크립트 (최종 버전)

이 스크립트는:
1. Atomic SPADL 데이터 로드
2. Feature 생성
3. Label 생성 (scores/concedes goals)
4. PyTorch 모델 학습 (K-Fold Cross Validation)
5. 모델 저장 및 평가
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from tqdm import tqdm

# socceraction imports
import socceraction.atomic.spadl as atomicspadl
import socceraction.atomic.vaep.features as atomic_features
import socceraction.atomic.vaep.labels as atomic_labels

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("atomic_vaep_training")


# ============================================================================
# 1. Feature 및 Label 생성
# ============================================================================

def create_features_and_labels(
    atomic_actions_df: pd.DataFrame,
    nb_prev_actions: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Atomic SPADL 액션으로부터 feature와 label 생성
    
    Returns:
        features_df: 생성된 feature DataFrame
        labels_scores: 득점 확률 라벨
        labels_concedes: 실점 확률 라벨
    """
    logger.info("=== Feature 및 Label 생성 시작 ===")
    
    # Game states 생성
    logger.info("Game states 생성 중...")
    gamestates = atomic_features.gamestates(
        atomic_actions_df, 
        nb_prev_actions=nb_prev_actions
    )
    
    # gamestates가 list인 경우 처리
    if isinstance(gamestates, list):
        logger.info(f"Game states 개수: {len(gamestates)}")
        logger.info(f"첫 번째 gamestate shape: {gamestates[0].shape if len(gamestates) > 0 else 'N/A'}")
    else:
        logger.info(f"Game states shape: {gamestates.shape}")
    
    # Features 생성 (문자열 feature 제외)
    logger.info("Features 생성 중...")
    xfns = [
        # atomic_features.actiontype,  # 문자열이므로 제외
        atomic_features.actiontype_onehot,
        atomic_features.bodypart_onehot,
        atomic_features.location,
        atomic_features.direction,
        atomic_features.team,
        atomic_features.time,
        atomic_features.time_delta,
    ]
    
    features_list = []
    for fn in tqdm(xfns, desc="Feature 함수 적용"):
        result = fn(gamestates)
        if isinstance(result, list):
            # list인 경우 concat
            result = pd.concat(result, ignore_index=True)
        features_list.append(result)
    
    features_df = pd.concat(features_list, axis=1)
    logger.info(f"Features shape: {features_df.shape}")
    logger.info(f"Feature columns: {features_df.columns.tolist()[:10]}... (총 {len(features_df.columns)}개)")
    
    # Labels 생성
    logger.info("Labels 생성 중...")
    yfns = [atomic_labels.scores, atomic_labels.concedes]
    
    labels_scores = yfns[0](atomic_actions_df)
    labels_concedes = yfns[1](atomic_actions_df)
    
    logger.info(f"Labels scores shape: {labels_scores.shape}")
    logger.info(f"Labels concedes shape: {labels_concedes.shape}")
    logger.info(f"Positive scores: {labels_scores.sum()}, Positive concedes: {labels_concedes.sum()}")
    
    return features_df, labels_scores, labels_concedes


# ============================================================================
# 2. PyTorch Dataset 및 Model
# ============================================================================

class VAEPDataset(Dataset):
    """VAEP 학습을 위한 PyTorch Dataset"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class VAEPMLP(nn.Module):
    """VAEP을 위한 Multi-Layer Perceptron"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64], dropout: float = 0.3):
        super(VAEPMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


# ============================================================================
# 3. 학습 함수
# ============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """한 에포크 학습"""
    model.train()
    total_loss = 0.0
    
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(features)
    
    return total_loss / len(train_loader.dataset)


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """모델 평가"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * len(features)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader.dataset)
    
    # 추가 메트릭 계산
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = {
        'loss': avg_loss,
        'brier_score': brier_score_loss(all_labels, all_preds),
        'log_loss': log_loss(all_labels, all_preds),
    }
    
    # ROC AUC (라벨이 모두 같은 경우 계산 불가)
    if len(np.unique(all_labels)) > 1:
        metrics['roc_auc'] = roc_auc_score(all_labels, all_preds)
    else:
        metrics['roc_auc'] = 0.0
    
    return avg_loss, metrics


def train_kfold(
    features: np.ndarray,
    labels_scores: np.ndarray,
    labels_concedes: np.ndarray,
    args: argparse.Namespace,
    device: torch.device
) -> Dict:
    """K-Fold Cross Validation으로 학습"""
    
    logger.info(f"\n=== K-Fold Cross Validation 학습 시작 (K={args.k_folds}) ===")
    
    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.random_seed)
    
    results = {
        'scores': {'models': [], 'metrics': []},
        'concedes': {'models': [], 'metrics': []}
    }
    
    # 두 개의 모델 학습 (scores, concedes)
    for target_name, labels in [('scores', labels_scores), ('concedes', labels_concedes)]:
        logger.info(f"\n{'='*60}")
        logger.info(f"학습 대상: {target_name.upper()}")
        logger.info(f"{'='*60}")
        
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(features)):
            logger.info(f"\n--- Fold {fold+1}/{args.k_folds} ---")
            
            # 데이터 분할
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
            logger.info(f"Positive samples - Train: {y_train.sum()}, Val: {y_val.sum()}")
            
            # Dataset & DataLoader
            train_dataset = VAEPDataset(X_train, y_train)
            val_dataset = VAEPDataset(X_val, y_val)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=4
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4
            )
            
            # 모델 초기화
            model = VAEPMLP(
                input_dim=features.shape[1],
                hidden_dims=args.hidden_dims,
                dropout=args.dropout
            ).to(device)
            
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            
            # 학습
            best_val_loss = float('inf')
            patience_counter = 0
            
            pbar = tqdm(range(args.epochs), desc=f"Fold {fold+1} Training")
            for epoch in pbar:
                train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
                
                pbar.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'val_auc': f'{val_metrics["roc_auc"]:.4f}'
                })
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 최고 모델 저장
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= args.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # 최고 모델 로드
            model.load_state_dict(best_model_state)
            
            # 최종 평가
            _, final_metrics = evaluate(model, val_loader, criterion, device)
            fold_metrics.append(final_metrics)
            
            logger.info(f"Fold {fold+1} Final Metrics:")
            for metric_name, metric_value in final_metrics.items():
                logger.info(f"  {metric_name}: {metric_value:.4f}")
            
            # 모델 저장
            results[target_name]['models'].append(model.state_dict())
        
        # Fold 평균 메트릭
        results[target_name]['metrics'] = fold_metrics
        avg_metrics = {
            metric: np.mean([m[metric] for m in fold_metrics])
            for metric in fold_metrics[0].keys()
        }
        
        logger.info(f"\n{target_name.upper()} - Average Metrics across {args.k_folds} folds:")
        for metric_name, metric_value in avg_metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    return results


# ============================================================================
# 4. Main 함수
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Atomic SPADL VAEP 모델 학습")
    
    # 데이터 관련
    parser.add_argument(
        "--input",
        type=str,
        default="../data/processed/atomic_spadl_all_games.csv",
        help="Atomic SPADL 데이터 경로"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../models/atomic_vaep",
        help="모델 저장 디렉토리"
    )
    
    # Feature 관련
    parser.add_argument(
        "--nb_prev_actions",
        type=int,
        default=3,
        help="이전 액션 수 (feature 생성용)"
    )
    
    # 학습 관련
    parser.add_argument("--k_folds", type=int, default=5, help="K-Fold CV의 K값")
    parser.add_argument("--batch_size", type=int, default=1024, help="배치 크기")
    parser.add_argument("--epochs", type=int, default=50, help="최대 에포크 수")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=0.001, help="학습률")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout 비율")
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[128, 64],
        help="히든 레이어 차원"
    )
    parser.add_argument("--random_seed", type=int, default=42, help="랜덤 시드")
    
    # 기타
    parser.add_argument("--sample_size", type=int, default=None, help="샘플 데이터 크기 (디버깅용)")
    parser.add_argument("--debug", action="store_true", help="디버그 모드")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.debug:
        args.epochs = 3
        args.k_folds = 2
        args.sample_size = 10000
    
    # 랜덤 시드 설정
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # Step 1: 데이터 로드
    # ========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("Step 1: Atomic SPADL 데이터 로드")
    logger.info(f"{'='*60}")
    logger.info(f"데이터 경로: {args.input}")
    
    atomic_actions_df = pd.read_csv(args.input)
    logger.info(f"전체 데이터 shape: {atomic_actions_df.shape}")
    logger.info(f"Columns: {atomic_actions_df.columns.tolist()}")
    
    # 샘플링 (디버깅용)
    if args.sample_size:
        logger.info(f"샘플링: {args.sample_size} 액션")
        atomic_actions_df = atomic_actions_df.head(args.sample_size)
    
    # ========================================================================
    # Step 2: Feature 및 Label 생성
    # ========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("Step 2: Feature 및 Label 생성")
    logger.info(f"{'='*60}")
    
    features_df, labels_scores, labels_concedes = create_features_and_labels(
        atomic_actions_df,
        nb_prev_actions=args.nb_prev_actions
    )
    
    # NumPy 배열로 변환
    features = features_df.values.astype(np.float32)
    labels_scores_arr = labels_scores.values.flatten().astype(np.float32)  # 1차원으로 변환
    labels_concedes_arr = labels_concedes.values.flatten().astype(np.float32)  # 1차원으로 변환
    
    logger.info(f"\n최종 데이터 shape:")
    logger.info(f"  Features: {features.shape}")
    logger.info(f"  Labels (scores): {labels_scores_arr.shape}")
    logger.info(f"  Labels (concedes): {labels_concedes_arr.shape}")
    
    # ========================================================================
    # Step 3: 모델 학습 (K-Fold CV)
    # ========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("Step 3: K-Fold Cross Validation 학습")
    logger.info(f"{'='*60}")
    
    results = train_kfold(
        features,
        labels_scores_arr,
        labels_concedes_arr,
        args,
        device
    )
    
    # ========================================================================
    # Step 4: 결과 저장
    # ========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("Step 4: 모델 및 결과 저장")
    logger.info(f"{'='*60}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 각 fold의 모델 저장
    for target_name in ['scores', 'concedes']:
        for fold, model_state in enumerate(results[target_name]['models']):
            model_path = output_dir / f"vaep_{target_name}_fold{fold}_{timestamp}.pth"
            torch.save(model_state, model_path)
            logger.info(f"모델 저장: {model_path}")
    
    # 메트릭 저장
    metrics_summary = {}
    for target_name in ['scores', 'concedes']:
        fold_metrics = results[target_name]['metrics']
        avg_metrics = {
            metric: np.mean([m[metric] for m in fold_metrics])
            for metric in fold_metrics[0].keys()
        }
        metrics_summary[target_name] = {
            'avg_metrics': avg_metrics,
            'fold_metrics': fold_metrics
        }
    
    metrics_path = output_dir / f"metrics_{timestamp}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2, default=str)
    logger.info(f"메트릭 저장: {metrics_path}")
    
    # 설정 저장
    config = vars(args)
    config_path = output_dir / f"config_{timestamp}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"설정 저장: {config_path}")
    
    logger.info(f"\n{'='*60}")
    logger.info("학습 완료!")
    logger.info(f"{'='*60}")
    
    # 최종 요약
    logger.info("\n=== 최종 요약 ===")
    for target_name, summary in metrics_summary.items():
        logger.info(f"\n{target_name.upper()}:")
        for metric, value in summary['avg_metrics'].items():
            logger.info(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
