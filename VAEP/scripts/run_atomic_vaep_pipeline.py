"""
Atomic SPADL 전처리 및 vAEP K-Fold CV 학습 파이프라인

이 스크립트는:
1. 전체 경기 데이터를 Atomic SPADL 형식으로 전처리
2. K-Fold Cross Validation을 사용하여 vAEP 모델 학습
3. 모든 진행사항을 tqdm으로 시각화
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm

# socceraction imports
import socceraction.atomic.spadl as atomicspadl
from socceraction.atomic.spadl import convert_to_atomic
import socceraction.atomic.vaep.features as atomic_features
from socceraction.data.wyscout import PublicWyscoutLoader
from socceraction.spadl.wyscout import convert_to_actions as wyscout_to_spadl

# 현재 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import setup_logger, ensure_dir


# ============================================================================
# 1. 데이터 전처리 (Atomic SPADL 변환)
# ============================================================================

def process_game_to_atomic_spadl(
    loader: PublicWyscoutLoader,
    game_id: int,
    league: str,
    logger: logging.Logger
) -> Optional[pd.DataFrame]:
    """단일 경기를 Atomic SPADL 형식으로 변환"""
    try:
        # 이벤트 로드
        events = loader.events(game_id)
        if events.empty:
            return None

        # 팀 정보 가져오기
        teams = loader.teams(game_id)
        if teams.empty:
            return None
        
        # 홈팀 ID 찾기
        if "side" in teams.columns:
            home_team_id = teams[teams.side == "home"]["team_id"].iloc[0]
        else:
            home_team_id = teams.iloc[0]["team_id"]

        # SPADL로 변환
        spadl_actions = wyscout_to_spadl(events, home_team_id)
        if spadl_actions.empty:
            return None

        # Atomic SPADL로 변환
        atomic_actions = convert_to_atomic(spadl_actions)
        if atomic_actions.empty:
            return None

        atomic_actions["league"] = league
        return atomic_actions

    except Exception as e:
        logger.warning(f"Error processing game {game_id}: {str(e)}")
        return None


def preprocess_all_games(data_dir: str, output_dir: str, logger: logging.Logger) -> str:
    """모든 경기를 Atomic SPADL로 전처리하고 저장"""
    logger.info("=" * 80)
    logger.info("STEP 1: Atomic SPADL 전처리 시작")
    logger.info("=" * 80)
    
    # Wyscout 로더 초기화
    loader = PublicWyscoutLoader(root=data_dir)
    
    # 경기 목록 로드
    logger.info("경기 목록 로딩 중...")
    competitions = loader.competitions()
    
    all_games = []
    for _, comp in tqdm(competitions.iterrows(), total=len(competitions), desc="대회 로딩"):
        try:
            games = loader.games(comp.competition_id, comp.season_id)
            for _, game in games.iterrows():
                all_games.append({
                    "game_id": game.game_id,
                    "competition": comp.competition_name
                })
        except Exception as e:
            logger.warning(f"대회 {comp.competition_name} 로딩 실패: {e}")
            continue
    
    logger.info(f"총 {len(all_games)} 경기 발견")
    
    # 리그 이름 매핑
    league_mapping = {
        "English first division": "England",
        "Spanish first division": "Spain",
        "French first division": "France",
        "German first division": "Germany",
        "Italian first division": "Italy",
        "European Championship": "European_Championship",
        "World Cup": "World_Cup",
    }
    
    # 전체 경기 처리
    all_atomic_actions = []
    success_count = 0
    
    for game_info in tqdm(all_games, desc="경기 전처리 (Atomic SPADL 변환)"):
        game_id = game_info["game_id"]
        competition = game_info["competition"]
        league = league_mapping.get(competition, competition)
        
        atomic_actions = process_game_to_atomic_spadl(loader, game_id, league, logger)
        if atomic_actions is not None:
            all_atomic_actions.append(atomic_actions)
            success_count += 1
    
    logger.info(f"성공적으로 처리된 경기: {success_count}/{len(all_games)}")
    
    # 전체 데이터 결합
    logger.info("데이터 결합 중...")
    df_all = pd.concat(all_atomic_actions, ignore_index=True)
    
    # 저장
    output_file = os.path.join(output_dir, "atomic_spadl_all_games.csv")
    logger.info(f"저장 중: {output_file}")
    df_all.to_csv(output_file, index=False)
    
    logger.info(f"전처리 완료! 총 액션 수: {len(df_all)}")
    logger.info(f"리그별 액션 수:\n{df_all['league'].value_counts()}")
    
    return output_file


# ============================================================================
# 2. Feature 생성
# ============================================================================

def create_features_and_labels(
    df: pd.DataFrame,
    horizon: int,
    logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Atomic SPADL 데이터로부터 feature와 label 생성"""
    logger.info("=" * 80)
    logger.info("Feature 및 Label 생성 중...")
    logger.info("=" * 80)
    
    # game_id별로 그룹화
    games = df.groupby("game_id")
    
    all_features = []
    all_labels_scores = []
    all_labels_concedes = []
    
    # ActionType 가져오기
    actiontypes = atomicspadl.actiontypes_df()
    
    for game_id, game_actions in tqdm(games, desc="Game별 Feature 생성"):
        try:
            # Atomic VAEP features 생성
            gamestates = atomic_features.gamestates(
                game_actions, 
                nb_prev_actions=3  # 이전 3개 액션 포함
            )
            features = atomic_features.features(gamestates, actiontypes)
            
            # Label 생성
            labels_scores = atomic_features.label_scores(
                game_actions, 
                actiontypes,
                nr=horizon
            )
            labels_concedes = atomic_features.label_concedes(
                game_actions,
                actiontypes,
                nr=horizon
            )
            
            all_features.append(features)
            all_labels_scores.append(labels_scores)
            all_labels_concedes.append(labels_concedes)
            
        except Exception as e:
            logger.warning(f"Game {game_id} feature 생성 실패: {e}")
            continue
    
    # 전체 데이터 결합
    X = pd.concat(all_features, ignore_index=True).values
    y_scores = pd.concat(all_labels_scores, ignore_index=True).values
    y_concedes = pd.concat(all_labels_concedes, ignore_index=True).values
    
    logger.info(f"Feature shape: {X.shape}")
    logger.info(f"Label scores shape: {y_scores.shape}")
    logger.info(f"Label concedes shape: {y_concedes.shape}")
    
    return X, y_scores, y_concedes


# ============================================================================
# 3. PyTorch 모델 정의
# ============================================================================

class VAEPDataset(Dataset):
    """VAEP 학습용 PyTorch Dataset"""
    def __init__(self, X, y_scores, y_concedes):
        self.X = torch.FloatTensor(X)
        self.y_scores = torch.FloatTensor(y_scores)
        self.y_concedes = torch.FloatTensor(y_concedes)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_scores[idx], self.y_concedes[idx]


class VAEPModel(nn.Module):
    """VAEP 예측 모델 (Scoring + Conceding)"""
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64]):
        super(VAEPModel, self).__init__()
        
        # Shared layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        # Scoring head
        self.scoring_head = nn.Linear(prev_dim, 1)
        
        # Conceding head
        self.conceding_head = nn.Linear(prev_dim, 1)
    
    def forward(self, x):
        shared_out = self.shared(x)
        scores = torch.sigmoid(self.scoring_head(shared_out))
        concedes = torch.sigmoid(self.conceding_head(shared_out))
        return scores, concedes


# ============================================================================
# 4. K-Fold Cross Validation 학습
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """단일 에포크 학습"""
    model.train()
    total_loss = 0
    
    for X_batch, y_scores_batch, y_concedes_batch in dataloader:
        X_batch = X_batch.to(device)
        y_scores_batch = y_scores_batch.to(device).unsqueeze(1)
        y_concedes_batch = y_concedes_batch.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        
        pred_scores, pred_concedes = model(X_batch)
        loss_scores = criterion(pred_scores, y_scores_batch)
        loss_concedes = criterion(pred_concedes, y_concedes_batch)
        loss = loss_scores + loss_concedes
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """모델 평가"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, y_scores_batch, y_concedes_batch in dataloader:
            X_batch = X_batch.to(device)
            y_scores_batch = y_scores_batch.to(device).unsqueeze(1)
            y_concedes_batch = y_concedes_batch.to(device).unsqueeze(1)
            
            pred_scores, pred_concedes = model(X_batch)
            loss_scores = criterion(pred_scores, y_scores_batch)
            loss_concedes = criterion(pred_concedes, y_concedes_batch)
            loss = loss_scores + loss_concedes
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train_kfold_cv(
    X: np.ndarray,
    y_scores: np.ndarray,
    y_concedes: np.ndarray,
    k: int = 5,
    epochs: int = 50,
    batch_size: int = 512,
    hidden_dims: List[int] = [128, 64],
    lr: float = 0.001,
    output_dir: str = None,
    logger: logging.Logger = None
) -> Dict[str, List[float]]:
    """K-Fold Cross Validation으로 모델 학습"""
    logger.info("=" * 80)
    logger.info(f"STEP 2: K-Fold CV (k={k}) 학습 시작")
    logger.info("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"사용 디바이스: {device}")
    
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    
    fold_results = {
        "train_losses": [],
        "val_losses": []
    }
    
    # K-Fold 순회
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Fold {fold}/{k} 시작")
        logger.info(f"{'='*80}")
        
        # 데이터 분할
        X_train, X_val = X[train_idx], X[val_idx]
        y_scores_train, y_scores_val = y_scores[train_idx], y_scores[val_idx]
        y_concedes_train, y_concedes_val = y_concedes[train_idx], y_concedes[val_idx]
        
        # Dataset 및 DataLoader 생성
        train_dataset = VAEPDataset(X_train, y_scores_train, y_concedes_train)
        val_dataset = VAEPDataset(X_val, y_scores_val, y_concedes_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 모델 초기화
        input_dim = X.shape[1]
        model = VAEPModel(input_dim, hidden_dims).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        # 학습
        best_val_loss = float('inf')
        fold_train_losses = []
        fold_val_losses = []
        
        pbar = tqdm(range(epochs), desc=f"Fold {fold} 학습")
        for epoch in pbar:
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = evaluate(model, val_loader, criterion, device)
            
            fold_train_losses.append(train_loss)
            fold_val_losses.append(val_loss)
            
            # 진행상황 업데이트
            pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}'
            })
            
            # Best 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if output_dir:
                    model_path = os.path.join(output_dir, f'vaep_model_fold{fold}.pt')
                    torch.save({
                        'fold': fold,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'input_dim': input_dim,
                        'hidden_dims': hidden_dims,
                    }, model_path)
        
        fold_results["train_losses"].append(fold_train_losses)
        fold_results["val_losses"].append(fold_val_losses)
        
        logger.info(f"Fold {fold} 완료 - Best Val Loss: {best_val_loss:.4f}")
    
    # 전체 결과 요약
    logger.info("\n" + "=" * 80)
    logger.info("K-Fold CV 결과 요약")
    logger.info("=" * 80)
    
    avg_train_losses = np.mean([losses[-1] for losses in fold_results["train_losses"]])
    avg_val_losses = np.mean([losses[-1] for losses in fold_results["val_losses"]])
    
    logger.info(f"평균 Train Loss: {avg_train_losses:.4f}")
    logger.info(f"평균 Val Loss: {avg_val_losses:.4f}")
    
    # 결과 저장
    if output_dir:
        results_file = os.path.join(output_dir, "kfold_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'k': k,
                'epochs': epochs,
                'avg_train_loss': float(avg_train_losses),
                'avg_val_loss': float(avg_val_losses),
                'fold_train_losses': [[float(l) for l in losses] for losses in fold_results["train_losses"]],
                'fold_val_losses': [[float(l) for l in losses] for losses in fold_results["val_losses"]],
            }, f, indent=2)
        logger.info(f"결과 저장: {results_file}")
    
    return fold_results


# ============================================================================
# 5. 메인 파이프라인
# ============================================================================

def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description="Atomic SPADL 전처리 및 vAEP K-Fold CV 학습")
    parser.add_argument("--data_dir", type=str, default="../data/wyscout", help="Wyscout 데이터 디렉토리")
    parser.add_argument("--output_dir", type=str, default="../data/processed", help="출력 디렉토리")
    parser.add_argument("--model_dir", type=str, default="../models", help="모델 저장 디렉토리")
    parser.add_argument("--k", type=int, default=5, help="K-Fold CV의 K 값")
    parser.add_argument("--horizon", type=int, default=10, help="라벨링 horizon")
    parser.add_argument("--epochs", type=int, default=50, help="학습 에포크 수")
    parser.add_argument("--batch_size", type=int, default=512, help="배치 크기")
    parser.add_argument("--lr", type=float, default=0.001, help="학습률")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 64], help="히든 레이어 차원")
    parser.add_argument("--skip_preprocess", action="store_true", help="전처리 스킵 (기존 데이터 사용)")
    return parser.parse_args()


def main():
    """메인 실행 함수"""
    args = parse_args()
    
    # 로거 설정
    logger = setup_logger("atomic_vaep_pipeline", None)
    logger.info("=" * 80)
    logger.info("Atomic SPADL 전처리 및 vAEP K-Fold CV 학습 파이프라인")
    logger.info("=" * 80)
    logger.info(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, args.data_dir))
    output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    model_dir = os.path.abspath(os.path.join(script_dir, args.model_dir))
    
    ensure_dir(output_dir)
    ensure_dir(model_dir)
    
    # 모델 버전 디렉토리
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = os.path.join(model_dir, f"atomic_vaep_{version}")
    ensure_dir(version_dir)
    
    logger.info(f"\n설정:")
    logger.info(f"  - 데이터 디렉토리: {data_dir}")
    logger.info(f"  - 출력 디렉토리: {output_dir}")
    logger.info(f"  - 모델 디렉토리: {version_dir}")
    logger.info(f"  - K-Fold: k={args.k}")
    logger.info(f"  - Epochs: {args.epochs}")
    logger.info(f"  - Batch Size: {args.batch_size}")
    logger.info(f"  - Learning Rate: {args.lr}")
    logger.info(f"  - Hidden Dims: {args.hidden_dims}")
    
    # STEP 1: 전처리
    if not args.skip_preprocess:
        atomic_spadl_file = preprocess_all_games(data_dir, output_dir, logger)
    else:
        atomic_spadl_file = os.path.join(output_dir, "atomic_spadl_all_games.csv")
        logger.info(f"전처리 스킵 - 기존 파일 사용: {atomic_spadl_file}")
    
    # 데이터 로드
    logger.info(f"\n데이터 로딩: {atomic_spadl_file}")
    df = pd.read_csv(atomic_spadl_file)
    logger.info(f"총 액션 수: {len(df)}")
    
    # STEP 2: Feature 및 Label 생성
    X, y_scores, y_concedes = create_features_and_labels(df, args.horizon, logger)
    
    # STEP 3: K-Fold CV 학습
    fold_results = train_kfold_cv(
        X, y_scores, y_concedes,
        k=args.k,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dims=args.hidden_dims,
        lr=args.lr,
        output_dir=version_dir,
        logger=logger
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("파이프라인 완료!")
    logger.info("=" * 80)
    logger.info(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"모델 저장 위치: {version_dir}")


if __name__ == "__main__":
    main()
