"""
Atomic SPADL 기반 선수별 VAEP 계산 스크립트

학습된 모델을 사용하여 모든 선수의 경기별, 시즌별 VAEP를 계산합니다.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# socceraction imports
import socceraction.atomic.vaep.features as atomic_features
from socceraction.data.wyscout import PublicWyscoutLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vaep_computation")


# ============================================================================
# Model Definition (동일한 구조)
# ============================================================================

class VAEPMLP(nn.Module):
    """VAEP을 위한 Multi-Layer Perceptron"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128], dropout: float = 0.3):
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
# Feature 생성
# ============================================================================

def create_features(atomic_actions_df: pd.DataFrame, nb_prev_actions: int = 3) -> pd.DataFrame:
    """Atomic SPADL 액션으로부터 feature 생성"""
    
    logger.info("Game states 생성 중...")
    gamestates = atomic_features.gamestates(
        atomic_actions_df, 
        nb_prev_actions=nb_prev_actions
    )
    
    logger.info("Features 생성 중...")
    xfns = [
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
            result = pd.concat(result, ignore_index=True)
        features_list.append(result)
    
    features_df = pd.concat(features_list, axis=1)
    logger.info(f"Features shape: {features_df.shape}")
    
    return features_df


# ============================================================================
# 모델 앙상블 예측
# ============================================================================

def load_models(model_dir: Path, model_type: str, device: torch.device) -> List[nn.Module]:
    """여러 fold 모델 로드"""
    
    models = []
    model_files = sorted(model_dir.glob(f"vaep_{model_type}_fold*_20251129_101333.pth"))
    
    logger.info(f"Loading {len(model_files)} {model_type} models...")
    
    for model_file in model_files:
        model = VAEPMLP(input_dim=133, hidden_dims=[256, 128], dropout=0.3).to(device)
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.eval()
        models.append(model)
    
    return models


def predict_with_ensemble(
    models: List[nn.Module],
    features: np.ndarray,
    device: torch.device,
    batch_size: int = 4096
) -> np.ndarray:
    """앙상블 모델로 예측 (평균)"""
    
    all_predictions = []
    
    for model in models:
        predictions = []
        
        for i in range(0, len(features), batch_size):
            batch = features[i:i + batch_size]
            batch_tensor = torch.FloatTensor(batch).to(device)
            
            with torch.no_grad():
                outputs = model(batch_tensor).squeeze().cpu().numpy()
            
            predictions.extend(outputs if outputs.ndim > 0 else [outputs.item()])
        
        all_predictions.append(predictions)
    
    # 앙상블 평균
    ensemble_predictions = np.mean(all_predictions, axis=0)
    
    return ensemble_predictions


# ============================================================================
# VAEP 계산
# ============================================================================

def compute_vaep_for_actions(
    atomic_actions_df: pd.DataFrame,
    scores_models: List[nn.Module],
    concedes_models: List[nn.Module],
    device: torch.device
) -> pd.DataFrame:
    """액션별 VAEP 계산"""
    
    logger.info("\n=== VAEP 계산 시작 ===")
    
    # Feature 생성
    features_df = create_features(atomic_actions_df)
    features = features_df.values.astype(np.float32)
    
    # Scores 예측
    logger.info("Scores 확률 예측 중...")
    scores_probs = predict_with_ensemble(scores_models, features, device)
    
    # Concedes 예측
    logger.info("Concedes 확률 예측 중...")
    concedes_probs = predict_with_ensemble(concedes_models, features, device)
    
    # VAEP 계산: offensive_value - defensive_value
    vaep = scores_probs - concedes_probs
    
    # 결과 DataFrame 생성
    result_df = atomic_actions_df[['game_id', 'action_id', 'period_id', 'time_seconds', 
                                     'team_id', 'player_id', 'type_id', 'x', 'y']].copy()
    result_df['scores_prob'] = scores_probs
    result_df['concedes_prob'] = concedes_probs
    result_df['vaep'] = vaep
    
    logger.info(f"VAEP 계산 완료: {len(result_df)} 액션")
    logger.info(f"VAEP 평균: {vaep.mean():.6f}, 표준편차: {vaep.std():.6f}")
    logger.info(f"VAEP 범위: [{vaep.min():.6f}, {vaep.max():.6f}]")
    
    return result_df


# ============================================================================
# 선수별 집계
# ============================================================================

def aggregate_player_vaep_by_game(vaep_df: pd.DataFrame) -> pd.DataFrame:
    """경기별 선수 VAEP 집계"""
    
    logger.info("\n=== 경기별 선수 VAEP 집계 ===")
    
    player_game_vaep = vaep_df.groupby(['game_id', 'player_id']).agg({
        'vaep': ['sum', 'mean', 'count'],
        'scores_prob': 'sum',
        'concedes_prob': 'sum'
    }).reset_index()
    
    # 컬럼명 정리
    player_game_vaep.columns = [
        'game_id', 'player_id', 'vaep_total', 'vaep_mean', 'num_actions',
        'total_scores_prob', 'total_concedes_prob'
    ]
    
    logger.info(f"총 {len(player_game_vaep)} 선수-경기 조합")
    
    return player_game_vaep


def aggregate_player_vaep_by_season(
    player_game_vaep: pd.DataFrame,
    loader: PublicWyscoutLoader
) -> pd.DataFrame:
    """시즌별 선수 VAEP 집계"""
    
    logger.info("\n=== 시즌별 선수 VAEP 집계 ===")
    
    # 경기 정보 로드 (시즌 정보 포함)
    logger.info("경기 메타데이터 로드 중...")
    games = loader.games()
    
    # game_id로 매핑 (season, competition 정보 추가)
    if 'season_id' in games.columns:
        game_season_map = games[['game_id', 'season_id', 'competition_id']].copy()
        player_game_vaep = player_game_vaep.merge(game_season_map, on='game_id', how='left')
        
        # 시즌별 집계
        player_season_vaep = player_game_vaep.groupby(['season_id', 'player_id']).agg({
            'vaep_total': 'sum',
            'vaep_mean': 'mean',
            'num_actions': 'sum',
            'total_scores_prob': 'sum',
            'total_concedes_prob': 'sum',
            'game_id': 'count'  # 경기 수
        }).reset_index()
        
        player_season_vaep.columns = [
            'season_id', 'player_id', 'vaep_total', 'vaep_mean', 'num_actions',
            'total_scores_prob', 'total_concedes_prob', 'num_games'
        ]
        
        # VAEP per 90 계산 (대략적)
        player_season_vaep['vaep_per90'] = (
            player_season_vaep['vaep_total'] / player_season_vaep['num_games']
        )
        
    else:
        # season 정보가 없으면 전체를 하나의 시즌으로 간주
        logger.warning("시즌 정보가 없어 전체를 단일 시즌으로 처리합니다.")
        player_season_vaep = player_game_vaep.groupby('player_id').agg({
            'vaep_total': 'sum',
            'vaep_mean': 'mean',
            'num_actions': 'sum',
            'total_scores_prob': 'sum',
            'total_concedes_prob': 'sum',
            'game_id': 'count'
        }).reset_index()
        
        player_season_vaep.columns = [
            'player_id', 'vaep_total', 'vaep_mean', 'num_actions',
            'total_scores_prob', 'total_concedes_prob', 'num_games'
        ]
        
        player_season_vaep['vaep_per90'] = (
            player_season_vaep['vaep_total'] / player_season_vaep['num_games']
        )
        player_season_vaep['season_id'] = 'all'
    
    logger.info(f"총 {len(player_season_vaep)} 선수-시즌 조합")
    
    return player_season_vaep


def enrich_with_player_info(
    player_vaep_df: pd.DataFrame,
    loader: PublicWyscoutLoader
) -> pd.DataFrame:
    """선수 정보 추가 (이름 등)"""
    
    logger.info("선수 정보 추가 중...")
    
    try:
        players = loader.players()
        
        if 'player_id' in players.columns and 'player_name' in players.columns:
            player_info = players[['player_id', 'player_name']].drop_duplicates()
            player_vaep_df = player_vaep_df.merge(
                player_info, 
                on='player_id', 
                how='left'
            )
        else:
            logger.warning("선수 이름 정보를 찾을 수 없습니다.")
    
    except Exception as e:
        logger.warning(f"선수 정보 로드 실패: {e}")
    
    return player_vaep_df


# ============================================================================
# Main 함수
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Atomic SPADL VAEP 계산")
    
    parser.add_argument(
        "--atomic_data",
        type=str,
        default="../data/processed/atomic_spadl_all_games.csv",
        help="Atomic SPADL 데이터 경로"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="../models/atomic_vaep",
        help="학습된 모델 디렉토리"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data/vaep_results",
        help="결과 저장 디렉토리"
    )
    parser.add_argument(
        "--wyscout_dir",
        type=str,
        default="../statsbomb/open-data/data",
        help="Wyscout 데이터 디렉토리 (선수/경기 정보용)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="예측 배치 크기"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="테스트용 샘플 크기 (None이면 전체 데이터)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # Step 1: 데이터 로드
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("Step 1: Atomic SPADL 데이터 로드")
    logger.info("="*60)
    
    atomic_actions_df = pd.read_csv(args.atomic_data)
    logger.info(f"전체 데이터 shape: {atomic_actions_df.shape}")
    
    if args.sample_size:
        logger.info(f"샘플링: {args.sample_size} 액션")
        atomic_actions_df = atomic_actions_df.head(args.sample_size)
    
    # ========================================================================
    # Step 2: 모델 로드
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("Step 2: 학습된 모델 로드")
    logger.info("="*60)
    
    model_dir = Path(args.model_dir)
    
    scores_models = load_models(model_dir, "scores", device)
    concedes_models = load_models(model_dir, "concedes", device)
    
    logger.info(f"Scores 모델: {len(scores_models)}개")
    logger.info(f"Concedes 모델: {len(concedes_models)}개")
    
    # ========================================================================
    # Step 3: VAEP 계산
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("Step 3: 액션별 VAEP 계산")
    logger.info("="*60)
    
    vaep_df = compute_vaep_for_actions(
        atomic_actions_df,
        scores_models,
        concedes_models,
        device
    )
    
    # 액션별 VAEP 저장
    action_vaep_file = output_dir / "action_vaep_atomic.csv"
    vaep_df.to_csv(action_vaep_file, index=False)
    logger.info(f"액션별 VAEP 저장: {action_vaep_file}")
    
    # ========================================================================
    # Step 4: 경기별 집계
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("Step 4: 경기별 선수 VAEP 집계")
    logger.info("="*60)
    
    player_game_vaep = aggregate_player_vaep_by_game(vaep_df)
    
    # Wyscout loader로 선수 정보 추가
    try:
        loader = PublicWyscoutLoader(root=args.wyscout_dir, getter="local")
        player_game_vaep = enrich_with_player_info(player_game_vaep, loader)
    except Exception as e:
        logger.warning(f"Wyscout loader 초기화 실패: {e}")
        loader = None
    
    # 경기별 VAEP 저장
    game_vaep_file = output_dir / "player_game_vaep_atomic.csv"
    player_game_vaep.to_csv(game_vaep_file, index=False)
    logger.info(f"경기별 선수 VAEP 저장: {game_vaep_file}")
    
    # 통계 출력
    logger.info(f"\n경기별 VAEP 통계:")
    logger.info(f"  평균 VAEP: {player_game_vaep['vaep_total'].mean():.4f}")
    logger.info(f"  중앙값: {player_game_vaep['vaep_total'].median():.4f}")
    logger.info(f"  표준편차: {player_game_vaep['vaep_total'].std():.4f}")
    logger.info(f"  최고값: {player_game_vaep['vaep_total'].max():.4f}")
    logger.info(f"  최저값: {player_game_vaep['vaep_total'].min():.4f}")
    
    # Top 10 선수-경기
    logger.info("\nTop 10 경기 성과:")
    top_10 = player_game_vaep.nlargest(10, 'vaep_total')[
        ['player_id', 'player_name', 'game_id', 'vaep_total', 'num_actions']
    ] if 'player_name' in player_game_vaep.columns else player_game_vaep.nlargest(10, 'vaep_total')[
        ['player_id', 'game_id', 'vaep_total', 'num_actions']
    ]
    print(top_10.to_string(index=False))
    
    # ========================================================================
    # Step 5: 시즌별 집계
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("Step 5: 시즌별 선수 VAEP 집계")
    logger.info("="*60)
    
    if loader:
        player_season_vaep = aggregate_player_vaep_by_season(player_game_vaep, loader)
        player_season_vaep = enrich_with_player_info(player_season_vaep, loader)
        
        # 시즌별 VAEP 저장
        season_vaep_file = output_dir / "player_season_vaep_atomic.csv"
        player_season_vaep.to_csv(season_vaep_file, index=False)
        logger.info(f"시즌별 선수 VAEP 저장: {season_vaep_file}")
        
        # 통계 출력
        logger.info(f"\n시즌별 VAEP 통계:")
        logger.info(f"  평균 VAEP: {player_season_vaep['vaep_total'].mean():.4f}")
        logger.info(f"  중앙값: {player_season_vaep['vaep_total'].median():.4f}")
        logger.info(f"  표준편차: {player_season_vaep['vaep_total'].std():.4f}")
        logger.info(f"  최고값: {player_season_vaep['vaep_total'].max():.4f}")
        logger.info(f"  최저값: {player_season_vaep['vaep_total'].min():.4f}")
        
        # Top 20 선수
        logger.info("\nTop 20 선수 (시즌별):")
        top_20 = player_season_vaep.nlargest(20, 'vaep_total')[
            ['player_id', 'player_name', 'season_id', 'vaep_total', 'vaep_per90', 'num_games', 'num_actions']
        ] if 'player_name' in player_season_vaep.columns else player_season_vaep.nlargest(20, 'vaep_total')[
            ['player_id', 'season_id', 'vaep_total', 'vaep_per90', 'num_games', 'num_actions']
        ]
        print(top_20.to_string(index=False))
    
    # ========================================================================
    # Step 6: 요약 통계 저장
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("Step 6: 요약 통계 저장")
    logger.info("="*60)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_actions': len(vaep_df),
        'total_players': player_game_vaep['player_id'].nunique(),
        'total_games': player_game_vaep['game_id'].nunique(),
        'vaep_stats': {
            'mean': float(vaep_df['vaep'].mean()),
            'median': float(vaep_df['vaep'].median()),
            'std': float(vaep_df['vaep'].std()),
            'min': float(vaep_df['vaep'].min()),
            'max': float(vaep_df['vaep'].max())
        },
        'player_game_stats': {
            'mean': float(player_game_vaep['vaep_total'].mean()),
            'median': float(player_game_vaep['vaep_total'].median()),
            'std': float(player_game_vaep['vaep_total'].std()),
            'min': float(player_game_vaep['vaep_total'].min()),
            'max': float(player_game_vaep['vaep_total'].max())
        }
    }
    
    if loader:
        summary['player_season_stats'] = {
            'mean': float(player_season_vaep['vaep_total'].mean()),
            'median': float(player_season_vaep['vaep_total'].median()),
            'std': float(player_season_vaep['vaep_total'].std()),
            'min': float(player_season_vaep['vaep_total'].min()),
            'max': float(player_season_vaep['vaep_total'].max())
        }
        summary['total_seasons'] = int(player_season_vaep['season_id'].nunique())
    
    summary_file = output_dir / "vaep_computation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"요약 통계 저장: {summary_file}")
    
    logger.info("\n" + "="*60)
    logger.info("VAEP 계산 완료!")
    logger.info("="*60)
    logger.info(f"\n저장된 파일:")
    logger.info(f"  1. {action_vaep_file}")
    logger.info(f"  2. {game_vaep_file}")
    if loader:
        logger.info(f"  3. {season_vaep_file}")
    logger.info(f"  4. {summary_file}")


if __name__ == "__main__":
    main()
