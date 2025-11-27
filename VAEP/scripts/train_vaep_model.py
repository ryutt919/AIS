"""
PyTorch를 사용한 VAEP 모델 학습 스크립트

이 스크립트는 VAEP (Valuing Actions by Estimating Probabilities) 논문의 방법론을 구현합니다:

1. State representation: 각 이벤트의 상태를 특징 벡터로 표현
2. Labeling: 각 이벤트 후 일정 시간 내 득점/실점 여부를 라벨로 생성
3. Model: MLP를 사용하여 P(score) 및 P(concede)를 예측
4. Value: V(state) = P(score) - P(concede)

참고: VAEP 논문의 수식
- V_t = P_score_t - P_concede_t
- VAEP(action_t) = V_{t+1} - V_t
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import setup_logger, ensure_dir, split_train_val, load_config


def parse_args():
    """명령줄 인자를 파싱합니다. config.yaml의 설정을 기본값으로 사용합니다."""
    # config 파일 로드
    config = load_config()
    train_config = config["training"]
    paths_config = config["paths"]

    parser = argparse.ArgumentParser(description="VAEP 모델 학습")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="설정 파일 경로 (config.yaml)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="전처리된 학습 데이터 경로",
    )
    parser.add_argument(
        "--output_dir", type=str, default="../models", help="모델 저장 디렉토리 (버전 폴더가 자동 생성됨)"
    )
    parser.add_argument(
        "--version", type=str, default=None, help="버전 이름 (기본값: 자동 생성)"
    )
    parser.add_argument(
        "--horizon", type=int, default=None, help="라벨링 horizon (이벤트 수)"
    )
    parser.add_argument(
        "--precomputed_dir",
        type=str,
        default=None,
        help="전처리된 horizon 데이터 디렉토리 (속도 향상)",
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=None,
        help="MLP 히든 레이어 차원",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="배치 크기")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader worker 수 (GPU 최적화)")
    parser.add_argument("--use_amp", action="store_true", help="Mixed Precision Training 사용")
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "step", "cosine"], help="학습률 스케줄러 타입")
    parser.add_argument("--epochs", type=int, default=None, help="학습 에포크 수")
    parser.add_argument("--lr", type=float, default=None, help="학습률")
    parser.add_argument("--val_ratio", type=float, default=None, help="검증 세트 비율")
    parser.add_argument("--random_seed", type=int, default=None, help="랜덤 시드")
    parser.add_argument("--log_file", type=str, default=None, help="로그 파일 경로")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="디버그 모드: 에포크 수를 3으로 제한"
    )

    args = parser.parse_args()

    # config 파일에서 기본값 설정 (인자가 없으면)
    if args.config:
        config = load_config(args.config)
        train_config = config["training"]
        paths_config = config["paths"]

    args.input = (
        args.input
        or f"../{paths_config['processed_dir']}/{config['preprocessing']['output_train']}"
    )
    
    # 버전 폴더 생성
    from datetime import datetime
    if not hasattr(args, 'version') or args.version is None:
        args.version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    base_output_dir = args.output_dir or f"../{paths_config['models_dir']}"
    args.output_dir = os.path.join(base_output_dir, args.version)
    args.horizon = args.horizon or train_config["horizon"]
    args.hidden_dims = args.hidden_dims or train_config["hidden_dims"]
    args.batch_size = args.batch_size or train_config["batch_size"]
    args.epochs = args.epochs or train_config["epochs"]
    args.lr = args.lr or train_config["learning_rate"]
    args.val_ratio = args.val_ratio or train_config["val_ratio"]
    args.random_seed = args.random_seed or train_config["random_seed"]

    # 디버그 모드: 에포크 수 제한
    if args.debug:
        args.epochs = 3

    return args, config


class VAEPDataset(Dataset):
    """
    VAEP 학습을 위한 PyTorch Dataset.

    각 샘플은 (state_features, y_score, y_concede) 튜플입니다.
    """

    def __init__(
        self, features: np.ndarray, labels_score: np.ndarray, labels_concede: np.ndarray
    ):
        """
        Args:
            features: 상태 특징 행렬 (n_samples, n_features)
            labels_score: 득점 라벨 (n_samples,)
            labels_concede: 실점 라벨 (n_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.labels_score = torch.FloatTensor(labels_score)
        self.labels_concede = torch.FloatTensor(labels_concede)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels_score[idx], self.labels_concede[idx]


class VAEPModel(nn.Module):
    """
    VAEP을 위한 MLP 모델.

    두 개의 출력 헤드를 가집니다:
    - score_head: P(득점) 예측
    - concede_head: P(실점) 예측

    VAEP 논문의 모델 구조:
    - 공유 인코더 레이어
    - 두 개의 독립적인 출력 헤드
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64]):
        """
        Args:
            input_dim: 입력 특징 차원
            hidden_dims: 히든 레이어 차원 리스트
        """
        super(VAEPModel, self).__init__()

        # 공유 인코더
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # 득점 예측 헤드
        self.score_head = nn.Linear(prev_dim, 1)

        # 실점 예측 헤드
        self.concede_head = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        순전파.

        Args:
            x: 입력 특징 (batch_size, input_dim)

        Returns:
            (score_logits, concede_logits) 튜플
        """
        encoded = self.encoder(x)
        score_logits = self.score_head(encoded).squeeze(-1)
        concede_logits = self.concede_head(encoded).squeeze(-1)
        return score_logits, concede_logits


def create_state_features(
    df: pd.DataFrame, logger: logging.Logger
) -> Tuple[np.ndarray, List[str]]:
    """
    이벤트 데이터에서 state 특징을 생성합니다.

    VAEP state representation:
    - 이벤트 타입 (원핫 인코딩)
    - 서브이벤트 타입 (원핫 인코딩)
    - 태그 (멀티-핫 인코딩)
    - 위치 특징 (start_x, start_y, end_x, end_y)
    - 골 관련 특징 (goal_distance, goal_angle)
    - 이동 특징 (distance, angle)
    - 성공 여부 (is_successful)
    - 피리어드 (원핫 인코딩)

    Args:
        df: 이벤트 DataFrame
        logger: 로거 객체

    Returns:
        (features, feature_names) 튜플
    """
    logger.info("Creating state features...")

    # 사용할 서브이벤트 ID 정의 (Duel, Foul, Free Kick, Others on the ball, Pass, Shot)
    SUBEVENT_IDS = [10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 27, 30, 31, 32, 33, 34, 35, 36, 70, 71, 72, 80, 81, 82, 83, 84, 85, 86, 100]
    
    # 사용할 태그 ID 정의
    TAG_IDS = [101, 102, 201, 301, 302, 403, 501, 502, 503, 504, 601, 602, 701, 702, 703, 801, 802, 901, 1101, 1102, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1301, 1302, 1401, 1501, 1601, 1701, 1702, 1703, 1801, 1802, 2001, 2101]

    feature_list = []
    feature_names = []

    # 1. 이벤트 타입 원핫 인코딩 (1-10 전체 사용)
    event_types = pd.get_dummies(df["eventId"], prefix="event")
    feature_list.append(event_types.values)
    feature_names.extend(event_types.columns.tolist())
    logger.info(f"  - Event types: {event_types.shape[1]} features")

    # 2. 서브이벤트 타입 원핫 인코딩 (지정된 ID만 사용)
    subevent_features = np.zeros((len(df), len(SUBEVENT_IDS)), dtype=np.float32)
    subevent_id_to_idx = {sid: idx for idx, sid in enumerate(SUBEVENT_IDS)}
    
    for i, subevent_id in enumerate(df["subEventId"]):
        if subevent_id in subevent_id_to_idx:
            subevent_features[i, subevent_id_to_idx[subevent_id]] = 1.0
    
    feature_list.append(subevent_features)
    feature_names.extend([f"subevent_{sid}" for sid in SUBEVENT_IDS])
    logger.info(f"  - Sub-event types: {len(SUBEVENT_IDS)} features")

    # 3. 태그 멀티-핫 인코딩 (지정된 ID만 사용)
    import json
    tag_features = np.zeros((len(df), len(TAG_IDS)), dtype=np.float32)
    tag_id_to_idx = {tid: idx for idx, tid in enumerate(TAG_IDS)}
    
    for i, tags_str in enumerate(df["tags_list"]):
        # JSON 문자열을 리스트로 파싱
        if isinstance(tags_str, str):
            try:
                tags = json.loads(tags_str)
            except:
                tags = []
        elif isinstance(tags_str, list):
            tags = tags_str
        else:
            tags = []
        
        for tag in tags:
            if tag in tag_id_to_idx:
                tag_features[i, tag_id_to_idx[tag]] = 1.0
    
    feature_list.append(tag_features)
    feature_names.extend([f"tag_{tid}" for tid in TAG_IDS])
    logger.info(f"  - Tag features: {len(TAG_IDS)} features")

    # 4. 위치 특징
    position_features = df[["start_x", "start_y", "end_x", "end_y"]].values
    feature_list.append(position_features)
    feature_names.extend(["start_x", "start_y", "end_x", "end_y"])
    logger.info(f"  - Position features: 4 features")

    # 5. 골 관련 특징
    goal_features = df[["goal_distance", "goal_angle"]].values
    feature_list.append(goal_features)
    feature_names.extend(["goal_distance", "goal_angle"])
    logger.info(f"  - Goal features: 2 features")

    # 6. 이동 특징
    movement_features = df[["distance", "angle"]].values
    feature_list.append(movement_features)
    feature_names.extend(["distance", "angle"])
    logger.info(f"  - Movement features: 2 features")

    # 7. 성공 여부
    success_features = df[["is_successful"]].values
    feature_list.append(success_features)
    feature_names.extend(["is_successful"])
    logger.info(f"  - Success features: 1 feature")

    # 8. 피리어드 원핫 인코딩 (고정된 5개 period 사용)
    PERIODS = ['1H', '2H', 'E1H', 'E2H', 'P']
    period_features = np.zeros((len(df), len(PERIODS)), dtype=np.float32)
    period_to_idx = {p: idx for idx, p in enumerate(PERIODS)}
    
    for i, period in enumerate(df["period"]):
        if period in period_to_idx:
            period_features[i, period_to_idx[period]] = 1.0
    
    feature_list.append(period_features)
    feature_names.extend([f"period_{p}" for p in PERIODS])
    logger.info(f"  - Period features: {len(PERIODS)} features")

    # 모든 특징 결합
    features = np.concatenate(feature_list, axis=1).astype(np.float32)

    logger.info(f"Total feature dimension: {features.shape[1]}")

    return features, feature_names


def create_labels(
    df: pd.DataFrame, horizon: int, logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray]:
    """
    각 이벤트에 대해 득점/실점 라벨을 생성합니다.

    VAEP 라벨링:
    - y_score_t = 1 if 팀이 horizon 내에 득점, else 0
    - y_concede_t = 1 if 팀이 horizon 내에 실점, else 0

    Args:
        df: 이벤트 DataFrame
        horizon: 라벨링 horizon (이벤트 수)
        logger: 로거 객체

    Returns:
        (labels_score, labels_concede) 튜플
    """
    logger.info(f"Creating labels with horizon={horizon} events...")

    labels_score = np.zeros(len(df), dtype=np.float32)
    labels_concede = np.zeros(len(df), dtype=np.float32)

    # 매치별로 처리
    for match_id in tqdm(df["matchId"].unique(),desc='Processing by match..'):
        match_mask = df["matchId"] == match_id
        match_df = df[match_mask].reset_index(drop=True)
        match_indices = df[match_mask].index

        for i in range(len(match_df)):
            current_team = match_df.loc[i, "teamId"]

            # horizon 내의 이벤트들 확인
            end_idx = min(i + horizon + 1, len(match_df))
            future_events = match_df.loc[i + 1 : end_idx - 1]

            if len(future_events) == 0:
                continue

            # 득점 확인 (우리 팀이 골)
            goals_by_team = future_events[
                (future_events["is_goal"] == 1)
                & (future_events["teamId"] == current_team)
            ]
            if len(goals_by_team) > 0:
                labels_score[match_indices[i]] = 1.0

            # 실점 확인 (상대 팀이 골)
            goals_by_opponent = future_events[
                (future_events["is_goal"] == 1)
                & (future_events["teamId"] != current_team)
            ]
            if len(goals_by_opponent) > 0:
                labels_concede[match_indices[i]] = 1.0

    n_scores = labels_score.sum()
    n_concedes = labels_concede.sum()

    logger.info(f"  - Scoring events: {int(n_scores)} ({n_scores/len(df)*100:.2f}%)")
    logger.info(
        f"  - Conceding events: {int(n_concedes)} ({n_concedes/len(df)*100:.2f}%)"
    )

    return labels_score, labels_concede


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    logger: logging.Logger,
    scaler=None,
) -> Tuple[float, float, float, float]:
    """
    한 에포크 학습을 수행합니다.

    Args:
        model: VAEP 모델
        dataloader: 학습 데이터로더
        criterion: 손실 함수
        optimizer: 옵티마이저
        device: 디바이스
        logger: 로거
        scaler: AMP GradScaler (optional)

    Returns:
        (total_loss, score_acc, concede_acc, avg_loss)
    """
    model.train()
    total_loss = 0.0
    score_correct = 0
    concede_correct = 0
    total_samples = 0

    for batch_idx, (features, labels_score, labels_concede) in enumerate(tqdm(dataloader, desc='Training', leave=False)):
        features = features.to(device, non_blocking=True)
        labels_score = labels_score.to(device, non_blocking=True)
        labels_concede = labels_concede.to(device, non_blocking=True)

        optimizer.zero_grad()
        
        # Mixed Precision Training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                score_logits, concede_logits = model(features)
                loss_score = criterion(score_logits, labels_score)
                loss_concede = criterion(concede_logits, labels_concede)
                loss = loss_score + loss_concede
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 순전파
            score_logits, concede_logits = model(features)

            # 손실 계산 (VAEP 논문: Binary Cross-Entropy)
            loss_score = criterion(score_logits, labels_score)
            loss_concede = criterion(concede_logits, labels_concede)
            loss = loss_score + loss_concede

            # 역전파
            loss.backward()
            optimizer.step()

        # 통계
        total_loss += loss.item()

        # 정확도 계산 (threshold=0.5)
        score_preds = (torch.sigmoid(score_logits) > 0.5).float()
        concede_preds = (torch.sigmoid(concede_logits) > 0.5).float()

        score_correct += (score_preds == labels_score).sum().item()
        concede_correct += (concede_preds == labels_concede).sum().item()
        total_samples += len(features)

    avg_loss = total_loss / len(dataloader)
    score_acc = score_correct / total_samples
    concede_acc = concede_correct / total_samples

    return total_loss, score_acc, concede_acc, avg_loss


def validate(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float, float]:
    """
    검증을 수행합니다.

    Args:
        model: VAEP 모델
        dataloader: 검증 데이터로더
        criterion: 손실 함수
        device: 디바이스

    Returns:
        (avg_loss, score_acc, concede_acc)
    """
    model.eval()
    total_loss = 0.0
    score_correct = 0
    concede_correct = 0
    total_samples = 0

    with torch.no_grad():
        for features, labels_score, labels_concede in tqdm(dataloader, desc='Validating', leave=False):
            features = features.to(device)
            labels_score = labels_score.to(device)
            labels_concede = labels_concede.to(device)

            # 순전파
            score_logits, concede_logits = model(features)

            # 손실 계산
            loss_score = criterion(score_logits, labels_score)
            loss_concede = criterion(concede_logits, labels_concede)
            loss = loss_score + loss_concede

            total_loss += loss.item()

            # 정확도 계산
            score_preds = (torch.sigmoid(score_logits) > 0.5).float()
            concede_preds = (torch.sigmoid(concede_logits) > 0.5).float()

            score_correct += (score_preds == labels_score).sum().item()
            concede_correct += (concede_preds == labels_concede).sum().item()
            total_samples += len(features)

    avg_loss = total_loss / len(dataloader)
    score_acc = score_correct / total_samples
    concede_acc = concede_correct / total_samples

    return avg_loss, score_acc, concede_acc


def main():
    """메인 실행 함수."""
    args, config = parse_args()

    # 출력 디렉토리 생성 (로거 설정 전에)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    ensure_dir(output_dir)

    # 로거 설정 (버전 폴더의 로그 파일)
    if args.log_file and args.log_file != "../logs/train_vaep_model.log":
        log_file = args.log_file
    else:
        log_file = os.path.join(output_dir, "training.log")
    
    logger = setup_logger("train_vaep", log_file)
    logger.info("=" * 80)
    logger.info("VAEP Model Training")
    logger.info("=" * 80)
    logger.info(f"Version: {args.version}")

    # 랜덤 시드 설정
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 절대 경로 계산
    input_path = os.path.abspath(os.path.join(script_dir, args.input))

    logger.info(f"Input: {input_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Horizon: {args.horizon} events")
    logger.info(f"Hidden dimensions: {args.hidden_dims}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.lr}")

    try:
        # 데이터 로드
        logger.info("\n" + "=" * 80)
        logger.info("Loading data...")
        logger.info("=" * 80)

        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} events")
        logger.info(f"  - Unique matches: {df['matchId'].nunique()}")
        logger.info(f"  - Unique players: {df['playerId'].nunique()}")
        logger.info(f"  - Leagues: {df['league'].unique().tolist()}")

        # State 특징 생성
        logger.info("\n" + "=" * 80)
        logger.info("Creating features...")
        logger.info("=" * 80)

        features, feature_names = create_state_features(df, logger)

        # 라벨 생성 또는 로드 (캐싱)
        logger.info("\n" + "=" * 80)
        logger.info("Creating or loading labels...")
        logger.info("=" * 80)
        
        # 라벨 캐시 파일 경로
        labels_cache_dir = os.path.join(script_dir, "../data/processed")
        labels_cache_path = os.path.join(labels_cache_dir, f"labels_h{args.horizon}.npz")
        
        if os.path.exists(labels_cache_path):
            logger.info(f"Loading cached labels from: {labels_cache_path}")
            labels_data = np.load(labels_cache_path)
            labels_score = labels_data['labels_score']
            labels_concede = labels_data['labels_concede']
            logger.info(f"  - Loaded {len(labels_score)} labels from cache")
            logger.info(f"  - Scoring events: {int(labels_score.sum())} ({labels_score.sum()/len(labels_score)*100:.2f}%)")
            logger.info(f"  - Conceding events: {int(labels_concede.sum())} ({labels_concede.sum()/len(labels_concede)*100:.2f}%)")
        else:
            logger.info("No cached labels found. Computing labels...")
            labels_score, labels_concede = create_labels(df, args.horizon, logger)
            
            # 라벨 저장
            logger.info(f"Saving labels to cache: {labels_cache_path}")
            np.savez_compressed(labels_cache_path, labels_score=labels_score, labels_concede=labels_concede)
            logger.info("Labels cached successfully")

        # Train/Val 분할
        logger.info("\n" + "=" * 80)
        logger.info("Splitting data...")
        logger.info("=" * 80)

        train_df, val_df = split_train_val(df, args.val_ratio, args.random_seed)

        train_indices = train_df.index
        val_indices = val_df.index

        train_features = features[train_indices]
        train_labels_score = labels_score[train_indices]
        train_labels_concede = labels_concede[train_indices]

        val_features = features[val_indices]
        val_labels_score = labels_score[val_indices]
        val_labels_concede = labels_concede[val_indices]

        logger.info(f"Training samples: {len(train_features)}")
        logger.info(f"Validation samples: {len(val_features)}")

        # Dataset 및 DataLoader 생성
        train_dataset = VAEPDataset(
            train_features, train_labels_score, train_labels_concede
        )
        val_dataset = VAEPDataset(val_features, val_labels_score, val_labels_concede)

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
        )

        # 모델 생성
        logger.info("\n" + "=" * 80)
        logger.info("Creating model...")
        logger.info("=" * 80)

        input_dim = features.shape[1]
        model = VAEPModel(input_dim, args.hidden_dims).to(device)

        logger.info(f"Model architecture:")
        logger.info(f"  - Input dimension: {input_dim}")
        logger.info(f"  - Hidden dimensions: {args.hidden_dims}")
        logger.info(
            f"  - Total parameters: {sum(p.numel() for p in model.parameters())}"
        )

        # 손실 함수 및 옵티마이저
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        # 러닝 스케줄러 설정
        scheduler = None
        if args.scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            logger.info("Using StepLR scheduler (step_size=10, gamma=0.5)")
        elif args.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            logger.info(f"Using CosineAnnealingLR scheduler (T_max={args.epochs})")
        
        # Mixed Precision Training
        scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
        if args.use_amp:
            logger.info("Using Automatic Mixed Precision (AMP) training")

        # 학습
        logger.info("\n" + "=" * 80)
        logger.info("Training...")
        logger.info("=" * 80)
        logger.info(f"Num workers: {args.num_workers}")
        logger.info(f"Mixed Precision: {args.use_amp}")

        best_val_loss = float("inf")
        best_epoch = 0
        
        # 학습 이력 저장
        history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_score_acc': [],
            'train_concede_acc': [],
            'val_score_acc': [],
            'val_concede_acc': []
        }

        for epoch in range(args.epochs):
            # 학습
            train_loss, train_score_acc, train_concede_acc, avg_train_loss = (
                train_epoch(model, train_loader, criterion, optimizer, device, logger, scaler)
            )

            # 검증
            val_loss, val_score_acc, val_concede_acc = validate(
                model, val_loader, criterion, device
            )
            
            # 이력 저장
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['train_score_acc'].append(train_score_acc)
            history['train_concede_acc'].append(train_concede_acc)
            history['val_score_acc'].append(val_score_acc)
            history['val_concede_acc'].append(val_concede_acc)

            logger.info(
                f"Epoch [{epoch+1}/{args.epochs}] "
                f"Train Loss: {avg_train_loss:.4f} "
                f"(Score Acc: {train_score_acc:.4f}, Concede Acc: {train_concede_acc:.4f}) | "
                f"Val Loss: {val_loss:.4f} "
                f"(Score Acc: {val_score_acc:.4f}, Concede Acc: {val_concede_acc:.4f})"
            )

            # 에포별 모델 저장
            epoch_model_path = os.path.join(output_dir, f"vaep_model_epoch_{epoch+1:03d}.pt")
            torch.save(model.state_dict(), epoch_model_path)
            
            # 최고 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                model_path = os.path.join(output_dir, "vaep_model_best.pt")
                torch.save(model.state_dict(), model_path)
                logger.info(f"  -> Best model saved (val_loss: {val_loss:.4f})")
            
            # 스케줄러 스텝
            if scheduler is not None:
                scheduler.step()
                logger.info(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # 설정 및 이력 저장
        logger.info("\n" + "=" * 80)
        logger.info("Saving configuration and history...")
        logger.info("=" * 80)

        config = {
            "version": args.version,
            "input_dim": input_dim,
            "hidden_dims": args.hidden_dims,
            "horizon": args.horizon,
            "feature_names": feature_names,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "num_train_samples": len(train_dataset),
            "num_val_samples": len(val_dataset),
        }

        config_path = os.path.join(output_dir, "vaep_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"Configuration saved to: {config_path}")
        
        # 학습 이력 저장
        history_path = os.path.join(output_dir, "training_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Training history saved to: {history_path}")

        logger.info("\n" + "=" * 80)
        logger.info("Training completed successfully!")
        logger.info("=" * 80)
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Best epoch: {best_epoch}")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
