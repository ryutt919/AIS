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
        "--output_dir", type=str, default=None, help="모델 저장 디렉토리"
    )
    parser.add_argument(
        "--horizon", type=int, default=None, help="라벨링 horizon (이벤트 수)"
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=None,
        help="MLP 히든 레이어 차원",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="배치 크기")
    parser.add_argument("--epochs", type=int, default=None, help="학습 에포크 수")
    parser.add_argument("--lr", type=float, default=None, help="학습률")
    parser.add_argument("--val_ratio", type=float, default=None, help="검증 세트 비율")
    parser.add_argument("--random_seed", type=int, default=None, help="랜덤 시드")
    parser.add_argument("--log_file", type=str, default=None, help="로그 파일 경로")

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
    args.output_dir = args.output_dir or f"../{paths_config['models_dir']}"
    args.horizon = args.horizon or train_config["horizon"]
    args.hidden_dims = args.hidden_dims or train_config["hidden_dims"]
    args.batch_size = args.batch_size or train_config["batch_size"]
    args.epochs = args.epochs or train_config["epochs"]
    args.lr = args.lr or train_config["learning_rate"]
    args.val_ratio = args.val_ratio or train_config["val_ratio"]
    args.random_seed = args.random_seed or train_config["random_seed"]

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
    - 위치 특징 (start_x, start_y, end_x, end_y)
    - 골 관련 특징 (goal_distance, goal_angle)
    - 이동 특징 (distance, angle)
    - 성공 여부 (is_successful)

    Args:
        df: 이벤트 DataFrame
        logger: 로거 객체

    Returns:
        (features, feature_names) 튜플
    """
    logger.info("Creating state features...")

    feature_list = []
    feature_names = []

    # 1. 이벤트 타입 원핫 인코딩
    event_types = pd.get_dummies(df["eventId"], prefix="event")
    feature_list.append(event_types.values)
    feature_names.extend(event_types.columns.tolist())
    logger.info(f"  - Event types: {event_types.shape[1]} features")

    # 2. 위치 특징
    position_features = df[["start_x", "start_y", "end_x", "end_y"]].values
    feature_list.append(position_features)
    feature_names.extend(["start_x", "start_y", "end_x", "end_y"])
    logger.info(f"  - Position features: 4 features")

    # 3. 골 관련 특징
    goal_features = df[["goal_distance", "goal_angle"]].values
    feature_list.append(goal_features)
    feature_names.extend(["goal_distance", "goal_angle"])
    logger.info(f"  - Goal features: 2 features")

    # 4. 이동 특징
    movement_features = df[["distance", "angle"]].values
    feature_list.append(movement_features)
    feature_names.extend(["distance", "angle"])
    logger.info(f"  - Movement features: 2 features")

    # 5. 성공 여부
    success_features = df[["is_successful"]].values
    feature_list.append(success_features)
    feature_names.extend(["is_successful"])
    logger.info(f"  - Success features: 1 feature")

    # 6. 피리어드 원핫 인코딩
    period_features = pd.get_dummies(df["period"], prefix="period")
    feature_list.append(period_features.values)
    feature_names.extend(period_features.columns.tolist())
    logger.info(f"  - Period features: {period_features.shape[1]} features")

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
    for match_id in df["matchId"].unique():
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

    Returns:
        (total_loss, score_acc, concede_acc, avg_loss)
    """
    model.train()
    total_loss = 0.0
    score_correct = 0
    concede_correct = 0
    total_samples = 0

    for batch_idx, (features, labels_score, labels_concede) in enumerate(dataloader):
        features = features.to(device)
        labels_score = labels_score.to(device)
        labels_concede = labels_concede.to(device)

        # 순전파
        score_logits, concede_logits = model(features)

        # 손실 계산 (VAEP 논문: Binary Cross-Entropy)
        loss_score = criterion(score_logits, labels_score)
        loss_concede = criterion(concede_logits, labels_concede)
        loss = loss_score + loss_concede

        # 역전파
        optimizer.zero_grad()
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
        for features, labels_score, labels_concede in dataloader:
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

    # 로거 설정
    logger = setup_logger("train_vaep", args.log_file)
    logger.info("=" * 80)
    logger.info("VAEP Model Training")
    logger.info("=" * 80)

    # 랜덤 시드 설정
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 절대 경로 계산
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.abspath(os.path.join(script_dir, args.input))
    output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))

    logger.info(f"Input: {input_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Horizon: {args.horizon} events")
    logger.info(f"Hidden dimensions: {args.hidden_dims}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.lr}")

    ensure_dir(output_dir)

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

        # 라벨 생성
        logger.info("\n" + "=" * 80)
        logger.info("Creating labels...")
        logger.info("=" * 80)

        labels_score, labels_concede = create_labels(df, args.horizon, logger)

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
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
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

        # 학습
        logger.info("\n" + "=" * 80)
        logger.info("Training...")
        logger.info("=" * 80)

        best_val_loss = float("inf")

        for epoch in range(args.epochs):
            # 학습
            train_loss, train_score_acc, train_concede_acc, avg_train_loss = (
                train_epoch(model, train_loader, criterion, optimizer, device, logger)
            )

            # 검증
            val_loss, val_score_acc, val_concede_acc = validate(
                model, val_loader, criterion, device
            )

            logger.info(
                f"Epoch [{epoch+1}/{args.epochs}] "
                f"Train Loss: {avg_train_loss:.4f} "
                f"(Score Acc: {train_score_acc:.4f}, Concede Acc: {train_concede_acc:.4f}) | "
                f"Val Loss: {val_loss:.4f} "
                f"(Score Acc: {val_score_acc:.4f}, Concede Acc: {val_concede_acc:.4f})"
            )

            # 최고 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = os.path.join(output_dir, "vaep_model.pt")
                torch.save(model.state_dict(), model_path)
                logger.info(f"  -> Best model saved (val_loss: {val_loss:.4f})")

        # 설정 저장
        logger.info("\n" + "=" * 80)
        logger.info("Saving configuration...")
        logger.info("=" * 80)

        config = {
            "input_dim": input_dim,
            "hidden_dims": args.hidden_dims,
            "horizon": args.horizon,
            "feature_names": feature_names,
            "best_val_loss": best_val_loss,
        }

        config_path = os.path.join(output_dir, "vaep_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"Configuration saved to: {config_path}")

        logger.info("\n" + "=" * 80)
        logger.info("Training completed successfully!")
        logger.info("=" * 80)
        logger.info(f"Best validation loss: {best_val_loss:.4f}")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
