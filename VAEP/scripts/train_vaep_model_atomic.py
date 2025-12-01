"""
PyTorch를 사용한 Atomic SPADL 기반 VAEP 모델 학습 스크립트

이 스크립트는 Atomic SPADL 형식의 데이터를 사용하여 VAEP 모델을 학습합니다.
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

# socceraction imports for Atomic SPADL features
import socceraction.atomic.spadl.config as atomicspadl
import socceraction.atomic.vaep.features as atomic_features

from utils import setup_logger, ensure_dir, split_train_val, load_config


def parse_args():
    """명령줄 인자를 파싱합니다."""
    config = load_config()
    train_config = config["training"]
    paths_config = config["paths"]

    parser = argparse.ArgumentParser(description="Atomic SPADL 기반 VAEP 모델 학습")
    parser.add_argument(
        "--config", type=str, default=None, help="설정 파일 경로 (config.yaml)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="전처리된 Atomic SPADL 학습 데이터 경로",
    )
    parser.add_argument(
        "--output_dir", type=str, default="../models", help="모델 저장 디렉토리"
    )
    parser.add_argument(
        "--version", type=str, default=None, help="버전 이름 (기본값: 자동 생성)"
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
    parser.add_argument(
        "--debug", action="store_true", help="디버그 모드: 에포크 수를 3으로 제한"
    )

    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        train_config = config["training"]
        paths_config = config["paths"]

    args.input = (
        args.input
        or f"../{paths_config['processed_dir']}/vaep_train_atomic_spadl.csv"
    )

    from datetime import datetime

    if not hasattr(args, "version") or args.version is None:
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

    if args.debug:
        args.epochs = 3

    return args, config


class VAEPDataset(Dataset):
    """VAEP 학습을 위한 PyTorch Dataset."""

    def __init__(
        self, features: np.ndarray, labels_score: np.ndarray, labels_concede: np.ndarray
    ):
        self.features = torch.FloatTensor(features)
        self.labels_score = torch.FloatTensor(labels_score)
        self.labels_concede = torch.FloatTensor(labels_concede)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels_score[idx], self.labels_concede[idx]


class VAEPModel(nn.Module):
    """VAEP을 위한 MLP 모델."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64]):
        super(VAEPModel, self).__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)
        self.score_head = nn.Linear(prev_dim, 1)
        self.concede_head = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        score_logits = self.score_head(encoded).squeeze(-1)
        concede_logits = self.concede_head(encoded).squeeze(-1)
        return score_logits, concede_logits


def create_atomic_spadl_features(
    df: pd.DataFrame, logger: logging.Logger, nb_prev_actions: int = 3
) -> Tuple[np.ndarray, List[str]]:
    """
    Atomic SPADL 데이터에서 state 특징을 생성합니다.

    socceraction의 Atomic VAEP feature transformer를 사용합니다.

    Args:
        df: Atomic SPADL DataFrame
        logger: 로거 객체
        nb_prev_actions: 게임 상태에 포함할 이전 액션 수

    Returns:
        (features, feature_names) 튜플
    """
    logger.info("Creating Atomic SPADL features...")

    # Atomic SPADL 형식에 맞게 컬럼 확인
    required_cols = [
        "game_id",
        "action_id",
        "period_id",
        "time_seconds",
        "team_id",
        "player_id",
        "x",
        "y",
        "dx",
        "dy",
        "type_id",
        "bodypart_id",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # type_name과 bodypart_name 추가 (없으면 생성)
    import socceraction.atomic.spadl.config as atomicspadl_config
    if "type_name" not in df.columns:
        df["type_name"] = df["type_id"].apply(
            lambda x: atomicspadl_config.actiontypes[x] if x < len(atomicspadl_config.actiontypes) else "unknown"
        )
    if "bodypart_name" not in df.columns:
        df["bodypart_name"] = df["bodypart_id"].apply(
            lambda x: atomicspadl_config.bodyparts[x] if x < len(atomicspadl_config.bodyparts) else "unknown"
        )

    # 게임별로 처리 (메모리 효율성을 위해 배치 처리)
    all_features = []
    all_feature_names = None
    unique_games = df["game_id"].unique()
    
    logger.info(f"Processing {len(unique_games)} games for feature generation...")

    for game_id in tqdm(unique_games, desc="Processing games"):
        try:
            game_actions = df[df["game_id"] == game_id].copy()
            
            # 액션이 너무 적으면 스킵
            if len(game_actions) < nb_prev_actions + 1:
                logger.warning(f"Game {game_id}: Too few actions ({len(game_actions)}), skipping")
                continue
                
            game_actions = game_actions.sort_values(
                ["period_id", "time_seconds", "action_id"]
            ).reset_index(drop=True)

            # Game states 생성
            gamestates = atomic_features.gamestates(game_actions, nb_prev_actions)

            # Feature transformers (Atomic VAEP 기본 feature set)
            xfns = [
                atomic_features.actiontype,
                atomic_features.actiontype_onehot,
                atomic_features.bodypart,
                atomic_features.bodypart_onehot,
                atomic_features.time,
                atomic_features.team,
                atomic_features.time_delta,
                atomic_features.location,
                atomic_features.polar,
                atomic_features.movement_polar,
                atomic_features.direction,
                atomic_features.goalscore,
            ]

            # Feature 생성
            game_features = pd.concat([fn(gamestates) for fn in xfns], axis=1)

            if all_feature_names is None:
                all_feature_names = game_features.columns.tolist()
            elif list(game_features.columns) != all_feature_names:
                # 컬럼 순서가 다르면 재정렬
                game_features = game_features[all_feature_names]

            all_features.append(game_features.values)
            
        except Exception as e:
            logger.warning(f"Error processing game {game_id}: {e}, skipping...")
            continue

    # 모든 게임의 feature 결합
    if len(all_features) == 0:
        raise ValueError("No features generated! Check input data.")
    
    features = np.concatenate(all_features, axis=0).astype(np.float32)

    logger.info(f"Total feature dimension: {features.shape[1]}")
    logger.info(f"Total samples: {features.shape[0]}")
    logger.info(f"Feature names: {len(all_feature_names)} features")

    return features, all_feature_names


def create_labels_atomic_spadl(
    df: pd.DataFrame, horizon: int, logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Atomic SPADL 데이터에서 득점/실점 라벨을 생성합니다.

    Args:
        df: Atomic SPADL DataFrame
        horizon: 라벨링 horizon (이벤트 수)
        logger: 로거 객체

    Returns:
        (labels_score, labels_concede) 튜플
    """
    logger.info(f"Creating labels with horizon={horizon} events...")

    labels_score = np.zeros(len(df), dtype=np.float32)
    labels_concede = np.zeros(len(df), dtype=np.float32)

    # Atomic SPADL에서 goal 타입 찾기
    import socceraction.atomic.spadl.config as atomicspadl_config
    goal_type_id = atomicspadl_config.actiontypes.index("goal")

    # 게임별로 처리
    for game_id in tqdm(df["game_id"].unique(), desc="Creating labels"):
        game_mask = df["game_id"] == game_id
        game_df = df[game_mask].copy()
        game_df = game_df.sort_values(
            ["period_id", "time_seconds", "action_id"]
        ).reset_index(drop=True)
        game_indices = df[game_mask].index.values

        for i in range(len(game_df)):
            # horizon 내의 액션 확인
            future_actions = game_df.iloc[i + 1 : i + 1 + horizon]

            if len(future_actions) == 0:
                continue

            team_id = game_df.iloc[i]["team_id"]

            # 득점 확인 (같은 팀의 goal 액션)
            goals = future_actions[
                (future_actions["type_id"] == goal_type_id)
                & (future_actions["team_id"] == team_id)
            ]

            # 실점 확인 (다른 팀의 goal 액션)
            concedes = future_actions[
                (future_actions["type_id"] == goal_type_id)
                & (future_actions["team_id"] != team_id)
            ]

            if len(goals) > 0:
                labels_score[game_indices[i]] = 1.0

            if len(concedes) > 0:
                labels_concede[game_indices[i]] = 1.0

    logger.info(f"  - Score labels: {labels_score.sum()} / {len(labels_score)}")
    logger.info(f"  - Concede labels: {labels_concede.sum()} / {len(labels_concede)}")

    return labels_score, labels_concede


def main():
    """메인 실행 함수."""
    args, config = parse_args()

    # 로거 설정
    logger = setup_logger("train_vaep_atomic", args.log_file)
    logger.info("=" * 80)
    logger.info("Starting Atomic SPADL VAEP model training")
    logger.info("=" * 80)

    # 출력 디렉토리 생성
    ensure_dir(args.output_dir)

    try:
        # 데이터 로드
        logger.info(f"\nLoading data from: {args.input}")
        df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df)} atomic actions")
        logger.info(f"Columns: {df.columns.tolist()}")

        # Feature 생성
        logger.info("\n" + "=" * 80)
        logger.info("Creating features")
        logger.info("=" * 80)
        X, feature_names = create_atomic_spadl_features(df, logger)

        # Label 생성
        logger.info("\n" + "=" * 80)
        logger.info("Creating labels")
        logger.info("=" * 80)
        y_score, y_concede = create_labels_atomic_spadl(df, args.horizon, logger)

        # 학습/검증 분할
        logger.info("\n" + "=" * 80)
        logger.info("Splitting train/validation sets")
        logger.info("=" * 80)
        train_df, val_df = split_train_val(df, args.val_ratio, args.random_seed)

        train_indices = train_df.index.values
        val_indices = val_df.index.values

        X_train, X_val = X[train_indices], X[val_indices]
        y_score_train, y_score_val = y_score[train_indices], y_score[val_indices]
        y_concede_train, y_concede_val = (
            y_concede[train_indices],
            y_concede[val_indices],
        )

        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Validation set: {len(X_val)} samples")

        # Dataset 생성
        train_dataset = VAEPDataset(X_train, y_score_train, y_concede_train)
        val_dataset = VAEPDataset(X_val, y_score_val, y_concede_val)

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # 모델 생성
        model = VAEPModel(input_dim=X_train.shape[1], hidden_dims=args.hidden_dims)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logger.info(f"Using device: {device}")

        # 손실 함수 및 옵티마이저
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # 학습
        logger.info("\n" + "=" * 80)
        logger.info("Training model")
        logger.info("=" * 80)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(args.epochs):
            # 학습 단계
            model.train()
            train_loss = 0.0
            for batch_features, batch_score, batch_concede in train_loader:
                batch_features = batch_features.to(device)
                batch_score = batch_score.to(device)
                batch_concede = batch_concede.to(device)

                optimizer.zero_grad()
                score_logits, concede_logits = model(batch_features)

                loss_score = criterion(score_logits, batch_score)
                loss_concede = criterion(concede_logits, batch_concede)
                loss = loss_score + loss_concede

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # 검증 단계
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_features, batch_score, batch_concede in val_loader:
                    batch_features = batch_features.to(device)
                    batch_score = batch_score.to(device)
                    batch_concede = batch_concede.to(device)

                    score_logits, concede_logits = model(batch_features)

                    loss_score = criterion(score_logits, batch_score)
                    loss_concede = criterion(concede_logits, batch_concede)
                    loss = loss_score + loss_concede

                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            logger.info(
                f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 모델 저장
                model_path = os.path.join(args.output_dir, "vaep_model.pt")
                torch.save(model.state_dict(), model_path)
                logger.info(f"  -> Saved model to {model_path}")
            else:
                patience_counter += 1
                if patience_counter >= config["training"]["early_stopping_patience"]:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # 모델 설정 저장
        config_dict = {
            "input_dim": X_train.shape[1],
            "hidden_dims": args.hidden_dims,
            "horizon": args.horizon,
            "feature_names": feature_names,
        }
        config_path = os.path.join(args.output_dir, "vaep_config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Saved config to {config_path}")

        logger.info("\n" + "=" * 80)
        logger.info("Training completed!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

