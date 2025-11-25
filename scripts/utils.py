"""
VAEP 구현을 위한 공통 유틸리티 함수 모음

이 모듈은 데이터 로딩, 전처리, 특징 추출 등의 공통 기능을 제공합니다.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import yaml


def setup_logger(
    name: str, log_file: Optional[str] = None, level: int = logging.INFO
) -> logging.Logger:
    """
    로거를 설정합니다.

    Args:
        name: 로거 이름
        log_file: 로그 파일 경로 (선택사항)
        level: 로깅 레벨

    Returns:
        설정된 로거 객체
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_json(file_path: str) -> Any:
    """JSON 파일을 로드합니다."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_wyscout_events(data_dir: str, league: str) -> pd.DataFrame:
    """
    특정 리그의 Wyscout 이벤트 데이터를 로드합니다.

    Args:
        data_dir: wyscout 데이터 디렉토리 경로
        league: 리그 이름 (예: 'England', 'Spain')

    Returns:
        이벤트 DataFrame
    """
    logger = logging.getLogger(__name__)
    event_file = os.path.join(data_dir, f"events_{league}.json")

    if not os.path.exists(event_file):
        raise FileNotFoundError(f"Event file not found: {event_file}")

    events = load_json(event_file)
    df = pd.DataFrame(events)
    logger.info(f"Loaded {len(df)} events from {league}")

    return df


def extract_position_features(
    positions: List[Dict],
) -> Tuple[float, float, float, float]:
    """
    위치 정보에서 시작/끝 좌표를 추출합니다.

    Args:
        positions: 위치 정보 리스트 [{'x': x1, 'y': y1}, {'x': x2, 'y': y2}]

    Returns:
        (start_x, start_y, end_x, end_y) 튜플 (0-1로 정규화됨)
    """
    if not positions:
        return 0.0, 0.0, 0.0, 0.0

    start_pos = positions[0]
    start_x = start_pos.get("x", 0) / 100.0  # 0-1로 정규화
    start_y = start_pos.get("y", 0) / 100.0

    if len(positions) > 1:
        end_pos = positions[1]
        end_x = end_pos.get("x", 0) / 100.0
        end_y = end_pos.get("y", 0) / 100.0
    else:
        end_x, end_y = start_x, start_y

    return start_x, start_y, end_x, end_y


def extract_tags(tags: List[Dict]) -> List[int]:
    """태그 정보에서 ID 리스트를 추출합니다."""
    if not tags:
        return []
    return [tag.get("id", 0) for tag in tags]


def compute_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """두 점 사이의 유클리드 거리를 계산합니다."""
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def compute_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    """두 점 사이의 각도를 계산합니다 (라디안)."""
    return np.arctan2(y2 - y1, x2 - x1)


def is_goal(tags: List[int]) -> bool:
    """이벤트가 골인지 확인합니다. (101: Goal, 102: Own goal)"""
    return 101 in tags or 102 in tags


def is_successful(tags: List[int]) -> bool:
    """이벤트가 성공적인지 확인합니다. (1801: Accurate)"""
    return 1801 in tags


def compute_goal_distance(x: float, y: float) -> float:
    """
    골 중앙까지의 거리를 계산합니다.

    Args:
        x: x 좌표 (0-1, 0이 자기편 골, 1이 상대편 골)
        y: y 좌표 (0-1)

    Returns:
        골까지의 거리
    """
    return compute_distance(x, y, 1.0, 0.5)


def compute_goal_angle(x: float, y: float) -> float:
    """
    골대를 보는 각도를 계산합니다 (라디안).

    Args:
        x: x 좌표 (0-1)
        y: y 좌표 (0-1)

    Returns:
        골대 각도 (라디안)
    """
    # 골대의 폭 (정규화된 좌표, 골대 폭 7.32m)
    goal_y_min = 0.5 - (7.32 / 68.0) / 2
    goal_y_max = 0.5 + (7.32 / 68.0) / 2

    # 골대 양 끝점까지의 각도
    angle1 = np.arctan2(goal_y_min - y, 1.0 - x)
    angle2 = np.arctan2(goal_y_max - y, 1.0 - x)

    return abs(angle2 - angle1)


def split_train_val(
    df: pd.DataFrame, val_ratio: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    매치 단위로 학습/검증 데이터를 분할합니다.

    Args:
        df: 이벤트 DataFrame
        val_ratio: 검증 세트 비율
        random_state: 랜덤 시드

    Returns:
        (train_df, val_df) 튜플
    """
    unique_matches = df["matchId"].unique()

    np.random.seed(random_state)
    shuffled_matches = np.random.permutation(unique_matches)

    n_val = int(len(shuffled_matches) * val_ratio)
    val_matches = shuffled_matches[:n_val]
    train_matches = shuffled_matches[n_val:]

    train_df = df[df["matchId"].isin(train_matches)].copy()
    val_df = df[df["matchId"].isin(val_matches)].copy()

    return train_df, val_df


def ensure_dir(directory: str) -> None:
    """디렉토리가 존재하지 않으면 생성합니다."""
    os.makedirs(directory, exist_ok=True)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    YAML 설정 파일을 로드합니다.

    Args:
        config_path: 설정 파일 경로 (없으면 기본 경로 사용)

    Returns:
        설정 딕셔너리
    """
    if config_path is None:
        # 기본 경로: scripts 폴더의 상위 폴더에 있는 config.yaml
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "..", "config.yaml")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except ImportError:
        # PyYAML이 없으면 기본값 반환
        print("Warning: PyYAML not installed. Using default configuration.")
        return get_default_config()
    except FileNotFoundError:
        print(
            f"Warning: Config file not found at {config_path}. Using default configuration."
        )
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """기본 설정을 반환합니다."""
    return {
        "paths": {
            "data_dir": "data/wyscout",
            "processed_dir": "data/processed",
            "models_dir": "models",
            "results_dir": "data/vaep_results",
            "logs_dir": "logs",
        },
        "preprocessing": {
            "output_train": "vaep_train_events.csv",
            "output_eval": "vaep_eval_events_england.csv",
        },
        "training": {
            "hidden_dims": [128, 64],
            "dropout": 0.3,
            "epochs": 50,
            "batch_size": 512,
            "learning_rate": 0.001,
            "val_ratio": 0.2,
            "horizon": 10,
            "random_seed": 42,
            "early_stopping_patience": 5,
        },
        "evaluation": {"batch_size": 1024, "matches_file": "matches_England.json"},
        "logging": {"level": "INFO", "save_logs": True},
    }
