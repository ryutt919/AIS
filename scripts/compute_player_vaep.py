"""
학습된 VAEP 모델을 사용하여 England 이벤트 데이터를 평가하고
선수별 VAEP를 계산하는 스크립트

VAEP (Valuing Actions by Estimating Probabilities) 계산:
1. 각 이벤트 전후의 state value를 계산: V_t = P_score_t - P_concede_t
2. 이벤트의 VAEP: VAEP(a_t) = V_{t+1} - V_t
3. 선수별 집계:
   - 경기당 VAEP: 각 경기에서 선수의 모든 이벤트 VAEP 합산
   - 시즌 평균 VAEP: 시즌 전체 경기의 평균
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
from tqdm import tqdm

from train_vaep_model import VAEPModel, create_state_features
from utils import setup_logger, ensure_dir, load_json, load_config


def parse_args():
    """명령줄 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description="VAEP 모델을 사용한 선수 평가")
    parser.add_argument(
        "--input",
        type=str,
        default="../data/processed/vaep_eval_events_england.csv",
        help="평가할 이벤트 데이터 경로",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../data/models/vaep_model.pt",
        help="학습된 모델 경로",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="../data/models/vaep_config.json",
        help="모델 설정 파일 경로",
    )
    parser.add_argument(
        "--matches_path",
        type=str,
        default="../data/wyscout/matches_England.json",
        help="매치 정보 파일 경로",
    )
    parser.add_argument(
        "--players_path",
        type=str,
        default="../data/wyscout/players.json",
        help="선수 정보 파일 경로",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data/vaep_results",
        help="결과 저장 디렉토리",
    )
    parser.add_argument("--log_file", type=str, default=None, help="로그 파일 경로")
    return parser.parse_args()


def load_model(
    model_path: str, config_path: str, device: torch.device, logger: logging.Logger
) -> Tuple[VAEPModel, Dict]:
    """
    학습된 VAEP 모델을 로드합니다.

    Args:
        model_path: 모델 파일 경로
        config_path: 설정 파일 경로
        device: 디바이스
        logger: 로거 객체

    Returns:
        (model, config) 튜플
    """
    logger.info("Loading model configuration...")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    logger.info(f"  - Input dimension: {config['input_dim']}")
    logger.info(f"  - Hidden dimensions: {config['hidden_dims']}")
    logger.info(f"  - Horizon: {config['horizon']}")

    logger.info("Loading model weights...")
    model = VAEPModel(config["input_dim"], config["hidden_dims"])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    logger.info("Model loaded successfully")

    return model, config


def compute_state_values(
    model: VAEPModel, features: np.ndarray, device: torch.device, batch_size: int = 1024
) -> np.ndarray:
    """
    모든 이벤트의 state value를 계산합니다.

    VAEP 논문의 state value:
    V(s_t) = P(score | s_t) - P(concede | s_t)

    Args:
        model: VAEP 모델
        features: 특징 행렬 (n_samples, n_features)
        device: 디바이스
        batch_size: 배치 크기

    Returns:
        state values (n_samples,)
    """
    model.eval()
    values = []

    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch_features = torch.FloatTensor(features[i : i + batch_size]).to(device)

            # 모델 예측
            score_logits, concede_logits = model(batch_features)

            # 확률로 변환
            score_probs = torch.sigmoid(score_logits)
            concede_probs = torch.sigmoid(concede_logits)

            # V = P(score) - P(concede)
            batch_values = score_probs - concede_probs
            values.append(batch_values.cpu().numpy())

    return np.concatenate(values)


def compute_event_vaep(
    df: pd.DataFrame, state_values: np.ndarray, logger: logging.Logger
) -> pd.DataFrame:
    """
    각 이벤트의 VAEP를 계산합니다.

    VAEP 논문의 핵심 수식:
    VAEP(a_t) = V(s_{t+1}) - V(s_t)

    여기서:
    - a_t: 시간 t의 액션(이벤트)
    - s_t: 액션 전의 게임 상태
    - s_{t+1}: 액션 후의 게임 상태
    - V(s): 상태 s의 가치 (득점 확률 - 실점 확률)

    Args:
        df: 이벤트 DataFrame
        state_values: 각 이벤트의 state value
        logger: 로거 객체

    Returns:
        VAEP 컬럼이 추가된 DataFrame
    """
    logger.info("Computing VAEP for each event...")

    df = df.copy()
    df["state_value"] = state_values
    df["vaep"] = 0.0

    # 매치별로 처리
    for match_id in df["matchId"].unique():
        match_mask = df["matchId"] == match_id
        match_indices = df[match_mask].index

        # 각 이벤트의 VAEP = V_{t+1} - V_t
        for i in range(len(match_indices) - 1):
            curr_idx = match_indices[i]
            next_idx = match_indices[i + 1]

            # VAEP(action_t) = V_{t+1} - V_t
            df.loc[curr_idx, "vaep"] = (
                df.loc[next_idx, "state_value"] - df.loc[curr_idx, "state_value"]
            )

        # 마지막 이벤트는 VAEP = 0 (다음 상태가 없음)
        df.loc[match_indices[-1], "vaep"] = 0.0

    logger.info(f"VAEP statistics:")
    logger.info(f"  - Mean: {df['vaep'].mean():.6f}")
    logger.info(f"  - Std: {df['vaep'].std():.6f}")
    logger.info(f"  - Min: {df['vaep'].min():.6f}")
    logger.info(f"  - Max: {df['vaep'].max():.6f}")

    return df


def load_player_roles(
    players_path: str, logger: logging.Logger
) -> Dict[int, str]:
    """
    선수 정보에서 포지션(role)을 로드합니다.

    Args:
        players_path: players.json 파일 경로
        logger: 로거 객체

    Returns:
        {playerId: roleCode} 딕셔너리
    """
    import json
    
    logger.info("Loading player roles...")
    
    with open(players_path, 'r', encoding='utf-8') as f:
        players_data = json.load(f)
    
    player_roles = {}
    for player in players_data:
        player_id = player.get('wyId')
        role = player.get('role', {})
        role_code = role.get('code2', 'UNKNOWN')
        player_roles[player_id] = role_code
    
    logger.info(f"Loaded roles for {len(player_roles)} players")
    gk_count = sum(1 for role in player_roles.values() if role == 'GK')
    logger.info(f"  - Goalkeepers: {gk_count}")
    
    return player_roles


def extract_player_minutes(
    matches_data: List[Dict], logger: logging.Logger
) -> pd.DataFrame:
    """
    매치 데이터에서 선수별 출전 시간을 추출합니다.
    
    이벤트 데이터의 matchPeriod를 기반으로 실제 출전 시간을 추정합니다.

    Args:
        matches_data: 매치 정보 리스트
        logger: 로거 객체

    Returns:
        (matchId, playerId, minutes_played) DataFrame
    """
    logger.info("Extracting player minutes from matches...")

    player_minutes = []

    for match in matches_data:
        match_id = match["wyId"]
        teams_data = match.get("teamsData", {})

        for team_id, team_info in teams_data.items():
            formation = team_info.get("formation", {})

            # 선발 라인업
            lineup = formation.get("lineup", [])
            for player_info in lineup:
                player_id = player_info["playerId"]
                # 선발 출전은 90분으로 가정 (교체 정보가 있으면 조정 가능)
                player_minutes.append(
                    {
                        "matchId": match_id,
                        "playerId": player_id,
                        "teamId": int(team_id),
                        "minutes_played": 90.0,  # 기본값
                    }
                )

            # 벤치 (출전하지 않은 선수는 0분)
            bench = formation.get("bench", [])
            for player_info in bench:
                player_id = player_info["playerId"]
                # 실제로는 교체 시간을 확인해야 하지만, 여기서는 간소화
                player_minutes.append(
                    {
                        "matchId": match_id,
                        "playerId": player_id,
                        "teamId": int(team_id),
                        "minutes_played": 0.0,
                    }
                )

    df = pd.DataFrame(player_minutes)
    logger.info(f"Extracted minutes for {len(df)} player-match records")

    return df

    """
    선수-경기별 VAEP를 계산합니다.
    (이벤트는 모두 포함, 골키퍼 선수의 집계만 제외)

    Args:
        df: VAEP가 계산된 이벤트 DataFrame
        player_minutes_df: 선수별 출전 시간 DataFrame
        player_roles: 선수별 포지션 딕셔너리
        logger: 로거 객체

    Returns:
        선수-경기별 VAEP DataFrame
    """
    logger.info("Computing player-match VAEP...")
    # 선수-경기별 VAEP 합산 (이벤트는 모두 포함)
    player_match_vaep = (
        df.groupby(["playerId", "matchId", "teamId"])
        .agg({"vaep": "sum", "id": "count"})  # 이벤트 수
        .reset_index()
    )

    player_match_vaep.rename(columns={"id": "num_events"}, inplace=True)

    # 이벤트 기반으로 실제 출전 시간 추정
    player_periods = df.groupby(["matchId", "playerId"])["period"].apply(lambda x: x.unique().tolist()).reset_index()
    player_periods['estimated_minutes'] = player_periods['period'].apply(lambda periods: len(periods) * 45.0)  # 각 피리어드는 45분

    # 출전 시간 병합 (이벤트 기반 추정 우선)
    player_match_vaep = player_match_vaep.merge(
        player_periods[["matchId", "playerId", "estimated_minutes"]],
        on=["matchId", "playerId"],
        how="left",
    )

    # 매치 데이터의 출전 시간도 병합 (백업)
    player_match_vaep = player_match_vaep.merge(
        player_minutes_df[["matchId", "playerId", "minutes_played"]],
        on=["matchId", "playerId"],
        how="left",
    )

    # 이벤트 기반 추정이 있으면 사용, 없으면 매치 데이터 사용, 둘 다 없으면 90분
    player_match_vaep["minutes_played"] = player_match_vaep["estimated_minutes"].fillna(
        player_match_vaep["minutes_played"]
    ).fillna(90.0)

    player_match_vaep.drop(columns=["estimated_minutes"], inplace=True)

    # VAEP per 90 minutes 계산
    player_match_vaep["vaep_per90"] = (
        player_match_vaep["vaep"]
        * 90.0
        / player_match_vaep["minutes_played"].replace(0, 1)
    )

    # 골키퍼 선수 집계만 제외
    player_match_vaep['role'] = player_match_vaep['playerId'].map(player_roles)
    n_before = len(player_match_vaep)
    player_match_vaep = player_match_vaep[player_match_vaep['role'] != 'GK'].copy()
    n_after = len(player_match_vaep)
    logger.info(f"Excluded {n_before-n_after} goalkeeper player-match rows from aggregation.")
    player_match_vaep.drop(columns=['role'], inplace=True)

    logger.info(f"Computed VAEP for {len(player_match_vaep)} player-match combinations (goalkeeper players excluded)")

    return player_match_vaep

def compute_player_season_vaep(
    player_match_vaep: pd.DataFrame, logger: logging.Logger
) -> pd.DataFrame:
    """
    선수별 시즌 평균 VAEP를 계산합니다.

    Args:
        player_match_vaep: 선수-경기별 VAEP DataFrame
        logger: 로거 객체

    Returns:
        선수별 시즌 VAEP DataFrame
    """
    logger.info("Computing player season VAEP...")

    # 선수별 집계
    player_season_vaep = (
        player_match_vaep.groupby("playerId")
        .agg(
            {
                "matchId": "count",  # 출전 경기 수
                "vaep": "sum",  # 시즌 총 VAEP
                "vaep_per90": "mean",  # 경기당 평균 VAEP per 90
                "minutes_played": "sum",  # 총 출전 시간
                "num_events": "sum",  # 총 이벤트 수
            }
        )
        .reset_index()
    )

    player_season_vaep.rename(
        columns={
            "matchId": "matches_played",
            "vaep": "season_vaep_total",
            "vaep_per90": "season_vaep_per90_avg",
        },
        inplace=True,
    )

    # 경기당 평균 VAEP 계산
    player_season_vaep["season_vaep_per_match"] = (
        player_season_vaep["season_vaep_total"] / player_season_vaep["matches_played"]
    )

    # 정렬 (시즌 VAEP per 90 기준 내림차순)
    player_season_vaep = player_season_vaep.sort_values(
        "season_vaep_per90_avg", ascending=False
    ).reset_index(drop=True)

    logger.info(f"Computed season VAEP for {len(player_season_vaep)} players")
    logger.info(f"\nTop 5 players by VAEP per 90:")
    for idx, row in player_season_vaep.head(5).iterrows():
        logger.info(
            f"  {idx+1}. Player {int(row['playerId'])}: "
            f"{row['season_vaep_per90_avg']:.6f} VAEP/90 "
            f"({int(row['matches_played'])} matches)"
        )

    return player_season_vaep


def main():
    """메인 실행 함수."""
    args = parse_args()

    # 로거 설정
    logger = setup_logger("compute_player_vaep", args.log_file)
    logger.info("=" * 80)
    logger.info("Computing Player VAEP")
    logger.info("=" * 80)

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 절대 경로 계산
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.abspath(os.path.join(script_dir, args.input))
    model_path = os.path.abspath(os.path.join(script_dir, args.model_path))
    config_path = os.path.abspath(os.path.join(script_dir, args.config_path))
    matches_path = os.path.abspath(os.path.join(script_dir, args.matches_path))
    players_path = os.path.abspath(os.path.join(script_dir, args.players_path))
    output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))

    logger.info(f"Input: {input_path}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Matches: {matches_path}")
    logger.info(f"Players: {players_path}")
    logger.info(f"Output directory: {output_dir}")

    ensure_dir(output_dir)

    try:
        # 모델 로드
        logger.info("\n" + "=" * 80)
        logger.info("Loading model...")
        logger.info("=" * 80)

        model, config = load_model(model_path, config_path, device, logger)

        # 데이터 로드
        logger.info("\n" + "=" * 80)
        logger.info("Loading evaluation data...")
        logger.info("=" * 80)

        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} events")
        logger.info(f"  - Unique matches: {df['matchId'].nunique()}")
        logger.info(f"  - Unique players: {df['playerId'].nunique()}")

        # 매치 정보 로드
        logger.info("\n" + "=" * 80)
        logger.info("Loading match information...")
        logger.info("=" * 80)

        matches_data = load_json(matches_path)
        logger.info(f"Loaded {len(matches_data)} matches")

        player_minutes_df = extract_player_minutes(matches_data, logger)
        
        # 선수 role 로드
        logger.info("\n" + "=" * 80)
        logger.info("Loading player roles...")
        logger.info("=" * 80)
        
        player_roles = load_player_roles(players_path, logger)

        # State 특징 생성
        logger.info("\n" + "=" * 80)
        logger.info("Creating state features...")
        logger.info("=" * 80)

        features, feature_names = create_state_features(df, logger)

        # State values 계산
        logger.info("\n" + "=" * 80)
        logger.info("Computing state values...")
        logger.info("=" * 80)

        state_values = compute_state_values(model, features, device)
        logger.info(f"Computed state values for {len(state_values)} events")
        logger.info(f"  - Mean: {state_values.mean():.6f}")
        logger.info(f"  - Std: {state_values.std():.6f}")

        # 이벤트별 VAEP 계산
        logger.info("\n" + "=" * 80)
        logger.info("Computing event VAEP...")
        logger.info("=" * 80)

        df_with_vaep = compute_event_vaep(df, state_values, logger)

        # 선수-경기별 VAEP 계산
        logger.info("\n" + "=" * 80)
        logger.info("Computing player-match VAEP...")
        logger.info("=" * 80)

        player_match_vaep = compute_player_match_vaep(
            df_with_vaep, player_minutes_df, player_roles, logger
        )

        # 저장
        player_match_output = os.path.join(output_dir, "player_match_vaep_england.csv")
        player_match_vaep.to_csv(player_match_output, index=False, encoding="utf-8")
        logger.info(f"Saved player-match VAEP to: {player_match_output}")

        # 선수별 시즌 VAEP 계산
        logger.info("\n" + "=" * 80)
        logger.info("Computing player season VAEP...")
        logger.info("=" * 80)

        player_season_vaep = compute_player_season_vaep(player_match_vaep, logger)

        # 저장
        player_season_output = os.path.join(
            output_dir, "player_season_vaep_england.csv"
        )
        player_season_vaep.to_csv(player_season_output, index=False, encoding="utf-8")
        logger.info(f"Saved player season VAEP to: {player_season_output}")

        logger.info("\n" + "=" * 80)
        logger.info("VAEP computation completed successfully!")
        logger.info("=" * 80)
        logger.info(f"\nOutput files:")
        logger.info(f"  1. {player_match_output}")
        logger.info(f"  2. {player_season_output}")

    except Exception as e:
        logger.error(f"Error during VAEP computation: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
