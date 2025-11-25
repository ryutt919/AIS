"""
Wyscout 이벤트 데이터를 VAEP 모델에 사용할 수 있는 형식으로 전처리합니다.

이 스크립트는:
1. wyscout 폴더의 모든 이벤트 데이터를 로드
2. England를 학습용과 평가용으로 분리
3. VAEP에 필요한 특징들을 추출하여 공통 스키마로 변환
4. data/processed 폴더에 저장
"""

import argparse
import logging
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import (
    setup_logger,
    load_json,
    load_wyscout_events,
    extract_position_features,
    extract_tags,
    is_goal,
    is_successful,
    compute_distance,
    compute_angle,
    compute_goal_distance,
    compute_goal_angle,
    ensure_dir,
    load_config,
)


def parse_args():
    """명령줄 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="Wyscout 데이터를 VAEP 형식으로 전처리"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/wyscout",
        help="Wyscout 데이터 디렉토리 경로",
    )
    parser.add_argument(
        "--output_dir", type=str, default="../data/processed", help="출력 디렉토리 경로"
    )
    parser.add_argument("--log_file", type=str, default=None, help="로그 파일 경로")
    return parser.parse_args()


def process_events(
    events_df: pd.DataFrame, league: str, logger: logging.Logger
) -> pd.DataFrame:
    """
    이벤트 데이터를 VAEP 스키마로 변환합니다.

    VAEP state representation을 위해 다음을 추출:
    - 이벤트 타입 및 하위 타입
    - 위치 정보 (시작/끝 좌표)
    - 팀 및 선수 정보
    - 성공 여부
    - 골 관련 특징 (거리, 각도)
    - 시간 정보

    Args:
        events_df: 원본 이벤트 DataFrame
        league: 리그 이름
        logger: 로거 객체

    Returns:
        전처리된 DataFrame
    """
    logger.info(f"Processing {len(events_df)} events from {league}")

    # 기본 컬럼 복사
    processed = events_df.copy()

    # 위치 정보 추출
    logger.info("Extracting position features...")
    positions = processed["positions"].apply(extract_position_features)
    processed["start_x"] = positions.apply(lambda x: x[0])
    processed["start_y"] = positions.apply(lambda x: x[1])
    processed["end_x"] = positions.apply(lambda x: x[2])
    processed["end_y"] = positions.apply(lambda x: x[3])

    # 태그 추출
    logger.info("Extracting tags...")
    processed["tags_list"] = processed["tags"].apply(extract_tags)
    processed["is_goal"] = processed["tags_list"].apply(is_goal).astype(int)
    processed["is_successful"] = processed["tags_list"].apply(is_successful).astype(int)

    # 이동 거리 및 방향
    logger.info("Computing movement features...")
    tqdm.pandas(desc="Distance", leave=False)
    processed["distance"] = processed.progress_apply(
        lambda row: compute_distance(
            row["start_x"], row["start_y"], row["end_x"], row["end_y"]
        ),
        axis=1,
    )
    tqdm.pandas(desc="Angle", leave=False)
    processed["angle"] = processed.progress_apply(
        lambda row: compute_angle(
            row["start_x"], row["start_y"], row["end_x"], row["end_y"]
        ),
        axis=1,
    )

    # 골까지의 거리와 각도
    logger.info("Computing goal-related features...")
    tqdm.pandas(desc="Goal distance", leave=False)
    processed["goal_distance"] = processed.progress_apply(
        lambda row: compute_goal_distance(row["start_x"], row["start_y"]), axis=1
    )
    tqdm.pandas(desc="Goal angle", leave=False)
    processed["goal_angle"] = processed.progress_apply(
        lambda row: compute_goal_angle(row["start_x"], row["start_y"]), axis=1
    )

    # 경기 시간 (초)
    logger.info("Processing time information...")
    processed["event_sec"] = processed["eventSec"].fillna(0)
    processed["period"] = (
        processed["matchPeriod"]
        .map({"1H": 1, "2H": 2, "E1": 3, "E2": 4, "P": 5})
        .fillna(0)
        .astype(int)
    )

    # 리그 정보 추가
    processed["league"] = league

    # 필요한 컬럼만 선택
    output_columns = [
        "id",
        "matchId",
        "playerId",
        "teamId",
        "league",
        "eventId",
        "eventName",
        "subEventId",
        "subEventName",
        "period",
        "event_sec",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "distance",
        "angle",
        "goal_distance",
        "goal_angle",
        "is_goal",
        "is_successful",
        "tags_list",
    ]

    processed = processed[output_columns].copy()

    logger.info(f"Processed {len(processed)} events")

    return processed


def load_all_matches(data_dir: str, logger: logging.Logger) -> pd.DataFrame:
    """
    모든 매치 정보를 로드하고 competitionId를 추가합니다.

    Args:
        data_dir: wyscout 데이터 디렉토리
        logger: 로거 객체

    Returns:
        매치 정보가 포함된 DataFrame
    """
    match_files = {
        "England": "matches_England.json",
        "Spain": "matches_Spain.json",
        "France": "matches_France.json",
        "Germany": "matches_Germany.json",
        "Italy": "matches_Italy.json",
        "European_Championship": "matches_European_Championship.json",
        "World_Cup": "matches_World_Cup.json",
    }

    all_matches = []
    for league, filename in match_files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            matches = load_json(filepath)
            for match in matches:
                match["league"] = league
            all_matches.extend(matches)
            logger.info(f"Loaded {len(matches)} matches from {league}")

    matches_df = pd.DataFrame(all_matches)
    logger.info(f"Total matches loaded: {len(matches_df)}")

    return matches_df


def main():
    """메인 실행 함수."""
    args = parse_args()

    # 로거 설정
    logger = setup_logger("preprocess_wyscout", args.log_file)
    logger.info("=" * 80)
    logger.info("Starting Wyscout data preprocessing for VAEP")
    logger.info("=" * 80)

    # 절대 경로 계산
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, args.data_dir))
    output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))

    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")

    # 출력 디렉토리 생성
    ensure_dir(output_dir)

    try:
        # 매치 정보 로드
        logger.info("\nLoading match information...")
        matches_df = load_all_matches(data_dir, logger)

        # 이벤트 파일 목록
        event_leagues = {
            "England": "events_England.json",
            "Spain": "events_Spain.json",
            "France": "events_France.json",
            "Germany": "events_Germany.json",
            "Italy": "events_Italy.json",
            "European_Championship": "events_European_Championship.json",
            "World_Cup": "events_World_Cup.json",
        }

        # 학습용 데이터 (England 제외)
        train_events = []

        # 평가용 데이터 (England만)
        eval_events_england = None

        logger.info("\n" + "=" * 80)
        logger.info("Processing events from each league")
        logger.info("=" * 80)

        for league, filename in tqdm(
            event_leagues.items(), desc="Processing leagues", unit="league"
        ):
            filepath = os.path.join(data_dir, filename)

            if not os.path.exists(filepath):
                logger.warning(f"File not found: {filepath}, skipping...")
                continue

            logger.info(f"\n{'=' * 40}")
            logger.info(f"Processing: {league}")
            logger.info(f"{'=' * 40}")

            # 이벤트 로드
            events_df = load_wyscout_events(data_dir, league)

            # 전처리
            processed_df = process_events(events_df, league, logger)

            # England는 평가용으로, 나머지는 학습용으로 분리
            if league == "England":
                eval_events_england = processed_df
                logger.info(f"Set {len(processed_df)} events for evaluation (England)")
            else:
                train_events.append(processed_df)
                logger.info(f"Added {len(processed_df)} events to training set")

        # 학습용 데이터 병합
        logger.info("\n" + "=" * 80)
        logger.info("Merging training data (excluding England)")
        logger.info("=" * 80)

        if train_events:
            train_df = pd.concat(train_events, ignore_index=True)
            logger.info(f"Total training events: {len(train_df)}")

            # 매치 ID로 정렬
            train_df = train_df.sort_values(
                ["matchId", "period", "event_sec"]
            ).reset_index(drop=True)

            # 저장
            train_output = os.path.join(output_dir, "vaep_train_events.csv")
            train_df.to_csv(train_output, index=False, encoding="utf-8")
            logger.info(f"Saved training data to: {train_output}")
            logger.info(f"  - Total events: {len(train_df)}")
            logger.info(f"  - Unique matches: {train_df['matchId'].nunique()}")
            logger.info(f"  - Unique players: {train_df['playerId'].nunique()}")
            logger.info(f"  - Leagues: {train_df['league'].unique().tolist()}")
        else:
            logger.warning("No training data collected!")

        # 평가용 데이터 저장
        if eval_events_england is not None:
            logger.info("\n" + "=" * 80)
            logger.info("Saving evaluation data (England)")
            logger.info("=" * 80)

            eval_events_england = eval_events_england.sort_values(
                ["matchId", "period", "event_sec"]
            ).reset_index(drop=True)

            eval_output = os.path.join(output_dir, "vaep_eval_events_england.csv")
            eval_events_england.to_csv(eval_output, index=False, encoding="utf-8")
            logger.info(f"Saved evaluation data to: {eval_output}")
            logger.info(f"  - Total events: {len(eval_events_england)}")
            logger.info(
                f"  - Unique matches: {eval_events_england['matchId'].nunique()}"
            )
            logger.info(
                f"  - Unique players: {eval_events_england['playerId'].nunique()}"
            )
        else:
            logger.warning("No evaluation data (England) found!")

        logger.info("\n" + "=" * 80)
        logger.info("Preprocessing completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
