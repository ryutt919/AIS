"""
Wyscout 이벤트 데이터를 Atomic SPADL 형식으로 전처리합니다.

이 스크립트는:
1. wyscout 폴더의 모든 이벤트 데이터를 로드
2. socceraction을 사용하여 SPADL로 변환
3. Atomic SPADL로 변환
4. England를 학습용과 평가용으로 분리
5. data/processed 폴더에 저장
"""

import argparse
import logging
import os
import sys
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

# socceraction imports
import socceraction.spadl as spadl
import socceraction.atomic.spadl as atomicspadl
from socceraction.data.wyscout import PublicWyscoutLoader
from socceraction.spadl.wyscout import convert_to_actions as wyscout_to_spadl

from utils import (
    setup_logger,
    ensure_dir,
    load_config,
)


def parse_args():
    """명령줄 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="Wyscout 데이터를 Atomic SPADL 형식으로 전처리"
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="디버그 모드: 5개 매치만 처리"
    )
    return parser.parse_args()


def get_all_game_ids(data_dir: str, logger: logging.Logger) -> pd.DataFrame:
    """
    모든 경기 ID를 로드합니다.

    Args:
        data_dir: wyscout 데이터 디렉토리
        logger: 로거 객체

    Returns:
        경기 정보가 포함된 DataFrame (game_id, league 컬럼)
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

    all_games = []
    for league, filename in match_files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            import json
            with open(filepath, 'r', encoding='utf-8') as f:
                matches = json.load(f)
            for match in matches:
                game_id = match.get("wyId") or match.get("matchId")
                if game_id:
                    all_games.append({"game_id": game_id, "league": league})
            logger.info(f"Loaded {len(matches)} matches from {league}")

    games_df = pd.DataFrame(all_games)
    logger.info(f"Total games loaded: {len(games_df)}")

    return games_df


def process_game_to_atomic_spadl(
    loader: PublicWyscoutLoader,
    game_id: int,
    league: str,
    logger: logging.Logger
) -> Optional[pd.DataFrame]:
    """
    단일 경기를 Atomic SPADL 형식으로 변환합니다.

    Args:
        loader: Wyscout 데이터 로더
        game_id: 경기 ID
        league: 리그 이름
        logger: 로거 객체

    Returns:
        Atomic SPADL 형식의 DataFrame 또는 None (오류 시)
    """
    try:
        # 1. 이벤트 로드
        events = loader.events(game_id)
        if events.empty:
            logger.warning(f"Game {game_id}: No events found")
            return None

        # 2. 팀 정보 가져오기 (홈팀 ID 필요)
        teams = loader.teams(game_id)
        if teams.empty:
            logger.warning(f"Game {game_id}: No teams found")
            return None
        
        # side 컬럼이 없으면 matches에서 직접 가져오기
        if "side" in teams.columns:
            home_team_id = teams[teams.side == "home"]["team_id"].iloc[0]
        else:
            # matches 파일에서 직접 홈팀 찾기
            try:
                import json
                # 모든 matches 파일에서 찾기
                match_files = [
                    "matches_England.json", "matches_Spain.json", "matches_France.json",
                    "matches_Germany.json", "matches_Italy.json",
                    "matches_European_Championship.json", "matches_World_Cup.json"
                ]
                home_team_id = None
                for match_file in match_files:
                    filepath = os.path.join(loader.root, match_file)
                    if os.path.exists(filepath):
                        with open(filepath, 'r', encoding='utf-8') as f:
                            matches = json.load(f)
                        for match in matches:
                            if match.get("wyId") == game_id or match.get("matchId") == game_id:
                                # teamsData에서 홈팀 찾기
                                teams_data = match.get("teamsData", {})
                                for team_id_str, team_data in teams_data.items():
                                    if team_data.get("side") == "home":
                                        home_team_id = int(team_id_str)
                                        break
                                if home_team_id:
                                    break
                    if home_team_id:
                        break
                
                if home_team_id is None:
                    # 폴백: 첫 번째 팀을 홈팀으로 가정
                    home_team_id = teams.iloc[0]["team_id"]
                    logger.warning(f"Game {game_id}: Could not find home team, using first team: {home_team_id}")
            except Exception as e:
                logger.warning(f"Game {game_id}: Error finding home team: {e}, using first team")
                home_team_id = teams.iloc[0]["team_id"]

        # 3. SPADL로 변환
        spadl_actions = wyscout_to_spadl(events, home_team_id)
        if spadl_actions.empty:
            logger.warning(f"Game {game_id}: No SPADL actions after conversion")
            return None

        # 4. Atomic SPADL로 변환
        atomic_actions = atomicspadl.convert_to_atomic(spadl_actions)
        if atomic_actions.empty:
            logger.warning(f"Game {game_id}: No atomic actions after conversion")
            return None

        # 5. 리그 정보 추가
        atomic_actions["league"] = league
        
        # Atomic SPADL 컬럼 확인 및 추가 정보 포함
        # Atomic SPADL 기본 컬럼: game_id, action_id, period_id, time_seconds, 
        # team_id, player_id, x, y, dx, dy, type_id, bodypart_id
        
        return atomic_actions

    except Exception as e:
        logger.error(f"Error processing game {game_id}: {str(e)}", exc_info=True)
        return None


def main():
    """메인 실행 함수."""
    args = parse_args()

    # 로거 설정
    logger = setup_logger("preprocess_wyscout_atomic", args.log_file)
    logger.info("=" * 80)
    logger.info("Starting Wyscout data preprocessing for Atomic SPADL")
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
        # Wyscout 로더 초기화
        logger.info("\nInitializing Wyscout loader...")
        loader = PublicWyscoutLoader(root=data_dir)

        # 로더의 인덱스에서 경기 목록 가져오기
        logger.info("\nLoading game information from loader...")
        try:
            # loader의 competitions와 games를 사용
            competitions = loader.competitions()
            logger.info(f"Found {len(competitions)} competitions")
            
            all_games = []
            for _, comp in competitions.iterrows():
                try:
                    games = loader.games(comp.competition_id, comp.season_id)
                    for _, game in games.iterrows():
                        all_games.append({
                            "game_id": game.game_id,
                            "league": comp.competition_name  # 또는 다른 매핑 필요
                        })
                    logger.info(f"  - {comp.competition_name}: {len(games)} games")
                except Exception as e:
                    logger.warning(f"  - Failed to load games for {comp.competition_name}: {e}")
                    continue
            
            games_df = pd.DataFrame(all_games)
            logger.info(f"Total games from loader: {len(games_df)}")
            
            # 리그 이름 매핑 (competition_name -> 우리 리그 이름)
            league_mapping = {
                "English first division": "England",
                "Spanish first division": "Spain",
                "French first division": "France",
                "German first division": "Germany",
                "Italian first division": "Italy",
                "European Championship": "European_Championship",
                "World Cup": "World_Cup",
            }
            # 매핑 적용
            games_df["league"] = games_df["league"].map(league_mapping).fillna(games_df["league"])
            
            # 매핑 결과 확인
            unmapped = games_df[~games_df["league"].isin(league_mapping.values())]["league"].unique()
            if len(unmapped) > 0:
                logger.warning(f"Unmapped competition names (will use original): {unmapped}")
            logger.info(f"League distribution: {games_df['league'].value_counts().to_dict()}")
            
        except Exception as e:
            logger.warning(f"Failed to use loader index, falling back to manual loading: {e}")
            # 폴백: 수동으로 경기 로드
            games_df = get_all_game_ids(data_dir, logger)

        # 디버그 모드 설정
        if args.debug:
            games_df = games_df.head(5)
            logger.warning("\n" + "=" * 80)
            logger.warning("DEBUG MODE ENABLED: Processing only 5 games")
            logger.warning("=" * 80 + "\n")

        # 학습용 데이터 (England 제외)
        train_actions = []

        # 평가용 데이터 (England만)
        eval_actions_england = None

        logger.info("\n" + "=" * 80)
        logger.info("Processing games and converting to Atomic SPADL")
        logger.info("=" * 80)

        # 리그별로 그룹화하여 처리
        for league in games_df["league"].unique():
            league_games = games_df[games_df["league"] == league]
            logger.info(f"\n{'=' * 40}")
            logger.info(f"Processing: {league} ({len(league_games)} games)")
            logger.info(f"{'=' * 40}")

            league_actions = []
            for _, game_row in tqdm(
                league_games.iterrows(),
                desc=f"Processing {league}",
                total=len(league_games),
                unit="game"
            ):
                game_id = game_row["game_id"]
                league_name = game_row["league"]

                atomic_actions = process_game_to_atomic_spadl(
                    loader, game_id, league_name, logger
                )

                if atomic_actions is not None and not atomic_actions.empty:
                    league_actions.append(atomic_actions)

            if league_actions:
                league_df = pd.concat(league_actions, ignore_index=True)
                logger.info(f"  - Processed {len(league_df)} atomic actions from {len(league_actions)} matches")

                # England는 평가용으로, 나머지는 학습용으로 분리
                if league == "England":
                    eval_actions_england = league_df
                    logger.info(f"Set {len(league_df)} atomic actions for evaluation (England)")
                else:
                    train_actions.append(league_df)
                    logger.info(f"Added {len(league_df)} atomic actions to training set")

        # 학습용 데이터 병합
        logger.info("\n" + "=" * 80)
        logger.info("Merging training data (excluding England)")
        logger.info("=" * 80)

        if train_actions:
            train_df = pd.concat(train_actions, ignore_index=True)
            logger.info(f"Total training atomic actions: {len(train_df)}")

            # 정렬
            train_df = train_df.sort_values(
                ["game_id", "period_id", "time_seconds", "action_id"]
            ).reset_index(drop=True)

            # 저장
            train_output = os.path.join(output_dir, "vaep_train_atomic_spadl.csv")
            train_df.to_csv(train_output, index=False, encoding="utf-8")
            logger.info(f"Saved training data to: {train_output}")
            logger.info(f"  - Total atomic actions: {len(train_df)}")
            logger.info(f"  - Unique games: {train_df['game_id'].nunique()}")
            logger.info(f"  - Unique players: {train_df['player_id'].nunique()}")
            logger.info(f"  - Leagues: {train_df['league'].unique().tolist()}")
            logger.info(f"  - Columns: {train_df.columns.tolist()}")
        else:
            logger.warning("No training data collected!")

        # 평가용 데이터 저장
        if eval_actions_england is not None:
            logger.info("\n" + "=" * 80)
            logger.info("Saving evaluation data (England)")
            logger.info("=" * 80)

            eval_actions_england = eval_actions_england.sort_values(
                ["game_id", "period_id", "time_seconds", "action_id"]
            ).reset_index(drop=True)

            eval_output = os.path.join(output_dir, "vaep_eval_atomic_spadl_england.csv")
            eval_actions_england.to_csv(eval_output, index=False, encoding="utf-8")
            logger.info(f"Saved evaluation data to: {eval_output}")
            logger.info(f"  - Total atomic actions: {len(eval_actions_england)}")
            logger.info(
                f"  - Unique games: {eval_actions_england['game_id'].nunique()}"
            )
            logger.info(
                f"  - Unique players: {eval_actions_england['player_id'].nunique()}"
            )
            logger.info(f"  - Columns: {eval_actions_england.columns.tolist()}")
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

