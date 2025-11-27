"""
선수 매치 통계 데이터 전처리 스크립트 (간소화 버전)

목적:
- player_match_stats 데이터 정제 및 통합
- 선수별 시즌 통계 생성
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


def load_filtered_data(season="2017-18", data_dir="../data/player_match_stats"):
    """필터링된 데이터 로드"""
    file_path = Path(data_dir) / f"{season}_filtered.csv"

    # 3번째 줄부터 데이터, 2번째 줄이 실제 컬럼명
    df = pd.read_csv(file_path, skiprows=2)

    # 컬럼명 수동 설정
    col_names = [
        "league",
        "season",
        "game",
        "team",
        "player",
        "jersey_number",
        "nation",
        "pos",
        "age",
        "xG",
        "npxG",
        "xAG",
        "SCA",
        "GCA",
        "Carries",
        "PrgC",
        "game_id",
    ]

    if len(df.columns) == len(col_names):
        df.columns = col_names
    else:
        print(f"⚠ 컬럼 수 불일치: 예상 {len(col_names)}, 실제 {len(df.columns)}")
        return None

    print(f"✓ {season} 데이터 로드: {df.shape}")
    return df


def clean_data(df):
    """데이터 정제"""
    df_clean = df.copy()

    # 1. 숫자형 컬럼 변환
    numeric_cols = [
        "jersey_number",
        "xG",
        "npxG",
        "xAG",
        "SCA",
        "GCA",
        "Carries",
        "PrgC",
    ]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # 2. 결측치 처리
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)

    # 3. 포지션 정리
    if "pos" in df_clean.columns:
        df_clean["main_pos"] = df_clean["pos"].apply(
            lambda x: x.split(",")[0].strip() if pd.notna(x) else "Unknown"
        )

    print(f"✓ 데이터 정제 완료: {df_clean.shape}")
    return df_clean


def create_player_season_stats(df, season):
    """선수별 시즌 통계"""

    # 선수별 집계
    player_stats = (
        df.groupby(["league", "season", "player", "team", "nation", "main_pos"])
        .agg(
            {
                "game": "count",  # 경기 수
                "xG": "sum",
                "npxG": "sum",
                "xAG": "sum",
                "SCA": "sum",
                "GCA": "sum",
                "Carries": "sum",
                "PrgC": "sum",
            }
        )
        .reset_index()
    )

    player_stats = player_stats.rename(columns={"game": "matches_played"})

    # 경기당 평균
    for col in ["xG", "npxG", "xAG", "SCA", "GCA", "Carries", "PrgC"]:
        player_stats[f"{col}_per_90"] = (
            player_stats[col] / player_stats["matches_played"]
        ) * 90

    # Progressive carry 비율
    player_stats["progressive_carry_rate"] = np.where(
        player_stats["Carries"] > 0, player_stats["PrgC"] / player_stats["Carries"], 0
    )

    print(f"✓ 선수별 통계 생성: {player_stats.shape}")
    return player_stats


def create_position_benchmarks(df):
    """포지션별 벤치마크"""

    pos_stats = (
        df.groupby("main_pos")
        .agg(
            {
                "matches_played": "count",
                "xG": ["mean", "median", "std"],
                "npxG": ["mean", "median"],
                "xAG": ["mean", "median"],
                "SCA": ["mean", "median"],
                "GCA": ["mean", "median"],
            }
        )
        .round(3)
    )

    print(f"✓ 포지션별 벤치마크 생성")
    return pos_stats


def main():
    """메인 실행 함수"""

    print("=" * 60)
    print("선수 통계 데이터 전처리 시작")
    print("=" * 60)

    output_dir = Path("../data/processed")
    output_dir.mkdir(exist_ok=True)

    all_match_data = []
    all_season_stats = []

    # 시즌별 처리
    for season in ["2017-18", "2018-19"]:
        print(f"\n{'='*60}")
        print(f"시즌: {season}")
        print(f"{'='*60}")

        # 1. 로드
        df = load_filtered_data(season)
        if df is None:
            continue

        # 2. 정제
        df_clean = clean_data(df)

        # 3. 시즌 통계
        season_stats = create_player_season_stats(df_clean, season)

        # 4. 저장
        df_clean.to_csv(output_dir / f"match_level_{season}.csv", index=False)
        season_stats.to_csv(
            output_dir / f"player_season_stats_{season}.csv", index=False
        )

        all_match_data.append(df_clean)
        all_season_stats.append(season_stats)

    # 전체 통합
    if all_match_data:
        combined_match = pd.concat(all_match_data, ignore_index=True)
        combined_season = pd.concat(all_season_stats, ignore_index=True)

        combined_match.to_csv(output_dir / "match_level_all.csv", index=False)
        combined_season.to_csv(output_dir / "player_season_stats_all.csv", index=False)

        # 포지션별 벤치마크
        pos_bench = create_position_benchmarks(combined_season)
        pos_bench.to_csv(output_dir / "position_benchmarks.csv")

        print(f"\n{'='*60}")
        print("✅ 전처리 완료!")
        print(f"{'='*60}")
        print(f"- 전체 경기 데이터: {combined_match.shape}")
        print(f"- 선수 시즌 통계: {combined_season.shape}")
        print(f"- 저장 위치: {output_dir}")


if __name__ == "__main__":
    main()
