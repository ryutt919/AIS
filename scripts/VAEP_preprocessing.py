VAEP_preprocessing.py
# ## 1. 라이브러리 및 데이터 로드

# %%
import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import warnings
import os

# 경로 설정
data_dir = Path("./data/wyscout")

# matplotlib 캐시 디렉터리를 임시 폴더로 설정 (권한 문제 회피)
import tempfile

temp_cache = tempfile.gettempdir()
os.environ["MPLCONFIGDIR"] = temp_cache

# matplotlib 폰트 캐시 완전 삭제 (안전하게)
import matplotlib

cache_dir = matplotlib.get_cachedir()
import shutil

# %%
# 리그 선택
league = "England"

# 이벤트 데이터 로드
with open(data_dir / f"events_{league}.json", "r") as f:
    events_raw = json.load(f)

# 매치 데이터 로드
with open(data_dir / f"matches_{league}.json", "r") as f:
    matches_raw = json.load(f)

print(f"로드된 이벤트 수: {len(events_raw):,}")
print(f"로드된 경기 수: {len(matches_raw):,}")

# %% [markdown]
# ## 2. 데이터 전처리

# %%
# 이벤트 데이터를 DataFrame으로 변환
events_df = pd.DataFrame(events_raw)
events_df.head()

# %%

# 위치 데이터 추출
events_df["start_x"] = events_df["positions"].apply(
    lambda x: x[0]["x"] if len(x) > 0 else None
)
events_df["start_y"] = events_df["positions"].apply(
    lambda x: x[0]["y"] if len(x) > 0 else None
)
events_df["end_x"] = events_df["positions"].apply(
    lambda x: x[1]["x"] if len(x) > 1 else None
)
events_df["end_y"] = events_df["positions"].apply(
    lambda x: x[1]["y"] if len(x) > 1 else None
)

# 태그 처리 - 여러 태그를 리스트로
events_df["tag_ids"] = events_df["tags"].apply(
    lambda x: [tag["id"] for tag in x] if isinstance(x, list) else []
)


# %%
# 경기 내 시간 순서 정렬
events_df = events_df.sort_values(["matchId", "matchPeriod", "eventSec"]).reset_index(
    drop=True
)

events_df.head()

# %%
# 매치 데이터 처리 - 각 경기의 홈/어웨이 팀 정보 (확장 버전)
match_info = []
for match in matches_raw:
    teams = match["teamsData"]
    home_team = [t for t in teams.values() if t["side"] == "home"][0]
    away_team = [t for t in teams.values() if t["side"] == "away"][0]

    match_info.append(
        {
            "matchId": match["wyId"],
            "home_team_id": home_team["teamId"],
            "away_team_id": away_team["teamId"],
            "home_score": home_team["score"],
            "away_score": away_team["score"],
            "home_score_ht": home_team.get("scoreHT", 0),  # 전반전 스코어
            "away_score_ht": away_team.get("scoreHT", 0),
            "winner": match["winner"],
        }
    )

matches_df = pd.DataFrame(match_info)

# 교체 정보를 별도 DataFrame으로 저장 (안전하게)
substitutions = []
for match in matches_raw:
    match_id = match["wyId"]
    teams = match["teamsData"]

    for team_data in teams.values():
        team_id = team_data["teamId"]
        formation = team_data.get("formation", {})

        # formation이 딕셔너리가 아니면 건너뛰기
        if not isinstance(formation, dict):
            continue

        subs = formation.get("substitutions", [])

        # substitutions가 리스트인지 확인
        if not isinstance(subs, list):
            continue

        for sub in subs:
            # sub가 딕셔너리인지 확인
            if isinstance(sub, dict):
                substitutions.append(
                    {
                        "matchId": match_id,
                        "teamId": team_id,
                        "playerIn": sub.get("playerIn", 0),
                        "playerOut": sub.get("playerOut", 0),
                        "minute": sub.get("minute", 0),
                    }
                )

substitutions_df = pd.DataFrame(substitutions)

print(f"경기 수: {len(matches_df)}")
print(f"총 교체 수: {len(substitutions_df)}")
print("\n매치 정보:")
display(matches_df.head())
print("\n교체 정보:")
display(substitutions_df.head())


# %% [markdown]
# ### 추가 특징: 현재 스코어, 교체 정보

# %% [markdown]
# ## 3. VAEP 특징 생성
#
# 각 액션에 대해 다음 특징들을 추출합니다:
# - 액션 타입 (패스, 슛, 드리블 등)
# - 위치 정보 (x, y 좌표)
# - 팀 정보
# - 시간 정보
# - 게임 상태

# %%
# 매치 정보 병합
events_df = events_df.merge(matches_df, on="matchId", how="left")

# 팀이 홈인지 어웨이인지 판단
events_df["is_home_team"] = events_df["teamId"] == events_df["home_team_id"]

# 상대 팀 ID
events_df["opponent_team_id"] = events_df.apply(
    lambda row: row["away_team_id"] if row["is_home_team"] else row["home_team_id"],
    axis=1,
)

print(f"✓ 매치 정보 병합 완료")


# %%
# 경기 시간 구간 (초 단위를 분 단위로)
period_map = {"1H": 0, "2H": 45, "E1": 90, "E2": 105, "P": 120}
events_df["game_minute"] = events_df.apply(
    lambda row: period_map.get(row["matchPeriod"], 0) + row["eventSec"] / 60, axis=1
)

# 경기 구간 (전반/후반/연장 등)
events_df["period_code"] = events_df["matchPeriod"].map(period_map)

# %%
from tqdm import tqdm


# 2. 교체 관련 특징 추가
def add_substitution_features(events_df, substitutions_df):
    """교체 정보 기반 특징 추가"""
    events_df = events_df.copy()
    events_df["subs_used"] = 0  # 해당 시점까지 사용한 교체 수
    events_df["is_subbed_player"] = (
        0  # 교체된 선수가 한 이벤트인지 (1=교체되어 들어온 선수, 0=아님)
    )

    for match_id in tqdm(events_df["matchId"].unique(), desc="Processing matches"):
        match_mask = events_df["matchId"] == match_id
        match_subs = substitutions_df[substitutions_df["matchId"] == match_id]

        for idx in events_df[match_mask].index:
            team_id = events_df.loc[idx, "teamId"]
            player_id = events_df.loc[idx, "playerId"]
            game_minute = events_df.loc[idx, "game_minute"]

            # 해당 팀의 교체 정보
            team_subs = match_subs[match_subs["teamId"] == team_id]

            # 이 시점까지 일어난 교체 수
            subs_before = team_subs[team_subs["minute"] <= game_minute]
            events_df.loc[idx, "subs_used"] = len(subs_before)

            # 이 선수가 교체되어 들어온 선수인지 확인
            if player_id in subs_before["playerIn"].values:
                events_df.loc[idx, "is_subbed_player"] = 1

    return events_df


print("교체 특징 계산 중...")
events_df = add_substitution_features(events_df, substitutions_df)
print(f"✓ 교체 특징 추가 완료")
print(f"최대 교체 수: {events_df['subs_used'].max()}")
print(f"교체 선수 이벤트 수: {events_df['is_subbed_player'].sum():,}")


# %%

events_df.head()


# %%
# 중요 태그 확인
def has_tag(tag_list, tag_id):
    return tag_id in tag_list


events_df["is_goal"] = events_df["tag_ids"].apply(lambda x: has_tag(x, 101))
events_df["is_assist"] = events_df["tag_ids"].apply(lambda x: has_tag(x, 301))
events_df["is_key_pass"] = events_df["tag_ids"].apply(lambda x: has_tag(x, 302))
events_df["is_counter_attack"] = events_df["tag_ids"].apply(lambda x: has_tag(x, 1901))

print(f"골: {events_df['is_goal'].sum()}")
print(f"어시스트: {events_df['is_assist'].sum()}")
print(f"키 패스: {events_df['is_key_pass'].sum()}")

# %%
# 골 골대 거리 계산 (정규화된 좌표: 0-100 → 실제 필드 크기로 변환)
# 필드 크기: 길이 105m, 너비 68m
FIELD_LENGTH = 105  # meters
FIELD_WIDTH = 68  # meters


def distance_to_goal(x, y, attacking_direction="right"):
    """골대까지의 유클리드 거리 계산 (미터 단위)"""
    # 정규화된 좌표(0-100)를 실제 미터로 변환
    x_meters = (x / 100) * FIELD_LENGTH
    y_meters = (y / 100) * FIELD_WIDTH

    if attacking_direction == "right":
        goal_x, goal_y = FIELD_LENGTH, FIELD_WIDTH / 2
    else:
        goal_x, goal_y = 0, FIELD_WIDTH / 2

    return np.sqrt((x_meters - goal_x) ** 2 + (y_meters - goal_y) ** 2)


# 공격 방향 판단 (1H는 홈팀이 오른쪽 공격, 2H는 반대)
def get_attacking_direction(is_home, period):
    if period == "1H":
        return "right" if is_home else "left"
    else:  # 2H, E1, E2, P
        return "left" if is_home else "right"


events_df["attacking_direction"] = events_df.apply(
    lambda row: get_attacking_direction(row["is_home_team"], row["matchPeriod"]), axis=1
)

events_df["distance_to_goal"] = events_df.apply(
    lambda row: (
        distance_to_goal(row["start_x"], row["start_y"], row["attacking_direction"])
        if pd.notna(row["start_x"])
        else None
    ),
    axis=1,
)


# %% [markdown]
# ## 4. 게임 상태 특징 생성
#
# 각 액션 시점의 게임 상태를 나타내는 특징들

# %%
# 이전/다음 액션 정보 추가 (시간 윈도우)
events_df["prev_event"] = events_df.groupby("matchId")["eventName"].shift(1)
events_df["next_event"] = events_df.groupby("matchId")["eventName"].shift(-1)

events_df["prev_team"] = events_df.groupby("matchId")["teamId"].shift(1)
events_df["next_team"] = events_df.groupby("matchId")["teamId"].shift(-1)

# 시간 차이
events_df["time_diff"] = events_df.groupby("matchId")["eventSec"].diff()

# 팀이 공을 소유하고 있는지
events_df["team_possession_change"] = events_df["teamId"] != events_df["prev_team"]

# %% [markdown]
# ## 5. 라벨 생성: 득점/실점 여부
#
# 각 액션 이후 N초 내에 득점/실점이 발생했는지 확인 (보통 10초)

# %%
# 득점 이벤트 찾기
goal_events = events_df[events_df["is_goal"]].copy()
print(f"총 골 수: {len(goal_events)}")

# 시간 윈도우 설정 (초)
TIME_WINDOW = 10


def label_goals_in_window(df, window=10):
    """각 액션 이후 window초 내에 득점/실점 라벨링"""
    df = df.copy()
    df["scores"] = False  # 이 팀이 득점
    df["concedes"] = False  # 이 팀이 실점

    for match_id in df["matchId"].unique():
        match_events = df[df["matchId"] == match_id].copy()
        goal_indices = match_events[match_events["is_goal"]].index

        for goal_idx in goal_indices:
            goal_time = match_events.loc[goal_idx, "eventSec"]
            goal_period = match_events.loc[goal_idx, "matchPeriod"]
            goal_team = match_events.loc[goal_idx, "teamId"]

            # 같은 피리어드에서 window 초 이전 이벤트들 찾기
            window_mask = (
                (match_events["matchPeriod"] == goal_period)
                & (match_events["eventSec"] >= goal_time - window)
                & (match_events["eventSec"] < goal_time)
            )

            window_indices = match_events[window_mask].index

            # 득점한 팀의 액션은 'scores', 상대팀은 'concedes'
            for idx in window_indices:
                if match_events.loc[idx, "teamId"] == goal_team:
                    df.loc[idx, "scores"] = True
                else:
                    df.loc[idx, "concedes"] = True

    return df


# 라벨링 수행 (시간이 걸릴 수 있음)
print("득점/실점 라벨링 중...")
events_df = label_goals_in_window(events_df, TIME_WINDOW)

print(f"득점으로 이어진 액션: {events_df['scores'].sum():,}")
print(f"실점으로 이어진 액션: {events_df['concedes'].sum():,}")

# %% [markdown]
# ## 6. 특징 벡터 준비
#
# 머신러닝 모델을 위한 특징 선택 및 인코딩

# %%
# 특징 선택 (확장 버전)
feature_columns = [
    "eventId",
    "subEventId",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "distance_to_goal",
    "is_home_team",
    "period_code",
    "game_minute",
    "team_possession_change",
    "time_diff",
    # 새로 추가된 특징들
    "score_diff",  # 현재 스코어 차이
    "home_score_current",  # 현재 홈 스코어
    "away_score_current",  # 현재 어웨이 스코어
    "subs_used",  # 사용한 교체 수
    "is_subbed_player",  # 교체되어 들어온 선수의 이벤트인지
]

# 결측치 처리
vaep_df = events_df[
    feature_columns + ["scores", "concedes", "matchId", "teamId"]
].copy()
vaep_df = vaep_df.fillna(0)

print(f"VAEP 데이터셋 크기: {vaep_df.shape}")
print(f"\n특징 개수: {len(feature_columns)}")
print(
    f"추가된 특징: score_diff, home_score_current, away_score_current, subs_used, is_subbed_player"
)
vaep_df.head()


# %%
# 새로운 특징들의 기초 통계
print("\n=== 새로 추가된 특징 통계 ===")
print(f"\n스코어 차이 분포:")
print(vaep_df["score_diff"].value_counts().sort_index().head(10))

print(f"\n교체 사용 분포:")
print(vaep_df["subs_used"].value_counts().sort_index())

print(f"\n교체 선수 이벤트:")
print(vaep_df["is_subbed_player"].value_counts())
print(f"교체 선수 비율: {vaep_df['is_subbed_player'].mean():.2%}")


# %%
# 클래스 불균형 확인
print("득점 라벨 분포:")
print(vaep_df["scores"].value_counts())
print(f"\n득점 비율: {vaep_df['scores'].mean():.4%}")

print("\n실점 라벨 분포:")
print(vaep_df["concedes"].value_counts())
print(f"실점 비율: {vaep_df['concedes'].mean():.4%}")

# %% [markdown]
# ## 7. 데이터 분할 및 저장

# %%
# Train/Test 분할 (경기 단위로)
from sklearn.model_selection import train_test_split

unique_matches = vaep_df["matchId"].unique()
train_matches, test_matches = train_test_split(
    unique_matches, test_size=0.2, random_state=42
)

train_df = vaep_df[vaep_df["matchId"].isin(train_matches)]
test_df = vaep_df[vaep_df["matchId"].isin(test_matches)]

print(f"훈련 경기 수: {len(train_matches)}")
print(f"테스트 경기 수: {len(test_matches)}")
print(f"\n훈련 이벤트 수: {len(train_df):,}")
print(f"테스트 이벤트 수: {len(test_df):,}")

# %%
# 데이터 저장
output_dir = Path("../data/VAEP")
output_dir.mkdir(exist_ok=True)

train_df.to_csv(output_dir / f"vaep_train_{league}.csv", index=False)
test_df.to_csv(output_dir / f"vaep_test_{league}.csv", index=False)
vaep_df.to_csv(output_dir / f"vaep_full_{league}.csv", index=False)

print(f"데이터 저장 완료: {output_dir}")

# %% [markdown]
# ## 8. 기본 통계 분석

# %%
# 이벤트 타입별 득점 기여도
event_stats = (
    events_df.groupby("event_label")
    .agg({"scores": ["sum", "mean"], "concedes": ["sum", "mean"], "eventId": "count"})
    .round(4)
)

event_stats.columns = ["득점_합", "득점_비율", "실점_합", "실점_비율", "총_횟수"]
event_stats = event_stats.sort_values("득점_비율", ascending=False)

print("이벤트 타입별 득점 기여도:")
event_stats

# %%
print(f"현재 사용 중인 폰트: {plt.rcParams['font.family']}")
print(f"Sans-serif 폰트 목록: {plt.rcParams['font.sans-serif']}")


# %%

# 위치별 득점 확률 시각화
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 득점으로 이어진 액션의 시작 위치
scoring_actions = events_df[events_df["scores"]]
axes[0].hexbin(
    scoring_actions["start_x"],
    scoring_actions["start_y"],
    gridsize=20,
    cmap="Reds",
    mincnt=1,
)
axes[0].set_title("득점으로 이어진 액션의 위치", fontsize=14, fontweight="bold")
axes[0].set_xlabel("X 좌표")
axes[0].set_ylabel("Y 좌표")
axes[0].set_xlim(0, 100)
axes[0].set_ylim(0, 100)

# 모든 액션의 분포
axes[1].hexbin(
    events_df["start_x"], events_df["start_y"], gridsize=20, cmap="Blues", mincnt=1
)
axes[1].set_title("전체 액션의 위치", fontsize=14, fontweight="bold")
axes[1].set_xlabel("X 좌표")
axes[1].set_ylabel("Y 좌표")
axes[1].set_xlim(0, 100)
axes[1].set_ylim(0, 100)

plt.tight_layout()
plt.show()

# %%
# 거리에 따른 득점 확률
distance_bins = pd.cut(events_df["distance_to_goal"], bins=10)
distance_stats = events_df.groupby(distance_bins, observed=False)["scores"].agg(
    ["sum", "count", "mean"]
)
distance_stats.columns = ["득점_수", "총_액션", "득점_확률"]

plt.figure(figsize=(12, 6))
distance_stats["득점_확률"].plot(kind="bar", color="steelblue")
plt.title("골대 거리별 득점 확률", fontsize=14, fontweight="bold")
plt.xlabel("골대까지 거리")
plt.ylabel("득점 확률")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

print("\n거리별 득점 통계:")
distance_stats

# %% [markdown]
# ## 다음 단계
#
# 1. **모델 학습**: XGBoost, Random Forest 등을 사용하여 득점/실점 확률 예측 모델 학습
# 2. **VAEP 값 계산**: 각 액션의 VAEP = ΔP(득점) - ΔP(실점)
# 3. **선수 평가**: 선수별 누적 VAEP 값으로 기여도 평가
# 4. **시각화**: 선수/팀별 VAEP 분포, 시간대별 변화 등
