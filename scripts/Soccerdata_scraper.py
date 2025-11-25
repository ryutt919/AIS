# %% [markdown]
# ## 데이터 수집 진행 상황 관리
#
# 중단된 지점부터 재개할 수 있도록 진행 상황을 저장합니다.

# %%
import soccerdata as sd
from tqdm import tqdm


# 시즌
def league_scraper():
    season = "2017"  # 2017/18 시즌
    leagues = [
        "ENG-Premier League",  # 잉글랜드 프리미어리그
        "ESP-La Liga",  # 스페인 라리가
        "FRA-Ligue 1",  # 프랑스 리그 1
        "GER-Bundesliga",  # 독일 분데스리가
        "ITA-Serie A",  # 이탈리아 세리에 A
    ]

    # %%
    for league in leagues[2:]:
        print(f"{league} processing ===============================================")
        fb = sd.FBref(league, season)
        player_match_stats = fb.read_player_match_stats(stat_type="summary")

        # %%
        # 데이터 확인
        print(f"데이터 크기: {player_match_stats.shape}")
        print(f"\n컬럼: {list(player_match_stats.columns)}")
        player_match_stats.head()

        # %%
        # CSV 파일로 저장
        season_year = int(season)
        output_path = f"./data/player_match_stats/{league}-{season_year}-{str(season_year + 1)[-2:]}.csv"
        try:
            player_match_stats.to_csv(output_path, index=True, encoding="utf-8-sig")
            print(f"저장 완료: {output_path}")
        except Exception as e:
            print("저장 실패:", e)


league_scraper()
