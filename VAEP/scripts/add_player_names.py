"""선수 이름 추가 및 최종 리포트 생성"""
import pandas as pd
import numpy as np
from socceraction.data.wyscout import PublicWyscoutLoader

# VAEP 데이터 로드
print("=== VAEP 데이터 로드 중 ===")
player_game_vaep = pd.read_csv('data/vaep_results/player_game_vaep_atomic.csv')

# Wyscout loader로 선수 정보 추가
try:
    loader = PublicWyscoutLoader(root='statsbomb/open-data/data')
    players = loader.players()
    
    # 선수 이름 매핑
    if 'player_id' in players.columns and 'player_name' in players.columns:
        player_names = players[['player_id', 'player_name']].drop_duplicates()
        player_game_vaep = player_game_vaep.merge(player_names, on='player_id', how='left')
        print(f"선수 정보 추가 완료: {player_names['player_id'].nunique()}명")
    
    # 업데이트된 파일 저장
    player_game_vaep.to_csv('data/vaep_results/player_game_vaep_atomic.csv', index=False)
    print("업데이트된 경기별 VAEP 저장 완료\n")
    
except Exception as e:
    print(f"선수 정보 추가 실패: {e}\n")

# 선수별 시즌 집계
print("=== 선수별 시즌 집계 ===")
player_season = player_game_vaep.groupby('player_id').agg({
    'vaep_total': ['sum', 'mean'],
    'num_actions': 'sum',
    'game_id': 'count'
}).reset_index()

player_season.columns = ['player_id', 'season_vaep_total', 'avg_vaep_per_game', 'total_actions', 'num_games']
player_season['vaep_per90'] = player_season['season_vaep_total'] / player_season['num_games']

# 선수 이름 추가
if 'player_name' in player_game_vaep.columns:
    player_names_unique = player_game_vaep[['player_id', 'player_name']].drop_duplicates()
    player_season = player_season.merge(player_names_unique, on='player_id', how='left')

# 최소 출전 경기 필터 (5경기 이상)
player_season_filtered = player_season[player_season['num_games'] >= 5].copy()

# 저장
player_season.to_csv('data/vaep_results/player_season_vaep_atomic.csv', index=False)
print(f"시즌별 VAEP 저장 완료: {len(player_season)} 선수\n")

# Top 선수 출력
print("=== TOP 30 선수 (시즌 총 VAEP) ===")
top_30 = player_season_filtered.nlargest(30, 'season_vaep_total')
if 'player_name' in top_30.columns:
    display_cols = ['player_name', 'season_vaep_total', 'avg_vaep_per_game', 'vaep_per90', 'num_games', 'total_actions']
else:
    display_cols = ['player_id', 'season_vaep_total', 'avg_vaep_per_game', 'vaep_per90', 'num_games', 'total_actions']
print(top_30[display_cols].to_string(index=False))

print("\n=== TOP 30 선수 (경기당 평균 VAEP) ===")
top_30_avg = player_season_filtered.nlargest(30, 'avg_vaep_per_game')
print(top_30_avg[display_cols].to_string(index=False))
