import pandas as pd
import numpy as np

# 버전 명시: v1.0
__version__ = "v1.0"

def create_vaep_labels(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    각 액션 이후 horizon 개 액션 이내에 득점(scores) 또는 실점(concedes)이 발생했는지 여부를 라벨링합니다.
    - 골 이벤트: type_id == 27 (goal), owngoal == 28
    """
    df = df.sort_values(by=['game_id', 'period_id', 'time_seconds', 'action_id']).reset_index(drop=True)
    
    n = len(df)
    scores = np.zeros(n, dtype=int)
    concedes = np.zeros(n, dtype=int)
    
    # 경기별로 분할하여 계산
    for game_id in df['game_id'].unique():
        idx = np.where(df['game_id'] == game_id)[0]
        sub = df.iloc[idx]
        
        types = sub['type_id'].values
        teams = sub['team_id'].values
        
        goal_mask = (types == 27)
        owngoal_mask = (types == 28)
        
        for i in range(len(idx)):
            window_slice = slice(i + 1, min(i + 1 + horizon, len(idx)))
            window_types = types[window_slice]
            window_teams = teams[window_slice]
            curr_team = teams[i]
            
            # scores: 현재 팀의 골 또는 상대팀의 자책골
            has_score = np.any((window_types == 27) & (window_teams == curr_team)) or \
                        np.any((window_types == 28) & (window_teams != curr_team))
            # concedes: 상대 팀의 골 또는 현재 팀의 자책골
            has_concede = np.any((window_types == 27) & (window_teams != curr_team)) or \
                          np.any((window_types == 28) & (window_teams == curr_team))
                          
            scores[idx[i]] = int(has_score)
            concedes[idx[i]] = int(has_concede)
            
    df['scores'] = scores
    df['concedes'] = concedes
    return df
