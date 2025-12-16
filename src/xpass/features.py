import pandas as pd
import numpy as np

# 버전 명시: v1.0
__version__ = "v1.0"

def extract_xpass_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    패스 액션에 대한 기하학적 및 공간적 피처를 추출합니다.
    - 패스 액션: type_id == 0 (pass) 및 type_id == 1 (cross) 등
    - 성공 여부: 다음 액션이 동일 팀의 receival(23)이거나 점유 유지인 경우 (여기서는 Atomic 표준 규칙 적용)
    """
    df = df.sort_values(by=['game_id', 'period_id', 'time_seconds', 'action_id']).reset_index(drop=True)
    
    # 패스 이벤트 필터링 (type_id 0: pass, 1: cross)
    is_pass = df['type_id'].isin([0, 1])
    pass_indices = np.where(is_pass)[0]
    
    records = []
    for idx in pass_indices:
        if idx >= len(df) - 1:
            continue
            
        cur = df.iloc[idx]
        nxt = df.iloc[idx + 1]
        
        # 성공 판정: 다음 액션이 같은 팀이고 리시브(23)이거나 패스(0)인 경우
        success = 1 if (nxt['team_id'] == cur['team_id'] and nxt['type_id'] in [0, 1, 23]) else 0
        
        # 피처 엔지니어링
        start_x, start_y = cur['x'], cur['y']
        dx, dy = cur['dx'], cur['dy']
        end_x, end_y = start_x + dx, start_y + dy
        
        dist = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        
        # 골대(105, 34)까지의 거리 및 각도
        goal_x, goal_y = 105.0, 34.0
        dist_to_goal = np.sqrt((goal_x - end_x)**2 + (goal_y - end_y)**2)
        
        records.append({
            'game_id': cur['game_id'],
            'player_id': cur['player_id'],
            'team_id': cur['team_id'],
            'start_x': start_x,
            'start_y': start_y,
            'end_x': end_x,
            'end_y': end_y,
            'dx': dx,
            'dy': dy,
            'dist': dist,
            'angle': angle,
            'dist_to_goal': dist_to_goal,
            'bodypart_id': cur['bodypart_id'],
            'success': success
        })
        
    return pd.DataFrame(records)
