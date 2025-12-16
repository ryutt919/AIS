import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

# 버전 명시: v1.0
__version__ = "v1.0"

def load_atomic_spadl_data(cfg: dict) -> pd.DataFrame:
    """
    설정 파일의 실행 모드에 따라 원본 또는 샘플 Atomic SPADL 데이터를 로드합니다.
    """
    mode = cfg.get("mode", "sample")
    if mode == "sample":
        file_path = cfg["paths"]["sample_file"]
    else:
        file_path = cfg["paths"]["raw_dir"] / "atomic_spadl_all_games.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Data file does not exist: {file_path}")

    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} actions across {df['game_id'].nunique()} games.")
    return df

def normalize_attack_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    공격 방향 정규화:
    각 팀이 전반/후반에 항상 +x 방향(105m 방향)으로 공격하도록 좌표를 변환합니다.
    """
    df = df.copy()
    # 좌표 결측치 보정
    for col in ['x', 'y', 'dx', 'dy']:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # 기본 필드 규격 (105m x 68m)
    field_length = 105.0
    field_width = 68.0

    # 팀별 경기 내 평균 x 진행 방향 추정
    team_mean_x = df.groupby(['game_id', 'period_id', 'team_id'])['x'].mean().reset_index()
    team_mean_x['needs_flip'] = team_mean_x['x'] > (field_length / 2.0)

    df = df.merge(team_mean_x[['game_id', 'period_id', 'team_id', 'needs_flip']], 
                  on=['game_id', 'period_id', 'team_id'], how='left')

    flip_mask = df['needs_flip'] == True
    df.loc[flip_mask, 'x'] = field_length - df.loc[flip_mask, 'x']
    df.loc[flip_mask, 'y'] = field_width - df.loc[flip_mask, 'y']
    df.loc[flip_mask, 'dx'] = -df.loc[flip_mask, 'dx']
    df.loc[flip_mask, 'dy'] = -df.loc[flip_mask, 'dy']

    df.drop(columns=['needs_flip'], inplace=True)
    return df

def split_by_game(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15, random_state=42):
    """
    데이터 누출 방지를 위해 game_id 단위로 Train, Validation, Test 세트를 분할합니다. (70:15:15)
    """
    unique_games = df['game_id'].unique()
    np.random.seed(random_state)
    shuffled_games = np.random.permutation(unique_games)

    n_games = len(shuffled_games)
    n_train = max(1, int(n_games * train_ratio))
    n_val = max(1, int(n_games * val_ratio)) if n_games > 2 else 0

    train_games = set(shuffled_games[:n_train])
    val_games = set(shuffled_games[n_train:n_train + n_val])
    test_games = set(shuffled_games[n_train + n_val:]) if n_games > 2 else val_games

    # 소규모 샘플 대응
    if not val_games:
        val_games = train_games
    if not test_games:
        test_games = val_games

    train_df = df[df['game_id'].isin(train_games)].copy()
    val_df = df[df['game_id'].isin(val_games)].copy()
    test_df = df[df['game_id'].isin(test_games)].copy()

    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    print(f"Dataset split: Train {len(train_games)} games ({len(train_df)} rows), "
          f"Val {len(val_games)} games ({len(val_df)} rows), "
          f"Test {len(test_games)} games ({len(test_df)} rows)")

    full_split_df = pd.concat([train_df, val_df, test_df], ignore_index=True).drop_duplicates(subset=['game_id', 'action_id'])
    return full_split_df, train_df, val_df, test_df

def preprocess_and_save(cfg: dict) -> pd.DataFrame:
    """
    전처리 전체 파이프라인을 실행하고 data/refined/{version}/ 에 저장합니다.
    """
    df = load_atomic_spadl_data(cfg)
    if cfg['xaction'].get('normalize_direction', True):
        df = normalize_attack_direction(df)

    split_df, _, _, _ = split_by_game(df)
    
    # 정제된 데이터 저장 (CSV로 경량 보관)
    output_path = cfg["paths"]["refined_dir"] / "atomic_spadl_cleaned.csv"
    split_df.to_csv(output_path, index=False)
    print(f"Refined dataset saved to: {output_path} ({len(split_df)} rows)")
    return split_df
