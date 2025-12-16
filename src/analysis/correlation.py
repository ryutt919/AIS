import pandas as pd
import numpy as np
import json
from scipy.stats import spearmanr, pearsonr

# 버전 명시: v1.0
__version__ = "v1.0"

def compute_correlation_analysis(cfg: dict) -> dict:
    """
    xAction 잔차(match_rate)와 VAEP 지표 간의 선수-경기 및 선수-시즌 단위 상관관계 분석
    """
    print("\n=== [Track 4] Correlation & Integration Analysis ===")
    refined_dir = cfg["paths"]["refined_dir"]
    analysis_dir = cfg["paths"]["analysis_dir"]
    
    match_summary_path = refined_dir / "per_match_player_summary.csv"
    game_vaep_path = refined_dir / "player_game_vaep_atomic.csv"
    season_vaep_path = refined_dir / "player_season_vaep_atomic.csv"
    
    if not match_summary_path.exists() or not game_vaep_path.exists():
        raise FileNotFoundError("Prerequisite summary files missing for correlation analysis.")
        
    df_match = pd.read_csv(match_summary_path)
    df_game_vaep = pd.read_csv(game_vaep_path)
    
    # 1. 선수-경기 단위 조인 및 상관관계 분석
    merged_match = pd.merge(df_match, df_game_vaep, on=['player_id', 'game_id'], how='inner')
    print(f"Match-level merged records: {len(merged_match)}")
    
    match_vars = ['num_actions_y', 'total_concedes_prob', 'total_scores_prob', 'vaep_total', 'vaep_mean']
    match_results = []
    
    for v in match_vars:
        if v in merged_match.columns:
            valid = merged_match[['match_rate', v]].dropna()
            if len(valid) > 2 and valid['match_rate'].std() > 0 and valid[v].std() > 0:
                sp_r, sp_p = spearmanr(valid['match_rate'], valid[v])
                pe_r, pe_p = pearsonr(valid['match_rate'], valid[v])
            else:
                sp_r, sp_p, pe_r, pe_p = 0.0, 1.0, 0.0, 1.0
            
            clean_name = v.replace('_y', '')
            match_results.append({
                "variable": clean_name,
                "spearman": float(sp_r),
                "pearson": float(pe_r),
                "n_samples": len(valid)
            })
            
    df_match_corr = pd.DataFrame(match_results)
    out_match_csv = analysis_dir / "correlation_match_level.csv"
    df_match_corr.to_csv(out_match_csv, index=False)
    print(f"Match-level correlation results:\n{df_match_corr}")
    
    # 2. 선수-시즌 단위 조인 및 상관관계 분석
    df_season_xaction = df_match.groupby('player_id').agg(
        avg_match_rate=('match_rate', 'mean')
    ).reset_index()
    
    df_season_vaep = pd.read_csv(season_vaep_path)
    merged_season = pd.merge(df_season_xaction, df_season_vaep, on='player_id', how='inner')
    print(f"Season-level merged records: {len(merged_season)}")
    
    season_vars = ['avg_vaep_per_game', 'vaep_per90', 'total_actions', 'season_vaep_total', 'num_games']
    season_results = []
    
    for v in season_vars:
        if v in merged_season.columns:
            valid = merged_season[['avg_match_rate', v]].dropna()
            if len(valid) > 2 and valid['avg_match_rate'].std() > 0 and valid[v].std() > 0:
                sp_r, sp_p = spearmanr(valid['avg_match_rate'], valid[v])
                pe_r, pe_p = pearsonr(valid['avg_match_rate'], valid[v])
            else:
                sp_r, sp_p, pe_r, pe_p = 0.0, 1.0, 0.0, 1.0
                
            season_results.append({
                "variable": v,
                "spearman": float(sp_r),
                "pearson": float(pe_r),
                "n_samples": len(valid)
            })
            
    df_season_corr = pd.DataFrame(season_results)
    out_season_csv = analysis_dir / "correlation_season_level.csv"
    df_season_corr.to_csv(out_season_csv, index=False)
    print(f"Season-level correlation results:\n{df_season_corr}")
    
    summary = {
        "match_level": match_results,
        "season_level": season_results
    }
    summary_path = analysis_dir / "correlation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    return summary
