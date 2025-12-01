"""
Atomic SPADL 변환 빠른 테스트 스크립트

단일 경기만 처리하여 변환이 제대로 동작하는지 확인합니다.
"""

import os
import sys
import logging

# socceraction 경로 추가
sys.path.insert(0, '/root/AIS/socceraction')

import pandas as pd
import socceraction.spadl as spadl
import socceraction.atomic.spadl as atomicspadl
from socceraction.data.wyscout import PublicWyscoutLoader
from socceraction.spadl.wyscout import convert_to_actions as wyscout_to_spadl


def test_single_game(data_dir: str, game_id: int = None):
    """단일 경기로 Atomic SPADL 변환 테스트"""
    
    print("=" * 80)
    print("Atomic SPADL 변환 빠른 테스트")
    print("=" * 80)
    
    # 로더 초기화
    print("\n1. Wyscout 로더 초기화...")
    try:
        loader = PublicWyscoutLoader(root=data_dir)
        print("✓ 로더 초기화 성공")
    except Exception as e:
        print(f"✗ 로더 초기화 실패: {e}")
        return False
    
    # 경기 목록 확인
    print("\n2. 사용 가능한 경기 확인...")
    try:
        competitions = loader.competitions()
        print(f"✓ {len(competitions)} 개 대회 발견")
        print(f"  대회 목록:")
        for _, comp in competitions.iterrows():
            print(f"    - {comp.competition_name} (ID: {comp.competition_id}, Season: {comp.season_id})")
        
        # 첫 번째 대회의 첫 번째 경기 가져오기
        if game_id is None:
            first_comp = competitions.iloc[0]
            games = loader.games(first_comp.competition_id, first_comp.season_id)
            if len(games) > 0:
                game_id = games.iloc[0].game_id
                print(f"\n  테스트 경기 ID: {game_id} (자동 선택)")
            else:
                print("✗ 경기를 찾을 수 없습니다")
                return False
    except Exception as e:
        print(f"✗ 경기 목록 확인 실패: {e}")
        return False
    
    # 이벤트 로드
    print(f"\n3. 경기 {game_id} 이벤트 로드...")
    try:
        events = loader.events(game_id)
        print(f"✓ {len(events)} 개 이벤트 로드")
        print(f"  이벤트 타입: {events['type_name'].value_counts().head()}")
    except Exception as e:
        print(f"✗ 이벤트 로드 실패: {e}")
        return False
    
    # 팀 정보
    print(f"\n4. 팀 정보 로드...")
    try:
        teams = loader.teams(game_id)
        print(f"  팀 정보 컬럼: {list(teams.columns)}")
        
        # side 컬럼이 있으면 사용, 없으면 matches에서 직접 찾기
        if "side" in teams.columns:
            home_team_id = teams[teams.side == "home"]["team_id"].iloc[0]
        else:
            # matches 파일에서 직접 홈팀 찾기
            import json
            match_files = [
                "matches_England.json", "matches_Spain.json", "matches_France.json",
                "matches_Germany.json", "matches_Italy.json",
                "matches_European_Championship.json", "matches_World_Cup.json"
            ]
            home_team_id = None
            for match_file in match_files:
                filepath = os.path.join(data_dir, match_file)
                if os.path.exists(filepath):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        matches = json.load(f)
                    for match in matches:
                        if match.get("wyId") == game_id or match.get("matchId") == game_id:
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
                home_team_id = teams.iloc[0]["team_id"]
                print(f"  ⚠ 홈팀을 찾지 못해 첫 번째 팀 사용: {home_team_id}")
        
        print(f"✓ 홈팀 ID: {home_team_id}")
    except Exception as e:
        print(f"✗ 팀 정보 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # SPADL 변환
    print(f"\n5. SPADL 변환...")
    try:
        spadl_actions = wyscout_to_spadl(events, home_team_id)
        print(f"✓ {len(spadl_actions)} 개 SPADL 액션 생성")
        print(f"  액션 타입 분포:")
        if "type_name" in spadl_actions.columns:
            print(spadl_actions["type_name"].value_counts().head(10))
        else:
            # type_id를 이름으로 변환
            spadl_actions = spadl.add_names(spadl_actions)
            print(spadl_actions["type_name"].value_counts().head(10))
    except Exception as e:
        print(f"✗ SPADL 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Atomic SPADL 변환
    print(f"\n6. Atomic SPADL 변환...")
    try:
        atomic_actions = atomicspadl.convert_to_atomic(spadl_actions)
        print(f"✓ {len(atomic_actions)} 개 Atomic SPADL 액션 생성")
        print(f"  액션 증가율: {len(atomic_actions) / len(spadl_actions):.2f}x")
        
        # type_name 추가
        if "type_name" not in atomic_actions.columns:
            import socceraction.atomic.spadl.config as atomicspadl_config
            atomic_actions["type_name"] = atomic_actions["type_id"].apply(
                lambda x: atomicspadl_config.actiontypes[x] if x < len(atomicspadl_config.actiontypes) else "unknown"
            )
        
        print(f"\n  Atomic SPADL 액션 타입 분포:")
        # type_name이 없으면 생성
        if "type_name" not in atomic_actions.columns:
            import socceraction.atomic.spadl.config as atomicspadl_config
            atomic_actions["type_name"] = atomic_actions["type_id"].apply(
                lambda x: atomicspadl_config.actiontypes[x] if x < len(atomicspadl_config.actiontypes) else "unknown"
            )
        
        print(atomic_actions["type_name"].value_counts().head(15))
        
        # receival 액션 확인
        receival_count = (atomic_actions["type_name"] == "receival").sum()
        print(f"\n  Receival 액션: {receival_count} 개")
        
    except Exception as e:
        print(f"✗ Atomic SPADL 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 컬럼 확인
    print(f"\n7. Atomic SPADL 컬럼 확인...")
    print(f"  컬럼 목록: {list(atomic_actions.columns)}")
    
    required_cols = ["game_id", "action_id", "period_id", "time_seconds", 
                     "team_id", "player_id", "x", "y", "dx", "dy", 
                     "type_id", "bodypart_id"]
    missing = [col for col in required_cols if col not in atomic_actions.columns]
    if missing:
        print(f"  ⚠ 경고: 누락된 컬럼: {missing}")
    else:
        print(f"  ✓ 필수 컬럼 모두 존재")
    
    # 샘플 데이터 출력
    print(f"\n8. 샘플 데이터 (처음 5개 액션):")
    print(atomic_actions[["game_id", "action_id", "type_name", "x", "y", "dx", "dy", "team_id", "player_id"]].head())
    
    print("\n" + "=" * 80)
    print("✓ 모든 테스트 통과!")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Atomic SPADL 변환 빠른 테스트")
    parser.add_argument("--data_dir", type=str, default="../data/wyscout", help="Wyscout 데이터 디렉토리")
    parser.add_argument("--game_id", type=int, default=None, help="테스트할 경기 ID (없으면 자동 선택)")
    
    args = parser.parse_args()
    
    # 절대 경로 계산
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, args.data_dir))
    
    success = test_single_game(data_dir, args.game_id)
    sys.exit(0 if success else 1)

