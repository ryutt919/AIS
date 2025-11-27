"""
VAEP 데이터 전처리 - Horizon별 레이블 사전 계산

Grid search에서 사용할 모든 horizon 값에 대해 레이블을 미리 계산하여
data/horizon_processed/ 폴더에 영구 저장합니다.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm


def parse_args():
    """명령줄 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description="VAEP Horizon별 전처리")
    parser.add_argument(
        "--input",
        type=str,
        default="../data/processed/vaep_train_events.csv",
        help="입력 이벤트 CSV 파일",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data/horizon_processed",
        help="전처리 결과 저장 디렉토리",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20],
        help="전처리할 horizon 값 리스트",
    )
    return parser.parse_args()


def compute_labels_for_horizon(events: pd.DataFrame, horizon: int) -> np.ndarray:
    """
    주어진 horizon에 대한 레이블(scores, concedes)을 계산합니다.
    
    Args:
        events: 이벤트 DataFrame
        horizon: 레이블링 horizon
        
    Returns:
        (N, 2) shape의 numpy array [scores, concedes]
    """
    print(f"  Computing labels for {len(events)} events...")
    
    # 매치별로 그룹화
    grouped = events.groupby("matchId")
    
    labels = []
    
    for match_id, match_events in tqdm(grouped, desc="  Processing matches", leave=False):
        match_events = match_events.sort_values("eventId").reset_index(drop=True)
        n_events = len(match_events)
        
        # 각 이벤트에 대해 horizon 내 골 계산
        for i in range(n_events):
            # horizon 범위 계산
            end_idx = min(i + horizon, n_events)
            future_events = match_events.iloc[i:end_idx]
            
            current_team = match_events.iloc[i]["teamId"]
            
            # 득점/실점 계산
            scores = 0
            concedes = 0
            
            for _, event in future_events.iterrows():
                # 골 이벤트인지 확인 (simple check)
                # event_type에 따라 다를 수 있지만, 일반적으로:
                # - Shot이면서 결과가 골인 경우
                # 또는 event_type이 직접 Goal인 경우
                is_goal = False
                
                # 간단한 방식: event_type_name에 'goal' 포함 또는
                # result_name에 'goal' 포함
                event_type = str(event.get("event_type_name", "")).lower()
                result = str(event.get("result_name", "")).lower()
                
                if "goal" in event_type or "goal" in result:
                    is_goal = True
                
                if is_goal:
                    if event["teamId"] == current_team:
                        scores += 1
                    else:
                        concedes += 1
            
            labels.append([scores, concedes])
    
    return np.array(labels, dtype=np.float32)


def main():
    """메인 실행 함수."""
    args = parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.abspath(os.path.join(script_dir, args.input))
    output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("VAEP Horizon별 전처리")
    print("=" * 80)
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Horizons: {args.horizons}")
    print()
    
    # 데이터 로드
    print("Loading events...")
    events = pd.read_csv(input_path)
    print(f"  Loaded {len(events)} events")
    print(f"  Matches: {events['matchId'].nunique()}")
    print()
    
    # 각 horizon별로 레이블 계산 및 저장
    for horizon in args.horizons:
        output_file = os.path.join(output_dir, f"labels_h{horizon}.npz")
        
        # 이미 존재하면 스킵
        if os.path.exists(output_file):
            print(f"✓ Horizon {horizon}: Already exists, skipping")
            continue
        
        print(f"Processing horizon={horizon}")
        
        try:
            # 레이블 계산
            labels = compute_labels_for_horizon(events, horizon)
            
            # .npz 형식으로 저장 (압축)
            np.savez_compressed(
                output_file,
                scores=labels[:, 0],
                concedes=labels[:, 1]
            )
            
            print(f"✓ Horizon {horizon}: Saved to {output_file}")
            print(f"  Shape: {labels.shape}")
            print()
            
        except Exception as e:
            print(f"✗ Horizon {horizon}: Error - {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("=" * 80)
    print("Precomputation completed!")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    for horizon in args.horizons:
        filepath = os.path.join(output_dir, f"labels_h{horizon}.npz")
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  - labels_h{horizon}.npz ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
