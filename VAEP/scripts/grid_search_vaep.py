"""
VAEP 모델 하이퍼파라미터 Grid Search 스크립트

다양한 하이퍼파라미터 조합을 자동으로 탐색하여 최적의 설정을 찾습니다.
"""

import argparse
import itertools
import json
import logging
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import pandas as pd

from utils import setup_logger, ensure_dir, load_config


def parse_args():
    """명령줄 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description="VAEP 모델 Grid Search")
    parser.add_argument(
        "--config",
        type=str,
        default="../config.yaml",
        help="설정 파일 경로",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../models/grid_search",
        help="Grid search 결과 저장 디렉토리",
    )
    parser.add_argument(
        "--max_runs",
        type=int,
        default=20,
        help="최대 실행 횟수 (기본값: 20)",
    )
    parser.add_argument(
        "--random_search",
        action="store_true",
        help="Random search 사용 (grid search 대신)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="각 실행의 에포크 수 (지정 시 grid의 'epochs' 값 무시)",
    )
    return parser.parse_args()


def get_grid_search_space() -> Dict[str, List]:
    """
    Grid search를 위한 하이퍼파라미터 공간을 정의합니다.
    
    Returns:
        하이퍼파라미터 이름과 탐색할 값 리스트의 딕셔너리
    """
    return {
        "learning_rate": [0.001, 0.0005, 0.0001, 0.00075, 0.002],
        "batch_size": [2048, 4096, 8192],
        "hidden_dims": [
            [128, 64],
            [256, 128], 
            [256, 128, 64],
            [512, 256],
            [384, 192, 96],
        ],
        "horizon": [5, 10, 15, 20],
        "epochs": [20, 35, 40],
        "scheduler": ["step", "cosine"],
    }


def run_training(
    params: Dict,
    version: str,
    epochs: int,
    logger: logging.Logger,
) -> Dict:
    """
    주어진 하이퍼파라미터로 학습을 실행합니다.
    
    Args:
        params: 하이퍼파라미터 딕셔너리
        version: 모델 버전 이름
        epochs: 에포크 수
        logger: 로거 객체
        
    Returns:
        학습 결과 딕셔너리
    """
    # 실제 사용할 epoch 수 (오버라이드 반영)
    actual_epochs = epochs if epochs is not None else params["epochs"]
    
    logger.info(f"Starting training with params: {params}")
    if epochs is not None and epochs != params["epochs"]:
        logger.info(f"  -> Epochs overridden: {params['epochs']} -> {actual_epochs}")
    
    # train_vaep_model.py 실행 명령어 구성
    cmd = [
        sys.executable,
        "train_vaep_model.py",
        "--version", version,
        "--epochs", str(actual_epochs),
        "--lr", str(params["learning_rate"]),
        "--batch_size", str(params["batch_size"]),
        "--horizon", str(params["horizon"]),
        "--hidden_dims", *[str(d) for d in params["hidden_dims"]],
        "--num_workers", "8",
        "--use_amp",  # A100 GPU 최적화
        "--precomputed_dir", "../data/horizon_processed",  # 전처리된 horizon 데이터 사용
    ]
    
    # 스케줄러 추가
    if params["scheduler"] != "none":
        cmd.extend(["--scheduler", params["scheduler"]])
    
    try:
        # subprocess로 학습 스크립트 실행 (실시간 출력 표시)
        logger.info(f"Executing: {' '.join(cmd)}")
        logger.info("Training in progress... (check terminal for live updates)")
        
        result = subprocess.run(
            cmd,
            capture_output=False,  # 실시간 출력 표시
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        
        if result.returncode != 0:
            logger.error(f"Training failed with return code: {result.returncode}")
            return {
                "status": "failed",
                "error": f"Training failed with return code {result.returncode}",
            }
        
        # 학습 결과 로드 (training_history.json)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.abspath(os.path.join(script_dir, "..", "models", version))
        history_path = os.path.join(model_dir, "training_history.json")
        
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
            
            # 최고 성능 추출
            best_val_loss = min(history["val_loss"])
            best_epoch = history["val_loss"].index(best_val_loss) + 1
            
            return {
                "status": "success",
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "final_train_loss": history["train_loss"][-1],
                "final_val_loss": history["val_loss"][-1],
            }
        else:
            logger.warning(f"Training history not found: {history_path}")
            return {
                "status": "completed_no_history",
            }
            
    except Exception as e:
        logger.error(f"Exception during training: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
        }


def load_previous_runs(output_dir: str) -> set:
    """
    이전에 실행했던 하이퍼파라미터 조합을 로드합니다.
    
    Args:
        output_dir: Grid search 결과 디렉토리
        
    Returns:
        이전에 실행한 조합의 해시 set
    """
    used_combinations = set()
    
    # 전용 조합 추적 파일 경로
    tracking_file = os.path.join(output_dir, "used_combinations.json")
    
    if os.path.exists(tracking_file):
        try:
            with open(tracking_file, "r") as f:
                used_list = json.load(f)
            
            # 리스트를 set으로 변환
            for params in used_list:
                param_tuple = tuple(sorted([
                    (k, tuple(v) if isinstance(v, list) else v)
                    for k, v in params.items()
                ]))
                used_combinations.add(param_tuple)
        except Exception as e:
            print(f"Warning: Could not load tracking file: {e}")
    
    return used_combinations


def save_used_combination(output_dir: str, params: Dict):
    """
    실행한 조합을 추적 파일에 저장합니다.
    
    Args:
        output_dir: Grid search 결과 디렉토리
        params: 실행한 파라미터 딕셔너리
    """
    tracking_file = os.path.join(output_dir, "used_combinations.json")
    
    # 기존 파일 로드
    used_list = []
    if os.path.exists(tracking_file):
        try:
            with open(tracking_file, "r") as f:
                used_list = json.load(f)
        except Exception:
            used_list = []
    
    # 새 조합 추가
    used_list.append(params)
    
    # 저장
    try:
        with open(tracking_file, "w") as f:
            json.dump(used_list, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save tracking file: {e}")


def params_to_hashable(params: Dict) -> tuple:
    """파라미터 딕셔너리를 해시 가능한 튜플로 변환합니다."""
    return tuple(sorted([
        (k, tuple(v) if isinstance(v, list) else v)
        for k, v in params.items()
    ]))


def main():
    """메인 실행 함수."""
    args = parse_args()
    
    # 출력 디렉토리 생성
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    ensure_dir(output_dir)
    
    # 로거 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"grid_search_{timestamp}.log")
    logger = setup_logger("grid_search", log_file)
    
    # 이전에 실행한 조합 로드
    used_combinations = load_previous_runs(output_dir)
    logger.info(f"Loaded {len(used_combinations)} previously executed combinations")
    
    logger.info("\n" + "=" * 80)
    logger.info("VAEP Model Grid Search")
    logger.info("\n" + "=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Max runs: {args.max_runs or 'All combinations'}")
    if args.epochs is None:
        logger.info("Epochs per run: using value from each grid combination")
    else:
        logger.info(f"Epochs per run: {args.epochs} (global override)")
    
    # Grid search 공간 정의
    search_space = get_grid_search_space()
    logger.info("\nGrid search space:")
    for param, values in search_space.items():
        logger.info(f"  - {param}: {values}")
    
    # 조합 생성 (Grid search vs Random search)
    param_names = list(search_space.keys())
    param_values = list(search_space.values())
    
    # 모든 조합 생성
    all_combinations = list(itertools.product(*param_values))
    logger.info(f"\nTotal possible combinations: {len(all_combinations)}")
    
    if args.random_search:
        # Random search: 랜덤하게 조합 선택 (중복 제거하며 max_runs만큼 새 조합 찾기)
        logger.info(f"Using Random Search")
        random.shuffle(all_combinations)
        selected_combinations = []
        
        for combo in all_combinations:
            params_test = dict(zip(param_names, combo))
            if params_to_hashable(params_test) not in used_combinations:
                selected_combinations.append(combo)
                if len(selected_combinations) >= args.max_runs:
                    break
        
        all_combinations = selected_combinations
        logger.info(f"Selected {len(all_combinations)} new random combinations")
    else:
        # Grid search: 순차적으로 실행하되, 이미 실행한 것은 제외하고 max_runs만큼 새 조합 선택
        logger.info(f"Using Grid Search")
        
        if args.max_runs:
            # 이미 실행하지 않은 조합만 필터링하여 max_runs만큼 선택
            new_combinations = []
            for combo in all_combinations:
                params_test = dict(zip(param_names, combo))
                if params_to_hashable(params_test) not in used_combinations:
                    new_combinations.append(combo)
                    if len(new_combinations) >= args.max_runs:
                        break
            
            all_combinations = new_combinations
            logger.info(f"Selected {len(all_combinations)} new combinations to run")
    
    # 결과 저장용 리스트
    results = []
    skipped_in_loop = 0
    
    # 각 조합에 대해 학습 실행 (이미 필터링된 새 조합만 실행)
    for idx, param_combo in tqdm(enumerate(all_combinations, 1), total=len(all_combinations), desc='Grid Searching'):
        params = dict(zip(param_names, param_combo))
        
        # 더블 체크: 혹시 이미 실행한 조합인지 다시 확인 (안전장치)
        param_hash = params_to_hashable(params)
        if param_hash in used_combinations:
            skipped_in_loop += 1
            logger.warning(f"\nDouble-check skip {idx}/{len(all_combinations)}: already executed {params}")
            continue
        
        # 버전 이름 생성
        version = f"grid_{timestamp}_{idx:03d}"
        
        logger.info("\n" + "=" * 80)
        logger.info(f"Run {idx}/{len(all_combinations)}: {version}")
        logger.info("\n" + "=" * 80)
        
        # 학습 실행
        # 결정된 epoch 수: 전역 오버라이드가 있으면 그것을 사용, 없으면 조합의 값을 사용
        run_epochs = args.epochs if args.epochs is not None else params.get("epochs")
        result = run_training(params, version, run_epochs, logger)
        
        # 현재 조합을 추적 파일에 저장
        save_used_combination(output_dir, params)
        
        # 현재 조합을 메모리의 set에도 추가
        param_hash = params_to_hashable(params)
        used_combinations.add(param_hash)
        
        # 결과 저장
        result_entry = {
            "run_id": idx,
            "version": version,
            "params": params,
            **result,
        }
        results.append(result_entry)
        
        logger.info(f"Result: {result}")
        
        # 중간 결과 저장
        results_df = pd.DataFrame(results)
        results_csv = os.path.join(output_dir, f"grid_search_results_{timestamp}.csv")
        results_df.to_csv(results_csv, index=False)
        
        # JSON으로도 저장 (nested 구조 보존)
        results_json = os.path.join(output_dir, f"grid_search_results_{timestamp}.json")
        with open(results_json, "w") as f:
            json.dump(results, f, indent=2)
    
    # 최종 결과 분석
    logger.info("\n" + "=" * 80)
    logger.info("Grid Search Summary")
    logger.info("\n" + "=" * 80)
    
    logger.info(f"Previously executed combinations loaded: {len(used_combinations) - len(results)}")
    logger.info(f"Combinations selected for this run: {len(all_combinations)}")
    if skipped_in_loop > 0:
        logger.warning(f"Skipped in loop (double-check caught): {skipped_in_loop}")
    logger.info(f"New runs executed: {len(results)}")
    
    successful_runs = [r for r in results if r["status"] == "success"]
    logger.info(f"Successful runs: {len(successful_runs)}/{len(results)}")
    
    if successful_runs:
        # 최고 성능 모델 찾기
        best_run = min(successful_runs, key=lambda x: x["best_val_loss"])
        
        logger.info("\nBest model:")
        logger.info(f"  - Version: {best_run['version']}")
        logger.info(f"  - Val Loss: {best_run['best_val_loss']:.4f}")
        logger.info(f"  - Best Epoch: {best_run['best_epoch']}")
        logger.info(f"  - Parameters:")
        for param, value in best_run["params"].items():
            logger.info(f"    - {param}: {value}")
        
        # Top 5 모델 출력
        logger.info("\nTop 5 models:")
        top_5 = sorted(successful_runs, key=lambda x: x["best_val_loss"])[:5]
        for i, run in enumerate(top_5, 1):
            logger.info(f"\n{i}. {run['version']} (val_loss: {run['best_val_loss']:.4f})")
            logger.info(f"   Params: {run['params']}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Grid search completed!")
    logger.info(f"Results saved to:")
    logger.info(f"  - {results_csv}")
    logger.info(f"  - {results_json}")
    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()
