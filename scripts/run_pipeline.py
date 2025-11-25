"""
VAEP 전체 파이프라인 실행 스크립트

config.yaml 설정에 따라 전체 파이프라인을 순차적으로 실행합니다:
1. 데이터 전처리 (preprocess_wyscout.py)
2. 모델 학습 (train_vaep_model.py)
3. 선수 VAEP 계산 (compute_player_vaep.py)
"""

import argparse
import os
import subprocess
import sys
from utils import load_config, setup_logger


def run_command(cmd: list, step_name: str, logger) -> bool:
    """
    명령을 실행하고 결과를 반환합니다.

    Args:
        cmd: 실행할 명령 리스트
        step_name: 단계 이름
        logger: 로거 객체

    Returns:
        성공 여부
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Step: {step_name}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"{'=' * 80}\n")

    try:
        result = subprocess.run(cmd, check=True)
        logger.info(f"\n✓ {step_name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"\n✗ {step_name} failed with error code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"\n✗ {step_name} failed: {str(e)}")
        return False


def main():
    """메인 실행 함수."""
    parser = argparse.ArgumentParser(description="VAEP 전체 파이프라인 실행")
    parser.add_argument(
        "--config", type=str, default="../config.yaml", help="설정 파일 경로"
    )
    parser.add_argument(
        "--skip-preprocess", action="store_true", help="전처리 단계 건너뛰기"
    )
    parser.add_argument("--skip-train", action="store_true", help="학습 단계 건너뛰기")
    parser.add_argument(
        "--skip-evaluate", action="store_true", help="평가 단계 건너뛰기"
    )
    args = parser.parse_args()

    # 로거 설정
    logger = setup_logger("vaep_pipeline")

    logger.info("=" * 80)
    logger.info("VAEP Pipeline Execution")
    logger.info("=" * 80)

    # 설정 로드
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        sys.exit(1)

    # 현재 디렉토리
    script_dir = os.path.dirname(os.path.abspath(__file__))

    success = True

    # 1. 전처리
    if not args.skip_preprocess:
        cmd = [sys.executable, os.path.join(script_dir, "preprocess_wyscout.py")]
        if not run_command(cmd, "Data Preprocessing", logger):
            success = False
            logger.error("Preprocessing failed. Stopping pipeline.")
            sys.exit(1)
    else:
        logger.info("Skipping preprocessing step...")

    # 2. 모델 학습
    if not args.skip_train:
        cmd = [sys.executable, os.path.join(script_dir, "train_vaep_model.py")]
        if args.config:
            cmd.extend(["--config", args.config])
        if not run_command(cmd, "Model Training", logger):
            success = False
            logger.error("Training failed. Stopping pipeline.")
            sys.exit(1)
    else:
        logger.info("Skipping training step...")

    # 3. 선수 VAEP 계산
    if not args.skip_evaluate:
        cmd = [sys.executable, os.path.join(script_dir, "compute_player_vaep.py")]
        if not run_command(cmd, "Player VAEP Computation", logger):
            success = False
            logger.error("Evaluation failed.")
            sys.exit(1)
    else:
        logger.info("Skipping evaluation step...")

    # 완료
    logger.info("\n" + "=" * 80)
    if success:
        logger.info("✓ VAEP Pipeline completed successfully!")
        logger.info("=" * 80)
        logger.info("\nOutput files:")

        paths = config["paths"]
        logger.info(
            f"  1. Training data: {paths['processed_dir']}/{config['preprocessing']['output_train']}"
        )
        logger.info(
            f"  2. Evaluation data: {paths['processed_dir']}/{config['preprocessing']['output_eval']}"
        )
        logger.info(f"  3. Model: {paths['models_dir']}/vaep_model.pt")
        logger.info(f"  4. Config: {paths['models_dir']}/vaep_config.json")
        logger.info(
            f"  5. Player-match VAEP: {paths['results_dir']}/player_match_vaep_england.csv"
        )
        logger.info(
            f"  6. Player-season VAEP: {paths['results_dir']}/player_season_vaep_england.csv"
        )
    else:
        logger.error("✗ VAEP Pipeline completed with errors.")
        logger.info("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
