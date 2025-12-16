import argparse
import sys
import time
from pathlib import Path

# 모듈 경로 추가
sys.path.append(str(Path(__file__).resolve().parent))

from src.utils.config_loader import load_config
from src.preprocessing.data_loader import preprocess_and_save
from src.xaction.pipeline import train_and_evaluate_xaction
from src.xpass.pipeline import train_and_evaluate_xpass
from src.vaep.pipeline import train_and_evaluate_vaep
from src.analysis.correlation import compute_correlation_analysis

# 버전 명시: v1.0
__version__ = "v1.0"

def main():
    parser = argparse.ArgumentParser(description="AIS Portfolio End-to-End Pipeline Runner")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--mode", type=str, default=None, choices=["sample", "full"], help="Execution mode override")
    parser.add_argument("--stage", type=str, default="all", 
                        choices=["all", "prep", "xaction", "xpass", "vaep", "analysis"],
                        help="Pipeline stage to execute")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.mode:
        cfg["mode"] = args.mode

    start_time = time.time()
    print("=" * 65)
    print(f"  AIS Pipeline Execution Started [Version: {cfg.get('version', 'v1.0')}]")
    print(f"  Mode: {cfg.get('mode', 'sample')}, Stage: {args.stage}")
    print("=" * 65)

    # 1. Preprocessing
    if args.stage in ["all", "prep"]:
        df = preprocess_and_save(cfg)
    else:
        refined_path = cfg["paths"]["refined_dir"] / "atomic_spadl_cleaned.csv"
        import pandas as pd
        df = pd.read_csv(refined_path)

    # 2. Track 1: xAction
    if args.stage in ["all", "xaction"]:
        train_and_evaluate_xaction(cfg, df)

    # 3. Track 2: xPass
    if args.stage in ["all", "xpass"]:
        train_and_evaluate_xpass(cfg, df)

    # 4. Track 3: VAEP
    if args.stage in ["all", "vaep"]:
        train_and_evaluate_vaep(cfg, df)

    # 5. Track 4: Correlation Analysis
    if args.stage in ["all", "analysis"]:
        compute_correlation_analysis(cfg)

    elapsed = time.time() - start_time
    print("=" * 65)
    print(f"  Pipeline Finished Successfully in {elapsed:.2f}s!")
    print(f"  Results saved under: {cfg['paths']['results_dir'] if 'results_dir' in cfg['paths'] else 'results/'}")
    print("=" * 65)

if __name__ == "__main__":
    main()
