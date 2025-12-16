import os
from pathlib import Path
import yaml

# 버전 명시: v1.0
__version__ = "v1.0"

def get_project_root() -> Path:
    """프로젝트 루트 디렉토리를 반환합니다."""
    return Path(__file__).resolve().parent.parent.parent

def load_config(config_path: str = None) -> dict:
    """
    중앙 설정 파일(config.yaml)을 로드하고 버전 정보를 기준으로
    동적 경로를 설정합니다. (Single Source of Truth 준수)
    """
    root = get_project_root()
    if config_path is None:
        config_path = root / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 버전 정보 확인
    version = cfg.get("version", "v1.0")

    # 상대 경로를 프로젝트 루트 기준 절대/상대 경로로 정규화
    cfg["paths"] = {
        "root": root,
        "sample_file": root / cfg["data"]["sample_file"],
        "types_meta": root / cfg["data"]["types_meta"],
        "bodyparts_meta": root / cfg["data"]["bodyparts_meta"],
        "raw_dir": root / cfg["data"]["raw_dir"],
        "refined_dir": root / "data" / "refined" / version,
        "models_dir": root / "results" / "models" / version,
        "metrics_dir": root / "results" / "metrics" / version,
        "analysis_dir": root / "results" / "analysis" / version,
    }

    # 디렉토리 자동 생성
    for p in ["refined_dir", "models_dir", "metrics_dir", "analysis_dir"]:
        cfg["paths"][p].mkdir(parents=True, exist_ok=True)

    return cfg
