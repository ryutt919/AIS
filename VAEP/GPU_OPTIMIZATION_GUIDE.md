# GPU 최적화 및 버전 관리 업데이트

**날짜**: 2025-11-25
**주요 변경사항**: VAEP 결과 버전별 관리, GPU 최적화, Grid Search 구현

## 1. VAEP 결과 버전별 관리

### 변경 내용
- **기존**: `data/vaep_results/` 폴더에 모든 결과 저장
- **변경**: `models/{version}/results/` 폴더에 버전별 결과 저장

### 영향받는 파일
- `scripts/compute_player_vaep.py`
  - `--output_dir` 기본값을 모델 버전 폴더 내 `results/`로 자동 설정
  - 결과 파일이 모델과 함께 저장되어 버전 추적 용이

- `notebooks/vaep_performance_analysis.ipynb`
  - 데이터 로드 경로를 `../models/{MODEL_VERSION}/results/`로 변경
  - `MODEL_VERSION` 변수로 분석할 버전 지정 가능

- `notebooks/hyperparameter_tuning_analysis.ipynb`
  - Training history 로드 경로를 `../models/{MODEL_VERSION}/`로 변경

### 폴더 구조
```
models/
  {version}/                      # 예: 20251125_completed
    vaep_model_best.pt           # 최고 성능 모델
    vaep_model_epoch_XXX.pt      # 에폭별 모델
    vaep_config.json             # 모델 설정
    training_history.json        # 학습 이력
    training.log                 # 학습 로그
    results/                     # VAEP 계산 결과
      player_match_vaep_england.csv
      player_season_vaep_england.csv
```

## 2. GPU 최적화 (A100 80GB)

### 최적화 항목

#### 2.1 Batch Size 증가
- **train_vaep_model.py**: 기본 batch_size 유지 (config.yaml에서 관리)
- **compute_player_vaep.py**: 기본 4096으로 증가 (기존 1024)
- A100의 80GB 메모리를 활용하여 처리 속도 향상

#### 2.2 DataLoader 최적화
```python
# num_workers 추가 (기본값: 8)
DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=8,      # CPU 멀티프로세싱
    pin_memory=True     # GPU 전송 속도 향상
)
```

#### 2.3 Mixed Precision Training (AMP)
- `--use_amp` 플래그 추가
- CUDA의 자동 혼합 정밀도(FP16/FP32) 활용
- 메모리 사용량 감소 및 학습 속도 향상
- 정확도 손실 없이 약 2배 속도 향상 기대

```python
# 사용 예시
python train_vaep_model.py --use_amp --num_workers 8
```

#### 2.4 비동기 GPU 전송
```python
# non_blocking=True 추가
features = features.to(device, non_blocking=True)
labels = labels.to(device, non_blocking=True)
```

### GPU 사양
```
NVIDIA A100-SXM4-80GB
- VRAM: 80GB
- CUDA Version: 12.4
- MIG 모드: Enabled (1개 인스턴스 40GB)
```

### 성능 개선 예상
- **Batch size 증가**: 1024 → 4096 (4배)
- **num_workers**: 0 → 8 (데이터 로딩 병렬화)
- **Mixed Precision**: FP32 → FP16/FP32 혼합 (약 2배 속도)
- **총 예상 개선**: 5-8배 학습 속도 향상

## 3. Grid Search 구현

### 새 파일: `scripts/grid_search_vaep.py`

#### 기능
- 하이퍼파라미터 조합을 자동으로 탐색
- 각 조합마다 독립적인 버전 폴더에 결과 저장
- 학습 완료 후 최적 모델 자동 선택

#### 탐색 공간
```python
{
    "learning_rate": [0.001, 0.0005, 0.0001],
    "batch_size": [2048, 4096, 8192],
    "hidden_dims": [
        [128, 64],
        [256, 128],
        [256, 128, 64],
        [512, 256, 128],
    ],
    "horizon": [10, 15, 20],
}
```

#### 사용법
```bash
# 전체 조합 실행 (3 × 3 × 4 × 3 = 108개)
python scripts/grid_search_vaep.py --epochs 20

# 최대 20개 조합만 실행
python scripts/grid_search_vaep.py --epochs 20 --max_runs 20

# 결과 저장 위치: models/grid_search/
```

#### 출력 파일
- `grid_search_results_{timestamp}.csv`: 모든 실행 결과 테이블
- `grid_search_results_{timestamp}.json`: 상세 결과 (nested 구조)
- `grid_search_{timestamp}.log`: 실행 로그

#### 결과 분석
- 자동으로 최고 성능 모델 식별
- Top 5 모델 출력
- Validation loss 기준 정렬

## 4. 설정 파일 변경

### config.yaml (변경 없음)
- 기본 설정은 그대로 유지
- Grid search 또는 개별 실행 시 커맨드라인에서 오버라이드 가능

### 커맨드라인 인자 우선순위
```
커맨드라인 > config.yaml > 하드코딩된 기본값
```

## 5. 사용 예시

### 5.1 일반 학습 (GPU 최적화)
```bash
python scripts/train_vaep_model.py \
  --batch_size 4096 \
  --num_workers 8 \
  --use_amp \
  --epochs 50
```

### 5.2 VAEP 계산 (GPU 최적화)
```bash
python scripts/compute_player_vaep.py \
  --model_version latest \
  --batch_size 4096
```

### 5.3 Grid Search
```bash
python scripts/grid_search_vaep.py \
  --epochs 30 \
  --max_runs 50
```

### 5.4 특정 버전 분석
노트북에서 `MODEL_VERSION` 변수를 변경:
```python
MODEL_VERSION = "20251125_completed"  # 또는 "grid_20251125_001"
```

## 6. 성능 벤치마크

### 현재 실행 결과 (A100 80GB)
- **데이터**: 643,150 이벤트
- **Batch size**: 4096
- **State value 계산**: ~1초 (기존 ~3초)
- **Event VAEP 계산**: 1분 50초 (변경 없음, CPU 바운드)
- **총 실행 시간**: ~2분 (기존 ~3분)

### Mixed Precision 사용 시 예상
- **학습 시간**: 50% 감소
- **메모리 사용량**: 40% 감소
- **모델 정확도**: 유사 (±0.1% 이내)

## 7. 마이그레이션 가이드

### 기존 프로젝트 업데이트
1. 기존 결과 파일 이동:
```bash
mkdir -p models/{version}/results
mv data/vaep_results/*.csv models/{version}/results/
```

2. 노트북 업데이트:
- `vaep_performance_analysis.ipynb`의 `MODEL_VERSION` 설정
- `hyperparameter_tuning_analysis.ipynb`의 `MODEL_VERSION` 설정

3. 새 설정으로 재실행:
```bash
python scripts/compute_player_vaep.py --model_version {version}
```

## 8. 주의사항

### GPU 메모리
- Batch size가 너무 크면 OOM 발생 가능
- MIG 모드 사용 시 40GB로 제한됨
- 필요시 `--batch_size` 조정

### Mixed Precision
- A100에서 권장 (Tensor Core 활용)
- 일부 연산에서 수치 불안정 가능성
- 문제 발생 시 `--use_amp` 제거

### Grid Search
- 전체 조합 실행 시 시간 소요
- `--max_runs`로 제한 권장
- 각 버전별 디스크 공간 필요 (~100MB/버전)

## 9. 다음 단계

### 추천 작업
1. **Mixed Precision 테스트**:
   ```bash
   python scripts/train_vaep_model.py --use_amp --epochs 10
   ```

2. **Grid Search 실행**:
   ```bash
   python scripts/grid_search_vaep.py --epochs 20 --max_runs 20
   ```

3. **결과 분석**:
   - `notebooks/hyperparameter_tuning_analysis.ipynb` 실행
   - Grid search 결과 비교

4. **최적 모델로 전체 평가**:
   ```bash
   python scripts/compute_player_vaep.py --model_version grid_XXXXXX_XXX
   ```

## 10. 참고 자료

### GPU 최적화
- [PyTorch AMP 가이드](https://pytorch.org/docs/stable/amp.html)
- [DataLoader 최적화](https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading)

### 버전 관리
- 버전 폴더는 자동으로 타임스탬프로 생성
- `--version` 인자로 커스텀 이름 지정 가능

### 디버깅
- 학습 로그: `models/{version}/training.log`
- Grid search 로그: `models/grid_search/grid_search_{timestamp}.log`
