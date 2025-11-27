# 프로젝트 리팩토링 요약

## 실행 날짜
2025-11-25

## 수행한 리팩토링

### 1. 파일 구조 정리

#### 로그 파일 정리
- **이전**: 로그 파일들이 `scripts/` 폴더에 분산
- **이후**: 모든 로그 파일을 `logs/` 폴더로 이동
- 이동된 파일:
  - `scripts/debug_pipeline.log` → `logs/debug_pipeline.log`
  - `scripts/pipeline_output.log` → `logs/pipeline_output.log`
  - `scripts/pipeline_output_v2.log` → `logs/pipeline_output_v2.log`

#### 노트북 파일 정리
- **이전**: `data/wyscout/soccer_nsd_code.ipynb`가 데이터 폴더에 위치
- **이후**: `notebooks/soccer_nsd_code.ipynb`로 이동
- 기존 노트북들과 함께 통합 관리:
  - `Soccerdata_scraper.ipynb`
  - `vaep_validation.ipynb`
  - `vaep_performance_analysis.ipynb` (신규 생성)

#### 모델 폴더 구조
- **이전**: 모델 폴더가 비어있음
- **이후**: `/AIS/models/` 폴더가 최상위 레벨에 위치 (data 폴더 밖)
- 모델 저장 경로가 `config.yaml`에 명시됨

### 2. 코드 수정

#### train_vaep_model.py 수정
- 들여쓰기 오류 수정 (line 54, 72)
- 수정 내용:
  ```python
  # Before (잘못된 들여쓰기)
      parser.add_argument(
          "--output_dir", type=str, default="../models", help="모델 저장 디렉토리"
      )
  
  # After (올바른 들여쓰기)
  parser.add_argument(
      "--output_dir", type=str, default="../models", help="모델 저장 디렉토리"
  )
  ```

#### 경로 참조 확인
모든 스크립트의 로그 경로가 이미 올바르게 설정되어 있음:
- `run_pipeline.py`: `../logs/pipeline.log`
- `preprocess_wyscout.py`: `../logs/preprocess_wyscout.log`
- `train_vaep_model.py`: `../logs/train_vaep_model.log`
- `compute_player_vaep.py`: `../logs/compute_player_vaep.log`

### 3. 새로운 파일 생성

#### vaep_performance_analysis.ipynb
전체 파이프라인 결과를 분석하기 위한 노트북 생성:
- 데이터 로드 및 기본 통계
- VAEP 분포 시각화
- 상위 선수 분석 (Top 20)
- VAEP vs 경기/이벤트 수 상관관계
- 경기당 VAEP 변동성 분석
- 경기별 VAEP 추이 (샘플 선수)
- 요약 통계

분석 지표:
- Mean VAEP per 90
- Total VAEP
- Standard deviation (변동성)
- Correlation analysis
- Distribution plots (histogram, box plot)
- Time series plots

## 현재 프로젝트 구조

```
/AIS/
├── config.yaml                    # 파이프라인 설정
├── logs/                         # 모든 로그 파일
│   ├── debug_pipeline.log
│   ├── pipeline.log
│   ├── pipeline_output.log
│   ├── pipeline_output_v2.log
│   ├── preprocess_wyscout.log
│   └── pipeline_full_run.log     # 전체 실행 로그
├── models/                       # 학습된 모델 저장
│   ├── vaep_model.pt            # (생성 예정)
│   └── vaep_config.json         # (생성 예정)
├── notebooks/                    # 모든 노트북 파일
│   ├── Soccerdata_scraper.ipynb
│   ├── soccer_nsd_code.ipynb
│   ├── vaep_validation.ipynb
│   └── vaep_performance_analysis.ipynb
├── scripts/                      # 실행 스크립트
│   ├── run_pipeline.py
│   ├── preprocess_wyscout.py
│   ├── train_vaep_model.py
│   ├── compute_player_vaep.py
│   └── utils.py
├── data/
│   ├── wyscout/                 # 원본 데이터
│   ├── processed/               # 전처리된 데이터
│   │   ├── vaep_train_events.csv    (2,608,144 events, 1,561 matches)
│   │   └── vaep_eval_events_england.csv (380 matches)
│   └── vaep_results/            # VAEP 계산 결과
│       ├── player_match_vaep_england.csv
│       └── player_season_vaep_england.csv
└── statsbomb/                   # StatsBomb 관련 코드
```

## 디버그 모드 vs 전체 모드

### 디버그 모드 (이전)
- `--debug` 플래그 사용
- 리그당 5개 매치만 처리
- 학습 에포크: 3

### 전체 모드 (현재)
- 디버그 플래그 없이 실행
- 전체 매치 처리:
  - 학습 데이터: 1,561 matches (2.6M events)
  - 평가 데이터: 380 matches (England)
- 학습 에포크: 50 (config.yaml 설정)

## 파이프라인 실행 상태

### 1단계: 전처리 (완료)
- ✅ 전체 데이터 전처리 완료
- 입력: Wyscout JSON 파일 (6개 리그 + 2개 대회)
- 출력: CSV 파일 (학습용/평가용 분리)

### 2단계: 모델 학습 (진행 중)
- 🔄 현재 실행 중
- 데이터: 2,608,144 events (1,561 matches)
- 특징 차원: 108
- 레이블링 horizon: 10 events
- 모델 구조: [128, 64] hidden layers
- 배치 크기: 512
- 에포크: 50

**진행 상황**: 레이블 생성 단계 (시간 소요 중)

### 3단계: VAEP 계산 (대기 중)
- ⏳ 모델 학습 완료 후 실행 예정

## 파이프라인 코드 검토

### 중복/불필요한 부분
검토 결과 중복이나 불필요한 코드는 발견되지 않음:
- 각 스크립트는 명확한 역할 분리
- utils.py에 공통 함수 통합
- 설정은 config.yaml로 중앙화

### 로직 검토
- ✅ 전처리: England를 평가용으로 분리하는 로직 올바름
- ✅ 학습: VAEP 논문의 방법론 올바르게 구현
- ✅ 평가: goalkeeper 제외 로직 적용됨
- ✅ 집계: 경기당/시즌 VAEP 계산 올바름

## 실행 방법

### 전체 파이프라인 실행
```bash
cd /root/AIS/scripts
source /root/ais_venv/bin/activate
python run_pipeline.py
```

### 단계별 실행
```bash
# 1. 전처리 (이미 완료)
python preprocess_wyscout.py

# 2. 모델 학습
python train_vaep_model.py --config ../config.yaml

# 3. VAEP 계산
python compute_player_vaep.py
```

### 전처리 건너뛰기 (현재 사용)
```bash
python run_pipeline.py --skip-preprocess
```

## 다음 단계

1. ✅ 프로젝트 구조 리팩토링 완료
2. 🔄 전체 파이프라인 실행 중
3. ⏳ 모델 학습 완료 대기
4. ⏳ VAEP 계산 실행
5. ⏳ 성능 분석 노트북으로 결과 분석

## 예상 산출물

### 모델 파일
- `/AIS/models/vaep_model.pt`: 학습된 PyTorch 모델
- `/AIS/models/vaep_config.json`: 모델 설정 정보

### 결과 파일
- `/AIS/data/vaep_results/player_match_vaep_england.csv`: 선수-경기별 VAEP
- `/AIS/data/vaep_results/player_season_vaep_england.csv`: 선수-시즌별 VAEP (평균)

### 분석 노트북
- `/AIS/notebooks/vaep_performance_analysis.ipynb`: 결과 분석 및 시각화
