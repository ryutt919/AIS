# VAEP (Valuing Actions by Estimating Probabilities) 구현

PyTorch를 사용한 축구 이벤트 데이터 기반 VAEP 모델 구현

## 📋 개요

이 프로젝트는 Wyscout 축구 이벤트 데이터를 사용하여 VAEP를 구현합니다:

- **학습 데이터**: England를 제외한 모든 리그 (Spain, France, Germany, Italy, European Championship, World Cup)
- **평가 데이터**: England Premier League
- **모델**: PyTorch MLP (Multi-Layer Perceptron)
- **목표**: 선수별 경기당 및 시즌 평균 VAEP 계산

## 🗂️ 프로젝트 구조

```
AIS/
├── data/
│   ├── wyscout/                    # 원본 Wyscout 데이터
│   ├── processed/                  # 전처리된 데이터
│   │   ├── vaep_train_events.csv         # 학습용 (England 제외)
│   │   └── vaep_eval_events_england.csv  # 평가용 (England)
│   ├── models/                     # 학습된 모델
│   │   ├── vaep_model.pt                 # PyTorch 모델 가중치
│   │   └── vaep_config.json              # 모델 설정
│   └── vaep_results/               # VAEP 계산 결과
│       ├── player_match_vaep_england.csv   # 선수-경기별 VAEP
│       └── player_season_vaep_england.csv  # 선수-시즌별 VAEP
└── scripts/
    ├── utils.py                    # 공통 유틸리티 함수
    ├── preprocess_wyscout.py       # 데이터 전처리
    ├── train_vaep_model.py         # 모델 학습
    └── compute_player_vaep.py      # 선수 VAEP 계산
```

## 🚀 실행 방법

### 1단계: 데이터 전처리

Wyscout 이벤트 데이터를 VAEP 형식으로 전처리합니다.

```powershell
cd scripts
python preprocess_wyscout.py
```

**출력:**

- `data/processed/vaep_train_events.csv` - 학습용 데이터 (England 제외)
- `data/processed/vaep_eval_events_england.csv` - 평가용 데이터 (England만)

**옵션:**

```powershell
python preprocess_wyscout.py `
    --data_dir ../data/wyscout `
    --output_dir ../data/processed `
    --log_file preprocess.log
```

### 2단계: VAEP 모델 학습

PyTorch MLP 모델을 학습합니다.

```powershell
python train_vaep_model.py
```

**출력:**

- `data/models/vaep_model.pt` - 학습된 모델 가중치
- `data/models/vaep_config.json` - 모델 설정 (특징, 하이퍼파라미터)

**옵션:**

```powershell
python train_vaep_model.py `
    --input ../data/processed/vaep_train_events.csv `
    --output_dir ../data/models `
    --horizon 10 `
    --hidden_dims 128 64 `
    --batch_size 512 `
    --epochs 50 `
    --lr 0.001 `
    --val_ratio 0.2 `
    --log_file train.log
```

**주요 파라미터:**

- `--horizon`: 라벨링 horizon (기본: 10 이벤트)
- `--hidden_dims`: MLP 히든 레이어 차원 (기본: 128 64)
- `--epochs`: 학습 에포크 수 (기본: 50)
- `--lr`: 학습률 (기본: 0.001)

### 3단계: 선수 VAEP 계산

학습된 모델로 England 데이터를 평가하고 선수별 VAEP를 계산합니다.

```powershell
python compute_player_vaep.py
```

**출력:**

- `data/vaep_results/player_match_vaep_england.csv` - 선수-경기별 VAEP
- `data/vaep_results/player_season_vaep_england.csv` - 선수-시즌별 VAEP

**옵션:**

```powershell
python compute_player_vaep.py `
    --input ../data/processed/vaep_eval_events_england.csv `
    --model_path ../data/models/vaep_model.pt `
    --config_path ../data/models/vaep_config.json `
    --matches_path ../data/wyscout/matches_England.json `
    --output_dir ../data/vaep_results `
    --log_file compute.log
```

## 📊 출력 데이터 설명

### player_match_vaep_england.csv

선수-경기별 VAEP 데이터:

| 컬럼             | 설명                 |
| ---------------- | -------------------- |
| `playerId`       | 선수 ID              |
| `matchId`        | 경기 ID              |
| `teamId`         | 팀 ID                |
| `vaep`           | 경기에서의 총 VAEP   |
| `num_events`     | 경기에서의 이벤트 수 |
| `minutes_played` | 출전 시간 (분)       |
| `vaep_per90`     | 90분당 VAEP          |

### player_season_vaep_england.csv

선수-시즌별 VAEP 데이터:

| 컬럼                    | 설명              |
| ----------------------- | ----------------- |
| `playerId`              | 선수 ID           |
| `matches_played`        | 출전 경기 수      |
| `season_vaep_total`     | 시즌 총 VAEP      |
| `season_vaep_per90_avg` | 평균 VAEP/90분    |
| `season_vaep_per_match` | 경기당 평균 VAEP  |
| `minutes_played`        | 총 출전 시간 (분) |
| `num_events`            | 총 이벤트 수      |

## 🧮 VAEP 계산 방법

### 1. State Representation

각 이벤트의 게임 상태를 특징 벡터로 표현:

- 이벤트 타입 (원핫 인코딩)
- 위치 정보 (start_x, start_y, end_x, end_y)
- 골까지의 거리 및 각도
- 이동 거리 및 방향
- 성공 여부

### 2. Labeling

각 이벤트 후 horizon(기본 10 이벤트) 내에:

- `y_score`: 우리 팀이 득점하면 1, 아니면 0
- `y_concede`: 상대 팀이 득점하면 1, 아니면 0

### 3. Model Training

PyTorch MLP 모델을 학습하여 예측:

- `P(score | state)`: 상태에서 득점 확률
- `P(concede | state)`: 상태에서 실점 확률

**손실 함수**: Binary Cross-Entropy with Logits
**옵티마이저**: Adam

### 4. Value Calculation

**State Value** (VAEP 논문의 핵심 수식):

```
V(s_t) = P(score | s_t) - P(concede | s_t)
```

**Event VAEP** (액션의 가치):

```
VAEP(a_t) = V(s_{t+1}) - V(s_t)
```

여기서:

- `a_t`: 시간 t의 액션(이벤트)
- `s_t`: 액션 전의 게임 상태
- `s_{t+1}`: 액션 후의 게임 상태

### 5. Player Aggregation

**경기당 VAEP**:

```
VAEP_match = Σ VAEP(actions in match)
VAEP_per90 = VAEP_match × 90 / minutes_played
```

**시즌 평균 VAEP**:

```
Season_VAEP_per_match = Total_VAEP / matches_played
Season_VAEP_per90_avg = Average(VAEP_per90 across matches)
```

## 🔧 기술 스택

- **Python 3.8+**
- **PyTorch**: 딥러닝 모델 구현
- **pandas**: 데이터 처리
- **numpy**: 수치 계산

## 📝 코드 특징

1. **모듈화**: 공통 기능은 `utils.py`로 분리
2. **타입 힌트**: 모든 함수에 타입 힌트 추가
3. **Docstring**: 각 함수에 상세한 설명 추가
4. **로깅**: 진행 상황과 통계를 상세히 기록
5. **예외 처리**: 적절한 에러 핸들링
6. **CLI 지원**: argparse를 통한 명령줄 실행

## 📚 참고 문헌

VAEP 논문:

- Decroos, T., Bransen, L., Van Haaren, J., & Davis, J. (2019).
  "Actions Speak Louder than Goals: Valuing Player Actions in Soccer"
  Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining

## ⚠️ 주의사항

1. **메모리**: 이벤트 데이터가 크므로 충분한 RAM 필요 (8GB 이상 권장)
2. **시간**: 전체 파이프라인 실행에 수십 분 소요될 수 있음
3. **GPU**: CUDA 사용 가능 시 자동으로 GPU 활용
4. **데이터**: wyscout 폴더에 모든 이벤트 데이터가 있어야 함

## 🐛 문제 해결

### 메모리 부족

- `--batch_size`를 줄이기 (예: 256, 128)
- 더 작은 `--hidden_dims` 사용

### 학습이 느림

- GPU가 있다면 CUDA 설치
- `--epochs`를 줄이기 (빠른 테스트용)

### 정확도가 낮음

- `--horizon`을 조정 (5-15 사이)
- `--epochs`를 늘리기
- 더 깊은 네트워크 사용 (예: `--hidden_dims 256 128 64`)

## 📧 문의

문제가 발생하면 로그 파일을 확인하세요:

```powershell
python preprocess_wyscout.py --log_file preprocess.log
python train_vaep_model.py --log_file train.log
python compute_player_vaep.py --log_file compute.log
```
