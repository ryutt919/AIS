# Atomic SPADL 변환 가이드

## 📋 개요

이 문서는 `/VAEP/scripts` 폴더의 데이터 전처리 방식을 Atomic SPADL 형식으로 변경하는 방법을 설명합니다.

## 🔄 변경 사항

### 1. 새로운 파일

#### `preprocess_wyscout_atomic.py`
- **기존**: `preprocess_wyscout.py` - Wyscout 원본 이벤트를 직접 처리
- **변경**: `preprocess_wyscout_atomic.py` - socceraction을 사용하여 SPADL → Atomic SPADL 변환

**주요 차이점:**
- `socceraction.data.wyscout.PublicWyscoutLoader` 사용
- `socceraction.spadl.wyscout.convert_to_actions`로 SPADL 변환
- `socceraction.atomic.spadl.convert_to_atomic`로 Atomic SPADL 변환
- 출력 형식: Atomic SPADL 스키마 (x, y, dx, dy 형식)

#### `train_vaep_model_atomic.py`
- **기존**: `train_vaep_model.py` - Wyscout 원본 이벤트 기반 feature 생성
- **변경**: `train_vaep_model_atomic.py` - Atomic SPADL 기반 feature 생성

**주요 차이점:**
- `socceraction.atomic.vaep.features` 모듈 사용
- Atomic SPADL의 x, y, dx, dy 좌표 시스템 활용
- Game state 기반 feature 생성

## 📊 데이터 형식 비교

### 기존 형식 (Wyscout 원본)
```python
컬럼:
- id, matchId, playerId, teamId
- eventId, subEventId
- start_x, start_y, end_x, end_y  # 0-1 정규화
- distance, angle, goal_distance, goal_angle
- is_goal, is_successful
- period, event_sec
```

### Atomic SPADL 형식
```python
컬럼:
- game_id, action_id, period_id, time_seconds
- team_id, player_id
- x, y, dx, dy  # 메터 단위 (x: 0-105, y: 0-68)
- type_id, bodypart_id
- type_name, bodypart_name (선택)
- league (추가)
```

## 🔧 필요한 수정 사항

### 1. 의존성 설치

```bash
# socceraction 패키지 설치
cd /root/AIS/socceraction
pip install -e .

# 또는 필요한 의존성만 설치
pip install pandas numpy scikit-learn pandera
```

### 2. 데이터 경로 확인

`socceraction`의 `PublicWyscoutLoader`는 다음과 같은 파일 구조를 기대합니다:

```
data/wyscout/
├── matches_England.json
├── matches_Spain.json
├── matches_France.json
├── matches_Germany.json
├── matches_Italy.json
├── matches_European_Championship.json
├── matches_World_Cup.json
├── events_England.json
├── events_Spain.json
├── events_France.json
├── events_Germany.json
├── events_Italy.json
├── events_European_Championship.json
├── events_World_Cup.json
├── players.json
└── teams.json
```

### 3. 코드 수정 필요 사항

#### `preprocess_wyscout_atomic.py` 수정 필요

**문제점 1: PublicWyscoutLoader의 인덱스 구조**
- `PublicWyscoutLoader`는 내부적으로 `_index`를 사용하여 경기를 찾습니다.
- 현재 코드는 모든 경기를 직접 로드하려고 시도하지만, 로더의 인덱스 구조와 맞지 않을 수 있습니다.

**해결 방법:**
```python
# loader의 인덱스에서 경기 목록 가져오기
loader = PublicWyscoutLoader(root=data_dir)
competitions = loader.competitions()

# 각 competition/season별로 경기 로드
for _, comp in competitions.iterrows():
    games = loader.games(comp.competition_id, comp.season_id)
    for _, game in games.iterrows():
        game_id = game.game_id
        # 처리...
```

**문제점 2: 경기 ID 매핑**
- Wyscout 원본 데이터의 `wyId`와 `matchId`가 다를 수 있습니다.
- `PublicWyscoutLoader`는 `game_id`를 사용합니다.

**해결 방법:**
```python
# loader의 게임 인덱스 확인
loader._match_index  # game_id -> competition_id, season_id 매핑
```

#### `train_vaep_model_atomic.py` 수정 필요

**문제점 1: Feature 생성 시 게임별 처리**
- Atomic SPADL feature는 게임 상태(gamestates)를 기반으로 생성됩니다.
- 현재 코드는 게임별로 처리하지만, 대용량 데이터에서는 메모리 문제가 발생할 수 있습니다.

**해결 방법:**
- 배치 처리 또는 청크 단위 처리 추가
- HDF5 저장 형식 고려

**문제점 2: Label 생성 로직**
- Atomic SPADL에서는 `goal` 타입이 별도 액션으로 분리됩니다.
- 기존 코드는 `is_goal` 태그를 찾지만, Atomic SPADL에서는 `type_id == goal_type_id`를 확인해야 합니다.

**현재 구현:**
```python
goal_type_id = atomicspadl.actiontypes.index("goal")
goals = future_actions[
    (future_actions["type_id"] == goal_type_id)
    & (future_actions["team_id"] == team_id)
]
```

이 부분은 이미 올바르게 구현되어 있습니다.

### 4. `compute_player_vaep.py` 수정 필요

**현재 상태:** 확인 필요

**예상 수정 사항:**
- Atomic SPADL 형식의 입력 데이터 처리
- `matchId` → `game_id` 컬럼명 변경
- Atomic SPADL의 액션 구조에 맞게 집계 로직 수정

## 🧪 테스트 방법

### 1. 전처리 테스트

```bash
cd /root/AIS/VAEP/scripts
python3 preprocess_wyscout_atomic.py \
    --debug \
    --data_dir ../data/wyscout \
    --output_dir ../data/processed
```

**예상 출력:**
- `data/processed/vaep_train_atomic_spadl.csv`
- `data/processed/vaep_eval_atomic_spadl_england.csv`

### 2. 모델 학습 테스트

```bash
python3 train_vaep_model_atomic.py \
    --input ../data/processed/vaep_train_atomic_spadl.csv \
    --output_dir ../data/models/atomic_vaep \
    --debug
```

## ⚠️ 주의사항

### 1. 메모리 사용량
- Atomic SPADL은 일반 SPADL보다 액션 수가 많습니다 (패스 발신/수신 분리).
- 대용량 데이터 처리 시 메모리 부족 가능성이 있습니다.

### 2. 처리 시간
- SPADL 변환 → Atomic SPADL 변환 과정이 추가되어 처리 시간이 증가합니다.
- 게임별 순차 처리로 인해 병렬화가 어렵습니다.

### 3. Feature 차원
- Atomic SPADL feature는 socceraction의 transformer를 사용하므로 feature 수가 다를 수 있습니다.
- 기존 모델과 호환되지 않을 수 있습니다.

## 📝 체크리스트

- [ ] socceraction 패키지 설치 확인
- [ ] 데이터 경로 확인
- [ ] `preprocess_wyscout_atomic.py` 실행 테스트
- [ ] 출력 CSV 파일 형식 확인
- [ ] `train_vaep_model_atomic.py` 실행 테스트
- [ ] Feature 차원 및 분포 확인
- [ ] `compute_player_vaep.py` 수정 및 테스트
- [ ] 전체 파이프라인 통합 테스트

## 🔍 디버깅 팁

### 1. 로더 초기화 확인
```python
loader = PublicWyscoutLoader(root=data_dir)
print(loader.competitions())  # 사용 가능한 대회 확인
```

### 2. 단일 경기 테스트
```python
game_id = 2499719  # 예시
events = loader.events(game_id)
print(f"Events: {len(events)}")
print(events.head())
```

### 3. SPADL 변환 확인
```python
spadl_actions = wyscout_to_spadl(events, home_team_id)
print(f"SPADL actions: {len(spadl_actions)}")
print(spadl_actions.head())
```

### 4. Atomic SPADL 변환 확인
```python
atomic_actions = atomicspadl.convert_to_atomic(spadl_actions)
print(f"Atomic actions: {len(atomic_actions)}")
print(atomic_actions.head())
print(f"Action types: {atomic_actions['type_name'].value_counts()}")
```

## 📚 참고 자료

- [socceraction 문서](https://socceraction.readthedocs.io/)
- [Atomic SPADL 설명](https://socceraction.readthedocs.io/en/latest/documentation/spadl/atomic_spadl.html)
- [Atomic VAEP 예제](https://github.com/ML-KULeuven/socceraction/tree/master/public-notebooks)

