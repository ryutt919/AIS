# Atomic SPADL 변환 작업 요약

## ✅ 완료된 작업

### 1. 전처리 스크립트 생성
- **파일**: `scripts/preprocess_wyscout_atomic.py`
- **기능**: 
  - Wyscout 원본 데이터를 socceraction을 통해 SPADL로 변환
  - SPADL을 Atomic SPADL로 변환
  - 학습용/평가용 데이터 분리 및 저장

### 2. 모델 학습 스크립트 생성
- **파일**: `scripts/train_vaep_model_atomic.py`
- **기능**:
  - Atomic SPADL 형식의 데이터 로드
  - socceraction의 Atomic VAEP feature transformer 사용
  - PyTorch 모델 학습

## 🔧 주요 변경 사항

### 데이터 형식 변경

**기존 (Wyscout 원본):**
- `start_x`, `start_y`, `end_x`, `end_y` (0-1 정규화)
- `eventId`, `subEventId` (Wyscout 원본 타입)
- `is_goal`, `is_successful` (태그 기반)

**변경 (Atomic SPADL):**
- `x`, `y`, `dx`, `dy` (메터 단위: x: 0-105, y: 0-68)
- `type_id`, `type_name` (SPADL 표준 타입, 33개)
- `goal` 타입이 별도 액션으로 분리

### Feature 생성 방식 변경

**기존:**
- 수동으로 eventId, subEventId, tags를 원핫 인코딩
- 직접 계산한 distance, angle, goal_distance 등 사용

**변경:**
- `socceraction.atomic.vaep.features` 모듈 사용
- Game state 기반 feature (이전 3개 액션 포함)
- 표준화된 feature transformer 사용

## ⚠️ 필요한 추가 작업

### 1. 의존성 설치
```bash
cd /root/AIS/socceraction
pip install -e .
```

### 2. `compute_player_vaep.py` 수정
- Atomic SPADL 형식의 입력 데이터 처리
- `matchId` → `game_id` 컬럼명 변경
- 액션 집계 로직 수정

### 3. 리그 이름 매핑 확인
현재 코드에서 `competition_name`을 리그 이름으로 매핑하는 부분이 있습니다:
```python
league_mapping = {
    "English first division": "England",
    "Spanish first division": "Spain",
    # ...
}
```
실제 데이터의 `competition_name` 값에 맞게 수정이 필요할 수 있습니다.

## 🧪 테스트 방법

### 1. 전처리 테스트
```bash
cd /root/AIS/VAEP/scripts
PYTHONPATH=/root/AIS/socceraction:$PYTHONPATH python3 preprocess_wyscout_atomic.py \
    --debug \
    --data_dir ../data/wyscout \
    --output_dir ../data/processed
```

### 2. 출력 확인
- `data/processed/vaep_train_atomic_spadl.csv`
- `data/processed/vaep_eval_atomic_spadl_england.csv`

**예상 컬럼:**
- `game_id`, `action_id`, `period_id`, `time_seconds`
- `team_id`, `player_id`
- `x`, `y`, `dx`, `dy`
- `type_id`, `bodypart_id`
- `type_name`, `bodypart_name` (선택)
- `league`

### 3. 모델 학습 테스트
```bash
PYTHONPATH=/root/AIS/socceraction:$PYTHONPATH python3 train_vaep_model_atomic.py \
    --input ../data/processed/vaep_train_atomic_spadl.csv \
    --output_dir ../data/models/atomic_vaep \
    --debug
```

## 📊 예상 차이점

### 액션 수 증가
- Atomic SPADL은 패스 발신/수신을 분리하므로 액션 수가 약 1.5-2배 증가할 수 있습니다.
- 예: 기존 1000개 이벤트 → Atomic SPADL 1500-2000개 액션

### Feature 차원 변경
- Atomic VAEP feature는 기존과 다른 transformer를 사용하므로 feature 수가 다를 수 있습니다.
- Game state 기반이므로 이전 액션 정보가 포함됩니다.

### 처리 시간 증가
- SPADL 변환 → Atomic SPADL 변환 과정이 추가되어 처리 시간이 증가합니다.
- Feature 생성도 더 복잡하므로 시간이 더 걸릴 수 있습니다.

## 🔍 확인 사항

### 1. 데이터 로더 동작 확인
```python
loader = PublicWyscoutLoader(root=data_dir)
competitions = loader.competitions()
print(competitions)

# 특정 경기 테스트
game_id = 2499719  # 예시
events = loader.events(game_id)
print(f"Events: {len(events)}")
```

### 2. SPADL 변환 확인
```python
spadl_actions = wyscout_to_spadl(events, home_team_id)
print(f"SPADL actions: {len(spadl_actions)}")
print(spadl_actions[['type_name', 'result_name']].value_counts())
```

### 3. Atomic SPADL 변환 확인
```python
atomic_actions = atomicspadl.convert_to_atomic(spadl_actions)
print(f"Atomic actions: {len(atomic_actions)}")
print(f"Action types: {atomic_actions['type_name'].value_counts()}")
print(f"Receival actions: {(atomic_actions['type_name'] == 'receival').sum()}")
```

## 📝 다음 단계

1. **의존성 설치 및 환경 설정**
2. **소규모 데이터로 전처리 테스트** (--debug 모드)
3. **출력 데이터 형식 확인**
4. **`compute_player_vaep.py` 수정**
5. **전체 파이프라인 통합 테스트**

## 📚 참고 파일

- `ATOMIC_SPADL_MIGRATION_GUIDE.md` - 상세한 마이그레이션 가이드
- `scripts/preprocess_wyscout_atomic.py` - 전처리 스크립트
- `scripts/train_vaep_model_atomic.py` - 학습 스크립트

