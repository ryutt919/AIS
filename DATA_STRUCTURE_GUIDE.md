# 축구 경기 데이터셋 구조 가이드

## 📊 데이터 규모 요약

### 총 경기 수: **1,941 경기**
| 리그 | 경기 수 |
|------|---------|
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England (Premier League) | 380 |
| 🇪🇸 Spain (La Liga) | 380 |
| 🇮🇹 Italy (Serie A) | 380 |
| 🇫🇷 France (Ligue 1) | 380 |
| 🇩🇪 Germany (Bundesliga) | 306 |
| 🏆 World Cup | 64 |
| 🇪🇺 European Championship | 51 |

### 총 이벤트 수: **3,251,294 이벤트**
| 리그 | 이벤트 수 |
|------|-----------|
| 🇮🇹 Italy | 647,372 |
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | 643,150 |
| 🇫🇷 France | 632,807 |
| 🇪🇸 Spain | 628,659 |
| 🇩🇪 Germany | 519,407 |
| 🏆 World Cup | 101,759 |
| 🇪🇺 European Championship | 78,140 |

### 기타 통계
- **선수**: 3,603명
- **팀**: 142개
- **감독**: 208명
- **대회**: 7개
- **Playerank 레코드**: 46,897개

---

## 📁 데이터 파일 구조

```
data/
├── matches_[리그명].json          # 경기 메타데이터 (1,941개 경기)
├── events_[리그명].json           # 이벤트 시퀀스 (3.2M+ 이벤트, 50MB+)
├── players.json                   # 선수 프로필 (3,603명)
├── teams.json                     # 팀 정보 (142개)
├── coaches.json                   # 감독 정보 (208명)
├── referees.json                  # 심판 정보
├── competitions.json              # 대회 정보 (7개)
├── playerank.json                 # 선수 평가 지표 (46,897개)
├── eventid2name.csv              # 이벤트 타입 매핑표
└── tags2name.csv                 # 태그 의미 매핑표
```

---

## 🔍 데이터 구조 상세 설명

## 1. `matches_[리그명].json` - 경기 메타데이터

각 경기의 기본 정보, 라인업, 점수, 교체 정보 등을 포함

### 데이터 구조
```json
{
  "wyId": 2499719,                    // 경기 고유 ID (Primary Key)
  "status": "Played",                 // 경기 상태 (항상 "Played")
  "seasonId": 181248,                 // 시즌 ID
  "competitionId": 364,               // 대회 ID
  "gameweek": 38,                     // 라운드/주차 (1-38)
  "roundId": 4405654,                 // 라운드 고유 ID
  "date": "May 13, 2018 at 3:00:00 PM GMT+1",
  "dateutc": "2018-05-13 14:00:00",
  "venue": "Stamford Bridge",         // 경기장
  "duration": "Regular",              // "Regular" / "ExtraTime" / "Penalties"
  "winner": 1646,                     // 승리팀 ID (0=무승부)
  "label": "Chelsea - Newcastle United, 1 - 2",
  
  "teamsData": {
    "1646": {                         // 홈팀 데이터
      "teamId": 1646,
      "side": "home",
      "coachId": 8880,                // 감독 ID
      "score": 1,                     // 최종 점수
      "scoreHT": 1,                   // 전반전 점수
      "scoreET": 0,                   // 연장전 점수
      "scoreP": 0,                    // 승부차기 점수
      "hasFormation": 1,              // 포메이션 데이터 존재 여부
      
      "formation": {
        "lineup": [                   // 선발 라인업 (11명)
          {
            "playerId": 9206,
            "goals": "1",             // 득점 수 ("null" 가능)
            "ownGoals": "0",          // 자책골 수
            "yellowCards": "0",       // 경고
            "redCards": "0"           // 퇴장
          }
        ],
        
        "bench": [                    // 벤치 선수 (7-12명)
          {
            "playerId": 77502,
            "goals": "null",
            "ownGoals": "0",
            "yellowCards": "0",
            "redCards": "0"
          }
        ],
        
        "substitutions": [            // 교체 (0-3회)
          {
            "playerOut": 9206,
            "playerIn": 9127,
            "minute": 61
          }
        ]
      }
    },
    
    "1659": {                         // 원정팀 데이터 (구조 동일)
      // ... 홈팀과 동일한 구조
    }
  },
  
  "referees": [                       // 심판진
    {
      "refereeId": 377214,
      "role": "referee"               // "referee" / "firstAssistant" / "secondAssistant"
    }
  ]
}
```

### MLP 활용 가능 특징
- `gameweek`: 시즌 진행도 (1-38)
- `score`, `scoreHT`: 득점 패턴
- `lineup` 선수 구성 (포지션별)
- `substitutions.minute`: 교체 타이밍 전략

---

## 2. `events_[리그명].json` - 경기 이벤트 시퀀스 ⭐ 핵심 데이터

경기 중 발생한 모든 이벤트를 시간순으로 기록 (패스, 슈팅, 태클 등)

### 데이터 구조
```json
{
  "id": 177959171,                    // 이벤트 고유 ID
  "matchId": 2499719,                 // 경기 ID (matches와 조인)
  "playerId": 25413,                  // 행위 선수 ID
  "teamId": 1609,                     // 팀 ID
  
  "eventId": 8,                       // 이벤트 메인 카테고리 (1-10)
  "eventName": "Pass",                // 이벤트 이름
  "subEventId": 85,                   // 세부 이벤트 타입 (10-100)
  "subEventName": "Simple pass",      // 세부 이벤트 이름
  
  "matchPeriod": "1H",                // "1H", "2H", "E1", "E2", "P"
  "eventSec": 2.758649,               // 이벤트 발생 시각 (초)
  
  "positions": [                      // 항상 2개 요소
    {
      "x": 49,                        // 시작 X 좌표 (0-100)
      "y": 49                         // 시작 Y 좌표 (0-100)
    },
    {
      "x": 49,                        // 끝 X 좌표
      "y": 41                         // 끝 Y 좌표
    }
  ],
  
  "tags": [                           // 이벤트 속성 태그 (0-5개)
    {
      "id": 1801                      // 1801 = "accurate" (정확)
    }
  ]
}
```

### 좌표 시스템
- **X축 (0-100)**: 필드의 좌 → 우 (0 = 자기편 골대, 100 = 상대편 골대)
- **Y축 (0-100)**: 필드의 아래 → 위
- **정규화된 좌표**: 모든 경기장 크기에 무관하게 0-100 범위

### 이벤트 타입 (10가지)

| eventId | eventName | 설명 |
|---------|-----------|------|
| 1 | Duel | 볼 경합 (공중/지상) |
| 2 | Foul | 파울 |
| 3 | Free Kick | 프리킥/코너/스로인/골킥/페널티 |
| 4 | Goalkeeper leaving line | 골키퍼 전진 |
| 5 | Interruption | 경기 중단 |
| 6 | Offside | 오프사이드 |
| 7 | Others on the ball | 가속/클리어런스/터치 |
| 8 | Pass | 패스 (6가지 하위 타입) |
| 9 | Save attempt | 선방 |
| 10 | Shot | 슈팅 |

### 서브이벤트 타입 (37가지)

#### Duel (1)
- 10: Air duel (공중볼 경합)
- 11: Ground attacking duel (지상 공격 경합)
- 12: Ground defending duel (지상 수비 경합)
- 13: Ground loose ball duel (루즈볼 경합)

#### Foul (2)
- 20: Foul (일반 파울)
- 21: Hand foul (핸드볼)
- 22: Late card foul (지연 카드 파울)
- 23: Out of game foul (경기 외 파울)
- 24: Protest (항의)
- 25: Simulation (시뮬레이션)
- 26: Time lost foul (시간 지연 파울)
- 27: Violent Foul (폭력적 파울)

#### Free Kick (3)
- 30: Corner (코너킥)
- 31: Free Kick (프리킥)
- 32: Free kick cross (프리킥 크로스)
- 33: Free kick shot (프리킥 슛)
- 34: Goal kick (골킥)
- 35: Penalty (페널티킥)
- 36: Throw in (스로인)

#### Others on the ball (7)
- 70: Acceleration (가속)
- 71: Clearance (클리어런스)
- 72: Touch (터치)

#### Pass (8)
- 80: Cross (크로스)
- 81: Hand pass (핸드 패스)
- 82: Head pass (헤딩 패스)
- 83: High pass (높은 패스)
- 84: Launch (롱패스)
- 85: Simple pass (단순 패스)
- 86: Smart pass (스마트 패스)

#### Save attempt (9)
- 90: Reflexes (반사 신경)
- 91: Save attempt (선방 시도)

#### Shot (10)
- 100: Shot (슈팅)

### 태그 시스템 (60가지)

#### 결과 관련
| Tag ID | Label | 설명 |
|--------|-------|------|
| 101 | Goal | 골 |
| 102 | own_goal | 자책골 |
| 301 | assist | 어시스트 |
| 302 | keyPass | 키패스 |

#### 정확도
| Tag ID | Label | 설명 |
|--------|-------|------|
| 1801 | accurate | 정확 |
| 1802 | not accurate | 부정확 |
| 2101 | blocked | 차단됨 |

#### 신체 부위
| Tag ID | Label | 설명 |
|--------|-------|------|
| 401 | Left | 왼발 |
| 402 | Right | 오른발 |
| 403 | head/body | 헤딩/몸 |

#### 높이
| Tag ID | Label | 설명 |
|--------|-------|------|
| 801 | high | 높음 |
| 802 | low | 낮음 |

#### 수비 액션
| Tag ID | Label | 설명 |
|--------|-------|------|
| 1401 | interception | 인터셉트 |
| 1501 | clearance | 클리어런스 |
| 1601 | sliding_tackle | 슬라이딩 태클 |

#### 카드
| Tag ID | Label | 설명 |
|--------|-------|------|
| 1701 | red_card | 레드카드 |
| 1702 | yellow_card | 옐로카드 |
| 1703 | second_yellow_card | 2번째 옐로 |

#### 듀얼 결과
| Tag ID | Label | 설명 |
|--------|-------|------|
| 701 | lost | 패배 |
| 702 | neutral | 중립 |
| 703 | won | 승리 |

#### 기타
| Tag ID | Label | 설명 |
|--------|-------|------|
| 1901 | counter_attack | 역습 |
| 2001 | dangerous_ball_lost | 위험한 볼 손실 |
| 201 | opportunity | 기회 |
| 901 | through | 스루패스 |
| 1001 | fairplay | 페어플레이 |

---

## 3. `players.json` - 선수 프로필

### 데이터 구조
```json
{
  "wyId": 32777,                      // 선수 고유 ID
  "firstName": "Harun",
  "middleName": "",
  "lastName": "Tekin",
  "shortName": "H. Tekin",
  
  "birthDate": "1989-06-17",
  "height": 187,                      // 키 (cm)
  "weight": 78,                       // 몸무게 (kg)
  "foot": "right",                    // "left" / "right" / "both"
  
  "role": {
    "name": "Goalkeeper",             // 포지션 전체 이름
    "code2": "GK",                    // 2자리 코드
    "code3": "GKP"                    // 3자리 코드
  },
  
  "currentTeamId": 4502,              // 현재 소속 클럽 ID
  "currentNationalTeamId": 4687,      // 국가대표팀 ID
  
  "birthArea": {                      // 출생 국가
    "id": "792",
    "name": "Turkey",
    "alpha2code": "TR",
    "alpha3code": "TUR"
  },
  
  "passportArea": {                   // 여권 국가
    "id": "792",
    "name": "Turkey",
    "alpha2code": "TR",
    "alpha3code": "TUR"
  }
}
```

### 포지션 코드
- **GK** (Goalkeeper): 골키퍼
- **DF** (Defender): 수비수
- **MF** (Midfielder): 미드필더
- **FW** (Forward): 공격수

### MLP 특징 추출 예시
```python
# 나이 계산
age = 2018 - int(birthDate[:4])

# BMI 계산
bmi = weight / (height/100)**2

# 포지션 원핫 인코딩
position_onehot = [1, 0, 0, 0]  # [GK, DF, MF, FW]

# 주발 인코딩
foot_encoding = {"left": 0, "right": 1, "both": 2}
```

---

## 4. `teams.json` - 팀 정보

### 데이터 구조
```json
{
  "wyId": 1613,                       // 팀 고유 ID
  "name": "Newcastle United",         // 팀명
  "officialName": "Newcastle United FC",
  "city": "Newcastle upon Tyne",      // 연고지
  "type": "club",                     // "club" or "national"
  
  "area": {                           // 국가 정보
    "id": "0",
    "name": "England",
    "alpha2code": "",
    "alpha3code": "XEN"
  }
}
```

---

## 5. `playerank.json` - 선수 경기별 평가 지표

선수의 경기별 성과를 수치화한 평가 점수 (Wyscout 자체 알고리즘)

### 데이터 구조
```json
{
  "matchId": 2057991,                 // 경기 ID
  "playerId": 10014,                  // 선수 ID
  "playerankScore": 0.0053,           // 평가 점수 (-0.1 ~ 0.1)
  "roleCluster": "right CB",          // 경기 내 역할
  "minutesPlayed": 90,                // 출전 시간 (분)
  "goalScored": 0                     // 득점 여부 (0 or 1)
}
```

### roleCluster 타입 (약 30가지)

#### 골키퍼
- `GK`: 골키퍼

#### 수비수
- `left CB`, `central CB`, `right CB`: 센터백 (좌/중앙/우)
- `left FB`, `right FB`: 풀백
- `left WB`, `right WB`: 윙백

#### 미드필더
- `defensive MF`: 수비형 미드필더
- `central MF`: 중앙 미드필더
- `left MF`, `right MF`: 좌우 미드필더
- `attacking MF`: 공격형 미드필더

#### 공격수
- `left W`, `right W`: 윙어 (좌우)
- `CF`: 중앙 공격수
- `left CF`, `right CF`: 측면 공격수

### MLP 활용
- **타겟 레이블**: `playerankScore`를 예측 목표로 사용
- **검증 데이터**: 모델 예측값과 실제 점수 비교
- **특징 추가**: `roleCluster`를 입력 특징으로 활용

---

## 6. `competitions.json` - 대회 정보

### 데이터 구조
```json
{
  "wyId": 364,                        // 대회 ID
  "name": "English first division",   // 대회명
  "format": "Domestic league",        // "Domestic league" / "International cup"
  "type": "club",                     // "club" or "national"
  
  "area": {                           // 국가/지역
    "id": "0",
    "name": "England",
    "alpha2code": "",
    "alpha3code": "XEN"
  }
}
```

### 포함된 대회
| wyId | 대회명 | 타입 |
|------|--------|------|
| 364 | English first division | Domestic league |
| 795 | Spanish first division | Domestic league |
| 524 | Italian first division | Domestic league |
| 412 | French first division | Domestic league |
| 426 | German first division | Domestic league |
| 102 | World Cup | International cup |
| 102 | European Championship | International cup |

---

## 7. `coaches.json` - 감독 정보

### 데이터 구조
```json
{
  "wyId": 14710,                      // 감독 ID
  "firstName": "Josef",
  "middleName": "",
  "lastName": "Heynckes",
  "shortName": "J. Heynckes",
  
  "birthDate": "1945-05-09",          // null 가능
  
  "birthArea": {
    "id": 276,
    "name": "Germany",
    "alpha2code": "DE",
    "alpha3code": "DEU"
  },
  
  "passportArea": {
    "id": 276,
    "name": "Germany",
    "alpha2code": "DE",
    "alpha3code": "DEU"
  },
  
  "currentTeamId": 0                  // 현재 소속팀 (0 = 무소속)
}
```

---

## 8. CSV 참조 파일

### `eventid2name.csv` - 이벤트 타입 매핑
```csv
event,subevent,event_label,subevent_label
1,10,Duel,Air duel
1,11,Duel,Ground attacking duel
8,85,Pass,Simple pass
10,100,Shot,Shot
```

### `tags2name.csv` - 태그 의미 매핑
```csv
Tag,Label,Description
101,Goal,Goal
301,assist,Assist
1801,accurate,Accurate pass/shot
402,Right,Right foot
```

---

## 🤖 MLP 모델 적용 가이드

### 1. 데이터 흐름 이해

```
경기 (matches.json)
  ├── 이벤트 시퀀스 (events_*.json)  ← 핵심 데이터!
  │     ├── 선수 정보 (players.json)
  │     ├── 팀 정보 (teams.json)
  │     └── 태그 정보 (tags2name.csv)
  └── 선수 평가 (playerank.json)  ← 타겟 레이블
```

### 2. 시퀀스 데이터 예시

각 경기는 시간순 이벤트 시퀀스로 구성:

```python
match_2499719 = [
  Event(t=2.76s, type="Pass", player=25413, x=49→49, y=49→41),
  Event(t=4.95s, type="Pass", player=3319, x=51→35, y=75→71),
  Event(t=6.54s, type="Pass", player=120339, x=35→41, y=71→95),
  ...
  Event(t=5420s, type="Shot", player=9206, tags=[101]),  # 골!
]
```

### 3. 특징 벡터 구성 예시

#### 이벤트별 특징 (300+ 차원)

```python
# 카테고리형 특징 (원핫 인코딩)
eventId_onehot        # 10차원 [0,0,0,0,0,0,0,1,0,0]  # Pass
subEventId_onehot     # 37차원
matchPeriod_onehot    # 5차원 [1H, 2H, E1, E2, P]

# 수치형 특징
eventSec              # 1차원 (0 ~ 6000)
start_x, start_y      # 2차원 (0-100)
end_x, end_y          # 2차원 (0-100)
distance              # 1차원 sqrt((x2-x1)^2 + (y2-y1)^2)
direction_angle       # 1차원 atan2(y2-y1, x2-x1)

# 선수 특징
player_age            # 1차원
player_height         # 1차원 (정규화)
player_weight         # 1차원 (정규화)
player_bmi            # 1차원
player_position       # 4차원 [GK, DF, MF, FW]
player_foot           # 3차원 [left, right, both]

# 경기 상황 특징
current_score_home    # 1차원
current_score_away    # 1차원
score_diff            # 1차원 (home - away)
time_remaining        # 1차원
is_home_team          # 1차원 (0 or 1)
gameweek              # 1차원 (1-38)

# 최근 이벤트 패턴 (Sliding Window)
prev_5_events         # 5 x (eventId + subEventId)
prev_event_time_gap   # 1차원

# 태그 (멀티핫 인코딩)
tags_vector           # 60차원 [0,0,1,0,0,1,...]
```

#### 경기별 집계 특징

```python
# 팀별 통계
total_passes          # 총 패스 수
pass_accuracy         # 패스 성공률 (tags: 1801)
total_shots           # 총 슈팅 수
shots_on_target       # 유효 슈팅 수
possession_pct        # 점유율 (이벤트 수 비율)
tackles_won           # 태클 성공 수
fouls_committed       # 파울 수

# 선수별 통계
player_touches        # 터치 수
player_passes         # 패스 시도
player_key_passes     # 키패스 (tag: 302)
player_shots          # 슈팅 수
player_goals          # 골 (tag: 101)
player_assists        # 어시스트 (tag: 301)
```

### 4. 모델링 시나리오

#### 시나리오 1: 이벤트 결과 예측
```python
# 입력: 이벤트 특징
X = [eventId, subEventId, position, player_features, ...]

# 출력: 이벤트 성공 여부
y = has_tag_1801  # 정확한 패스인가?
```

#### 시나리오 2: 다음 이벤트 예측
```python
# 입력: 과거 N개 이벤트 시퀀스
X = [event_t-4, event_t-3, event_t-2, event_t-1, event_t]

# 출력: 다음 이벤트 타입
y = next_event_id  # 10개 클래스 분류
```

#### 시나리오 3: 경기 결과 예측
```python
# 입력: 전반전 통계 (45분까지)
X = [team_stats, player_stats, score_HT, ...]

# 출력: 최종 승패
y = winner  # 3개 클래스 (home/draw/away)
```

#### 시나리오 4: 선수 성과 예측 ⭐ 추천
```python
# 입력: 선수의 경기 내 모든 이벤트 집계
X = [player_events_aggregated, opponent_strength, ...]

# 출력: Playerank 점수
y = playerankScore  # 회귀 (-0.1 ~ 0.1)
```

### 5. 데이터 전처리 팁

#### 좌표 정규화
```python
# 이미 0-100으로 정규화되어 있음
x_norm = x / 100.0  # 0.0 ~ 1.0
y_norm = y / 100.0
```

#### 시간 정규화
```python
# 전반전: 0-2700초, 후반전: 2700-5400초
time_norm = eventSec / 5400.0  # 0.0 ~ 1.0
```

#### 결측치 처리
```python
# goals, ownGoals 등이 "null" 문자열로 저장됨
goals = 0 if goals == "null" else int(goals)
```

#### 시퀀스 패딩
```python
# 경기별 이벤트 수가 다름 (평균 ~1600개)
# 고정 길이로 패딩 필요
max_seq_len = 2000
padded_sequence = pad_sequences(events, maxlen=max_seq_len)
```

### 6. 추천 워크플로우

#### Phase 1: 탐색적 데이터 분석 (EDA)
1. 단일 경기 이벤트 시퀀스 시각화
2. 이벤트 타입 분포 분석
3. 태그 공출현 패턴 분석
4. 선수/팀별 통계 계산

#### Phase 2: 베이스라인 모델
1. 단순 집계 특징으로 시작
2. 작은 데이터셋 (England 380경기만)
3. 간단한 분류 문제 (패스 성공 예측)
4. 다층 퍼셉트론 (MLP) 3-4층

#### Phase 3: 고도화
1. 시퀀스 특징 추가 (RNN/LSTM도 고려)
2. 전체 리그 데이터 활용
3. 앙상블 모델
4. 하이퍼파라미터 튜닝

### 7. 코드 스니펫

#### 데이터 로딩
```python
import json
import pandas as pd

# 경기 데이터
with open('matches_England.json') as f:
    matches = json.load(f)

# 이벤트 데이터
with open('events_England.json') as f:
    events = json.load(f)
    
# 선수 데이터
with open('players.json') as f:
    players = json.load(f)
    
# 매핑 테이블
event_map = pd.read_csv('eventid2name.csv')
tag_map = pd.read_csv('tags2name.csv')
```

#### 특정 경기 이벤트 추출
```python
match_id = 2499719
match_events = [e for e in events if e['matchId'] == match_id]
print(f"경기 {match_id}: {len(match_events)} 이벤트")
```

#### 패스 성공률 계산
```python
passes = [e for e in events if e['eventId'] == 8]
accurate_passes = [p for p in passes 
                   if any(t['id'] == 1801 for t in p['tags'])]
accuracy = len(accurate_passes) / len(passes)
print(f"패스 성공률: {accuracy:.2%}")
```

#### 선수별 이벤트 집계
```python
from collections import defaultdict

player_stats = defaultdict(lambda: {'events': 0, 'passes': 0, 'goals': 0})

for event in events:
    pid = event['playerId']
    player_stats[pid]['events'] += 1
    
    if event['eventId'] == 8:  # Pass
        player_stats[pid]['passes'] += 1
    
    if any(t['id'] == 101 for t in event['tags']):  # Goal
        player_stats[pid]['goals'] += 1
```

---

## 📝 주의사항

### 데이터 이슈
1. **대용량 파일**: `events_*.json` 파일들은 50MB+ (VS Code에서 열기 어려움)
2. **문자열 타입**: `goals`, `ownGoals` 등이 숫자가 아닌 `"null"` 문자열
3. **좌표계**: X축이 공격 방향 (0=자기편, 100=상대편)
4. **시간**: `eventSec`는 누적 시간 (전반 0-2700, 후반 2700-5400)

### 모델링 시 고려사항
1. **클래스 불균형**: 골(tag:101)은 매우 희소 → 오버샘플링 필요
2. **시퀀스 길이**: 경기별 이벤트 수 차이 큼 → 패딩/자르기 필요
3. **시간 의존성**: 이벤트 순서가 중요 → RNN/LSTM/Transformer 고려
4. **팀 효과**: 같은 팀의 선수들은 상관관계 높음 → 팀별 정규화

---

## 🎯 시작하기

### 추천 첫 번째 작업
1. England 리그 380경기로 시작
2. Pass 이벤트만 필터링 (가장 많음)
3. 패스 성공 여부 (tag: 1801) 이진 분류
4. 특징: [eventSec, start_x, start_y, end_x, end_y, player_position]
5. 간단한 3층 MLP 구축

### 평가 지표
- 분류: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- 회귀: MAE, RMSE, R²
- 시퀀스: Perplexity (다음 이벤트 예측)

---

## 📚 참고 자료

### 데이터 출처
- Wyscout (현 Hudl Wyscout)
- 논문: "A public data set of spatio-temporal match events in soccer competitions"

### 관련 연구 키워드
- Event-based soccer analytics
- Expected Goals (xG)
- Player performance rating
- Pass network analysis
- Sequence modeling in sports

---

**문서 작성일**: 2025년 11월 20일  
**데이터 기준 시즌**: 2017/18 시즌
