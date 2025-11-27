# 선수 통계 데이터 전처리 가이드

## 📋 목차

1. [전처리 개요](#전처리-개요)
2. [생성된 파일](#생성된-파일)
3. [데이터 구조](#데이터-구조)
4. [전처리 프로세스](#전처리-프로세스)
5. [활용 방법](#활용-방법)
6. [실제 데이터 예시](#실제-데이터-예시)

---

## 🎯 전처리 개요

**목적**: `player_match_stats` 폴더의 FBRef 스크래핑 데이터를 분석 가능한 형태로 변환

**입력 데이터**:

- `data/player_match_stats/2017-18_filtered.csv`
- `data/player_match_stats/2018-19_filtered.csv`

**출력 데이터**:

- 경기 레벨 데이터 (Match-Level)
- 선수 시즌 통계 (Player Season Stats)
- 포지션별 벤치마크 (Position Benchmarks)

**실행 명령**:

```bash
python scripts/preprocess_player_stats.py
```

---

## 📂 생성된 파일

### 1. 경기 레벨 데이터 (Match-Level)

| 파일명                    | 행 수  | 설명         |
| ------------------------- | ------ | ------------ |
| `match_level_2017-18.csv` | 10,448 | 2017-18 시즌 |
| `match_level_2018-19.csv` | 10,480 | 2018-19 시즌 |
| `match_level_all.csv`     | 20,928 | 전체 통합    |

**데이터 구조**: 선수 1명 × 경기 1개 = 1행

- 예: Aaron Ramsey가 38경기 출전 → 38행 생성

### 2. 선수 시즌 통계 (Player Season Stats)

| 파일명                            | 행 수 | 설명         |
| --------------------------------- | ----- | ------------ |
| `player_season_stats_2017-18.csv` | 1,534 | 2017-18 시즌 |
| `player_season_stats_2018-19.csv` | 1,320 | 2018-19 시즌 |
| `player_season_stats_all.csv`     | 2,854 | 전체 통합    |

**데이터 구조**: 선수 1명 × 포지션 1개 × 시즌 1개 = 1행

- 예: Aaron Cresswell이 CB, LB, LM, WB로 출전 → 4행 생성

### 3. 포지션별 벤치마크

| 파일명                    | 설명                          |
| ------------------------- | ----------------------------- |
| `position_benchmarks.csv` | 포지션별 평균/중앙값/표준편차 |

---

## 📊 데이터 구조

### 경기 레벨 데이터 (18개 컬럼)

#### 기본 정보 (9개)

- `league` - 리그명 (ENG-Premier League)
- `season` - 시즌 (1718, 1819)
- `game` - 경기 정보 (날짜 + 매치업)
- `team` - 소속 팀
- `player` - 선수명
- `jersey_number` - 등번호
- `nation` - 국적
- `pos` - 포지션(들) (예: "DM,CM")
- `age` - 나이 (년-일 형식)

#### 통계 지표 (8개)

- `xG` - Expected Goals (예상 득점)
- `npxG` - Non-Penalty xG (PK 제외 예상 득점)
- `xAG` - Expected Assisted Goals (예상 어시스트)
- `SCA` - Shot Creating Actions (슛 생성 액션)
- `GCA` - Goal Creating Actions (골 생성 액션)
- `Carries` - 볼 운반 횟수
- `PrgC` - Progressive Carries (전진 드리블)
- `game_id` - 경기 고유 ID

#### 파생 변수 (1개)

- `main_pos` - 주 포지션 (첫 번째 포지션만 추출)

---

### 선수 시즌 통계 (22개 컬럼)

#### 기본 정보 (6개)

- `league`, `season`, `player`, `team`, `nation`, `main_pos`

#### 누적 통계 (8개)

- `matches_played` - 출전 경기 수
- `xG` - 시즌 누적 Expected Goals
- `npxG` - 시즌 누적 Non-Penalty xG
- `xAG` - 시즌 누적 Expected Assisted Goals
- `SCA` - 시즌 누적 Shot Creating Actions
- `GCA` - 시즌 누적 Goal Creating Actions
- `Carries` - 시즌 누적 볼 운반 횟수
- `PrgC` - 시즌 누적 Progressive Carries

#### 90분당 평균 (7개)

- `xG_per_90` = (xG / matches_played) × 90
- `npxG_per_90` = (npxG / matches_played) × 90
- `xAG_per_90` = (xAG / matches_played) × 90
- `SCA_per_90` = (SCA / matches_played) × 90
- `GCA_per_90` = (GCA / matches_played) × 90
- `Carries_per_90` = (Carries / matches_played) × 90
- `PrgC_per_90` = (PrgC / matches_played) × 90

#### 파생 변수 (1개)

- `progressive_carry_rate` = PrgC / Carries
  - 드리블 중 전진한 비율 (0~1)
  - 높을수록 공격적인 드리블

---

### 포지션별 벤치마크

| 포지션 | 선수-시즌 수 | xG 평균 | SCA 평균 | GCA 평균 |
| ------ | ------------ | ------- | -------- | -------- |
| **FW** | 293          | 2.72    | 19.65    | 2.43     |
| **AM** | 268          | 0.57    | 10.69    | 1.13     |
| **LW** | 233          | 0.72    | 10.56    | 1.43     |
| **RW** | 234          | 0.67    | 10.36    | 1.14     |
| **CM** | 294          | 0.51    | 15.10    | 1.41     |
| **LM** | 285          | 0.45    | 9.37     | 0.99     |
| **RM** | 295          | 0.38    | 9.02     | 0.96     |
| **DM** | 202          | 0.31    | 9.63     | 0.76     |
| **LB** | 141          | 0.25    | 16.77    | 1.70     |
| **RB** | 147          | 0.17    | 14.33    | 1.39     |
| **WB** | 139          | 0.21    | 7.31     | 0.72     |
| **CB** | 247          | 0.70    | 9.50     | 0.90     |
| **GK** | 76           | 0.00    | 3.04     | 0.24     |

**인사이트**:

- FW(공격수)가 모든 공격 지표에서 압도적
- LB/RB(풀백)도 높은 SCA → 현대 축구의 특징
- GK(골키퍼)는 거의 0 → 예상대로

---

## 🔧 전처리 프로세스

### STEP 1: 데이터 로딩

```python
# Multi-level 헤더 (3줄) 처리
df = pd.read_csv(file_path, skiprows=2)

# 컬럼명 수동 매핑
df.columns = ['league', 'season', 'game', 'team', 'player',
              'jersey_number', 'nation', 'pos', 'age',
              'xG', 'npxG', 'xAG', 'SCA', 'GCA',
              'Carries', 'PrgC', 'game_id']
```

**문제 해결**:

- 원본 CSV가 3줄 헤더 구조 (카테고리, 서브카테고리, 지표명)
- 첫 2줄을 스킵하고 3번째 줄부터 데이터 로딩
- 컬럼명을 명시적으로 지정하여 일관성 확보

---

### STEP 2: 데이터 정제

```python
# 1. 숫자형 변환
numeric_cols = ['jersey_number', 'xG', 'npxG', 'xAG',
                'SCA', 'GCA', 'Carries', 'PrgC']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 2. 결측치 처리
df[numeric_cols] = df[numeric_cols].fillna(0)

# 3. 포지션 정리
df['main_pos'] = df['pos'].apply(
    lambda x: x.split(',')[0].strip() if pd.notna(x) else 'Unknown'
)
```

**처리 내용**:

- **숫자형 변환**: 문자열로 저장된 숫자를 float으로 변환
- **결측치**: 0으로 대체 (출전하지 않은 경기)
- **포지션 단순화**:
  - 'DM,CM' → 'DM' (첫 번째만 사용)
  - 여러 포지션 → 대표 포지션

---

### STEP 3: 선수별 집계

```python
player_stats = df.groupby([
    'league', 'season', 'player', 'team', 'nation', 'main_pos'
]).agg({
    'game': 'count',      # 경기 수
    'xG': 'sum',          # 누적 xG
    'npxG': 'sum',        # 누적 npxG
    'xAG': 'sum',         # 누적 xAG
    'SCA': 'sum',         # 누적 SCA
    'GCA': 'sum',         # 누적 GCA
    'Carries': 'sum',     # 누적 Carries
    'PrgC': 'sum'         # 누적 PrgC
}).reset_index()

player_stats = player_stats.rename(columns={'game': 'matches_played'})
```

**핵심 로직**:

- **그룹화 기준**: 선수 + 포지션 + 시즌
  - 같은 선수가 여러 포지션에서 뛴 경우 각각 집계
  - Aaron Cresswell: CB(20경기), LB(8경기), LM(4경기), WB(4경기)
- **집계 방식**:
  - 경기 수 = count
  - 나머지 지표 = sum (누적)

---

### STEP 4: 파생 변수 생성

#### 4-1. 90분당 평균

```python
for col in ['xG', 'npxG', 'xAG', 'SCA', 'GCA', 'Carries', 'PrgC']:
    player_stats[f'{col}_per_90'] = (
        player_stats[col] / player_stats['matches_played']
    ) * 90
```

**의미**:

- 출전 시간을 정규화하여 공정 비교
- 교체 출전 선수와 풀타임 선수 비교 가능
- 90분 = 1경기 기준

**예시**:

```
선수A: 10경기 출전, xG = 5.0 → xG_per_90 = 45.0
선수B: 20경기 출전, xG = 8.0 → xG_per_90 = 36.0
→ 선수A가 더 효율적!
```

#### 4-2. Progressive Carry Rate

```python
player_stats['progressive_carry_rate'] = np.where(
    player_stats['Carries'] > 0,
    player_stats['PrgC'] / player_stats['Carries'],
    0
)
```

**의미**:

- 드리블 중 전진한 비율
- 0 ~ 1 사이 값
- 높을수록 공격적인 플레이

**해석**:

- 0.10 = 드리블 10회 중 1회 전진
- 0.30 = 드리블 10회 중 3회 전진 (매우 공격적)

---

### STEP 5: 포지션별 벤치마크

```python
pos_stats = df.groupby('main_pos').agg({
    'matches_played': 'count',
    'xG': ['mean', 'median', 'std'],
    'npxG': ['mean', 'median'],
    'xAG': ['mean', 'median'],
    'SCA': ['mean', 'median'],
    'GCA': ['mean', 'median']
}).round(3)
```

**용도**:

- 포지션별 "정상" 범위 파악
- 선수 평가 시 기준선 제공
- 이상치(Outlier) 탐지

**활용 예시**:

```
질문: "손흥민의 xG가 0.67인데 괜찮은가?"
답변: RW 평균 0.67, 중앙값 0.1 → 평균보다 훨씬 좋음!
```

---

## 💡 활용 방법

### 1. 선수 퍼포먼스 분석

```python
# Aaron Ramsey의 시즌 통계
ramsey = player_stats[player_stats['player'] == 'Aaron Ramsey']

# 포지션별 비교
print(ramsey[['main_pos', 'matches_played', 'xG', 'xG_per_90']])
```

**출력**:

```
main_pos  matches_played  xG    xG_per_90
CM        13              3.8   26.31
DM        6               1.8   27.00
AM        1               0.1   9.00
LM        3               0.3   9.00
```

**인사이트**: CM/DM에서 가장 생산적

---

### 2. VAEP 데이터와 병합

```python
import pandas as pd

# VAEP 데이터 로드
vaep_df = pd.read_csv('../VAEP/vaep_full_England.csv')

# 선수 통계 로드
player_stats = pd.read_csv('../data/processed/player_season_stats_all.csv')

# 병합 (game_id 기준)
merged = vaep_df.merge(
    match_level_df,
    left_on='matchId',
    right_on='game_id',
    how='left'
)

# 분석: VAEP vs xG 상관관계
import matplotlib.pyplot as plt
plt.scatter(merged['xG'], merged['VAEP'])
plt.xlabel('xG')
plt.ylabel('VAEP')
plt.title('VAEP vs Expected Goals')
plt.show()
```

---

### 3. 포지션별 상대 평가

```python
# 벤치마크 로드
benchmarks = pd.read_csv('../data/processed/position_benchmarks.csv',
                         index_col=0)

# 특정 선수 평가
player = player_stats[player_stats['player'] == 'Mohamed Salah'].iloc[0]
position = player['main_pos']

# 포지션 평균과 비교
avg_xG = benchmarks.loc[position, ('xG', 'mean')]
player_xG = player['xG']

print(f"{player['player']} ({position})")
print(f"xG: {player_xG:.2f} (평균: {avg_xG:.2f})")
print(f"차이: {(player_xG - avg_xG):.2f} (+{(player_xG/avg_xG - 1)*100:.1f}%)")
```

---

### 4. 시즌 비교

```python
# 2017-18 vs 2018-19
seasons = player_stats.groupby('season').agg({
    'xG': 'mean',
    'SCA': 'mean',
    'GCA': 'mean'
})

print("시즌별 평균 지표:")
print(seasons)
```

---

## 📈 실제 데이터 예시

### 예시 1: Aaron Ramsey (Arsenal, 2017-18)

```csv
main_pos,matches_played,xG,npxG,xAG,SCA,GCA,xG_per_90,progressive_carry_rate
CM,13,3.8,3.8,2.7,33,7,26.31,0.048
DM,6,1.8,1.8,1.0,19,1,27.00,0.078
AM,1,0.1,0.1,0.0,1,1,9.00,0.052
LM,3,0.3,0.3,1.4,10,4,9.00,0.064
RM,1,0.1,0.1,0.0,0,0,9.00,0.037
```

**분석**:

- **주 포지션**: CM (13경기)
- **생산성**: CM/DM에서 xG_per_90 = 26~27 (매우 높음)
- **공격 기여**: CM에서 SCA 33, GCA 7
- **결론**: 중앙 미드필더로서 매우 효율적

---

### 예시 2: Aaron Cresswell (West Ham, 2017-18)

```csv
main_pos,matches_played,xG,npxG,xAG,SCA,GCA,progressive_carry_rate
CB,20,0.4,0.4,1.7,32,6,0.040
LB,8,0.2,0.2,0.4,16,0,0.030
LM,4,0.0,0.0,0.9,14,3,0.069
WB,4,0.2,0.2,0.3,4,0,0.032
```

**분석**:

- **주 포지션**: CB (20경기)
- **다재다능**: 4개 포지션 소화
- **공격 기여**: LM에서 progressive_carry_rate 가장 높음
- **결론**: 수비수지만 왼쪽 측면 어디든 가능

---

### 예시 3: 포지션별 Top 5 (xG 기준)

```csv
포지션,선수,팀,xG,matches_played,xG_per_90
FW,Mohamed Salah,Liverpool,24.1,36,60.25
FW,Harry Kane,Tottenham,22.8,35,58.54
FW,Sergio Agüero,Manchester City,19.2,30,57.60
FW,Romelu Lukaku,Manchester United,15.3,32,43.03
FW,Jamie Vardy,Leicester City,14.7,35,37.80
```

---

## 🔄 업데이트 방법

새로운 시즌 데이터 추가:

```bash
# 1. 새 시즌 데이터 다운로드
# data/player_match_stats/2019-20_filtered.csv

# 2. 스크립트 수정
# preprocess_player_stats.py 의 seasons 리스트에 추가
seasons = ['2017-18', '2018-19', '2019-20']

# 3. 재실행
python scripts/preprocess_player_stats.py
```

---

## ⚠️ 주의사항

### 1. 포지션 중복

같은 선수가 여러 포지션에서 뛴 경우 **각각 별도 행**으로 집계됩니다.

```python
# Aaron Ramsey 총 경기수
ramsey_total = ramsey['matches_played'].sum()  # 24경기

# 각 포지션별
ramsey.groupby('main_pos')['matches_played'].sum()
# CM: 13, DM: 6, AM: 1, LM: 3, RM: 1
```

### 2. 90분당 지표의 해석

`xG_per_90`는 경기당 90분 기준이므로:

- 교체 출전이 많은 선수: 과대평가 가능
- 풀타임 출전 선수: 적절한 평가

**권장**: `matches_played`도 함께 확인

### 3. 결측치

- 출전하지 않은 경기: 0으로 처리
- 기록되지 않은 지표: 0으로 처리

---

## 📚 참고 문헌

- **FBref**: https://fbref.com/
- **Expected Goals (xG)**: https://theanalyst.com/eu/2023/06/what-is-expected-goals-xg/
- **Progressive Actions**: https://statsbomb.com/articles/soccer/progressive-passing/

---

## 🤝 문의 및 기여

문제 발견 시:

1. GitHub Issues 등록
2. Pull Request 제출

---

**생성일**: 2025-11-22  
**버전**: 1.0  
**작성자**: AIS Project Team
