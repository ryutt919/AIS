# Action-Valuation and Sequential Decision Modeling in Football Event Logs

축구 이벤트 시퀀스에서 차기 행동 예측과 패스 성공 확률 추론을 수행하고, 행동 가치 평가 모델과의 다각적 상관관계를 규명한 연구 포트폴리오임.

## Research Question

- 축구 경기 내에서 선수의 행동 전개 패턴을 얼마나 정확하게 예측할 수 있는가?
- 패스 상황의 기하학적/공간적 맥락이 성공 확률에 미치는 영향을 고정밀로 정량화할 수 있는가?
- 경기 중 선수의 행동 예측 일관성(Match Rate)이 실제 득실점 기여 가치(VAEP)와 양의 상관관계를 형성하는가?

## Method

### 1. Data Processing and Standardization
- Wyscout 원천 이벤트 로그 4,367,255건을 **Atomic SPADL(Spatial Post-Event Action Description Language)** 정형 규격으로 변환함.
- 경기장 좌표계를 105m x 68m 표준 규격으로 일원화하고, 공격 팀 진행 방향을 **+x축 방향으로 일관되게 정규화**함.
- 경기 간 데이터 누출(Data Leakage)을 차단하기 위해 **Game-level Split(70:15:15)** 방식을 적용함.

### 2. Multi-Task Next Action Prediction (xAction)
- 과거 액션 시퀀스(Sequence Length 10)를 기반으로 차기 행동 유형과 범주를 동시 추정하는 **Transformer/GRU/LSTM** 구조를 설계함.
- 불균형이 극심한 클래스 분포에 대응하기 위해 Cross-Entropy 손실 가중치 합산과 **Macro F1 기반 Early Stopping**을 도입함.
- 모델 예측과 실제 행동 간 일치 여부를 선수/경기 단위로 집계하여 행동 일관성 지표인 **Match Rate**를 도출함.

### 3. Expected Pass Completion Modeling (xPass)
- 패스 기하 벡터, 골대 거리/각도, 전진 변위를 반영하는 기하학적 피처셋을 구축함.
- **XGBoost(Hist/Logloss)** 알고리즘을 적용하고 Optuna 기반 베이지안 최적화로 정밀 하이퍼파라미터 튜닝을 수행함.
- 판별력(AUC-ROC)과 확률 보정 품질(LogLoss, Brier Score)을 다각도로 검증함.

### 4. Valuing Actions by Estimating Probabilities (VAEP)
- 액션 발생 후 단기 Horizon(5/10) 내 득점 및 실점 발생 확률을 각각 추정하는 **MLP 앙상블 모델**을 구축함.
- 5-Fold 교차 검증을 기반으로 득점 확률 증가분과 실점 확률 감소분을 결합해 **단위 액션 가치**를 산정함.
- 선수별/경기별/시즌별 가치 총량과 90분당 평균 기여도를 집계함.

## Experiments and Results

### 1. Next Action Prediction Performance
- 시퀀스 인코더 비교 실험에서 **Transformer 모델이 가장 높은 성능**을 기록함.
- 모델별 검증 Macro F1 결과:
  - GRU: 0.4728
  - LSTM: 0.4757
  - **Transformer**: **0.5315**
- 전체 평가 표본 4,347,710건 기준 전체 일치율(Match Rate)은 **0.7958**에 도달함.

### 2. Expected Pass Performance
- 패스 완료 여부(양성 비율 85.49%)에 대해 높은 변별력과 안정적인 확률 분포를 달성함.
- 테스트 세트(N=743,274) 기준 **AUC-ROC 0.9759**, **LogLoss 0.1997**, **Brier Score 0.0614**를 기록함.

### 3. Action Value Model Performance
- 5-Fold 평균 평가 지표:
  - Scores 모델: **ROC-AUC 0.6913**, LogLoss 0.05224, Brier Score 0.00960
  - Concedes 모델: **ROC-AUC 0.6253**, LogLoss 0.01390, Brier Score 0.00194

### 4. Correlation Analysis
- 시즌 단위(N=3,671)에서 선수별 행동 일관성(Match Rate)과 경기당 평균 VAEP 간 **Spearman 상관계수 0.4287, Pearson 상관계수 0.4274**의 뚜렷한 양의 상관을 확인증명함.
- 경기 단위(N=55,368)에서는 총 행동량(Num Actions)과의 상관계수가 0.3512로 나타난 반면, 액션당 평균 가치(VAEP Mean)와는 -0.2648의 음의 상관을 보여 **고위험/고가치 전개 시 예측 난이도 증가** 현상을 포착함.

## Problems and Solutions

- 극심한 이벤트 클래스 불균형
  - 패스(37.6%), 인터셉트(32.6%) 편향으로 희귀 액션에 대한 모델 예측력 저하가 발생함.
  - 빈도 임계치 미만 클래스를 정형 병합하되 득점/슛 등 핵심 이벤트는 보존하는 **빈도 기반 계층 필터링**을 적용해 Macro F1을 개선함.

- 원천 이벤트 로그의 좌표 방향 불일치
  - 팀별/진영별 공격 방향 전환으로 인해 동일 전술 패턴이 상이한 좌표 궤적으로 왜곡됨.
  - 전후반 진영 좌표를 **단일 공격 방향(+x)으로 동적 변환**하여 공간 피처 일반화 성능을 확보함.

- 고양성률 데이터에서의 확률 과신 문제
  - 85% 이상의 높은 패스 성공률로 인해 단순 분류 정확도만으로는 확률 신뢰성을 담보하기 어려움.
  - 순위 기반 AUC 지표 외에 **LogLoss와 Brier Score를 병행 최적화**하여 극단적 과신 예측을 억제함.

## Limitations

- 약 211만 건의 데이터 품질 이상치 및 매핑 노이즈가 완전히 제거되지 못함.
- 차기 단일 액션 예측에 국한되어 다단계 전술 전개(Multi-step Rollout) 시뮬레이션이 미흡함.
- 수비 압박 강도 및 상대 선수 좌표 부재로 패스 성공 난이도의 완전한 물리적 반영에 한계가 있음.

## How to Run

### Requirements
- Python 3.10 이상
- PyTorch 2.0+, XGBoost 2.0+, Pandas, Scikit-learn

```bash
# 가상환경 구성 및 패키지 설치
pip install -r requirements.txt

# 샘플 데이터 기반 전체 파이프라인 E2E 검증 실행
python run_pipeline.py --mode sample

# 단계별 개별 파이프라인 실행
python run_pipeline.py --stage xaction
python run_pipeline.py --stage xpass
python run_pipeline.py --stage vaep
python run_pipeline.py --stage analysis
```

## References

1. Decroos, T., Bransen, L., Van Haaren, J., & Davis, J. (2019). Actions speak louder than goals: Valuing player actions in soccer. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 1851-1861).
2. Decroos, T., & Davis, J. (2020). Valuing on-the-ball actions in soccer: a critical comparison of VAEP and xG. In Proceedings of the AAAI Workshop on Artificial Intelligence in Team Sports.
3. Pappalardo, L., Cintia, P., Rossi, A., et al. (2019). A public data set of spatio-temporal match events in world cup soccer matches. Scientific Data, 6(1), 236.
