# ☔ 서울 다음날 강수 예측

서울 일별 기상 관측(1994-01-01 \~ 2024-01-01, 30년치, 10,958행)을 분석해 **내일 비가
올지**(분류)와 **온다면 얼마나 올지**(회귀)를 예측하고, 예측이 왜 그 수준에서 멈추는지를
데이터에서 찾았다. 오늘(t)까지의 날씨로 내일(t+1)을 맞추는 문제라 입력에 t+1일자 값은
절대 넣지 않았다.

전체 작업은 [Claude Code](https://claude.com/product/claude-code)와의 대화로 진행했다.
분석 세부 내용은 [`REPORT.md`](./REPORT.md)에, 시간 누수·임계값·확률 보정 등 심화 검증은
[`ADVANCED_ANALYSIS.md`](./ADVANCED_ANALYSIS.md)에 정리했다.

## 01. 문제 정의

1. **내일 비가 올지** (분류) — 오늘까지의 날씨로 내일 강수 여부를 맞춘다.
2. **온다면 얼마나 올지** (회귀) — 내일 강수량(inch)을 맞춘다.

## 02. EDA 핵심 발견

| 확인한 것 | 결과 |
|---|---|
| 강수 발생 비율 | 다음날 비/눈 옴 36.3% — 한쪽으로 심하게 쏠리지 않음 |
| 계절성 | 7~8월 강수 비율 약 59%, 겨울 23~28% — 여름이 겨울의 2배 이상 |
| 강수 여부와 상관 큰 변수 | `cloudcover`(r=0.38), `precipprob`(0.33), `humidity`(0.29) |
| 강수량과 상관 큰 변수 | 대체로 강수 여부보다 약함 — 오늘 `precip`(r=0.26)이 최대 |

**가장 중요한 시사점**: 강수 *여부*는 구름량·습도 등과 상관관계가 어느 정도(r=0.3~0.4)
있지만, 강수 *양*은 어떤 입력 변수와도 상관관계가 약하다(r≤0.3). 이 데이터는 구름의
비율은 담고 있어도 구름 두께·종류, 상층 기압계·전선 위치는 담고 있지 않아서다. 이 결과가
아래 회귀 성능이 왜 낮은 수준에서 멈추는지를 미리 설명한다 (자세한 근거는
[REPORT.md 3장](./REPORT.md)).

## 03. 예측 모델

기준선(baseline) → 랜덤포레스트 → XGBoost 순으로, 같은 입력 변수 19개와 같은 8:2 분할
(`random_state=42`)로 비교했다. (로지스틱회귀, 튜닝 전 모델 등 전체 비교는
[REPORT.md 5장](./REPORT.md) 참고)

## 04. 모델 평가

**내일 비가 올지 (분류, F1 기준)**

| 모델 | accuracy | f1 |
|---|---:|---:|
| 다수 클래스 기준선 | 0.637 | 0.000 |
| persistence 기준선 (오늘=내일) | 0.687 | 0.570 |
| 랜덤포레스트 (튜닝) | 0.735 | 0.573 |
| **XGBoost (튜닝, 최종)** | **0.740** | **0.593** |

F1=0.593이 "좋은 성능"이라는 뜻은 아니다. precision 0.684 / recall 0.523 — 실제 비 온
날의 47%는 놓친다. 기준선보다는 낫지만 절대 성능은 낮다.

**온다면 얼마나 올지 (회귀, R² 기준)**

| 모델 | RMSE | R² |
|---|---:|---:|
| persistence 기준선 (오늘=내일) | 0.895 | -0.530 |
| 평균값 기준선 | 0.723 | -0.0004 |
| **랜덤포레스트 (튜닝)** | **0.668** | **0.146** |
| XGBoost (튜닝) | 0.669 | 0.143 |

R²=0.15는 이 데이터 변수 구성의 사실상의 천장이다(근거: [REPORT.md 3·7장](./REPORT.md)).
튜닝을 더 하거나, 분류×회귀를 결합하거나(R²=0.154), Tweedie 같은 대안 모델을 붙여봐도
이 천장을 크게 넘기지 못했다 — 실험 과정은
[ADVANCED_ANALYSIS.md](./ADVANCED_ANALYSIS.md)에 남겼다.

## 05. 결과 및 인사이트

1. **강수 여부는 과거 기상정보로 기준선보다 나은 예측이 가능하지만, 실제 비 온 날의
   상당수(47%)를 놓친다.** 우산 알림처럼 재현율이 중요한 용도로 바로 쓰기엔 부족하다.
2. **강수량은 현재 데이터만으로는 정확히 예측하기 어렵다.** 튜닝·모델 교체·결합 방식
   변경 어느 것으로도 R²=0.14~0.15의 천장을 넘지 못했다 — 모델 설계 문제가 아니라 이
   변수 구성 자체의 정보 한계다.
3. **더 나은 예측을 위해서는 입력 데이터 자체를 바꿔야 한다.** 레이더·위성 자료나 상층
   기압 자료처럼 구름 두께·종류·전선 위치를 담은 정보가 추가로 필요하다.

시간 누수 검증(무작위 분할이 성능을 부풀렸는가), 워킹 포워드 5-fold, 임계값 최적화 함정,
확률 보정, Tweedie 비교는 모두 이 세 결론을 뒤집지 않았다 — 근거는
[ADVANCED_ANALYSIS.md](./ADVANCED_ANALYSIS.md)에 정리했다.

## 그림으로 보기

| 계절별 강수 패턴 | 입력 변수 상관관계 |
|---|---|
| ![seasonality](./outputs/eda/seasonality_rain.png) | ![correlation](./outputs/eda/feature_correlation.png) |

| 모델 비교 (분류) | 모델 비교 (회귀) |
|---|---|
| ![clf comparison](./outputs/model/model_comparison_classification.png) | ![reg comparison](./outputs/model/model_comparison_regression.png) |

## 이 프로젝트를 어떻게 진행했나

- [`CLAUDE.md`](./CLAUDE.md)에 작업 규칙(로그 형식, 스크립트/산출물 분류 기준, 타깃 누수
  금지 원칙)을 먼저 정해두고 세션 내내 그 기준을 따랐다.
- 다중 원본 CSV를 병합하며 2022\~2024년 파일 하나가 미터법으로 섞여 있는 걸 발견해 단위
  자동 감지·변환 로직을 추가했다 — 상관관계·모델 결과가 왜곡되기 전에 잡아낸 문제다.
- "상관관계가 낮으면 모델링이 무의미한가"라는 질문에 실제로 답했다: 최강 단일 변수
  (`precip`, r=0.26, r²≈0.068)보다 19개 변수를 조합한 모델(R²=0.146)이 2배 이상 나은 걸
  확인해, "상관관계로 기대치를 낮추되 모델은 한 번 돌려본다"는 순서가 옳았음을 검증했다.
- 무작위 분할이 시간 누수를 만들었는지 직접 검증했고, 다른 프로젝트와 반대 결과가 나온
  이유(타깃의 날짜 간 자기상관 차이)까지 설명했다 ([ADVANCED_ANALYSIS.md](./ADVANCED_ANALYSIS.md)).

## 폴더 구조

```
21차시-실습/
├── README.md                                    # 이 파일
├── REPORT.md                                    # 상세 분석 리포트
├── ADVANCED_ANALYSIS.md                         # 심화 검증 (시간누수·임계값·보정·Tweedie·결합)
├── CLAUDE.md                                    # 이 프로젝트의 작업 규칙
├── LICENSE
├── requirements.txt
├── data/
│   ├── raw/                                     # 원본 CSV 15개(2년 단위) + train/test
│   └── preprocessed/                            # 병합·타깃 생성·분할 결과
├── scripts/                                     # 01부터 순서대로 실행
│   ├── 01_merge_seoul_weather.py
│   ├── 02_load_data.py
│   ├── 03_data_quality_check.py
│   ├── 04_build_next_day_target.py
│   ├── 05_precip_scatter.py
│   ├── 06_rain_ratio_eda.py
│   ├── 07_seasonality_eda.py
│   ├── 08_season_precip_boxplot.py
│   ├── 09_season_weekday_rain_heatmap.py
│   ├── 10_feature_correlation_eda.py
│   ├── 11_train_test_split.py
│   ├── 12_baseline.py
│   ├── 13_logistic_regression_rain.py
│   ├── 14_random_forest_classifier_rain.py
│   ├── 15_xgboost_classifier_rain.py
│   ├── 16_linear_regression_precip.py
│   ├── 17_random_forest_regression_precip.py
│   ├── 18_xgboost_regression_precip.py
│   ├── 19_hyperparameter_tuning.py
│   ├── 20_model_comparison_chart.py
│   ├── 21_feature_importance_chart.py
│   ├── 22_time_leakage_evaluation.py             # → ADVANCED_ANALYSIS.md 1장
│   ├── 23_combined_expected_precip.py            # → ADVANCED_ANALYSIS.md 5장
│   ├── 24_probability_calibration.py             # → ADVANCED_ANALYSIS.md 3장
│   ├── 25_tweedie_regression_precip.py           # → ADVANCED_ANALYSIS.md 4장
│   └── 26_rolling_window_evaluation.py           # → ADVANCED_ANALYSIS.md 2장
├── outputs/
│   ├── quality/                                 # 결측치·중복·데이터사전 (csv)
│   ├── eda/                                     # 시각화·상관관계·계절성 (png, csv)
│   └── model/                                   # 성능지표·중요도·튜닝모델·차트
├── docs/
│   ├── problem-definition.md                    # 문제 정의·가설
│   └── CLAUDE_CODE_WORKFLOW.md                  # 계획 문서
└── worklogs/                                    # 날짜별 작업 기록 (로컬 전용)
```

## 다시 돌려보려면

```bash
pip install -r requirements.txt

python scripts/01_merge_seoul_weather.py
python scripts/02_load_data.py
python scripts/03_data_quality_check.py
python scripts/04_build_next_day_target.py
python scripts/05_precip_scatter.py
python scripts/06_rain_ratio_eda.py
python scripts/07_seasonality_eda.py
python scripts/08_season_precip_boxplot.py
python scripts/09_season_weekday_rain_heatmap.py
python scripts/10_feature_correlation_eda.py
python scripts/11_train_test_split.py
python scripts/12_baseline.py
python scripts/13_logistic_regression_rain.py
python scripts/14_random_forest_classifier_rain.py
python scripts/15_xgboost_classifier_rain.py
python scripts/16_linear_regression_precip.py
python scripts/17_random_forest_regression_precip.py
python scripts/18_xgboost_regression_precip.py
python scripts/19_hyperparameter_tuning.py
python scripts/20_model_comparison_chart.py
python scripts/21_feature_importance_chart.py
python scripts/22_time_leakage_evaluation.py
python scripts/23_combined_expected_precip.py
python scripts/24_probability_calibration.py
python scripts/25_tweedie_regression_precip.py
python scripts/26_rolling_window_evaluation.py
```

- Python 3.14에서 검증했다.
- 스크립트는 자기 위치 기준 상대경로로 `data/`·`outputs/`를 찾으므로, `21차시-실습/` 폴더
  구조만 유지하면 실행 위치와 무관하게 같은 결과가 나온다.
- `20`·`21`·`22`·`23`은 `19`가 저장한 `outputs/model/*_tuned.joblib`를 읽으므로 `19` 다음에
  실행해야 한다. `22`는 `11`이 만든 `train.csv`/`test.csv`도 같이 읽는다.
- 차트는 Windows 기본 한글 글꼴(`Malgun Gothic`)로 그린다. 다른 OS에서 실행한다면 각
  시각화 스크립트 상단의 `plt.rcParams["font.family"]` 값을 보유 중인 한글 글꼴로 바꿔야
  라벨이 안 깨진다.

## 데이터 출처

`data/raw/`의 서울 일별 기상 CSV 15개(1994-01-01 \~ 2024-01-01, 2년 단위)는 WISET 강의를
통해 제공받았다. 라이선스는 [LICENSE](./LICENSE) 참고.

## 아직 못 한 것

- **재현율이 낮다.** 분류 최종 모델도 실제 비 온 날의 47%를 놓친다(recall 0.523). 우산
  알림처럼 재현율이 중요한 용도로 그대로 쓰기엔 부족하다.
- **분류 모델을 실제로 배포한다면 임계값 재보정이 필수다.** 기본 임계값 0.5는 강수가
  다수 클래스가 아닌 이 데이터에서 최적이 아니다 — 근거는
  [ADVANCED_ANALYSIS.md 2장](./ADVANCED_ANALYSIS.md#2-워킹-포워드-재검증과-임계값-함정) 참고.
- **회귀 성능을 더 올리려면 이 데이터셋만으론 부족하다.** 레이더/위성 자료나 상층 기압
  자료처럼 구름 두께·종류·전선 위치를 담은 입력이 추가로 필요하다고 판단한다.
