"""
pipe-line.md 확장 3단계(problem-definition.md 가설 3): 1번(강수 여부 확률)과 2번(강수량
회귀)을 곱해 "기대 강수량"을 만든다.

16~19번 스크립트의 회귀는 비 안 오는 날(precip_tomorrow=0)까지 포함한 전체 날로 학습했다
(무조건부). 여기서는 CLAUDE_CODE_WORKFLOW.md 계획대로 **조건부 회귀**(비 오는 날로만 학습)를
새로 만들어 분류 확률과 곱하는 방식을 시도하고, 무조건부 회귀와 실제로 어느 쪽이 나은지
비교한다.

기대 강수량 = P(내일 비 옴, 19단계 튜닝된 분류기) x E(강수량 | 비 옴, 여기서 새로 학습한
조건부 회귀). 12단계에서 무작위 분할이 시간 누수로 부풀려지지 않았음을 확인했으므로 그대로
재사용한다. 원본(train.csv/test.csv)은 읽기만 하고 고치지 않는다.

[2번째 외부 피드백 반영] RMSE/R2만으론 폭우처럼 드문 큰 값 몇 개가 오차를 과대평가할 수
있어, 중앙값 기반 지표 MedAE(median_absolute_error)를 추가한다. 19단계가 저장해 둔
튜닝된 RF/XGB 회귀 모델(joblib)을 재학습 없이 다시 불러와 MedAE만 새로 계산한다 — 저장된
모델이 없는 나머지 참조값(선형회귀 등)은 MedAE를 비워 둔다.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from xgboost import XGBRegressor

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "preprocessed"
MODEL_DIR = ROOT / "outputs" / "model"

ZERO_FILL_COLS = ["snow", "snowdepth"]
DROP_HIGH_MISSING = ["severerisk", "windgust", "uvindex", "solarradiation", "solarenergy"]
EXCLUDE = {
    "rain_tomorrow", "precip_tomorrow", "name", "datetime", "sunrise", "sunset",
    "conditions", "description", "icon", "stations", "preciptype",
} | set(DROP_HIGH_MISSING)
SEED = 42

# 19_hyperparameter_tuning.py에서 찾은 XGBRegressor best_params를 조건부 회귀에도 그대로 재사용
XGB_REG_PARAMS = {"learning_rate": 0.01, "max_depth": 4, "n_estimators": 300, "subsample": 0.8}


def prepare_features(df: pd.DataFrame, target: str):
    df = df.copy()
    df[ZERO_FILL_COLS] = df[ZERO_FILL_COLS].fillna(0)
    feature_cols = [
        c for c in df.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])
    ]
    return df[feature_cols], df[target], feature_cols


def score(y_true, y_pred) -> dict:
    return {
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "mae": mean_absolute_error(y_true, y_pred),
        "medae": median_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


if __name__ == "__main__":
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["datetime"])
    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["datetime"])

    # --- 1. 조건부 회귀: 비 오는 날(rain_tomorrow==1)로만 학습 ---
    train_rain = train[train["rain_tomorrow"] == 1]
    print(f"조건부 회귀 학습 표본: {len(train_rain)}행 (train 전체 {len(train)}행 중 비 오는 날만)")

    X_train_cond, y_train_cond, feature_cols = prepare_features(train_rain, "precip_tomorrow")
    X_test, y_test, _ = prepare_features(test, "precip_tomorrow")

    cond_reg = XGBRegressor(**XGB_REG_PARAMS, random_state=SEED, n_jobs=-1)
    cond_reg.fit(X_train_cond, y_train_cond)
    amount_if_rain = cond_reg.predict(X_test)

    # 조건부 회귀 자체 성능(비 오는 날 예측이 목적이므로, 실제로 비 온 날만 놓고 평가)
    test_rain_mask = test["rain_tomorrow"] == 1
    cond_reg_on_rain_days = score(test.loc[test_rain_mask, "precip_tomorrow"], amount_if_rain[test_rain_mask.to_numpy()])
    print(f"\n조건부 회귀 자체 성능 (테스트의 비 오는 날 {test_rain_mask.sum()}행만): {cond_reg_on_rain_days}")

    # --- 2. 분류 확률: 19단계에서 튜닝된 XGBoost 분류기를 재학습 없이 그대로 불러옴 ---
    clf = joblib.load(MODEL_DIR / "xgb_classifier_rain_tuned.joblib")
    X_test_clf, _, _ = prepare_features(test, "rain_tomorrow")
    rain_proba = clf.predict_proba(X_test_clf)[:, 1]

    # --- 3. 결합: 기대 강수량 = P(비 옴) x E(강수량 | 비 옴) ---
    combined_pred = rain_proba * amount_if_rain
    combined_metrics = score(y_test, combined_pred)
    print(f"\n=== 기대 강수량(결합) 성능 (테스트 전체 {len(test)}행) ===")
    print(combined_metrics)

    # --- 4. 비교 대상: 19단계 무조건부 회귀(전체 날로 학습, 재학습 없이 참조값) ---
    # 튜닝된 RF/XGB는 저장된 joblib을 재학습 없이 다시 불러와 MedAE까지 새로 계산.
    # 저장된 모델이 없는 나머지(선형회귀 등)는 19단계 결과값을 그대로 참조(medae 비움).
    rf_reg_tuned = joblib.load(MODEL_DIR / "rf_regression_precip_tuned.joblib")
    xgb_reg_tuned = joblib.load(MODEL_DIR / "xgb_regression_precip_tuned.joblib")
    unconditional = {
        "선형회귀": {"rmse": 0.686111, "mae": 0.270976, "medae": None, "r2": 0.100026},
        "랜덤포레스트": {"rmse": 0.686655, "mae": 0.253747, "medae": None, "r2": 0.098598},
        "XGBoost": {"rmse": 0.730463, "mae": 0.277396, "medae": None, "r2": -0.020088},
        "랜덤포레스트(튜닝)": score(y_test, rf_reg_tuned.predict(X_test)),
        "XGBoost(튜닝)": score(y_test, xgb_reg_tuned.predict(X_test)),
    }

    comparison = pd.DataFrame([
        {"방식": name, **metrics} for name, metrics in unconditional.items()
    ] + [{"방식": "기대 강수량(결합, 조건부 XGB)", **combined_metrics}])
    comparison = comparison.sort_values("r2", ascending=False).reset_index(drop=True)

    print(f"\nMedAE(중앙값 절대오차) — 결합: {combined_metrics['medae']:.4f} / "
          f"RF(튜닝): {unconditional['랜덤포레스트(튜닝)']['medae']:.4f} / "
          f"XGB(튜닝): {unconditional['XGBoost(튜닝)']['medae']:.4f}")

    print("\n=== 회귀 방식 전체 비교 (테스트 전체 기준) ===")
    print(comparison.to_string(index=False))

    OUTPUT_DIR = MODEL_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_DIR / "combined_expected_precip_comparison.csv", index=False)
    print(f"\n저장: {(OUTPUT_DIR / 'combined_expected_precip_comparison.csv').relative_to(ROOT)}")
