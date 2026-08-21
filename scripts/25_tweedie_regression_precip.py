"""
피드백 반영 2/3: "0이 많은 타깃(precip_tomorrow)에 맞춘 two-part 계열 모델을 최소 1개는
비교해봤는가?"

9장(23_combined_expected_precip.py)의 결합(분류 확률 x 조건부 회귀)이 이미 일종의
two-part 모델이지만, scikit-learn이 이런 분포(0이 많고 나머지는 오른쪽으로 긴 연속값)에
맞춰 직접 제공하는 TweedieRegressor(1<power<2, 복합 포아송-감마 분포 가정)를 별도로
붙여 비교한다. 16~19단계와 같은 입력 변수 선정 기준을 쓰고, 무조건부(비 안 오는 날 포함
전체)로 학습해 17·18과 같은 조건에서 비교한다.

원본(train.csv/test.csv)과 23단계가 저장한 비교표(combined_expected_precip_comparison.csv)는
읽기만 하고 고치지 않는다.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "preprocessed"
MODEL_DIR = ROOT / "outputs" / "model"

TARGET = "precip_tomorrow"
ZERO_FILL_COLS = ["snow", "snowdepth"]
DROP_HIGH_MISSING = ["severerisk", "windgust", "uvindex", "solarradiation", "solarenergy"]
EXCLUDE = {
    "rain_tomorrow", "precip_tomorrow", "name", "datetime", "sunrise", "sunset",
    "conditions", "description", "icon", "stations", "preciptype",
} | set(DROP_HIGH_MISSING)

# power=1.5: 복합 포아송-감마(compound Poisson-Gamma) — 0이 많고 나머지는 연속인 강수량류
# 데이터에 표준적으로 쓰이는 값(기상/보험 청구액 모델링에서 흔히 쓰임).
TWEEDIE_POWERS = [1.1, 1.5, 1.9]


def prepare_features(df: pd.DataFrame):
    df = df.copy()
    df[ZERO_FILL_COLS] = df[ZERO_FILL_COLS].fillna(0)
    feature_cols = [
        c for c in df.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])
    ]
    return df[feature_cols], df[TARGET], feature_cols


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

    X_train, y_train, feature_cols = prepare_features(train)
    X_test, y_test, _ = prepare_features(test)

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []
    for power in TWEEDIE_POWERS:
        model = TweedieRegressor(power=power, link="log", max_iter=2000).fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        metrics = score(y_test, pred)
        print(f"TweedieRegressor(power={power}): {metrics}")
        results.append({"방식": f"Tweedie(power={power})", **metrics})

    best_tweedie = max(results, key=lambda r: r["r2"])
    print(f"\n가장 나은 power: {best_tweedie['방식']} (R2={best_tweedie['r2']:.4f})")

    prior = pd.read_csv(MODEL_DIR / "combined_expected_precip_comparison.csv")
    prior = prior.rename(columns={"rmse": "rmse", "mae": "mae", "r2": "r2"})

    comparison = pd.concat(
        [prior, pd.DataFrame(results)], ignore_index=True
    ).sort_values("r2", ascending=False).reset_index(drop=True)

    print("\n=== 회귀 방식 전체 비교 (Tweedie 포함, 테스트 전체 기준) ===")
    print(comparison.to_string(index=False))

    OUTPUT_DIR = MODEL_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_DIR / "regression_comparison_with_tweedie.csv", index=False)
    print(f"\n저장: {(OUTPUT_DIR / 'regression_comparison_with_tweedie.csv').relative_to(ROOT)}")
