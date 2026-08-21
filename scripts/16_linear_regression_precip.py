"""
선형회귀로 다음날 강수량(precip_tomorrow)을 예측한다.
13~15와 같은 입력 변수 선정 기준을 쓰되 타깃만 회귀형으로 바꾼다.
원본(train.csv/test.csv)은 읽기만 하고 고치지 않는다.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "preprocessed"
OUTPUT_DIR = ROOT / "outputs" / "model"

TARGET = "precip_tomorrow"
ZERO_FILL_COLS = ["snow", "snowdepth"]
DROP_HIGH_MISSING = ["severerisk", "windgust", "uvindex", "solarradiation", "solarenergy"]
EXCLUDE = {
    "rain_tomorrow", "precip_tomorrow", "name", "datetime", "sunrise", "sunset",
    "conditions", "description", "icon", "stations", "preciptype",
} | set(DROP_HIGH_MISSING)
SEED = 42


def read_split():
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["datetime"])
    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["datetime"])
    return train, test


def prepare_features(df: pd.DataFrame):
    df = df.copy()
    df[ZERO_FILL_COLS] = df[ZERO_FILL_COLS].fillna(0)
    feature_cols = [
        c for c in df.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])
    ]
    return df[feature_cols], df[TARGET], feature_cols


def score(y_true, y_pred) -> pd.DataFrame:
    return pd.DataFrame([{
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }])


if __name__ == "__main__":
    train, test = read_split()
    X_train, y_train, feature_cols = prepare_features(train)
    X_test, y_test, _ = prepare_features(test)

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression().fit(X_train_scaled, y_train)
    metrics = score(y_test, model.predict(X_test_scaled))
    coefs = (
        pd.DataFrame({"feature": feature_cols, "coefficient": model.coef_})
        .sort_values("coefficient", key=abs, ascending=False)
    )

    print(f"입력 {len(feature_cols)}개, train={len(X_train)} test={len(X_test)}")
    print(metrics.to_string(index=False))
    print(coefs.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / "linear_regression_precip_metrics.csv", index=False)
    coefs.to_csv(OUTPUT_DIR / "linear_regression_precip_coefficients.csv", index=False)
