"""
로지스틱회귀로 다음날 강수 여부(rain_tomorrow)를 예측한다.
11_train_test_split.py가 저장한 고정 분할(train.csv/test.csv)을 그대로 쓴다.

입력 변수 선정 기준:
- 결측 50% 이상인 severerisk/windgust/uvindex/solarradiation/solarenergy는 기본 입력에서
  제외한다(03의 품질 점검 결과). 이 변수들은 10의 상관관계 분석에서도 상위권이 아니었다.
- snow/snowdepth는 결측 25% 안팎인데, 관측 안 된 게 아니라 "눈이 안 와서 값이 없는 경우"로
  보여 0으로 채운다.
- 나머지 연속형 입력(cloudcover/precipprob/precipcover/humidity/precip/dew/winddir/tempmin/
  sealevelpressure/feelslikemin/temp/feelslike/feelslikemax/tempmax/visibility/snow/
  snowdepth/moonphase/windspeed) 19개를 그대로 쓴다.
- 로지스틱회귀는 스케일에 민감하므로 StandardScaler로 표준화한다.

원본(train.csv/test.csv)은 읽기만 하고 고치지 않는다.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "preprocessed"
OUTPUT_DIR = ROOT / "outputs" / "model"

TARGET = "rain_tomorrow"
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
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }])


if __name__ == "__main__":
    train, test = read_split()
    X_train, y_train, feature_cols = prepare_features(train)
    X_test, y_test, _ = prepare_features(test)

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=SEED).fit(X_train_scaled, y_train)
    metrics = score(y_test, model.predict(X_test_scaled))
    coefs = (
        pd.DataFrame({"feature": feature_cols, "coefficient": model.coef_[0]})
        .sort_values("coefficient", key=abs, ascending=False)
    )

    print(f"입력 {len(feature_cols)}개, train={len(X_train)} test={len(X_test)}")
    print(metrics.to_string(index=False))
    print(coefs.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / "logistic_regression_rain_metrics.csv", index=False)
    coefs.to_csv(OUTPUT_DIR / "logistic_regression_rain_coefficients.csv", index=False)
