"""
랜덤포레스트 분류기로 다음날 강수 여부(rain_tomorrow)를 예측한다.
13과 같은 입력 변수·같은 분할을 써서 로지스틱회귀와 나란히 비교할 수 있게 한다.
트리 기반 모델이라 스케일링은 하지 않는다.
원본(train.csv/test.csv)은 읽기만 하고 고치지 않는다.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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
N_TREES = 300


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

    model = RandomForestClassifier(n_estimators=N_TREES, random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    metrics = score(y_test, model.predict(X_test))
    importances = (
        pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
    )

    print(f"입력 {len(feature_cols)}개, n_estimators={N_TREES}, train={len(X_train)} test={len(X_test)}")
    print(metrics.to_string(index=False))
    print(importances.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / "rf_classifier_rain_metrics.csv", index=False)
    importances.to_csv(OUTPUT_DIR / "rf_classifier_rain_importances.csv", index=False)
