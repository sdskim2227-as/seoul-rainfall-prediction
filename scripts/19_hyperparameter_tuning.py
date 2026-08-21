"""
랜덤포레스트·XGBoost(분류·회귀 각 2개, 총 4개)에 GridSearchCV로 하이퍼파라미터 튜닝을
적용한다. train 안에서만 5-fold 교차검증을 하고, 최종 평가는 튜닝에 한 번도 쓰이지 않은
test set으로만 한다.

특히 18(기본 XGBoost 회귀)이 test R²=-0.02로 평균값 기준선보다도 나빴던 문제를 해결하는
게 목적이다 — train R²=0.996 vs test R²=-0.02로 심한 과적합이었으므로, XGBoost 그리드에는
learning_rate를 낮추고 max_depth를 얕게 하는 조합을 포함해 규제를 건다.

원본(train.csv/test.csv)은 읽기만 하고 고치지 않는다. 튜닝된 모델은 outputs/model/에
joblib으로 저장해 11(변수중요도)·15(잔차분석)에서 다시 학습하지 않고 그대로 불러 쓴다.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score,
)
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "preprocessed"
OUTPUT_DIR = ROOT / "outputs" / "model"

ZERO_FILL_COLS = ["snow", "snowdepth"]
DROP_HIGH_MISSING = ["severerisk", "windgust", "uvindex", "solarradiation", "solarenergy"]
EXCLUDE = {
    "rain_tomorrow", "precip_tomorrow", "name", "datetime", "sunrise", "sunset",
    "conditions", "description", "icon", "stations", "preciptype",
} | set(DROP_HIGH_MISSING)
SEED = 42
CV = 5

RF_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [10, 20, None],
    "min_samples_leaf": [1, 5],
}
# learning_rate를 낮추고 max_depth를 얕게 하는 조합을 포함 — 18의 과적합(train R2=0.996,
# test R2=-0.02) 원인이 규제 없는 기본값이었으므로 여기서 직접 규제를 건다.
XGB_GRID = {
    "n_estimators": [100, 300],
    "max_depth": [2, 4, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
}


def read_split():
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["datetime"])
    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["datetime"])
    return train, test


def prepare_features(df: pd.DataFrame, target: str):
    df = df.copy()
    df[ZERO_FILL_COLS] = df[ZERO_FILL_COLS].fillna(0)
    feature_cols = [
        c for c in df.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])
    ]
    return df[feature_cols], df[target], feature_cols


def tune_classifier(name, estimator, grid, X_train, y_train, X_test, y_test):
    search = GridSearchCV(estimator, grid, scoring="f1", cv=CV, n_jobs=-1)
    search.fit(X_train, y_train)
    best = search.best_estimator_

    pred_train = best.predict(X_train)
    pred_test = best.predict(X_test)

    row = {
        "model": name,
        "best_params": str(search.best_params_),
        "cv_best_f1": search.best_score_,
        "train_accuracy": accuracy_score(y_train, pred_train),
        "test_accuracy": accuracy_score(y_test, pred_test),
        "test_precision": precision_score(y_test, pred_test, zero_division=0),
        "test_recall": recall_score(y_test, pred_test, zero_division=0),
        "test_f1": f1_score(y_test, pred_test, zero_division=0),
    }
    return best, row


def tune_regressor(name, estimator, grid, X_train, y_train, X_test, y_test):
    search = GridSearchCV(estimator, grid, scoring="r2", cv=CV, n_jobs=-1)
    search.fit(X_train, y_train)
    best = search.best_estimator_

    pred_train = best.predict(X_train)
    pred_test = best.predict(X_test)

    row = {
        "model": name,
        "best_params": str(search.best_params_),
        "cv_best_r2": search.best_score_,
        "train_r2": r2_score(y_train, pred_train),
        "test_rmse": mean_squared_error(y_test, pred_test) ** 0.5,
        "test_mae": mean_absolute_error(y_test, pred_test),
        "test_r2": r2_score(y_test, pred_test),
    }
    return best, row


if __name__ == "__main__":
    train, test = read_split()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 분류: rain_tomorrow ---
    X_train_c, y_train_c, _ = prepare_features(train, "rain_tomorrow")
    X_test_c, y_test_c, _ = prepare_features(test, "rain_tomorrow")

    rf_clf, rf_clf_row = tune_classifier(
        "RandomForestClassifier", RandomForestClassifier(random_state=SEED, n_jobs=-1),
        RF_GRID, X_train_c, y_train_c, X_test_c, y_test_c,
    )
    xgb_clf, xgb_clf_row = tune_classifier(
        "XGBClassifier", XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric="logloss"),
        XGB_GRID, X_train_c, y_train_c, X_test_c, y_test_c,
    )

    clf_results = pd.DataFrame([rf_clf_row, xgb_clf_row])
    print("=== 분류 튜닝 결과 (rain_tomorrow) ===")
    print(clf_results.to_string(index=False))

    joblib.dump(rf_clf, OUTPUT_DIR / "rf_classifier_rain_tuned.joblib")
    joblib.dump(xgb_clf, OUTPUT_DIR / "xgb_classifier_rain_tuned.joblib")

    # --- 회귀: precip_tomorrow ---
    X_train_r, y_train_r, _ = prepare_features(train, "precip_tomorrow")
    X_test_r, y_test_r, _ = prepare_features(test, "precip_tomorrow")

    rf_reg, rf_reg_row = tune_regressor(
        "RandomForestRegressor", RandomForestRegressor(random_state=SEED, n_jobs=-1),
        RF_GRID, X_train_r, y_train_r, X_test_r, y_test_r,
    )
    xgb_reg, xgb_reg_row = tune_regressor(
        "XGBRegressor", XGBRegressor(random_state=SEED, n_jobs=-1),
        XGB_GRID, X_train_r, y_train_r, X_test_r, y_test_r,
    )

    reg_results = pd.DataFrame([rf_reg_row, xgb_reg_row])
    print("\n=== 회귀 튜닝 결과 (precip_tomorrow) ===")
    print(reg_results.to_string(index=False))
    print(
        "\n비교: 18(기본 XGBoost 회귀) train R2=0.996 / test R2=-0.020 이었음. "
        f"튜닝 후 XGBRegressor train R2={xgb_reg_row['train_r2']:.3f} / "
        f"test R2={xgb_reg_row['test_r2']:.3f}"
    )

    joblib.dump(rf_reg, OUTPUT_DIR / "rf_regression_precip_tuned.joblib")
    joblib.dump(xgb_reg, OUTPUT_DIR / "xgb_regression_precip_tuned.joblib")

    clf_results.to_csv(OUTPUT_DIR / "tuned_classifier_metrics.csv", index=False)
    reg_results.to_csv(OUTPUT_DIR / "tuned_regressor_metrics.csv", index=False)
    print(f"\n저장: {OUTPUT_DIR / 'tuned_classifier_metrics.csv'}")
    print(f"저장: {OUTPUT_DIR / 'tuned_regressor_metrics.csv'}")
    print(f"저장: 튜닝된 모델 4개 (*.joblib) -> {OUTPUT_DIR}")
