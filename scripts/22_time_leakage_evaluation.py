"""
pipe-line.md 12단계(조건부: 시계열 데이터라 필수): 지금까지 쓴 무작위 8:2 분할(11번
스크립트)이 "테스트 날짜의 바로 이웃 날이 학습 데이터에 섞여 들어가 정답이나 다름없는
값을 미리 보는" 시간 누수로 성능을 부풀렸는지 확인한다.

두 가지를 비교한다:
1. neighbor-copy 기준선 — 학습을 전혀 하지 않고, test의 각 날짜에 대해 train에서
   날짜가 가장 가까운 행의 타깃값을 그대로 베낀다. 무작위 분할과 날짜순 분할
   (이전 24년 학습 -> 이후 6년 예측) 각각에서 이 기준선을 계산해 비교한다.
2. 튜닝된 최종 모델(19단계에서 찾은 best_params를 그대로 재사용, 재탐색하지 않음)을
   날짜순 분할로 다시 학습·평가해 무작위 분할 결과와 실제로 얼마나 차이나는지 잰다.

5단계(타깃 설계, 04_build_next_day_target.py)에서 이미 "타깃 자체의 누수"(t+1 값을
입력에 쓰는 것)는 검증했으므로, 여기서는 "분할 방식이 만든" 누수만 다룬다.
원본(seoul_weather_with_target.csv, train.csv, test.csv)은 읽기만 하고 고치지 않는다.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score,
)
from xgboost import XGBClassifier, XGBRegressor

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

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

# 19_hyperparameter_tuning.py의 GridSearchCV 결과를 그대로 재사용 (여기서 재탐색하지 않음)
RF_CLF_PARAMS = {"max_depth": 20, "min_samples_leaf": 1, "n_estimators": 200}
XGB_CLF_PARAMS = {"learning_rate": 0.1, "max_depth": 6, "n_estimators": 100, "subsample": 0.8}
RF_REG_PARAMS = {"max_depth": 20, "min_samples_leaf": 5, "n_estimators": 200}
XGB_REG_PARAMS = {"learning_rate": 0.01, "max_depth": 4, "n_estimators": 300, "subsample": 0.8}

INK = "#0b0b0b"
INK_DIM = "#52514e"
INK_FAINT = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
COLOR_RANDOM = "#c3c2b7"
COLOR_CHRONO = "#2a78d6"


def prepare_features(df: pd.DataFrame, target: str):
    df = df.copy()
    df[ZERO_FILL_COLS] = df[ZERO_FILL_COLS].fillna(0)
    feature_cols = [
        c for c in df.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])
    ]
    return df[feature_cols], df[target], feature_cols


def chronological_split(df: pd.DataFrame, test_size: float = 0.2):
    """날짜순 정렬된 df를 앞쪽(과거)=train, 뒤쪽(최근)=test로 나눈다."""
    cutoff = int(len(df) * (1 - test_size))
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()


def neighbor_copy_predictions(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str) -> np.ndarray:
    """학습 없이, test 각 행에 대해 train에서 날짜가 가장 가까운 행의 타깃값을 그대로 베낀다."""
    train_sorted = train_df.sort_values("datetime")
    train_dates = train_sorted["datetime"].to_numpy()
    train_values = train_sorted[target].to_numpy()

    idx_right = np.searchsorted(train_dates, test_df["datetime"].to_numpy())
    idx_right = np.clip(idx_right, 0, len(train_dates) - 1)
    idx_left = np.clip(idx_right - 1, 0, len(train_dates) - 1)

    test_dates = test_df["datetime"].to_numpy()
    dist_right = np.abs(train_dates[idx_right] - test_dates)
    dist_left = np.abs(train_dates[idx_left] - test_dates)
    nearest_idx = np.where(dist_left <= dist_right, idx_left, idx_right)
    nearest_gap_days = np.minimum(dist_left, dist_right).astype("timedelta64[D]").astype(int)

    return train_values[nearest_idx], nearest_gap_days


def persistence_predictions(test_df: pd.DataFrame, task: str):
    """12_baseline.py와 동일한 persistence(오늘=내일) — test 각 행의 '오늘' 값만 쓴다.
    train을 전혀 참조하지 않으므로 분할 방식과 무관하게 항상 계산 가능한, 진짜 도메인
    기준선이다 (washington의 lag-1 기준선과 동일한 성격)."""
    if task == "classification":
        return (test_df["precipprob"] == 100).astype(int)
    return test_df["precip"]


def score_classification(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def score_regression(y_true, y_pred) -> dict:
    return {
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def render(rows: pd.DataFrame, metric: str, title: str, ylabel: str, out_path: Path) -> None:
    labels = rows["label"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    random_vals = rows[f"{metric}_random"]
    chrono_vals = rows[f"{metric}_chrono"]

    ax.bar(x - width / 2, random_vals, width, color=COLOR_RANDOM, label="무작위 분할")
    ax.bar(x + width / 2, chrono_vals, width, color=COLOR_CHRONO, label="날짜순 분할")

    for xi, v in zip(x - width / 2, random_vals):
        ax.text(xi, v + (0.01 if v >= 0 else -0.02), f"{v:.3f}", ha="center", fontsize=8, color=INK_DIM)
    for xi, v in zip(x + width / 2, chrono_vals):
        ax.text(xi, v + (0.01 if v >= 0 else -0.02), f"{v:.3f}", ha="center", fontsize=8, color=INK_DIM)

    ax.set_xticks(x, labels, color=INK_DIM, fontsize=10)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_title(title, fontsize=14, color=INK, loc="left", pad=14)
    ax.set_ylabel(ylabel, color=INK_DIM)
    ax.legend(loc="best", frameon=False, fontsize=9, labelcolor=INK_DIM)

    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(colors=INK_FAINT)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    full = pd.read_csv(DATA_DIR / "seoul_weather_with_target.csv", parse_dates=["datetime"])
    train_random = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["datetime"])
    test_random = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["datetime"])
    train_chrono, test_chrono = chronological_split(full)

    print(f"무작위 분할: train {len(train_random)} / test {len(test_random)}")
    print(f"날짜순 분할: train {len(train_chrono)} ({train_chrono['datetime'].min().date()}~{train_chrono['datetime'].max().date()}) "
          f"/ test {len(test_chrono)} ({test_chrono['datetime'].min().date()}~{test_chrono['datetime'].max().date()})")

    # --- 1. neighbor-copy 기준선 (학습 없음) ---
    print("\n=== neighbor-copy 기준선 (분류: rain_tomorrow) ===")
    pred, gap = neighbor_copy_predictions(train_random, test_random, "rain_tomorrow")
    nc_clf_random = score_classification(test_random["rain_tomorrow"], pred)
    nc_clf_random["avg_gap_days"] = gap.mean()
    print(f"무작위 분할 — 평균 이웃 날짜 간격 {gap.mean():.2f}일, {nc_clf_random}")

    pred, gap = neighbor_copy_predictions(train_chrono, test_chrono, "rain_tomorrow")
    nc_clf_chrono = score_classification(test_chrono["rain_tomorrow"], pred)
    nc_clf_chrono["avg_gap_days"] = gap.mean()
    print(f"날짜순 분할 — 평균 이웃 날짜 간격 {gap.mean():.2f}일, {nc_clf_chrono}")

    print("\n=== neighbor-copy 기준선 (회귀: precip_tomorrow) ===")
    pred, gap = neighbor_copy_predictions(train_random, test_random, "precip_tomorrow")
    nc_reg_random = score_regression(test_random["precip_tomorrow"], pred)
    nc_reg_random["avg_gap_days"] = gap.mean()
    print(f"무작위 분할 — 평균 이웃 날짜 간격 {gap.mean():.2f}일, {nc_reg_random}")

    pred, gap = neighbor_copy_predictions(train_chrono, test_chrono, "precip_tomorrow")
    nc_reg_chrono = score_regression(test_chrono["precip_tomorrow"], pred)
    nc_reg_chrono["avg_gap_days"] = gap.mean()
    print(f"날짜순 분할 — 평균 이웃 날짜 간격 {gap.mean():.2f}일, {nc_reg_chrono}")

    if nc_clf_chrono["avg_gap_days"] > 30:
        print(
            "\n[주의] 날짜순 분할에서는 test의 모든 날짜가 train보다 미래라, '가장 가까운 "
            "train 날짜'가 사실상 train 마지막 하루로 고정된다(평균 간격 "
            f"{nc_clf_chrono['avg_gap_days']:.0f}일). 즉 이웃을 베끼는 게 아니라 3년 전 "
            "하루를 test 전체에 그대로 복붙한 것과 같아 수치가 왜곡돼 있다 — 아래 "
            "persistence(오늘=내일) 기준선이 더 정직한 비교 대상이다."
        )

    # persistence(오늘=내일) — train을 안 쓰므로 분할과 무관하게 항상 계산 가능한 진짜 기준선
    print("\n=== persistence(오늘=내일) 기준선 — 분할별 재계산 ===")
    p_clf_random = score_classification(test_random["rain_tomorrow"], persistence_predictions(test_random, "classification"))
    p_clf_chrono = score_classification(test_chrono["rain_tomorrow"], persistence_predictions(test_chrono, "classification"))
    p_reg_random = score_regression(test_random["precip_tomorrow"], persistence_predictions(test_random, "regression"))
    p_reg_chrono = score_regression(test_chrono["precip_tomorrow"], persistence_predictions(test_chrono, "regression"))
    print(f"분류 — 무작위: {p_clf_random} / 날짜순: {p_clf_chrono}")
    print(f"회귀 — 무작위: {p_reg_random} / 날짜순: {p_reg_chrono}")

    # --- 2. 튜닝된 모델을 날짜순 분할로 재학습 ---
    print("\n=== 튜닝된 모델 재학습 (날짜순 분할) ===")
    X_train_c, y_train_c, _ = prepare_features(train_chrono, "rain_tomorrow")
    X_test_c, y_test_c, _ = prepare_features(test_chrono, "rain_tomorrow")

    rf_clf = RandomForestClassifier(**RF_CLF_PARAMS, random_state=SEED, n_jobs=-1).fit(X_train_c, y_train_c)
    xgb_clf = XGBClassifier(**XGB_CLF_PARAMS, random_state=SEED, n_jobs=-1, eval_metric="logloss").fit(X_train_c, y_train_c)
    rf_clf_chrono = score_classification(y_test_c, rf_clf.predict(X_test_c))
    xgb_clf_chrono = score_classification(y_test_c, xgb_clf.predict(X_test_c))

    X_train_r, y_train_r, _ = prepare_features(train_chrono, "precip_tomorrow")
    X_test_r, y_test_r, _ = prepare_features(test_chrono, "precip_tomorrow")

    rf_reg = RandomForestRegressor(**RF_REG_PARAMS, random_state=SEED, n_jobs=-1).fit(X_train_r, y_train_r)
    xgb_reg = XGBRegressor(**XGB_REG_PARAMS, random_state=SEED, n_jobs=-1).fit(X_train_r, y_train_r)
    rf_reg_chrono = score_regression(y_test_r, rf_reg.predict(X_test_r))
    xgb_reg_chrono = score_regression(y_test_r, xgb_reg.predict(X_test_r))

    print("RF 분류(날짜순):", rf_clf_chrono)
    print("XGB 분류(날짜순):", xgb_clf_chrono)
    print("RF 회귀(날짜순):", rf_reg_chrono)
    print("XGB 회귀(날짜순):", xgb_reg_chrono)

    # 무작위 분할 기준 모델 성능(19단계 결과, 재실행 없이 하드코딩된 참조값)
    rf_clf_random = {"accuracy": 0.734945, "precision": 0.689046, "recall": 0.490566, "f1": 0.573108}
    xgb_clf_random = {"accuracy": 0.739507, "precision": 0.684211, "recall": 0.523270, "f1": 0.593015}
    rf_reg_random = {"rmse": 0.668202, "mae": 0.237774, "r2": 0.146397}
    xgb_reg_random = {"rmse": 0.669345, "mae": 0.235753, "r2": 0.143474}

    # --- 정리 및 저장 ---
    clf_rows = pd.DataFrame([
        {"label": "neighbor-copy(train)", "f1_random": nc_clf_random["f1"], "f1_chrono": nc_clf_chrono["f1"]},
        {"label": "persistence(오늘=내일)", "f1_random": p_clf_random["f1"], "f1_chrono": p_clf_chrono["f1"]},
        {"label": "랜덤포레스트(튜닝)", "f1_random": rf_clf_random["f1"], "f1_chrono": rf_clf_chrono["f1"]},
        {"label": "XGBoost(튜닝)", "f1_random": xgb_clf_random["f1"], "f1_chrono": xgb_clf_chrono["f1"]},
    ])
    reg_rows = pd.DataFrame([
        {"label": "neighbor-copy(train)", "r2_random": nc_reg_random["r2"], "r2_chrono": nc_reg_chrono["r2"]},
        {"label": "persistence(오늘=내일)", "r2_random": p_reg_random["r2"], "r2_chrono": p_reg_chrono["r2"]},
        {"label": "랜덤포레스트(튜닝)", "r2_random": rf_reg_random["r2"], "r2_chrono": rf_reg_chrono["r2"]},
        {"label": "XGBoost(튜닝)", "r2_random": xgb_reg_random["r2"], "r2_chrono": xgb_reg_chrono["r2"]},
    ])

    print("\n=== 분류: 분할 방식별 f1 비교 ===")
    print(clf_rows.to_string(index=False))
    print("\n=== 회귀: 분할 방식별 r2 비교 ===")
    print(reg_rows.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clf_rows.to_csv(OUTPUT_DIR / "time_leakage_classification.csv", index=False)
    reg_rows.to_csv(OUTPUT_DIR / "time_leakage_regression.csv", index=False)
    render(
        clf_rows, "f1", "분할 방식별 성능 비교 — 다음날 강수 여부 (F1)", "F1 score",
        OUTPUT_DIR / "time_leakage_classification.png",
    )
    render(
        reg_rows, "r2", "분할 방식별 성능 비교 — 다음날 강수량 (R²)", "R² (test)",
        OUTPUT_DIR / "time_leakage_regression.png",
    )

    print(f"\n저장: {(OUTPUT_DIR / 'time_leakage_classification.csv').relative_to(ROOT)}")
    print(f"저장: {(OUTPUT_DIR / 'time_leakage_regression.csv').relative_to(ROOT)}")
    print(f"저장: {(OUTPUT_DIR / 'time_leakage_classification.png').relative_to(ROOT)}")
    print(f"저장: {(OUTPUT_DIR / 'time_leakage_regression.png').relative_to(ROOT)}")
