"""
피드백 반영 3/3: "22단계 시간순 검증은 split이 1번뿐이라, 특정 연도의 기후 패턴에
결과가 좌우됐을 수 있다" — 워킹 포워드(확장 윈도우) 방식으로 5번 반복해 평균±표준편차를
낸다.

방법: 전체 기간(1994-01-01~2023-12-31, 10,957행)을 날짜순으로 정렬한 뒤, 테스트 구간
크기(약 3년, 1,095행)를 고정하고 5개 fold로 나눈다. 각 fold는 그 이전의 모든 과거를
train으로 쓰는 확장 윈도우(expanding window)라 fold가 진행될수록 train이 늘어난다.
19단계에서 찾은 best_params를 재탐색 없이 그대로 재사용해 매 fold마다 재학습하고,
22단계와 마찬가지로 train을 쓰지 않는 persistence(오늘=내일) 기준선도 fold마다 같이
계산해 비교한다(모델만 보고 판단하지 않는다).

원본(seoul_weather_with_target.csv)은 읽기만 하고 고치지 않는다.

[2번째 외부 피드백 반영] F1은 분류 임계값(기본 0.5)에 민감한데, 지금까지의 모든 분류
스크립트(13~19)는 predict()의 기본 임계값만 썼다. "persistence가 모델보다 낫다"(위
발견)는 결론이 이 고정 임계값 때문일 수도 있어, fold마다 **test는 절대 보지 않고 train
안에서만** 임계값을 고른다 — 각 fold의 train을 다시 날짜순 85:15로 나눠 앞 85%로 모델을
학습하고, 뒤 15%(train 안의 미래, test 아님)에서 F1이 최대가 되는 임계값을 찾은 뒤, 그
임계값을 test 예측에만 적용한다. 최종 평가용 모델 자체는 기존과 동일하게 fold의 train
전체로 다시 학습한다(임계값 탐색용 모델과 별개).
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
N_SPLITS = 5

# 19_hyperparameter_tuning.py / 22_time_leakage_evaluation.py의 GridSearchCV 결과를
# 그대로 재사용 (여기서 재탐색하지 않음)
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
COLOR_MODEL = "#2a78d6"
COLOR_BASELINE = "#c3c2b7"


def prepare_features(df: pd.DataFrame, target: str):
    df = df.copy()
    df[ZERO_FILL_COLS] = df[ZERO_FILL_COLS].fillna(0)
    feature_cols = [
        c for c in df.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])
    ]
    return df[feature_cols], df[target], feature_cols


def persistence_predictions(test_df: pd.DataFrame, task: str):
    """22_time_leakage_evaluation.py와 동일한 persistence(오늘=내일) 기준선."""
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


def select_threshold(y_val, proba_val) -> float:
    """train 내부 검증(val)에서 F1이 최대가 되는 임계값을 찾는다. test는 전혀 보지 않는다."""
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        f1 = f1_score(y_val, (proba_val >= threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_threshold, best_f1 = threshold, f1
    return best_threshold


def expanding_windows(df: pd.DataFrame, n_splits: int, test_size: int):
    """fold별 (train, test) — train은 항상 그 fold의 test 이전 전체(확장 윈도우)."""
    n = len(df)
    initial_train_end = n - n_splits * test_size
    for k in range(n_splits):
        test_start = initial_train_end + k * test_size
        test_end = test_start + test_size
        yield df.iloc[:test_start].copy(), df.iloc[test_start:test_end].copy()


if __name__ == "__main__":
    full = pd.read_csv(DATA_DIR / "seoul_weather_with_target.csv", parse_dates=["datetime"])
    full = full.sort_values("datetime").reset_index(drop=True)

    test_size = len(full) // (N_SPLITS * 2)  # 약 3년(1,095행)씩, 5 fold
    print(f"전체 {len(full)}행, fold당 test {test_size}행(약 {test_size / 365.25:.1f}년), {N_SPLITS} fold")

    clf_rows, reg_rows = [], []

    for fold, (train_df, test_df) in enumerate(expanding_windows(full, N_SPLITS, test_size), start=1):
        train_span = f"{train_df['datetime'].min().date()}~{train_df['datetime'].max().date()}"
        test_span = f"{test_df['datetime'].min().date()}~{test_df['datetime'].max().date()}"
        print(f"\n=== fold {fold}: train {len(train_df)}행({train_span}) -> test {len(test_df)}행({test_span}) ===")

        X_train_c, y_train_c, _ = prepare_features(train_df, "rain_tomorrow")
        X_test_c, y_test_c, _ = prepare_features(test_df, "rain_tomorrow")
        X_train_r, y_train_r, _ = prepare_features(train_df, "precip_tomorrow")
        X_test_r, y_test_r, _ = prepare_features(test_df, "precip_tomorrow")

        rf_clf = RandomForestClassifier(**RF_CLF_PARAMS, random_state=SEED, n_jobs=-1).fit(X_train_c, y_train_c)
        xgb_clf = XGBClassifier(**XGB_CLF_PARAMS, random_state=SEED, n_jobs=-1, eval_metric="logloss").fit(X_train_c, y_train_c)
        rf_reg = RandomForestRegressor(**RF_REG_PARAMS, random_state=SEED, n_jobs=-1).fit(X_train_r, y_train_r)
        xgb_reg = XGBRegressor(**XGB_REG_PARAMS, random_state=SEED, n_jobs=-1).fit(X_train_r, y_train_r)

        # 임계값 탐색 전용: fold train을 다시 날짜순 85:15로 나눠(뒤 15%가 val, test 아님)
        # 앞 85%로 학습한 모델의 val 확률로 F1 최대 임계값을 찾는다 (test는 전혀 안 봄).
        fit_df, val_df = next(expanding_windows(train_df, 1, int(len(train_df) * 0.15)))
        X_fit_c, y_fit_c, _ = prepare_features(fit_df, "rain_tomorrow")
        X_val_c, y_val_c, _ = prepare_features(val_df, "rain_tomorrow")
        rf_clf_fit = RandomForestClassifier(**RF_CLF_PARAMS, random_state=SEED, n_jobs=-1).fit(X_fit_c, y_fit_c)
        xgb_clf_fit = XGBClassifier(**XGB_CLF_PARAMS, random_state=SEED, n_jobs=-1, eval_metric="logloss").fit(X_fit_c, y_fit_c)
        rf_threshold = select_threshold(y_val_c, rf_clf_fit.predict_proba(X_val_c)[:, 1])
        xgb_threshold = select_threshold(y_val_c, xgb_clf_fit.predict_proba(X_val_c)[:, 1])

        p_clf = score_classification(y_test_c, persistence_predictions(test_df, "classification"))
        p_reg = score_regression(y_test_r, persistence_predictions(test_df, "regression"))
        rf_clf_m = score_classification(y_test_c, rf_clf.predict(X_test_c))
        xgb_clf_m = score_classification(y_test_c, xgb_clf.predict(X_test_c))
        rf_reg_m = score_regression(y_test_r, rf_reg.predict(X_test_r))
        xgb_reg_m = score_regression(y_test_r, xgb_reg.predict(X_test_r))

        rf_proba_test = rf_clf.predict_proba(X_test_c)[:, 1]
        xgb_proba_test = xgb_clf.predict_proba(X_test_c)[:, 1]
        rf_f1_opt = f1_score(y_test_c, (rf_proba_test >= rf_threshold).astype(int), zero_division=0)
        xgb_f1_opt = f1_score(y_test_c, (xgb_proba_test >= xgb_threshold).astype(int), zero_division=0)

        print(f"  persistence f1={p_clf['f1']:.3f} / "
              f"RF f1={rf_clf_m['f1']:.3f}(임계값0.5) -> {rf_f1_opt:.3f}(임계값{rf_threshold:.2f}) / "
              f"XGB f1={xgb_clf_m['f1']:.3f}(임계값0.5) -> {xgb_f1_opt:.3f}(임계값{xgb_threshold:.2f})")
        print(f"  persistence r2={p_reg['r2']:.3f} / RF r2={rf_reg_m['r2']:.3f} / XGB r2={xgb_reg_m['r2']:.3f}")

        clf_rows += [
            {"fold": fold, "test_span": test_span, "label": "persistence(오늘=내일)",
             "f1": p_clf["f1"], "f1_threshold_opt": p_clf["f1"], "threshold": None},
            {"fold": fold, "test_span": test_span, "label": "랜덤포레스트(튜닝)",
             "f1": rf_clf_m["f1"], "f1_threshold_opt": rf_f1_opt, "threshold": rf_threshold},
            {"fold": fold, "test_span": test_span, "label": "XGBoost(튜닝)",
             "f1": xgb_clf_m["f1"], "f1_threshold_opt": xgb_f1_opt, "threshold": xgb_threshold},
        ]
        reg_rows += [
            {"fold": fold, "test_span": test_span, "label": "persistence(오늘=내일)", "r2": p_reg["r2"]},
            {"fold": fold, "test_span": test_span, "label": "랜덤포레스트(튜닝)", "r2": rf_reg_m["r2"]},
            {"fold": fold, "test_span": test_span, "label": "XGBoost(튜닝)", "r2": xgb_reg_m["r2"]},
        ]

    clf_df = pd.DataFrame(clf_rows)
    reg_df = pd.DataFrame(reg_rows)

    clf_summary = clf_df.groupby("label").agg(
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        f1_threshold_opt_mean=("f1_threshold_opt", "mean"), f1_threshold_opt_std=("f1_threshold_opt", "std"),
    ).reset_index()
    reg_summary = reg_df.groupby("label")["r2"].agg(["mean", "std"]).reset_index()

    print(f"\n=== {N_SPLITS} fold 요약: 분류 F1 (임계값0.5 vs train-내부 최적화, 평균±표준편차) ===")
    print(clf_summary.to_string(index=False))
    print(f"\n=== {N_SPLITS} fold 요약: 회귀 R2 (평균±표준편차) ===")
    print(reg_summary.to_string(index=False))

    diff = clf_summary.set_index("label")["f1_mean"]
    print(f"\npersistence - XGBoost(튜닝) f1 평균 차이 = {diff['persistence(오늘=내일)'] - diff['XGBoost(튜닝)']:+.3f} "
          f"(fold 표준편차 ~{clf_summary['f1_std'].mean():.3f}보다 작으면 우연 범위)")

    def render(df: pd.DataFrame, metric: str, title: str, ylabel: str, out_path: Path) -> None:
        labels = df["label"].unique()
        fig, ax = plt.subplots(figsize=(9, 5), dpi=150, facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        for label, color in zip(labels, [COLOR_BASELINE, COLOR_MODEL, "#1a4d8f"]):
            sub = df[df["label"] == label].sort_values("fold")
            ax.plot(sub["fold"], sub[metric], marker="o", color=color, lw=1.5, label=label)
        ax.set_xticks(sorted(df["fold"].unique()))
        ax.set_xlabel("fold (과거 -> 최근 순, 확장 윈도우)", color=INK_DIM)
        ax.set_ylabel(ylabel, color=INK_DIM)
        ax.set_title(title, fontsize=14, color=INK, loc="left", pad=14)
        ax.axhline(0, color=AXIS, lw=1)
        ax.legend(loc="best", frameon=False, fontsize=9, labelcolor=INK_DIM)
        ax.grid(color=GRID)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(AXIS)
        ax.tick_params(colors=INK_FAINT)
        fig.tight_layout()
        fig.savefig(out_path, facecolor=SURFACE)
        plt.close(fig)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clf_df.to_csv(OUTPUT_DIR / "rolling_window_classification_folds.csv", index=False)
    reg_df.to_csv(OUTPUT_DIR / "rolling_window_regression_folds.csv", index=False)
    clf_summary.to_csv(OUTPUT_DIR / "rolling_window_classification_summary.csv", index=False)
    reg_summary.to_csv(OUTPUT_DIR / "rolling_window_regression_summary.csv", index=False)
    render(clf_df, "f1", f"워킹 포워드 {N_SPLITS} fold — 분류(F1, 임계값 0.5 고정)", "F1 score",
           OUTPUT_DIR / "rolling_window_classification.png")
    render(clf_df, "f1_threshold_opt", f"워킹 포워드 {N_SPLITS} fold — 분류(F1, train-내부 임계값 최적화)", "F1 score",
           OUTPUT_DIR / "rolling_window_classification_threshold_opt.png")
    render(reg_df, "r2", f"워킹 포워드 {N_SPLITS} fold — 회귀(R²)", "R²",
           OUTPUT_DIR / "rolling_window_regression.png")

    for name in [
        "rolling_window_classification_folds.csv", "rolling_window_regression_folds.csv",
        "rolling_window_classification_summary.csv", "rolling_window_regression_summary.csv",
        "rolling_window_classification.png", "rolling_window_classification_threshold_opt.png",
        "rolling_window_regression.png",
    ]:
        print(f"저장: {(OUTPUT_DIR / name).relative_to(ROOT)}")
