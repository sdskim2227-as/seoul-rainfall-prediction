"""
pipe-line.md 11단계: 10단계(19_hyperparameter_tuning.py)에서 저장한 튜닝된 모델 4개
(RF·XGB × 분류·회귀)를 다시 불러와 변수중요도를 비교한다. 모델을 재학습하지 않는다.

같은 타깃(분류: rain_tomorrow, 회귀: precip_tomorrow)에 대해 RF·XGB를 나란히 그려,
8단계(튜닝 전) 결과와 마찬가지로 "두 트리 모델이 같은 변수를 중요하게 보는지"를 비교한다.
12:54 워크로그에서 확인한 precipprob 저평가(precip과 정보가 겹쳐 트리가 precip 하나로
대체) 현상이 튜닝 후에도 그대로인지 확인하는 것이 목적.
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "preprocessed"
MODEL_DIR = ROOT / "outputs" / "model"

ZERO_FILL_COLS = ["snow", "snowdepth"]
DROP_HIGH_MISSING = ["severerisk", "windgust", "uvindex", "solarradiation", "solarenergy"]
EXCLUDE = {
    "rain_tomorrow", "precip_tomorrow", "name", "datetime", "sunrise", "sunset",
    "conditions", "description", "icon", "stations", "preciptype",
} | set(DROP_HIGH_MISSING)

INK = "#0b0b0b"
INK_DIM = "#52514e"
INK_FAINT = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
COLOR_RF = "#2a78d6"
COLOR_XGB = "#f0a63c"


def feature_cols_of(df: pd.DataFrame) -> list[str]:
    df = df.copy()
    df[ZERO_FILL_COLS] = df[ZERO_FILL_COLS].fillna(0)
    return [c for c in df.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]


def load_importances(rf_path: Path, xgb_path: Path, feature_cols: list[str]) -> pd.DataFrame:
    rf = joblib.load(rf_path)
    xgb = joblib.load(xgb_path)
    result = pd.DataFrame({
        "feature": feature_cols,
        "rf_importance": rf.feature_importances_,
        "xgb_importance": xgb.feature_importances_,
    })
    result["max_importance"] = result[["rf_importance", "xgb_importance"]].max(axis=1)
    return result.sort_values("max_importance", ascending=False).drop(columns="max_importance")


def render(df: pd.DataFrame, title: str, out_path: Path) -> None:
    ordered = df.iloc[::-1]
    y = range(len(ordered))
    height = 0.35

    fig, ax = plt.subplots(figsize=(9, 8), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.barh(
        [v + height / 2 for v in y], ordered["rf_importance"],
        height=height, color=COLOR_RF, label="랜덤포레스트(튜닝)",
    )
    ax.barh(
        [v - height / 2 for v in y], ordered["xgb_importance"],
        height=height, color=COLOR_XGB, label="XGBoost(튜닝)",
    )

    ax.set_yticks(list(y), ordered["feature"], color=INK_DIM, fontsize=9)
    ax.axvline(0, color=AXIS, lw=1)

    ax.set_title(title, fontsize=14, color=INK, loc="left", pad=14)
    ax.set_xlabel("변수중요도", color=INK_DIM)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_DIM)

    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(colors=INK_FAINT)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["datetime"])
    feature_cols = feature_cols_of(train)

    OUTPUT_DIR = MODEL_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    clf_imp = load_importances(
        MODEL_DIR / "rf_classifier_rain_tuned.joblib",
        MODEL_DIR / "xgb_classifier_rain_tuned.joblib",
        feature_cols,
    )
    print("=== 분류 변수중요도 (rain_tomorrow, 튜닝 모델) ===")
    print(clf_imp.to_string(index=False))
    clf_imp.to_csv(OUTPUT_DIR / "tuned_feature_importance_classification.csv", index=False)
    render(
        clf_imp, "다음날 강수 여부 예측 — 튜닝 모델 변수중요도",
        OUTPUT_DIR / "tuned_feature_importance_classification.png",
    )

    reg_imp = load_importances(
        MODEL_DIR / "rf_regression_precip_tuned.joblib",
        MODEL_DIR / "xgb_regression_precip_tuned.joblib",
        feature_cols,
    )
    print("\n=== 회귀 변수중요도 (precip_tomorrow, 튜닝 모델) ===")
    print(reg_imp.to_string(index=False))
    reg_imp.to_csv(OUTPUT_DIR / "tuned_feature_importance_regression.csv", index=False)
    render(
        reg_imp, "다음날 강수량 예측 — 튜닝 모델 변수중요도",
        OUTPUT_DIR / "tuned_feature_importance_regression.png",
    )

    print(f"\n저장: {(OUTPUT_DIR / 'tuned_feature_importance_classification.csv').relative_to(ROOT)}")
    print(f"저장: {(OUTPUT_DIR / 'tuned_feature_importance_regression.csv').relative_to(ROOT)}")
    print(f"저장: {(OUTPUT_DIR / 'tuned_feature_importance_classification.png').relative_to(ROOT)}")
    print(f"저장: {(OUTPUT_DIR / 'tuned_feature_importance_regression.png').relative_to(ROOT)}")
