"""
피드백 반영 1/3: "결합(9장)의 P(비 옴) 확률이 실제 확률처럼 보정돼 있나?"

19단계에서 튜닝된 XGBoost 분류기는 5장·9장에서 predict_proba() 확률을 그대로 쓴다
(9장 23_combined_expected_precip.py가 이 확률에 조건부 회귀값을 곱함). 트리 모델의
predict_proba는 진짜 확률처럼 보정돼 있지 않을 수 있다는 지적이 있어, 여기서 직접 확인한다.

- Brier score: 확률 예측 자체의 품질(낮을수록 좋음, 0~1). "항상 기저 비율만 예측"과
  비교해 실제로 더 나은지 같이 본다.
- reliability diagram(calibration curve): 예측 확률 구간별로 실제 발생 비율이 얼마나
  일치하는지 시각적으로 확인.

임계값 관련 메모: 5·9장의 모든 분류(13~19단계)는 predict()의 sklearn/XGBoost 기본값인
0.5 임계값을 그대로 썼다. test로 임계값을 따로 탐색하거나 튜닝한 적이 없으므로
"test 정보로 임계값을 최적화했다"는 우려는 이 프로젝트에 해당하지 않는다.

원본(train.csv/test.csv)과 19단계가 저장한 모델(outputs/model/*.joblib)은 읽기만 한다.
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

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
COLOR_MODEL = "#2a78d6"
COLOR_PERFECT = "#c3c2b7"


def prepare_features(df: pd.DataFrame, target: str):
    df = df.copy()
    df[ZERO_FILL_COLS] = df[ZERO_FILL_COLS].fillna(0)
    feature_cols = [
        c for c in df.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])
    ]
    return df[feature_cols], df[target], feature_cols


if __name__ == "__main__":
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["datetime"])
    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["datetime"])

    X_test, y_test, _ = prepare_features(test, "rain_tomorrow")

    models = {
        "XGBoost(튜닝)": MODEL_DIR / "xgb_classifier_rain_tuned.joblib",
        "랜덤포레스트(튜닝)": MODEL_DIR / "rf_classifier_rain_tuned.joblib",
    }

    base_rate = train["rain_tomorrow"].mean()
    baseline_brier = brier_score_loss(y_test, np.full(len(y_test), base_rate))
    print(f"기저 비율(train rain_tomorrow 평균) = {base_rate:.4f}")
    print(f"기저 비율만 예측했을 때 Brier score = {baseline_brier:.4f} (비교 기준선)")

    rows = []
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot([0, 1], [0, 1], "--", color=COLOR_PERFECT, lw=1.5, label="완벽한 보정")

    colors = [COLOR_MODEL, "#8f8d86"]
    for (name, path), color in zip(models.items(), colors):
        clf = joblib.load(path)
        proba = clf.predict_proba(X_test)[:, 1]
        brier = brier_score_loss(y_test, proba)
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")

        print(f"\n{name}: Brier score = {brier:.4f} "
              f"({'기저 비율보다 좋음' if brier < baseline_brier else '기저 비율보다 나쁨'})")
        rows.append({"model": name, "brier_score": brier, "baseline_brier_score": baseline_brier})

        ax.plot(mean_pred, frac_pos, marker="o", color=color, lw=1.5, label=f"{name} (Brier={brier:.3f})")

    ax.set_xlabel("예측 확률(구간 평균)", color=INK_DIM)
    ax.set_ylabel("실제 발생 비율(구간 내)", color=INK_DIM)
    ax.set_title("확률 보정(calibration) 곡선 — 다음날 강수 확률", fontsize=13, color=INK, loc="left", pad=14)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_DIM)
    ax.grid(color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(colors=INK_FAINT)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    OUTPUT_DIR = MODEL_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "probability_calibration.png", facecolor=SURFACE)
    plt.close(fig)

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "probability_calibration_brier.csv", index=False)
    print(f"\n저장: {(OUTPUT_DIR / 'probability_calibration_brier.csv').relative_to(ROOT)}")
    print(f"저장: {(OUTPUT_DIR / 'probability_calibration.png').relative_to(ROOT)}")
