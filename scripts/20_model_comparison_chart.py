"""
pipe-line.md 9단계: 7단계 기준선 + 8단계 모델 6개 + 10단계 튜닝 모델 2개를
하나의 표·차트로 비교한다(10단계가 순서상 먼저 끝나 튜닝 결과까지 포함).

분류는 accuracy가 아니라 f1을 기준 정렬로 쓴다 — 12_baseline.py에서 이미
"다수 클래스" 기준선이 accuracy만 높고 f1=0(쓸모없음)인 걸 확인했기 때문.
회귀는 r2를 기준으로 쓴다 — 12_baseline.py에서 persistence 기준선이 오히려
평균값 기준선보다 나쁜 r2를 보였으므로, rmse만으로는 이 함정을 놓칠 수 있다.

각 스크립트가 이미 저장해 둔 metrics CSV를 다시 읽기만 하고, 모델을 재학습하지 않는다.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "outputs" / "model"

INK = "#0b0b0b"
INK_DIM = "#52514e"
INK_FAINT = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
COLOR_BASELINE = "#c3c2b7"
COLOR_MODEL = "#2a78d6"
COLOR_TUNED = "#1a4d8f"


def load_classification() -> pd.DataFrame:
    baseline = pd.read_csv(MODEL_DIR / "baseline_classification.csv")
    rows = [
        {"model": "다수 클래스(기준선)", "kind": "baseline", **baseline.iloc[0][["accuracy", "precision", "recall", "f1"]]},
        {"model": "persistence(기준선)", "kind": "baseline", **baseline.iloc[1][["accuracy", "precision", "recall", "f1"]]},
    ]
    for label, kind, filename in [
        ("로지스틱회귀", "model", "logistic_regression_rain_metrics.csv"),
        ("랜덤포레스트", "model", "rf_classifier_rain_metrics.csv"),
        ("XGBoost", "model", "xgb_classifier_rain_metrics.csv"),
    ]:
        m = pd.read_csv(MODEL_DIR / filename).iloc[0]
        rows.append({"model": label, "kind": kind, **m[["accuracy", "precision", "recall", "f1"]]})

    tuned = pd.read_csv(MODEL_DIR / "tuned_classifier_metrics.csv")
    tuned_label = {"RandomForestClassifier": "랜덤포레스트(튜닝)", "XGBClassifier": "XGBoost(튜닝)"}
    for _, r in tuned.iterrows():
        rows.append({
            "model": tuned_label[r["model"]], "kind": "tuned",
            "accuracy": r["test_accuracy"], "precision": r["test_precision"],
            "recall": r["test_recall"], "f1": r["test_f1"],
        })

    df = pd.DataFrame(rows)
    return df.sort_values("f1", ascending=False).reset_index(drop=True)


def load_regression() -> pd.DataFrame:
    baseline = pd.read_csv(MODEL_DIR / "baseline_regression.csv")
    rows = [
        {"model": "평균값(기준선)", "kind": "baseline", "rmse": baseline.iloc[0]["rmse"], "mae": baseline.iloc[0]["mae"], "r2": baseline.iloc[0]["r2"]},
        {"model": "persistence(기준선)", "kind": "baseline", "rmse": baseline.iloc[1]["rmse"], "mae": baseline.iloc[1]["mae"], "r2": baseline.iloc[1]["r2"]},
    ]
    for label, kind, filename in [
        ("선형회귀", "model", "linear_regression_precip_metrics.csv"),
        ("랜덤포레스트", "model", "rf_regression_precip_metrics.csv"),
        ("XGBoost", "model", "xgb_regression_precip_metrics.csv"),
    ]:
        m = pd.read_csv(MODEL_DIR / filename).iloc[0]
        rows.append({"model": label, "kind": kind, "rmse": m["RMSE"], "mae": m["MAE"], "r2": m["R2"]})

    tuned = pd.read_csv(MODEL_DIR / "tuned_regressor_metrics.csv")
    tuned_label = {"RandomForestRegressor": "랜덤포레스트(튜닝)", "XGBRegressor": "XGBoost(튜닝)"}
    for _, r in tuned.iterrows():
        rows.append({
            "model": tuned_label[r["model"]], "kind": "tuned",
            "rmse": r["test_rmse"], "mae": r["test_mae"], "r2": r["test_r2"],
        })

    df = pd.DataFrame(rows)
    return df.sort_values("r2", ascending=False).reset_index(drop=True)


def bar_color(kind: str) -> str:
    return {"baseline": COLOR_BASELINE, "model": COLOR_MODEL, "tuned": COLOR_TUNED}[kind]


def render(df: pd.DataFrame, metric: str, title: str, xlabel: str, out_path: Path) -> None:
    ordered = df.iloc[::-1]  # barh는 아래부터 그리므로 뒤집어서 위에서부터 큰 값이 오게
    y = range(len(ordered))
    colors = [bar_color(k) for k in ordered["kind"]]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.barh(list(y), ordered[metric], color=colors, height=0.6)
    ax.margins(x=0.18)
    span = ordered[metric].max() - ordered[metric].min()
    offset = max(span, 1e-6) * 0.02
    for i, v in zip(y, ordered[metric]):
        ax.text(
            v + (offset if v >= 0 else -offset), i, f"{v:.3f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=9, color=INK_DIM,
        )

    ax.set_yticks(list(y), ordered["model"], color=INK_DIM, fontsize=10)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_title(title, fontsize=14, color=INK, loc="left", pad=14)
    ax.set_xlabel(xlabel, color=INK_DIM)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_BASELINE, label="기준선"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_MODEL, label="기본 모델"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_TUNED, label="튜닝 모델"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9, labelcolor=INK_DIM)

    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(colors=INK_FAINT)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    OUTPUT_DIR = MODEL_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    clf = load_classification()
    print("=== 분류 비교 (rain_tomorrow, f1 기준 정렬) ===")
    print(clf.to_string(index=False))
    clf.to_csv(OUTPUT_DIR / "model_comparison_classification.csv", index=False)
    render(
        clf, "f1", "다음날 강수 여부 예측 — 모델 비교 (F1)", "F1 score",
        OUTPUT_DIR / "model_comparison_classification.png",
    )

    reg = load_regression()
    print("\n=== 회귀 비교 (precip_tomorrow, r2 기준 정렬) ===")
    print(reg.to_string(index=False))
    reg.to_csv(OUTPUT_DIR / "model_comparison_regression.csv", index=False)
    render(
        reg, "r2", "다음날 강수량 예측 — 모델 비교 (R²)", "R² (test)",
        OUTPUT_DIR / "model_comparison_regression.png",
    )

    print(f"\n저장: {(OUTPUT_DIR / 'model_comparison_classification.csv').relative_to(ROOT)}")
    print(f"저장: {(OUTPUT_DIR / 'model_comparison_regression.csv').relative_to(ROOT)}")
    print(f"저장: {(OUTPUT_DIR / 'model_comparison_classification.png').relative_to(ROOT)}")
    print(f"저장: {(OUTPUT_DIR / 'model_comparison_regression.png').relative_to(ROOT)}")
