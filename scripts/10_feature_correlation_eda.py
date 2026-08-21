"""
입력 후보 변수들이 다음날 강수 여부(rain_tomorrow)·강수량(precip_tomorrow)과 실제로
상관관계가 있는지 확인한다. docs/problem-definition.md의 "확인해볼 것 2"에 대응한다.

- precipprob/cloudcover/humidity처럼 problem-definition.md에서 미리 지목한 변수를 포함해
  모든 연속형 입력 후보의 상관계수를 계산하고, 절댓값이 큰 순서로 정렬한다.
- 원본은 읽기만 하고 고치지 않는다.
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "preprocessed" / "seoul_weather_with_target.csv"
OUTPUT_DIR = ROOT / "outputs" / "eda"

EXCLUDE = {
    "rain_tomorrow", "precip_tomorrow", "name", "datetime", "sunrise", "sunset",
    "conditions", "description", "icon", "stations", "preciptype",
}
# problem-definition.md에서 미리 지목한 변수 — 차트에서 강조 표시
HIGHLIGHT = {"precipprob", "cloudcover", "humidity"}

INK = "#0b0b0b"
INK_DIM = "#52514e"
INK_FAINT = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
COLOR_RAIN = "#2a78d6"
COLOR_PRECIP = "#f0a63c"
HIGHLIGHT_BG = "#eef4fc"


def read_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["datetime"])


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        c for c in df.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])
    ]
    corr_rain = df[candidates + ["rain_tomorrow"]].corr()["rain_tomorrow"].drop("rain_tomorrow")
    corr_precip = df[candidates + ["precip_tomorrow"]].corr()["precip_tomorrow"].drop("precip_tomorrow")

    result = pd.DataFrame(
        {"corr_rain_tomorrow": corr_rain, "corr_precip_tomorrow": corr_precip}
    )
    result["abs_max"] = result.abs().max(axis=1)
    return result.sort_values("abs_max", ascending=False).drop(columns="abs_max")


def render(corr: pd.DataFrame, out_path: Path) -> None:
    ordered = corr.iloc[::-1]  # 위에서부터 큰 값이 오도록(barh는 아래부터 그림)
    y = range(len(ordered))
    height = 0.35

    fig, ax = plt.subplots(figsize=(9, 8), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for i, var in enumerate(ordered.index):
        if var in HIGHLIGHT:
            ax.axhspan(i - 0.5, i + 0.5, color=HIGHLIGHT_BG, zorder=0)

    ax.barh(
        [v + height / 2 for v in y], ordered["corr_rain_tomorrow"],
        height=height, color=COLOR_RAIN, label="rain_tomorrow (강수 여부)",
    )
    ax.barh(
        [v - height / 2 for v in y], ordered["corr_precip_tomorrow"],
        height=height, color=COLOR_PRECIP, label="precip_tomorrow (강수량)",
    )

    labels = [f"* {v}" if v in HIGHLIGHT else v for v in ordered.index]
    ax.set_yticks(list(y), labels, color=INK_DIM, fontsize=9)
    ax.axvline(0, color=AXIS, lw=1)

    ax.set_title("입력 후보 변수와 다음날 강수 타깃의 상관계수", fontsize=14, color=INK, loc="left", pad=14)
    ax.set_xlabel("피어슨 상관계수 r", color=INK_DIM)
    ax.text(
        0.99, 1.02, "* = problem-definition.md에서 미리 지목한 변수", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9, color=INK_FAINT,
    )
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
    df = read_dataset()
    corr = compute_correlations(df)
    print(corr)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    corr.to_csv(OUTPUT_DIR / "feature_target_correlation.csv")
    out_path = OUTPUT_DIR / "feature_correlation.png"
    render(corr, out_path)
    print(f"\n저장: {out_path.relative_to(ROOT)}")
    print(f"저장: {(OUTPUT_DIR / 'feature_target_correlation.csv').relative_to(ROOT)}")
