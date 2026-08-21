"""
다음날 강수 발생 비율(rain_tomorrow 클래스 균형)을 확인한다.
docs/problem-definition.md의 "확인해볼 것 1: 강수 발생 비율"에 대응한다.

- rain_tomorrow==1(비/눈 옴)과 0(안 옴)의 비율을 계산한다.
- 한쪽으로 심하게 쏠려 있으면(예: 90/10) 분류 모델을 accuracy만으로 평가하면 안 된다는
  근거가 되므로, 이후 모델 스크립트에서 recall/precision/F1을 같이 봐야 하는지 판단하는 데 쓴다.
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

INK = "#0b0b0b"
INK_DIM = "#52514e"
INK_FAINT = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
BAR_COLORS = ["#c3c2b7", "#2a78d6"]


def read_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["datetime"])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["rain_tomorrow"].value_counts().sort_index()
    pct = (counts / len(df) * 100).round(2)
    return pd.DataFrame(
        {
            "rain_tomorrow": counts.index.map({0: "비 안 옴", 1: "비/눈 옴"}),
            "count": counts.values,
            "pct": pct.values,
        }
    )


def render(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    bars = ax.bar(summary["rain_tomorrow"], summary["pct"], color=BAR_COLORS, width=0.5)
    for bar, pct in zip(bars, summary["pct"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{pct:.1f}%",
            ha="center", fontsize=12, color=INK,
        )

    ax.set_title("다음날 강수 발생 비율", fontsize=14, color=INK, loc="left", pad=14)
    ax.set_ylabel("비율 (%)", color=INK_DIM)
    ax.set_ylim(0, max(summary["pct"]) + 15)

    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(colors=INK_FAINT)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    df = read_dataset()
    summary = summarize(df)
    print(summary.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "rain_ratio_summary.csv", index=False)
    out_path = OUTPUT_DIR / "rain_ratio.png"
    render(summary, out_path)
    print(f"저장: {out_path.relative_to(ROOT)}, {(OUTPUT_DIR / 'rain_ratio_summary.csv').relative_to(ROOT)}")
