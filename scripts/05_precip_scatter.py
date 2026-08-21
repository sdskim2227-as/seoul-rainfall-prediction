"""
타깃(precip_tomorrow, 내일 강수량)과 가장 관련 있어 보이는 연속형 입력 변수의 산점도를 그린다.

- 후보 연속형 변수 전체와 precip_tomorrow의 상관계수를 계산해 절댓값이 가장 큰 변수를 고른다
  (실행 시점 결과: precip(오늘 강수량), r≈0.26 — "비가 온 다음날엔 비가 이어질 확률이 높다"는
  강수의 지속성을 보여준다).
- 후보에서 rain_tomorrow/precip_tomorrow(타깃 자신), 날짜·문자열 컬럼은 제외한다.
- 원본(data/preprocessed/seoul_weather_with_target.csv)은 읽기만 하고 고치지 않는다.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "preprocessed" / "seoul_weather_with_target.csv"
OUTPUT_DIR = ROOT / "outputs" / "eda"

TARGET = "precip_tomorrow"
EXCLUDE = {
    "rain_tomorrow",
    "precip_tomorrow",
    "name",
    "datetime",
    "sunrise",
    "sunset",
    "conditions",
    "description",
    "icon",
    "stations",
    "preciptype",
}

INK = "#0b0b0b"
INK_DIM = "#52514e"
INK_FAINT = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
POINT_COLOR = "#2a78d6"


def read_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["datetime"])


def find_most_correlated_feature(df: pd.DataFrame) -> tuple[str, float]:
    candidates = [
        c for c in df.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])
    ]
    corr = df[candidates + [TARGET]].corr()[TARGET].drop(TARGET)
    top_col = corr.abs().idxmax()
    return top_col, corr[top_col]


def fit_trend(x: pd.Series, y: pd.Series):
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept


def render(df: pd.DataFrame, feature: str, r: float, out_path: Path) -> None:
    slope, intercept = fit_trend(df[feature], df[TARGET])

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.scatter(df[feature], df[TARGET], s=20, color=POINT_COLOR, alpha=0.4, linewidths=0)

    xs = np.array([df[feature].min(), df[feature].max()])
    ax.plot(xs, slope * xs + intercept, color=INK_DIM, ls="--", lw=2)

    ax.set_title(f"오늘 {feature}가 내일 강수량과 가장 관련이 크다", fontsize=14, color=INK, loc="left", pad=14)
    ax.set_xlabel(f"오늘 {feature}", color=INK_DIM)
    ax.set_ylabel("내일 강수량 (precip_tomorrow, inch)", color=INK_DIM)
    ax.text(
        0.02, 0.95, f"r = {r:.2f}, 기울기 = {slope:.3f}", transform=ax.transAxes,
        va="top", fontsize=10, color=INK_FAINT,
    )

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
    feature, r = find_most_correlated_feature(df)
    print(f"가장 관련 있는 연속형 변수: {feature} (r={r:.3f})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{feature}_{TARGET}_scatter.png"
    render(df, feature, r, out_path)
    print(f"저장: {out_path.relative_to(ROOT)}")
