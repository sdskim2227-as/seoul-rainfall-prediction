"""
월별·계절별로 강수 패턴이 뚜렷한지 확인한다.
docs/problem-definition.md의 "확인해볼 것 3: 계절성"에 대응한다.

- 월별 rain_tomorrow 발생 비율과 평균 precip_tomorrow를 계산해 장마철(6~7월) 등 계절 패턴이
  뚜렷하게 보이는지 확인한다.
- 뚜렷하면 month/season을 입력 변수로 추가할지 판단하는 근거로 쓴다.
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

SEASON_MAP = {
    12: "겨울", 1: "겨울", 2: "겨울",
    3: "봄", 4: "봄", 5: "봄",
    6: "여름", 7: "여름", 8: "여름",
    9: "가을", 10: "가을", 11: "가을",
}
SEASON_ORDER = ["봄", "여름", "가을", "겨울"]

INK = "#0b0b0b"
INK_DIM = "#52514e"
INK_FAINT = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
BAR_COLOR = "#2a78d6"
LINE_COLOR = "#d64545"


def read_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    df["month"] = df["datetime"].dt.month
    df["season"] = df["month"].map(SEASON_MAP)
    return df


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("month").agg(
        rain_ratio_pct=("rain_tomorrow", lambda s: round(s.mean() * 100, 2)),
        avg_precip_tomorrow=("precip_tomorrow", lambda s: round(s.mean(), 3)),
        n_days=("rain_tomorrow", "size"),
    )
    return grouped.reindex(range(1, 13)).reset_index()


def seasonal_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("season").agg(
        rain_ratio_pct=("rain_tomorrow", lambda s: round(s.mean() * 100, 2)),
        avg_precip_tomorrow=("precip_tomorrow", lambda s: round(s.mean(), 3)),
        n_days=("rain_tomorrow", "size"),
    )
    return grouped.reindex(SEASON_ORDER).reset_index()


def render(monthly: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.bar(monthly["month"], monthly["rain_ratio_pct"], color=BAR_COLOR, width=0.6, label="강수 비율(%)")

    ax2 = ax.twinx()
    ax2.plot(
        monthly["month"], monthly["avg_precip_tomorrow"], color=LINE_COLOR, marker="o",
        lw=2, label="평균 강수량(inch)",
    )
    ax2.set_ylabel("평균 강수량 (inch)", color=LINE_COLOR)
    ax2.tick_params(colors=LINE_COLOR)
    ax2.spines[["top"]].set_visible(False)

    ax.set_title("월별 다음날 강수 비율·평균 강수량 — 장마철(6~7월) 패턴 확인", fontsize=13, color=INK, loc="left", pad=14)
    ax.set_xlabel("월", color=INK_DIM)
    ax.set_ylabel("강수 비율 (%)", color=INK_DIM)
    ax.set_xticks(range(1, 13))

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

    monthly = monthly_summary(df)
    seasonal = seasonal_summary(df)

    print("=== 월별 ===")
    print(monthly.to_string(index=False))
    print("\n=== 계절별 ===")
    print(seasonal.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUTPUT_DIR / "monthly_rain_summary.csv", index=False)
    seasonal.to_csv(OUTPUT_DIR / "seasonal_rain_summary.csv", index=False)

    out_path = OUTPUT_DIR / "seasonality_rain.png"
    render(monthly, out_path)
    print(f"\n저장: {out_path.relative_to(ROOT)}")
    print(f"저장: {(OUTPUT_DIR / 'monthly_rain_summary.csv').relative_to(ROOT)}")
    print(f"저장: {(OUTPUT_DIR / 'seasonal_rain_summary.csv').relative_to(ROOT)}")
