"""
계절(season)별로 다음날 강수량(precip_tomorrow) 분포가 얼마나 다른지 박스플롯으로 비교한다.
평균 강수량이 큰 계절일수록 색을 진하게 써서 강도 차이를 드러낸다.
원본은 읽기만 하고 고치지 않는다.

강수량은 대부분 0에 몰려 있고 소수의 날만 많이 오는 분포(오른쪽 꼬리가 긴 분포)라, 극단값까지
다 보이게 y축을 그리면 박스 자체가 눌려 안 보인다. 99번째 백분위수(3.27inch) 근처로
y축을 제한하고, 그 위의 극단값은 "표시 범위 밖" 문구로 알린다.
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
BLUE_RAMP = ["#cde2fb", "#5598e7", "#1c5cab", "#0d366b"]  # 평균 강수량 적음 -> 많음

SEASON_MAP = {
    12: "겨울", 1: "겨울", 2: "겨울",
    3: "봄", 4: "봄", 5: "봄",
    6: "여름", 7: "여름", 8: "여름",
    9: "가을", 10: "가을", 11: "가을",
}
SEASON_ORDER = ["봄", "여름", "가을", "겨울"]
Y_LIMIT = 3.5  # 99번째 백분위수(3.27) 근처


def read_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    df["season"] = df["datetime"].dt.month.map(SEASON_MAP)
    return df


def render(df: pd.DataFrame, out_path: Path) -> None:
    samples = [df.loc[df["season"] == s, "precip_tomorrow"] for s in SEASON_ORDER]
    means = [s.mean() for s in samples]
    rank = pd.Series(means).rank().astype(int) - 1  # 평균이 작을수록 옅은 색
    n_over_limit = [int((s > Y_LIMIT).sum()) for s in samples]

    ticklabels = [f"{season}\n(n={len(s)})" for season, s in zip(SEASON_ORDER, samples)]
    overflow_note = "표시 범위(" + f"{Y_LIMIT}inch" + ") 밖: " + ", ".join(
        f"{season} {over}건" for season, over in zip(SEASON_ORDER, n_over_limit)
    )

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    box = ax.boxplot(
        samples, patch_artist=True, widths=0.5,
        medianprops={"color": INK, "linewidth": 2},
        whiskerprops={"color": AXIS}, capprops={"color": AXIS},
        flierprops={"marker": "o", "markersize": 4, "markerfacecolor": INK_FAINT, "markeredgecolor": "none"},
    )
    for patch, r in zip(box["boxes"], rank):
        patch.set_facecolor(BLUE_RAMP[r])
        patch.set_edgecolor(BLUE_RAMP[r])

    ax.set_xticks(range(1, len(SEASON_ORDER) + 1), ticklabels, color=INK_DIM)
    ax.set_ylim(0, Y_LIMIT)
    ax.set_title("여름철 다음날 강수량이 다른 계절보다 뚜렷하게 많다", fontsize=14, color=INK, loc="left", pad=14)
    ax.set_ylabel("다음날 강수량 (precip_tomorrow, inch)", color=INK_DIM)
    ax.text(
        0.98, 0.97, overflow_note, transform=ax.transAxes,
        ha="right", va="top", fontsize=9, color=INK_FAINT,
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "season_precip_boxplot.png"
    render(df, out_path)
    print(f"저장: {out_path.relative_to(ROOT)}")
