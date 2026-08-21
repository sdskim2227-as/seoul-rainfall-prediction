"""
계절(season) x 요일(weekday) 조합별 다음날 강수 발생 비율(rain_tomorrow 평균, %)을 히트맵으로 그린다.
두 변수 모두 범주형이라 행렬로 겹쳐 봐야 상호작용(예: "여름 특정 요일"만 유독 높다 등)이 드러난다.
원본은 읽기만 하고 고치지 않는다.
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]

SEASON_MAP = {
    12: "겨울", 1: "겨울", 2: "겨울",
    3: "봄", 4: "봄", 5: "봄",
    6: "여름", 7: "여름", 8: "여름",
    9: "가을", 10: "가을", 11: "가을",
}
SEASON_ORDER = ["봄", "여름", "가을", "겨울"]
WEEKDAYS = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}


def read_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    df["season"] = df["datetime"].dt.month.map(SEASON_MAP)
    df["weekday"] = df["datetime"].dt.dayofweek
    return df


def build_matrix(df: pd.DataFrame) -> pd.DataFrame:
    matrix = df.pivot_table(index="season", columns="weekday", values="rain_tomorrow", aggfunc="mean") * 100
    return matrix.reindex(index=SEASON_ORDER, columns=list(WEEKDAYS))


def readable_text_color(rgba) -> str:
    r, g, b = rgba[:3]
    return "#ffffff" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.6 else INK


def render(matrix: pd.DataFrame, out_path: Path) -> None:
    cmap = LinearSegmentedColormap.from_list("rain_scale", BLUE_RAMP)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    grid = ax.imshow(matrix.values, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(WEEKDAYS)), [WEEKDAYS[c] for c in matrix.columns], color=INK_DIM)
    ax.set_yticks(range(len(SEASON_ORDER)), matrix.index, color=INK_DIM)
    for i, row in enumerate(matrix.index):
        for j, col in enumerate(matrix.columns):
            value = matrix.loc[row, col]
            color = readable_text_color(grid.cmap(grid.norm(value)))
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=9, color=color)

    ax.set_title("계절 x 요일 다음날 강수 발생 비율(%)", fontsize=14, color=INK, loc="left", pad=14)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    bar = fig.colorbar(grid, ax=ax, fraction=0.046, pad=0.03)
    bar.ax.set_title("비율(%)", fontsize=9, color=INK_DIM, loc="left")
    bar.ax.tick_params(colors=INK_FAINT, labelsize=8)
    bar.outline.set_edgecolor(AXIS)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    df = read_dataset()
    matrix = build_matrix(df)
    print(matrix.round(1))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "season_weekday_rain_heatmap.png"
    render(matrix, out_path)
    print(f"저장: {out_path.relative_to(ROOT)}")
