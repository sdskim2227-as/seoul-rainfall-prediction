"""
data/preprocessed/seoul_weather_merged.csv에 "다음날 강수" 타깃 컬럼을 추가한다.

- rain_tomorrow: 내일 비/눈이 오는지(0/1). precipprob(다음 행 값)을 오늘 행으로 끌어온 것 —
  오늘 시점에서는 아직 모르는 미래 값이므로 "입력"이 아니라 "맞혀야 할 답(타깃)"이다.
- precip_tomorrow: 내일 강수량(inch). 같은 방식으로 precip(다음 행 값)을 끌어옴.
- 데이터의 마지막 날짜(2024-01-01)는 다음날 정보가 없어 타깃이 결측 → 그 행은 제거한다.
- 입력 피처 쪽 컬럼은 어떤 것도 앞당기지 않는다. shift(-1)이 실제로 "다음 행"을 가져왔는지
  무작위 지점 몇 곳을 직접 비교해서 확인한다.
"""

from pathlib import Path

import pandas as pd

DATA_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "preprocessed" / "seoul_weather_merged.csv"
)
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "preprocessed"
    / "seoul_weather_with_target.csv"
)


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    df["rain_tomorrow"] = (df["precipprob"].shift(-1) == 100).astype("Int64")
    df["precip_tomorrow"] = df["precip"].shift(-1)

    print("=== 누수 점검: i번째 행의 타깃이 실제 i+1번째 행 값과 같은지 확인 ===")
    for i in [0, len(df) // 2, len(df) - 3]:
        today = df.loc[i, "datetime"].date()
        tomorrow = df.loc[i + 1, "datetime"].date()
        matches = df.loc[i, "precip_tomorrow"] == df.loc[i + 1, "precip"]
        print(
            f"  idx={i} {today}의 precip_tomorrow({df.loc[i, 'precip_tomorrow']}) "
            f"== {tomorrow}(다음날) precip({df.loc[i + 1, 'precip']}) → {matches}"
        )

    before = len(df)
    last_date = df["datetime"].max()
    df = df.dropna(subset=["rain_tomorrow", "precip_tomorrow"]).reset_index(drop=True)
    dropped = before - len(df)

    rain_ratio = df["rain_tomorrow"].mean()

    print(f"\n입력 행 수: {before}")
    print(f"타깃 결측으로 제거된 행: {dropped} (다음날 정보가 없는 마지막 날짜 {last_date.date()})")
    print(f"최종 행 수: {len(df)}")
    print(f"다음날 강수 비율(rain_tomorrow==1): {rain_ratio:.1%}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"저장 위치: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
