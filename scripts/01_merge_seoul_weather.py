"""
data/raw/의 2년 단위 원본 CSV 15개를 하나로 합쳐 data/preprocessed/에 저장한다.

- 원본(data/raw/*.csv)은 읽기만 하고 절대 고치지 않는다.
- 2년 단위 파일끼리 경계 날짜가 겹친다(예: 1996-01-01이 1994-96 파일 마지막 행이자
  1996-98 파일 첫 행으로 중복 존재) → datetime 기준 중복 제거.
- 파일마다 단위계(미국식 화씨/inch/mph/mile vs 미터법 섭씨/mm/kmh/km)가 다를 수 있다
  (2022-01-01~2024-01-01 파일이 미터법으로 수집됨을 품질 점검 중 발견). tempmax 중앙값으로
  파일별 단위계를 감지해 미국식으로 통일한다.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_PATH = DATA_DIR / "preprocessed" / "seoul_weather_merged.csv"

# 서울 기후상 tempmax(°F) 파일 전체 중앙값이 이 값보다 낮으면 섭씨(미터법)로 판단한다.
# 미터법이면 연간 중앙값이 약 15°C 부근, 미국식이면 약 60~65°F 부근이라 40을 경계로 삼는다.
CELSIUS_MEDIAN_THRESHOLD = 40

TEMP_COLUMNS = ["tempmax", "tempmin", "temp", "feelslikemax", "feelslikemin", "feelslike", "dew"]
MM_TO_INCH_COLUMNS = ["precip", "snow", "snowdepth"]
KMH_TO_MPH_COLUMNS = ["windspeed", "windgust"]
KM_TO_MILE_COLUMNS = ["visibility"]


def convert_metric_to_us(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in TEMP_COLUMNS:
        df[col] = df[col] * 9 / 5 + 32
    for col in MM_TO_INCH_COLUMNS:
        df[col] = df[col] / 25.4
    for col in KMH_TO_MPH_COLUMNS:
        df[col] = df[col] / 1.60934
    for col in KM_TO_MILE_COLUMNS:
        df[col] = df[col] / 1.60934
    return df


def load_and_normalize(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    median_tempmax = df["tempmax"].median()
    if median_tempmax < CELSIUS_MEDIAN_THRESHOLD:
        print(f"  [단위 변환] {path.name}: tempmax 중앙값 {median_tempmax:.1f} → 미터법으로 판단, 미국식으로 변환")
        df = convert_metric_to_us(df)
    return df


def main() -> None:
    raw_files = sorted(RAW_DIR.glob("seoul *.csv"))
    if not raw_files:
        raise FileNotFoundError(f"{RAW_DIR}에 seoul *.csv 원본이 없습니다.")

    frames = [load_and_normalize(f) for f in raw_files]
    merged = pd.concat(frames, ignore_index=True)
    rows_before = len(merged)

    merged["datetime"] = pd.to_datetime(merged["datetime"])
    merged = merged.sort_values("datetime")

    dup_count = merged.duplicated(subset="datetime").sum()
    merged = merged.drop_duplicates(subset="datetime", keep="first").reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)

    print(f"원본 파일 수: {len(raw_files)}")
    print(f"병합 전 행 수: {rows_before}")
    print(f"경계 중복 제거: {dup_count}행")
    print(f"최종 행 수: {len(merged)}")
    print(f"날짜 범위: {merged['datetime'].min().date()} ~ {merged['datetime'].max().date()}")
    print(f"저장 위치: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
