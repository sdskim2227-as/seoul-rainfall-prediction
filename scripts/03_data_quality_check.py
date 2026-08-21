"""
data/preprocessed/seoul_weather_merged.csv의 결측치·중복행·이상 범위값을 점검하고,
컬럼별 설명·타입·관측범위를 정리한 데이터 사전을 만든다.

산출물(outputs/quality/):
- missing_values.csv   : 컬럼별 결측 개수·비율
- duplicate_rows.csv   : 전체 행 중복, datetime 중복 여부
- range_check.csv      : 도메인 규칙(습도 0~100% 등) 위반 건수
- data_dictionary.csv  : 컬럼별 설명·타입·관측된 최소/최대값
"""

from pathlib import Path

import pandas as pd

DATA_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "preprocessed" / "seoul_weather_merged.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "quality"

# 컬럼 설명. 단위는 Visual Crossing 기본(미국식: °F, inch, mph, mile) 기준.
COLUMN_DESCRIPTIONS = {
    "name": "지역명(관측 지점)",
    "datetime": "관측 날짜",
    "tempmax": "일 최고기온(°F)",
    "tempmin": "일 최저기온(°F)",
    "temp": "일 평균기온(°F)",
    "feelslikemax": "체감 최고기온(°F)",
    "feelslikemin": "체감 최저기온(°F)",
    "feelslike": "체감 평균기온(°F)",
    "dew": "이슬점(°F)",
    "humidity": "상대습도(%)",
    "precip": "강수량(inch)",
    "precipprob": "강수확률(%, 0~100) — 이 과거 관측 데이터에서는 0 또는 100만 나타나 사실상 강수 발생 여부 플래그로 쓰임",
    "precipcover": "하루 중 강수가 있었던 시간 비율(%)",
    "preciptype": "강수 형태(rain/snow 등) — 강수 없는 날은 결측",
    "snow": "적설량(inch)",
    "snowdepth": "적설 깊이(inch)",
    "windgust": "순간 최대 풍속(mph)",
    "windspeed": "평균 풍속(mph)",
    "winddir": "풍향(도, 0~360)",
    "sealevelpressure": "해수면 기압(hPa)",
    "cloudcover": "운량(%)",
    "visibility": "가시거리(mile)",
    "solarradiation": "일사량(W/m^2)",
    "solarenergy": "누적 태양에너지(MJ/m^2)",
    "uvindex": "자외선지수",
    "severerisk": "악기상 위험도(%)",
    "sunrise": "일출 시각",
    "sunset": "일몰 시각",
    "moonphase": "달의 위상(0~1, 0/1=삭 0.5=보름)",
    "conditions": "날씨 요약 범주",
    "description": "날씨 설명 문장",
    "icon": "날씨 아이콘 코드",
    "stations": "관측에 쓰인 기상관측소 ID 목록",
}

# 도메인상 허용 범위. None은 하한/상한 없음.
RANGE_RULES = {
    "humidity": (0, 100),
    "cloudcover": (0, 100),
    "precipcover": (0, 100),
    "precipprob": (0, 100),
    "winddir": (0, 360),
    "moonphase": (0, 1),
    "severerisk": (0, 100),
    "precip": (0, None),
    "snow": (0, None),
    "snowdepth": (0, None),
    "windspeed": (0, None),
    "windgust": (0, None),
    "solarradiation": (0, None),
    "solarenergy": (0, None),
    "uvindex": (0, None),
    "visibility": (0, None),
}


def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    null_count = df.isna().sum()
    return pd.DataFrame(
        {
            "column": df.columns,
            "non_null_count": df.notna().sum().values,
            "null_count": null_count.values,
            "null_pct": (null_count / len(df) * 100).round(2).values,
        }
    ).sort_values("null_count", ascending=False)


def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    full_row_dup = int(df.duplicated().sum())
    datetime_dup = int(df.duplicated(subset="datetime").sum())
    return pd.DataFrame(
        [
            {"check": "전체 행 완전 중복", "count": full_row_dup},
            {"check": "datetime 중복", "count": datetime_dup},
        ]
    )


def check_ranges(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, (low, high) in RANGE_RULES.items():
        series = df[col].dropna()
        violations = pd.Series(dtype=bool)
        if low is not None:
            violations = series < low
        if high is not None:
            over = series > high
            violations = over if violations.empty else (violations | over)
        rows.append(
            {
                "column": col,
                "allowed_range": f"[{low if low is not None else '-inf'}, {high if high is not None else 'inf'}]",
                "violation_count": int(violations.sum()),
                "observed_min": series.min(),
                "observed_max": series.max(),
            }
        )

    # 논리적 순서 위반: 최저값이 최고값보다 큰 경우
    for low_col, high_col in [("tempmin", "tempmax"), ("feelslikemin", "feelslikemax")]:
        pair = df[[low_col, high_col]].dropna()
        violations = pair[low_col] > pair[high_col]
        rows.append(
            {
                "column": f"{low_col} > {high_col}",
                "allowed_range": f"{low_col} <= {high_col}",
                "violation_count": int(violations.sum()),
                "observed_min": None,
                "observed_max": None,
            }
        )

    return pd.DataFrame(rows)


def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        series = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "description": COLUMN_DESCRIPTIONS.get(col, ""),
                "observed_min": series.min() if is_numeric else "",
                "observed_max": series.max() if is_numeric else "",
                "null_pct": round(series.isna().mean() * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    missing = check_missing(df)
    missing.to_csv(OUTPUT_DIR / "missing_values.csv", index=False)
    print("=== 결측치 (상위 5) ===")
    print(missing.head())

    duplicates = check_duplicates(df)
    duplicates.to_csv(OUTPUT_DIR / "duplicate_rows.csv", index=False)
    print("\n=== 중복행 ===")
    print(duplicates)

    ranges = check_ranges(df)
    ranges.to_csv(OUTPUT_DIR / "range_check.csv", index=False)
    print("\n=== 이상 범위값 (위반 건수 > 0) ===")
    print(ranges[ranges["violation_count"] > 0])

    data_dict = build_data_dictionary(df)
    data_dict.to_csv(OUTPUT_DIR / "data_dictionary.csv", index=False)
    print(f"\n산출물 저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
