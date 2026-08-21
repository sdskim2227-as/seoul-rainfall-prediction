"""
data/preprocessed/seoul_weather_merged.csv를 읽어 컬럼 구조·행 수·데이터 타입을 확인한다.

- 결측치·중복값 점검은 이 스크립트가 아니라 다음 단계(02_data_quality_check.py)에서 다룬다.
- 여기서는 "이 데이터가 어떤 모양인가"만 파악한다.
"""

from pathlib import Path

import pandas as pd

DATA_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "preprocessed" / "seoul_weather_merged.csv"
)
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "quality" / "column_structure.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])

    print(f"행 수: {len(df)}")
    print(f"컬럼 수: {len(df.columns)}")
    print(f"날짜 범위: {df['datetime'].min().date()} ~ {df['datetime'].max().date()}")
    print()
    print(df.info())

    structure = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "non_null_count": df.notna().sum().values,
            "n_unique": [df[c].nunique() for c in df.columns],
        }
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    structure.to_csv(OUTPUT_PATH, index=False)
    print(f"\n컬럼 구조 저장 위치: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
