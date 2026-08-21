"""
train/test 분할을 고정해 이후 모든 모델 스크립트가 같은 분할을 재사용하게 한다.
pipe-line.md 7단계: 시간 순서가 있는 데이터지만 우선 무작위 분할로 시작하고,
12단계(시간 누수 검증)에서 날짜순 분할과 성능을 비교한다.

- test_size=0.2, random_state=42로 washington 프로젝트와 같은 재현성 관례를 따른다.
- 분류 타깃(rain_tomorrow) 비율이 train/test에서 크게 어긋나지 않도록 stratify를 건다.
- 결과를 data/preprocessed/train.csv, test.csv로 저장해 모든 모델 스크립트가 다시 나누지
  않고 이 파일만 읽게 한다.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "preprocessed"
    / "seoul_weather_with_target.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "preprocessed"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["rain_tomorrow"]
    )

    train_path = OUTPUT_DIR / "train.csv"
    test_path = OUTPUT_DIR / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"전체 {len(df)}행 -> train {len(train_df)}행 / test {len(test_df)}행")
    print(f"train rain_tomorrow 비율: {train_df['rain_tomorrow'].mean():.3%}")
    print(f"test  rain_tomorrow 비율: {test_df['rain_tomorrow'].mean():.3%}")
    print(f"저장: {train_path}, {test_path}")


if __name__ == "__main__":
    main()
