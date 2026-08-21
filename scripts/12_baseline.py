"""
아무 모델도 학습하지 않은 기준선(baseline) 성능을 계산한다.
pipe-line.md 7단계: 이후 실제 모델(8단계~)의 성능이 이 기준선보다 확실히 나은지 판단하는
잣대로 쓴다. train/test는 11_train_test_split.py가 저장한 분할을 그대로 재사용한다.

두 타깃 각각에 대해 두 가지 기준선을 계산한다.
- 분류(rain_tomorrow): ① 다수 클래스(train에서 더 많은 쪽)로만 예측, ② persistence(오늘
  강수 여부를 그대로 내일 예측으로 씀)
- 회귀(precip_tomorrow): ① train 평균값으로만 예측, ② persistence(오늘 강수량을 그대로
  내일 예측으로 씀)
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "preprocessed"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "model"


def classification_baselines(train: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    majority_class = train["rain_tomorrow"].mode()[0]
    majority_pred = pd.Series(majority_class, index=test.index)

    # persistence: 오늘 precipprob==100(오늘 비/눈 옴)을 내일 예측으로 그대로 씀
    persistence_pred = (test["precipprob"] == 100).astype(int)

    rows = []
    for name, pred in [("다수 클래스", majority_pred), ("persistence(오늘=내일)", persistence_pred)]:
        rows.append(
            {
                "target": "rain_tomorrow",
                "baseline": name,
                "accuracy": accuracy_score(test["rain_tomorrow"], pred),
                "precision": precision_score(test["rain_tomorrow"], pred, zero_division=0),
                "recall": recall_score(test["rain_tomorrow"], pred, zero_division=0),
                "f1": f1_score(test["rain_tomorrow"], pred, zero_division=0),
            }
        )
    return rows


def regression_baselines(train: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    mean_value = train["precip_tomorrow"].mean()
    mean_pred = pd.Series(mean_value, index=test.index)

    # persistence: 오늘 precip을 내일 예측으로 그대로 씀
    persistence_pred = test["precip"]

    rows = []
    for name, pred in [("평균값", mean_pred), ("persistence(오늘=내일)", persistence_pred)]:
        mse = mean_squared_error(test["precip_tomorrow"], pred)
        rows.append(
            {
                "target": "precip_tomorrow",
                "baseline": name,
                "rmse": mse ** 0.5,
                "mae": mean_absolute_error(test["precip_tomorrow"], pred),
                "r2": r2_score(test["precip_tomorrow"], pred),
            }
        )
    return rows


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["datetime"])
    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["datetime"])

    clf_rows = classification_baselines(train, test)
    reg_rows = regression_baselines(train, test)

    clf_df = pd.DataFrame(clf_rows)
    reg_df = pd.DataFrame(reg_rows)

    print("=== 분류 기준선 (rain_tomorrow) ===")
    print(clf_df.round(3).to_string(index=False))
    print("\n=== 회귀 기준선 (precip_tomorrow) ===")
    print(reg_df.round(3).to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clf_df.to_csv(OUTPUT_DIR / "baseline_classification.csv", index=False)
    reg_df.to_csv(OUTPUT_DIR / "baseline_regression.csv", index=False)
    print(f"\n저장: {OUTPUT_DIR / 'baseline_classification.csv'}")
    print(f"저장: {OUTPUT_DIR / 'baseline_regression.csv'}")


if __name__ == "__main__":
    main()
