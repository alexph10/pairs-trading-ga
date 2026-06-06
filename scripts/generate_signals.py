"""
Convert the spread into a rolling z-score and generate long/short signals.
Enter when the spread is stretched (|z| > entry), exit when it reverts towards
the mean (|z| < exit). Position is held between entry and exit
"""

from pathlib import Path

import numpy as np
import pandas as pd

SPREAD_PATH = Path("data/processed/spread.csv")
OUTPUT_PATH = Path("data/processed/signals.csv")

WINDOW = 30
ENTRY_Z = 2.0
EXIT_Z = 0.5


def add_zscore(df: pd.DataFrame) -> pd.DataFrame:
    mean = df["spread"].rolling(WINDOW).mean()
    std = df["spread"].rolling(WINDOW).std()
    df["zscore"] = (df["spread"] - mean) / std

    return df


def generate_positions(z: pd.Series) -> pd.Series:
    # Stateful : carry the position forward until an exit condition is met
    position = 0
    out = []
    for value in z:
        if np.isnan(value):
            out.append(0)
            continue
        if position == 0:
            if value > ENTRY_Z:
                position = -1
            elif value < -ENTRY_Z:
                position = 1
        elif abs(value) < EXIT_Z:
            position = 0

        out.append(position)
    return pd.Series(out, index=z.index)


def main() -> None:
    df = pd.read_csv(SPREAD_PATH, parse_dates=["date"])
    df = add_zscore(df)
    df["position"] = generate_positions(df["zscore"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    trades = int((df["position"].diff().abs() > 0).sum())
    print(f"Generated signals: {trades} position changes over {len(df)} days")


if __name__ == "__main__":
    main()
