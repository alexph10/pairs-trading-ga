from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/prices.csv")
OUTPUT_PATH = Path("data/processed/adj_close.csv")
MAX_MISSING_FRAC = 0.02  # drop tickers missing more than 2% of dates


def load_wide_panel() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, parse_dates=["date"])

    # Long -> wide: one column per ticker, values = adjusted close.
    wide = df.pivot(index="date", columns="ticker", values="adj_close")
    wide = wide.sort_index()

    # Drop tickers with too many gaps, then forward-fill small holidays/halts
    keep = wide.columns[wide.isna().mean() <= MAX_MISSING_FRAC]
    wide = wide[keep].ffill()

    # Keep only dates where every surviving ticker has a price.
    wide = wide.dropna(how="any")
    return wide


def main() -> None:
    wide = load_wide_panel()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(OUTPUT_PATH)

    print(f"saved aligned panel: {wide.shape[0]} dates x {wide.shape[1]} tickers")


if __name__ == "__main__":
    main()
