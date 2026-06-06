"""
Build the tradeable spread for a chosen pair. Regress price A on price B (OLS)
to estimate the hedge ratio (beta), then form spread = A - beta * B. A stationary
spread is what mean-reversion trading exploits.
"""

from pathlib import Path

import pandas as pd
import statsmodels.api as sm

PANEL_PATH = Path("data/processed/adj_close.csv")
PAIRS_PATH = Path("data/processed/pairs.csv")
OUTPUT_PATH = Path("data/processed/spread.csv")


def estimate_hedge_ratio(price_a: pd.Series, price_b: pd.Series) -> float:
    # OLS: price_a = alpha + beta * price_b, add_constant gives us the intercept
    model = sm.OLS(price_a, sm.add_constant(price_b)).fit()
    return model.params[price_b.name]


def build_spread(panel: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    beta = estimate_hedge_ratio(panel[a], panel[b])
    spread = panel[a] - beta * panel[b]

    out = pd.DataFrame({"date": panel.index, "spread": spread.values})
    out["ticker_a"] = a
    out["ticker_b"] = b
    out["beta"] = beta
    return out


def main() -> None:
    panel = pd.read_csv(PANEL_PATH, index_col="date", parse_dates=True)
    pairs = pd.read_csv(PAIRS_PATH)

    best = pairs.iloc[0]
    a, b = best["ticker_a"], best["ticker_b"]
    print(f"Building spread for {a}--{b} (p={best['pvalue']:.4f})")

    spread = build_spread(panel, a, b)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    spread.to_csv(OUTPUT_PATH, index=False)
    print(f"beta = {spread['beta'].iloc[0]:.4f} | saved {len(spread)} rows")


if __name__ == "__main__":
    main()
