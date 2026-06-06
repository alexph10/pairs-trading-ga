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


def select_best_pair(panel: pd.DataFrame, pairs: pd.DataFrame):
    # Walk pairs from most to least cointegrated; take the first one whose
    # hedge ratio is positive (a genuine co-moving relationship).
    for _, row in pairs[pairs["cointegrated"]].iterrows():
        a, b = row["ticker_a"], row["ticker_b"]
        beta = estimate_hedge_ratio(panel[a], panel[b])
        if beta > 0:
            return a, b, beta, row["pvalue"]
    raise ValueError("No cointegrated pair with positive beta found")


def main() -> None:
    panel = pd.read_csv(PANEL_PATH, index_col="date", parse_dates=True)
    pairs = pd.read_csv(PAIRS_PATH)

    a, b, beta, pvalue = select_best_pair(panel, pairs)
    print(f"Building spread for {a}-{b} (p={pvalue:.4f}, beta={beta:.4f})")

    spread = panel[a] - beta * panel[b]
    out = pd.DataFrame({"date": panel.index, "spread": spread.values})
    out["ticker_a"], out["ticker_b"], out["beta"] = a, b, beta

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"beta = {beta:.4f} | saved {len(out)} rows")


if __name__ == "__main__":
    main()
