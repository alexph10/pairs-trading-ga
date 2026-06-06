"""
Scan all candidate pairs for cointegration using the Engle-Granger test and
rank them by p-value. Low p-value => the spread tends to mean-revert, which is
the core requirement for a pairs trade
"""

from itertools import combinations
from pathlib import Path

import pandas as pd
from statsmodels.tsa.stattools import coint

PANEL_PATH = Path("data/processed/adj_close.csv")
OUTPUT_PATH = Path("data/processed/pairs.csv")
PVALUE_THRESHOLD = 0.05

# Pairs are only tested within a sector; cross-sector cointegration is usually spurious.
SECTORS = {
    "tech":     ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD"],
    "banks":    ["JPM", "BAC", "WFC", "C", "GS", "MS"],
    "payments": ["V", "MA"],
    "energy":   ["XOM", "CVX", "COP", "SLB"],
    "staples":  ["KO", "PEP", "PG", "CL"],
    "retail":   ["WMT", "TGT", "COST", "HD", "LOW"],
    "pharma":   ["JNJ", "PFE", "MRK", "ABBV"],
    "telecom":  ["VZ", "T"],
}


def same_sector_pairs(columns) -> list[tuple[str, str]]:
    available = set(columns)
    pairs = []
    for members in SECTORS.values():
        present = [t for t in members if t in available]
        pairs.extend(combinations(present, 2))
    return pairs


def scan_pairs(panel: pd.DataFrame) -> pd.DataFrame:
    results = []
    for a, b in same_sector_pairs(panel.columns):
        # coint returns (test_statistic, p_value, critical_values)
        _, pvalue, _ = coint(panel[a], panel[b])
        results.append({"ticker_a": a, "ticker_b": b, "pvalue": pvalue})

    pairs = pd.DataFrame(results).sort_values("pvalue").reset_index(drop=True)
    pairs["cointegrated"] = pairs["pvalue"] < PVALUE_THRESHOLD

    return pairs


def main() -> None:
    panel = pd.read_csv(PANEL_PATH, index_col="date", parse_dates=True)
    pairs = scan_pairs(panel)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUTPUT_PATH, index=False)

    n_sig = int(pairs["cointegrated"].sum())
    print(f"Scanned {len(pairs)} pairs; {n_sig} cointegrated (p < {PVALUE_THRESHOLD})")
    print(pairs.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
