"""
Backtest the baseline pairs strategy. Convert daily positions on the spread into
a PnL curve (net of transaction costs) and report total return, annualized Sharpe,
and maximum drawdown.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SIGNALS_PATH = Path("data/processed/signals.csv")
OUTPUT_PATH = Path("data/processed/backtest.csv")

COST_PER_TRADE = 0.0005
TRADING_DAYS = 252


def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    df["spread_change"] = df["spread"].diff()
    df["gross_pnl"] = df["position"].shift(1) * df["spread_change"]

    # Cost is charged whenever the position size changes (entries and exits)
    df["trade"] = df["position"].diff().abs()
    df["cost"] = df["trade"] * COST_PER_TRADE

    df["net_pnl"] = (df["gross_pnl"] - df["cost"]).fillna(0)
    df["equity"] = df["net_pnl"].cumsum()
    return df


def summarize(df: pd.DataFrame) -> dict:
    pnl = df["net_pnl"]

    total_return = df["equity"].iloc[-1]
    sharpe = (pnl.mean() / pnl.std()) * np.sqrt(TRADING_DAYS) if pnl.std() else 0.0

    running_peak = df["equity"].cummax()
    drawdown = df["equity"] - running_peak
    max_drawdown = drawdown.min()

    n_trades = int((df["trade"] > 0).sum())

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "n_trades": n_trades,
    }


def main() -> None:
    df = pd.read_csv(SIGNALS_PATH, parse_dates=["date"])
    df = run_backtest(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    stats = summarize(df)
    print(f"Total PnL (spread units): {stats['total_return']:.2f}")
    print(f"Annualized Sharpe:        {stats['sharpe']:.2f}")
    print(f"Max drawdown:             {stats['max_drawdown']:.2f}")
    print(f"Number of trades:         {stats['n_trades']}")


if __name__ == "__main__":
    main()
