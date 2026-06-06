"""
Dollar / return-based backtest of the baseline pairs strategy.

The spread-unit backtest measures PnL in price points, which is fine for ranking
parameters but not for real-world interpretation. Here we normalize PnL by the
gross notional actually deployed each day, producing a percentage return series.
From that we report an account-level Sharpe, total return, and max drawdown, plus
a dollar equity curve for a chosen starting capital.

Reads the chosen pair and hedge ratio from spread.csv and prices from
adj_close.csv, and reuses the signal/sizing logic from optimize_ga_split.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import optimize_ga_split as ga

PANEL_PATH = Path("data/processed/adj_close.csv")
SPREAD_PATH = Path("data/processed/spread.csv")
OUTPUT_PATH = Path("data/processed/backtest_dollars.csv")

STARTING_CAPITAL = 100_000.0

# Strategy parameters (plug in the GA's best params here).
WINDOW = 30
ENTRY_Z = 2.0
EXIT_Z = 0.5
STOP_Z = 4.0
VOL_TARGET = 1.0


def run_dollar_backtest(price_a: pd.Series, price_b: pd.Series, beta: float) -> pd.DataFrame:
    """Build a percentage-return and dollar-equity backtest for one pair."""
    spread = price_a - beta * price_b
    change = spread.diff()
    rolling_std = change.rolling(WINDOW).std()

    zscore = (spread - spread.rolling(WINDOW).mean()) / spread.rolling(WINDOW).std()
    direction = ga.generate_positions(zscore, ENTRY_Z, EXIT_Z, STOP_Z)

    # Inverse-vol sizing (units of the spread), capped at the leverage limit.
    size = (VOL_TARGET / rolling_std).clip(upper=ga.MAX_SIZE).fillna(0.0)
    exposure = direction * size

    # Dollar PnL: holding 1 spread unit = long 1 share A, short beta shares B,
    # so its daily PnL equals the spread change. Scale by the position size.
    dollar_pnl = exposure.shift(1) * change

    # Gross notional deployed = |A| + |beta*B| per spread unit, times size.
    notional = (price_a.abs() + abs(beta) * price_b.abs()) * exposure.abs()
    cost = exposure.diff().abs() * (price_a.abs() + abs(beta) * price_b.abs()) * ga.COST_PER_TRADE

    net_pnl = (dollar_pnl - cost).fillna(0)

    # Return = net PnL relative to yesterday's deployed notional (0 when flat).
    deployed = notional.shift(1).replace(0, np.nan)
    ret = (net_pnl / deployed).fillna(0)

    out = pd.DataFrame(
        {
            "date": spread.index,
            "zscore": zscore.values,
            "exposure": exposure.values,
            "ret": ret.values,
        }
    )
    out["equity"] = STARTING_CAPITAL * (1 + out["ret"]).cumprod()
    return out


def summarize(df: pd.DataFrame) -> dict:
    ret = df["ret"]
    total_return = df["equity"].iloc[-1] / STARTING_CAPITAL - 1

    sharpe = (ret.mean() / ret.std()) * np.sqrt(ga.TRADING_DAYS) if ret.std() else 0.0

    running_peak = df["equity"].cummax()
    drawdown = df["equity"] / running_peak - 1
    max_drawdown = drawdown.min()

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "final_equity": df["equity"].iloc[-1],
    }


def main() -> None:
    panel = pd.read_csv(PANEL_PATH, index_col="date", parse_dates=True)
    spread_meta = pd.read_csv(SPREAD_PATH)
    a = spread_meta["ticker_a"].iloc[0]
    b = spread_meta["ticker_b"].iloc[0]
    beta = float(spread_meta["beta"].iloc[0])

    print(f"Dollar backtest for {a}-{b} (beta={beta:.4f}), capital=${STARTING_CAPITAL:,.0f}")
    df = run_dollar_backtest(panel[a], panel[b], beta)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    stats = summarize(df)
    print(f"  total return   = {stats['total_return'] * 100:.2f}%")
    print(f"  annualized Sharpe = {stats['sharpe']:.2f}")
    print(f"  max drawdown   = {stats['max_drawdown'] * 100:.2f}%")
    print(f"  final equity   = ${stats['final_equity']:,.0f}")


if __name__ == "__main__":
    main()
