import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BACKTEST_PATH = Path("data/processed/backtest_dollars.csv")
SPREAD_PATH = Path("data/processed/spread.csv")
OUTPUT_PATH = Path("out/backtest_report.png")

# Should match the thresholds used in backtest_dollars.py.
ENTRY_Z = 2.0
EXIT_Z = 0.5


def load() -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(BACKTEST_PATH, parse_dates=["date"]).set_index("date")
    meta = pd.read_csv(SPREAD_PATH)
    pair = f"{meta['ticker_a'].iloc[0]}-{meta['ticker_b'].iloc[0]}"
    return df, pair


def mark_trades(df: pd.DataFrame) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Entry = flat -> in a position; exit = in a position -> flat."""
    prev = df["exposure"].shift(1).fillna(0)
    entries = df.index[(prev == 0) & (df["exposure"] != 0)]
    exits = df.index[(prev != 0) & (df["exposure"] == 0)]
    return entries, exits


def build_figure(df: pd.DataFrame, pair: str) -> plt.Figure:
    entries, exits = mark_trades(df)

    drawdown = (df["equity"] / df["equity"].cummax() - 1) * 100

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    fig.suptitle(f"Pairs Backtest Report: {pair}", fontsize=15, fontweight="bold")

    # Panel 1: equity curve.
    axes[0].plot(df.index, df["equity"], color="tab:blue", linewidth=1.2)
    axes[0].axhline(df["equity"].iloc[0], color="grey", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Equity ($)")
    axes[0].set_title("Account Equity")
    axes[0].grid(alpha=0.3)

    # Panel 2: drawdown.
    axes[1].fill_between(df.index, drawdown, 0, color="tab:red", alpha=0.4)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_title(f"Drawdown (max {drawdown.min():.2f}%)")
    axes[1].grid(alpha=0.3)

    # Panel 3: z-score with threshold bands and trade markers.
    axes[2].plot(df.index, df["zscore"], color="black", linewidth=0.8, label="z-score")
    for level in (ENTRY_Z, -ENTRY_Z):
        axes[2].axhline(level, color="tab:orange", linestyle="--", linewidth=0.8)
    for level in (EXIT_Z, -EXIT_Z):
        axes[2].axhline(level, color="tab:green", linestyle=":", linewidth=0.8)
    axes[2].axhline(0, color="grey", linewidth=0.6)
    axes[2].scatter(
        entries, df.loc[entries, "zscore"], marker="^", color="tab:green",
        s=40, zorder=3, label="entry",
    )
    axes[2].scatter(
        exits, df.loc[exits, "zscore"], marker="v", color="tab:red",
        s=40, zorder=3, label="exit",
    )
    axes[2].set_ylabel("z-score")
    axes[2].set_title("Spread Z-Score with Entry/Exit Bands and Trades")
    axes[2].set_xlabel("Date")
    axes[2].legend(loc="upper left", ncol=3)
    axes[2].grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the backtest report.")
    parser.add_argument(
        "--show", action="store_true", help="open an interactive popup instead of saving"
    )
    args = parser.parse_args()

    df, pair = load()
    fig = build_figure(df, pair)

    entries, _ = mark_trades(df)
    drawdown = (df["equity"] / df["equity"].cummax() - 1) * 100
    print(f"  trades: {len(entries)} entries")
    print(f"  final equity: ${df['equity'].iloc[-1]:,.0f} | max drawdown: {drawdown.min():.2f}%")

    if args.show:
        plt.show()
    else:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_PATH, dpi=130)
        print(f"Saved report to {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
