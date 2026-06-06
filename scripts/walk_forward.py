"""
Walk-forward validation of the pairs-trading GA.

Instead of a single train/test split, slide overlapping windows through time:
optimize parameters on each train window, then score them on the following
(unseen) test window. Robustness = consistent out-of-sample Sharpe across every
fold. This is the gold-standard defense against curve-fitting one period.

Reuses the GA machinery and backtest from optimize_ga_split.
"""

import random

import numpy as np
import pandas as pd

import optimize_ga_split as ga

# Fold geometry (in trading days, ~252/year)
TRAIN_DAYS = 4 * 252      # train window length
TEST_DAYS = 252           # test window length
STEP_DAYS = 252           # how far each fold advances


def load_panel_and_pairs():
    panel = pd.read_csv(ga.PANEL_PATH, index_col="date", parse_dates=True)
    pairs = pd.read_csv(ga.PAIRS_PATH)
    return panel, pairs


def build_fold_spreads(panel, pairs, train_idx, test_idx) -> list[dict]:
    """Build pruned spreads for one fold; beta and pruning use train data only."""
    spreads = []
    for _, row in pairs[pairs["cointegrated"]].iterrows():
        a, b = row["ticker_a"], row["ticker_b"]
        beta = ga.estimate_hedge_ratio(panel.loc[train_idx, a], panel.loc[train_idx, b])
        if beta <= 0:
            continue

        spread = panel[a] - beta * panel[b]
        train_spread = spread.loc[train_idx]

        prune_params = (
            ga.PRUNE_WINDOW, ga.PRUNE_ENTRY, ga.PRUNE_EXIT,
            ga.STOP_BOUNDS[1], ga.VOL_TARGET_BOUNDS[0],
        )
        if ga.backtest_sharpe(train_spread, *prune_params) < ga.MIN_TRAIN_SHARPE:
            continue

        spreads.append(
            {"pair": f"{a}-{b}", "train": train_spread, "test": spread.loc[test_idx]}
        )
    return spreads


def run_fold(panel, pairs, train_idx, test_idx) -> dict | None:
    """Optimize on the train window, evaluate on the test window."""
    spreads = build_fold_spreads(panel, pairs, train_idx, test_idx)
    if not spreads:
        return None

    toolbox = ga.build_toolbox(spreads)
    hall_of_fame = ga.evolve(toolbox)
    params = ga.decode(hall_of_fame[0])

    return {
        "params": params,
        "n_pairs": len(spreads),
        "train_sharpe": ga.average_sharpe(spreads, "train", params),
        "test_sharpe": ga.average_sharpe(spreads, "test", params),
    }


def main() -> None:
    random.seed(ga.RANDOM_SEED)
    panel, pairs = load_panel_and_pairs()
    dates = panel.index

    results = []
    start = 0
    fold = 0
    while start + TRAIN_DAYS + TEST_DAYS <= len(dates):
        train_idx = dates[start : start + TRAIN_DAYS]
        test_idx = dates[start + TRAIN_DAYS : start + TRAIN_DAYS + TEST_DAYS]

        outcome = run_fold(panel, pairs, train_idx, test_idx)
        if outcome is not None:
            results.append(outcome)
            w, e, x, s, v = outcome["params"]
            print(
                f"fold {fold}: test {test_idx[0].date()}..{test_idx[-1].date()} | "
                f"pairs={outcome['n_pairs']} | "
                f"train={outcome['train_sharpe']:.2f} test={outcome['test_sharpe']:.2f} | "
                f"W={w} E={e:.2f} X={x:.2f} S={s:.2f} V={v:.2f}"
            )
        fold += 1
        start += STEP_DAYS

    if not results:
        raise ValueError("No valid folds; check data length vs fold geometry")

    test_scores = [r["test_sharpe"] for r in results]
    print("\nWalk-forward summary")
    print(f"  folds                = {len(test_scores)}")
    print(f"  mean test Sharpe     = {np.mean(test_scores):.3f}")
    print(f"  std  test Sharpe     = {np.std(test_scores):.3f}")
    print(f"  worst / best         = {min(test_scores):.3f} / {max(test_scores):.3f}")
    positive = sum(s > 0 for s in test_scores)
    print(f"  positive folds       = {positive}/{len(test_scores)}")


if __name__ == "__main__":
    main()
