"""
Robust GA optimization of the pairs-trading signal parameters.

Combines three defenses against overfitting:
  1. Train/test split  - parameters are evolved only on the in-sample period
                         and scored once out-of-sample.
  2. Multi-pair fitness - fitness is the AVERAGE train Sharpe across every
                         cointegrated, positive-beta pair, so the winning
                         parameters must generalize across spreads.
  3. Window penalty     - tiny rolling windows that fit noise are penalized.

Hedge ratios are estimated on the train slice only, so no test data leaks into
spread construction.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from deap import base, creator, tools

PANEL_PATH = Path("data/processed/adj_close.csv")
PAIRS_PATH = Path("data/processed/pairs.csv")
SPLIT_DATE = pd.Timestamp("2023-01-01")

COST_PER_TRADE = 0.0005
TRADING_DAYS = 252

# Gene bounds: (window, entry_z, exit_z)
WINDOW_BOUNDS = (5, 60)
ENTRY_BOUNDS = (1.0, 3.5)
EXIT_BOUNDS = (0.0, 1.5)

# Fitness shaping
MIN_WINDOW = 15          # windows below this are penalized
WINDOW_PENALTY = 0.05    # Sharpe penalty per day below MIN_WINDOW

# GA hyperparameters
POP_SIZE = 40
N_GEN = 25
CX_PROB = 0.6
MUT_PROB = 0.3
RANDOM_SEED = 42
OVERFIT_RATIO = 0.5      # flag if avg test Sharpe < this fraction of avg train


def estimate_hedge_ratio(price_a: pd.Series, price_b: pd.Series) -> float:
    """OLS hedge ratio (beta) of price_a on price_b."""
    model = sm.OLS(price_a, sm.add_constant(price_b)).fit()
    return float(model.params[price_b.name])


def build_pair_spreads() -> list[dict]:
    """Build train/test spread slices for every cointegrated, positive-beta pair.

    Beta is estimated on the train slice only to avoid look-ahead leakage.
    """
    panel = pd.read_csv(PANEL_PATH, index_col="date", parse_dates=True)
    pairs = pd.read_csv(PAIRS_PATH)

    train_mask = panel.index < SPLIT_DATE
    spreads = []
    for _, row in pairs[pairs["cointegrated"]].iterrows():
        a, b = row["ticker_a"], row["ticker_b"]
        beta = estimate_hedge_ratio(panel.loc[train_mask, a], panel.loc[train_mask, b])
        if beta <= 0:
            continue
        spread = panel[a] - beta * panel[b]
        spreads.append(
            {
                "pair": f"{a}-{b}",
                "train": spread.loc[train_mask],
                "test": spread.loc[~train_mask],
            }
        )
    return spreads


def decode(individual: list[float]) -> tuple[int, float, float]:
    """Clamp genes to their bounds; the window gene becomes an int."""
    window = int(round(np.clip(individual[0], *WINDOW_BOUNDS)))
    entry_z = float(np.clip(individual[1], *ENTRY_BOUNDS))
    exit_z = float(np.clip(individual[2], *EXIT_BOUNDS))
    return window, entry_z, exit_z


def generate_positions(zscore: pd.Series, entry_z: float, exit_z: float) -> pd.Series:
    """Hold a position from entry (|z| > entry_z) until reversion (|z| < exit_z)."""
    position = 0
    positions = []
    for value in zscore:
        if np.isnan(value):
            positions.append(0)
            continue
        if position == 0:
            if value > entry_z:
                position = -1
            elif value < -entry_z:
                position = 1
        elif abs(value) < exit_z:
            position = 0
        positions.append(position)
    return pd.Series(positions, index=zscore.index)


def backtest_sharpe(
    spread: pd.Series, window: int, entry_z: float, exit_z: float
) -> float:
    """Annualized Sharpe of the spread strategy for one parameter set."""
    if exit_z >= entry_z:
        return -10.0

    zscore = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    positions = generate_positions(zscore, entry_z, exit_z)

    gross_pnl = positions.shift(1) * spread.diff()
    cost = positions.diff().abs() * COST_PER_TRADE
    net_pnl = (gross_pnl - cost).fillna(0)

    if net_pnl.std() == 0:
        return -10.0
    return (net_pnl.mean() / net_pnl.std()) * np.sqrt(TRADING_DAYS)


def average_sharpe(spreads: list[dict], key: str, params: tuple) -> float:
    """Mean Sharpe across all pair spreads for the given slice ('train'/'test')."""
    scores = [backtest_sharpe(s[key], *params) for s in spreads]
    return float(np.mean(scores))


def build_toolbox(spreads: list[dict]) -> base.Toolbox:
    """Wire up DEAP; fitness is the penalized average train Sharpe."""
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    def evaluate(individual: list[float]) -> tuple[float]:
        window, entry_z, exit_z = decode(individual)
        sharpe = average_sharpe(spreads, "train", (window, entry_z, exit_z))
        if window < MIN_WINDOW:
            sharpe -= (MIN_WINDOW - window) * WINDOW_PENALTY
        return (sharpe,)

    toolbox = base.Toolbox()
    toolbox.register("attr_window", random.uniform, *WINDOW_BOUNDS)
    toolbox.register("attr_entry", random.uniform, *ENTRY_BOUNDS)
    toolbox.register("attr_exit", random.uniform, *EXIT_BOUNDS)
    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (toolbox.attr_window, toolbox.attr_entry, toolbox.attr_exit),
        n=1,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.4)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox


def evolve(toolbox: base.Toolbox) -> tools.HallOfFame:
    """Run the GA and return the hall of fame holding the best individual."""
    population = toolbox.population(n=POP_SIZE)
    hall_of_fame = tools.HallOfFame(1)

    for individual in population:
        individual.fitness.values = toolbox.evaluate(individual)

    for _ in range(N_GEN):
        selected = toolbox.select(population, len(population))
        offspring = [toolbox.clone(ind) for ind in selected]

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX_PROB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUT_PROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        for individual in offspring:
            if not individual.fitness.valid:
                individual.fitness.values = toolbox.evaluate(individual)

        population[:] = offspring
        hall_of_fame.update(population)

    return hall_of_fame


def main() -> None:
    random.seed(RANDOM_SEED)

    spreads = build_pair_spreads()
    print(f"Optimizing over {len(spreads)} pairs: {[s['pair'] for s in spreads]}")

    toolbox = build_toolbox(spreads)
    hall_of_fame = evolve(toolbox)

    window, entry_z, exit_z = decode(hall_of_fame[0])
    params = (window, entry_z, exit_z)
    train_sharpe = average_sharpe(spreads, "train", params)
    test_sharpe = average_sharpe(spreads, "test", params)

    print(f"Best params: WINDOW={window}, ENTRY_Z={entry_z:.2f}, EXIT_Z={exit_z:.2f}")
    print(f"  avg in-sample  (train) Sharpe = {train_sharpe:.3f}")
    print(f"  avg out-sample (test)  Sharpe = {test_sharpe:.3f}")
    print("  per-pair out-of-sample Sharpe:")
    for s in spreads:
        print(f"    {s['pair']:>12}: {backtest_sharpe(s['test'], *params):.3f}")

    if test_sharpe < train_sharpe * OVERFIT_RATIO:
        print("  WARNING: large train-to-test drop suggests overfitting.")
    else:
        print("  OK: parameters hold up reasonably out-of-sample.")


if __name__ == "__main__":
    main()
