"""
Robust GA optimization of the pairs-trading strategy.

Defenses against overfitting:
  1. Train/test split   - parameters are evolved only on the in-sample period
                          and scored once out-of-sample.
  2. Pair pruning       - only pairs that are tradeable in-sample (train Sharpe
                          above a floor under default settings) enter the GA, so
                          dead pairs cannot drag the fitness around. Pruning uses
                          train data only (no look-ahead).
  3. Multi-pair fitness - fitness is the AVERAGE train Sharpe across surviving
                          pairs, so winning parameters must generalize.
  4. Window penalty     - tiny rolling windows that fit noise are penalized.

The genome has five interpretable genes:
  window, entry_z, exit_z, stop_z, vol_target
where stop_z is a stop-loss band (cut a losing trade if |z| blows past it) and
vol_target drives inverse-volatility position sizing (smaller bets when the
spread is choppy). Hedge ratios are estimated on train data only.
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
MAX_SIZE = 3.0           # cap on inverse-vol position size (leverage limit)

# Gene bounds: (window, entry_z, exit_z, stop_z, vol_target)
WINDOW_BOUNDS = (5, 60)
ENTRY_BOUNDS = (1.0, 3.5)
EXIT_BOUNDS = (0.0, 1.5)
STOP_BOUNDS = (3.0, 6.0)
VOL_TARGET_BOUNDS = (0.5, 4.0)

# Pair pruning (uses train data only)
PRUNE_WINDOW = 30
PRUNE_ENTRY = 2.0
PRUNE_EXIT = 0.5
MIN_TRAIN_SHARPE = 0.20

# Fitness shaping
MIN_WINDOW = 15
WINDOW_PENALTY = 0.05

# GA hyperparameters
POP_SIZE = 40
N_GEN = 25
CX_PROB = 0.6
MUT_PROB = 0.3
RANDOM_SEED = 42
OVERFIT_RATIO = 0.5


def estimate_hedge_ratio(price_a: pd.Series, price_b: pd.Series) -> float:
    """OLS hedge ratio (beta) of price_a on price_b."""
    model = sm.OLS(price_a, sm.add_constant(price_b)).fit()
    return float(model.params[price_b.name])


def decode(individual: list[float]) -> tuple[int, float, float, float, float]:
    """Clamp genes to their bounds; the window gene becomes an int."""
    window = int(round(np.clip(individual[0], *WINDOW_BOUNDS)))
    entry_z = float(np.clip(individual[1], *ENTRY_BOUNDS))
    exit_z = float(np.clip(individual[2], *EXIT_BOUNDS))
    stop_z = float(np.clip(individual[3], *STOP_BOUNDS))
    vol_target = float(np.clip(individual[4], *VOL_TARGET_BOUNDS))
    return window, entry_z, exit_z, stop_z, vol_target


def generate_positions(
    zscore: pd.Series, entry_z: float, exit_z: float, stop_z: float
) -> pd.Series:
    """Position direction with mean-reversion entry/exit and a stop-loss.

    Enter at |z| > entry_z, take profit at |z| < exit_z, and stop out if the
    trade moves further against us past stop_z.
    """
    position = 0
    positions = []
    for value in zscore:
        if np.isnan(value):
            positions.append(0)
            continue
        if position == 0:
            if value > entry_z:
                position = -1          # spread rich -> short it
            elif value < -entry_z:
                position = 1           # spread cheap -> long it
        elif position == -1 and value > stop_z:
            position = 0               # stop-loss: spread kept rising
        elif position == 1 and value < -stop_z:
            position = 0               # stop-loss: spread kept falling
        elif abs(value) < exit_z:
            position = 0               # take profit: reverted to mean
        positions.append(position)
    return pd.Series(positions, index=zscore.index)


def backtest_sharpe(
    spread: pd.Series,
    window: int,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    vol_target: float,
) -> float:
    """Annualized Sharpe with stop-loss and inverse-volatility sizing."""
    # Bands must be ordered: exit inside entry inside stop.
    if not (exit_z < entry_z < stop_z):
        return -10.0

    change = spread.diff()
    rolling_std = change.rolling(window).std()

    zscore = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    direction = generate_positions(zscore, entry_z, exit_z, stop_z)

    # Inverse-vol sizing: smaller bets when the spread is choppy, capped.
    size = (vol_target / rolling_std).clip(upper=MAX_SIZE).fillna(0.0)
    exposure = direction * size

    gross_pnl = exposure.shift(1) * change
    cost = exposure.diff().abs() * COST_PER_TRADE
    net_pnl = (gross_pnl - cost).fillna(0)

    if net_pnl.std() == 0:
        return -10.0
    return (net_pnl.mean() / net_pnl.std()) * np.sqrt(TRADING_DAYS)


def build_pair_spreads() -> list[dict]:
    """Build train/test spreads for cointegrated, positive-beta, tradeable pairs.

    Beta and the pruning Sharpe are computed on the train slice only.
    """
    panel = pd.read_csv(PANEL_PATH, index_col="date", parse_dates=True)
    pairs = pd.read_csv(PAIRS_PATH)

    train_mask = panel.index < SPLIT_DATE
    prune_params = (PRUNE_WINDOW, PRUNE_ENTRY, PRUNE_EXIT, STOP_BOUNDS[1], VOL_TARGET_BOUNDS[0])

    spreads = []
    for _, row in pairs[pairs["cointegrated"]].iterrows():
        a, b = row["ticker_a"], row["ticker_b"]
        beta = estimate_hedge_ratio(panel.loc[train_mask, a], panel.loc[train_mask, b])
        if beta <= 0:
            continue

        spread = panel[a] - beta * panel[b]
        train_spread = spread.loc[train_mask]

        # Pair pruning: drop pairs that are not even tradeable in-sample.
        if backtest_sharpe(train_spread, *prune_params) < MIN_TRAIN_SHARPE:
            continue

        spreads.append(
            {"pair": f"{a}-{b}", "train": train_spread, "test": spread.loc[~train_mask]}
        )
    return spreads


def average_sharpe(spreads: list[dict], key: str, params: tuple) -> float:
    """Mean Sharpe across all pair spreads for the given slice ('train'/'test')."""
    scores = [backtest_sharpe(s[key], *params) for s in spreads]
    return float(np.mean(scores))


def build_toolbox(spreads: list[dict]) -> base.Toolbox:
    """Wire up DEAP; fitness is the penalized average train Sharpe."""
    # Guard so the GA can be rebuilt repeatedly (e.g. per walk-forward fold).
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    def evaluate(individual: list[float]) -> tuple[float]:
        params = decode(individual)
        sharpe = average_sharpe(spreads, "train", params)
        window = params[0]
        if window < MIN_WINDOW:
            sharpe -= (MIN_WINDOW - window) * WINDOW_PENALTY
        return (sharpe,)

    toolbox = base.Toolbox()
    toolbox.register("attr_window", random.uniform, *WINDOW_BOUNDS)
    toolbox.register("attr_entry", random.uniform, *ENTRY_BOUNDS)
    toolbox.register("attr_exit", random.uniform, *EXIT_BOUNDS)
    toolbox.register("attr_stop", random.uniform, *STOP_BOUNDS)
    toolbox.register("attr_vol", random.uniform, *VOL_TARGET_BOUNDS)
    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (
            toolbox.attr_window,
            toolbox.attr_entry,
            toolbox.attr_exit,
            toolbox.attr_stop,
            toolbox.attr_vol,
        ),
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
    if not spreads:
        raise ValueError("No tradeable pairs survived pruning; loosen MIN_TRAIN_SHARPE")
    print(f"Optimizing over {len(spreads)} pruned pairs: {[s['pair'] for s in spreads]}")

    toolbox = build_toolbox(spreads)
    hall_of_fame = evolve(toolbox)

    window, entry_z, exit_z, stop_z, vol_target = decode(hall_of_fame[0])
    params = (window, entry_z, exit_z, stop_z, vol_target)
    train_sharpe = average_sharpe(spreads, "train", params)
    test_sharpe = average_sharpe(spreads, "test", params)

    print(
        f"Best params: WINDOW={window}, ENTRY_Z={entry_z:.2f}, EXIT_Z={exit_z:.2f}, "
        f"STOP_Z={stop_z:.2f}, VOL_TARGET={vol_target:.2f}"
    )
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
