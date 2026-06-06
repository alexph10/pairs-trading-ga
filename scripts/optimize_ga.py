"""
Optimize the pairs-trading signal params with a Genetic Algo (DEAP)
Each individual is a parameter set [window, entry_z, exit_z]; fitness is the
annualized Sharpe of the resulting backtest. The GA evolves better parameters
than hand-tuning, while keeping every knob interpretable
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
from deap import base, creator, tools

SPREAD_PATH = Path("data/processed/spread.csv")
COST_PER_TRADE = 0.0005
TRADING_DAYS = 252


# Gene bounds [Wwindow, entry_z, exit_z]
BOUNDS = [(5, 60), (1.0, 3.5), (0.0, 1.5)]

# GA hyperparameters
POP_SIZE = 40
N_GEN = 25
CX_PROB = 0.6
MUT_PROB = 0.3

SPREAD = pd.read_csv(SPREAD_PATH, parse_dates=["date"])["spread"]


def decode(individual):
    # Clamp genes to bounds and turn the window gene into an int
    window = int(round(np.clip(individual[0], *BOUNDS[0])))
    entry_z = float(np.clip(individual[1], *BOUNDS[1]))
    exit_z = float(np.clip(individual[2], *BOUNDS[2]))
    return window, entry_z, exit_z


def backtest_sharpe(window: int, entry_z: float, exit_z: float) -> float:
    if exit_z >= entry_z:
        return -10.0

    spread = SPREAD
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    z = (spread - mean) / std

    # Stateful position: hold until reversion (same logic as generate_signals)
    position, positions = 0, []
    for value in z:
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
    positions = pd.Series(positions, index=spread.index)

    gross = positions.shift(1) * spread.diff()
    cost = positions.diff().abs() * COST_PER_TRADE
    net = (gross - cost).fillna(0)

    if net.std() == 0:
        return -10.0
    return (net.mean() / net.std()) * np.sqrt(TRADING_DAYS)


def evaluate(individual):
    window, entry_z, exit_z = decode(individual)
    return (backtest_sharpe(window, entry_z, exit_z),)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_window", random.uniform, *BOUNDS[0])
toolbox.register("attr_entry", random.uniform, *BOUNDS[1])
toolbox.register("attr_exit", random.uniform, *BOUNDS[2])
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


def main() -> None:
    random.seed(42)
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)

    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    for gen in range(N_GEN):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX_PROB:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < MUT_PROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Re-evaluate only the individuals that changed
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)
        pop[:] = offspring
        hof.update(pop)
        best = hof[0]
        print(f"gen {gen:02d} | best Sharpe = {best.fitness.values[0]:.3f}")

    window, entry_z, exit_z = decode(hof[0])
    print("\nBest params found:")
    print(f" WINDOW = {window}")
    print(f" ENTRY_Z = {entry_z:.2f}")
    print(f"  EXIT_Z  = {exit_z:.2f}")
    print(f"  Sharpe  = {hof[0].fitness.values[0]:.3f}")


if __name__ == "__main__":
    main()
