A research-oriented pairs trading repository for learning, reimplementing, and extending canonical statistical arbitrage workflows on large-cap equities. The project focuses on spread-based mean-reversion, cointegration-driven pair selection, z-score signal generation, and baseline backtesting, with planned extensions using Genetic Algorithm optimization for selected trading and risk-management parameters.

#### Overview

Converge studies a simple but widely used form of statistical arbitrage: pairs trading. In this setting, two related assets are monitored for deviations in their historical relationship, and long/short trades are entered when the spread between them moves far enough from its typical range and is expected to revert. This approach is commonly framed as a market-neutral or relative-value strategy built around spread convergence rather than outright directional prediction.

The repository is structured as a learning-first research lab. The first goal is to reproduce a clean baseline workflow: obtain historical price data, identify candidate pairs, construct a spread, generate z-score trading signals, and backtest the resulting strategy. Once that baseline is working, the project will extend the pipeline with optimization methods, especially Genetic Algorithms, to explore improvements in threshold selection, position sizing, and trade management.

#### Project goals

- Reimplement a canonical pairs trading pipeline from scratch.
- Build a reproducible historical data ingestion workflow.
- Evaluate candidate pairs using statistical tests and spread diagnostics.
- Construct spread-based mean-reversion trading signals.
- Backtest a baseline long/short statistical arbitrage strategy.
- Extend the baseline with Genetic Algorithm-based optimization.

#### What the project is

This repository is best understood as a **pairs trading research lab** rather than a finished trading system. It is designed to make the workflow explicit and reproducible:

1. Fetch historical equity price data from an external data source.
2. Prepare and validate aligned time series.
3. Select candidate pairs using relationship and stationarity criteria.
4. Construct and standardize the spread.
5. Generate long/short entry and exit signals.
6. Backtest the baseline strategy.
7. Explore optimization extensions with Genetic Algorithms.

#### Current scope

The current version focuses on:

- Large-cap equity pairs
- Daily historical price data
- Spread-based mean-reversion logic
- Simple baseline backtesting
- Research-oriented experimentation in notebooks and scripts

The intended Genetic Algorithm extension is not meant to “discover” an entire trading strategy from scratch. Instead, it is meant to optimize interpretable parts of the pipeline, such as:

- trading thresholds,
- position sizing rules,
- volatility scaling,
- or drawdown-aware trade management.
