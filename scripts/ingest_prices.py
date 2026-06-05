"""
Fetch daily historical equity price data from the Alpha Vantage API and save it
as a single normalized dataset for downstream pairs trading research.

This script retrieves daily adjusted OHLCV data for a fixed list of ticker
symbols, converts each API response into a standardized pandas DataFrame, filters
the data to the configured date range, combines all symbols into one long-format
table, and writes the result to data/raw/prices.csv.

The output dataset is intended to support later stages of the project, including
pair selection, spread construction, z-score signal generation, and baseline
backtesting for statistical arbitrage experiments.
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://www.alphavantage.co/query"
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

TICKERS = ["AAPL", "MSFT", "GOOGL"]
START_DATE = "2016-01-01"
END_DATE = "2026-01-01"
OUTPUT_PATH = Path("data/raw/prices.csv")

if not API_KEY:
    raise ValueError("Missing ALPHA_VANTAGE_API_KEY in environment variables")


def fetch_symbol_data(symbol: str) -> dict:
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": "full",
        "apikey": API_KEY,
    }

    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "Error Message" in data:
        raise ValueError(f"API error for {symbol}: {data['Error Message']}")

    if "Note" in data:
        raise ValueError(f"Rate limit hit for {symbol}: {data['Note']}")

    if "Time Series (Daily)" not in data:
        raise ValueError(f"Unexpected response for {symbol}: {data}")

    return data["Time Series (Daily)"]


def transform_symbol_data(symbol: str, raw_data: dict) -> pd.DataFrame:
    df = pd.DataFrame(raw_data).T
    df.index.name = "date"
    df = df.reset_index()

    df = df.rename(
        columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. adjusted close": "adj_close",
            "6. volume": "volumn",
        }
    )

    df["ticker"] = symbol
    df["date"] = pd.to_datetime(df["date"])

    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
    df = df.sort_values("date").reset_index(drop=True)

    return df[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]


all_frames = []

for symbol in TICKERS:
    print(f"Fetching {symbol}...")
    raw_data = fetch_symbol_data(symbol)
    df_symbol = transform_symbol_data(symbol, raw_data)
    all_frames.append(df_symbol)
    time.sleep(12)

prices = pd.concat(all_frames, ignore_index=True)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

prices.to_csv(OUTPUT_PATH, index=False)

print(
    f"Saved {len(prices)} rows for {prices['ticker'].nunique()} tickers to {OUTPUT_PATH}"
)
