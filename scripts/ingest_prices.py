"""
Fetch and normalize daily adjusted equity price data for pairs trading research.

Primary source:  Tiingo  (keyed, SLA-backed, 30+ years of clean adjusted EOD data)
Fallback source: Yahoo Finance via yfinance (no key, used if Tiingo is unavailable)

Both sources are normalized into the same standardized long-format DataFrame and
saved to data/raw/prices.csv for downstream pair selection and backtesting.
"""

import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# Large-cap universe grouped by sector. Pairs are most likely to cointegrate
# within a sector, so candidates are drawn from same-sector names.
TICKERS = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD",
    # Banks / financials
    "JPM", "BAC", "WFC", "C", "GS", "MS",
    # Payments
    "V", "MA",
    # Energy
    "XOM", "CVX", "COP", "SLB",
    # Consumer staples
    "KO", "PEP", "PG", "CL",
    # Retail
    "WMT", "TGT", "COST", "HD", "LOW",
    # Healthcare / pharma
    "JNJ", "PFE", "MRK", "ABBV",
    # Telecom
    "VZ", "T",
]
START_DATE = pd.Timestamp("2016-01-01")
END_DATE = pd.Timestamp("2026-01-01")
OUTPUT_PATH = Path("data/raw/prices.csv")

TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")
TIINGO_BASE_URL = "https://api.tiingo.com/tiingo/daily"

COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]


def _finalize(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Coerce types, clip to date range, and return standardized columns."""
    df["ticker"] = symbol
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()

    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
    df = df.dropna(subset=["close", "adj_close"])
    return df[COLUMNS]


# Primary source: Tiingo
def fetch_tiingo_symbol(symbol: str) -> pd.DataFrame:
    """Fetch one symbol's adjusted EOD history from the Tiingo REST API."""
    url = f"{TIINGO_BASE_URL}/{symbol}/prices"
    params = {
        "startDate": START_DATE.strftime("%Y-%m-%d"),
        "endDate": END_DATE.strftime("%Y-%m-%d"),
        "format": "json",
        "token": TIINGO_API_KEY,
    }
    headers = {"Content-Type": "application/json"}

    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()

    if not isinstance(data, list) or not data:
        raise ValueError(f"Tiingo returned no data for {symbol}: {data}")

    df = pd.DataFrame(data).rename(
        columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "adjClose": "adj_close",
            "volume": "volume",
        }
    )
    return _finalize(df, symbol)


def fetch_tiingo(tickers: list[str]) -> pd.DataFrame:
    if not TIINGO_API_KEY:
        raise ValueError("Missing TIINGO_API_KEY in environment variables")

    frames = []
    for symbol in tickers:
        print(f"[tiingo] fetching {symbol}...")
        frames.append(fetch_tiingo_symbol(symbol))

    prices = pd.concat(frames, ignore_index=True)
    return prices.sort_values(["ticker", "date"]).reset_index(drop=True)


# Fallback source: Yahoo Finance (yfinance)
def fetch_yfinance(tickers: list[str]) -> pd.DataFrame:
    import yfinance as yf

    data = yf.download(
        tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,  # keep raw close and a separate Adj Close column
        group_by="ticker",
        progress=False,
        threads=True,
    )

    if data.empty:
        raise ValueError("yfinance returned no data; check tickers / date range")

    frames = []
    for symbol in tickers:
        print(f"[yfinance] processing {symbol}...")
        df = data[symbol].copy() if len(tickers) > 1 else data.copy()
        df = df.dropna(how="all").reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        frames.append(_finalize(df, symbol))

    prices = pd.concat(frames, ignore_index=True)
    return prices.sort_values(["ticker", "date"]).reset_index(drop=True)


# Orchestration
def fetch_prices(tickers: list[str]) -> pd.DataFrame:
    """Try Tiingo first; fall back to yfinance on any failure."""
    try:
        return fetch_tiingo(tickers)
    except Exception as exc:  # noqa: BLE001 - any failure should trigger fallback
        print(f"[warn] Tiingo unavailable ({exc}); falling back to yfinance...")
        return fetch_yfinance(tickers)


def main() -> None:
    print(f"Fetching {len(TICKERS)} tickers...")
    prices = fetch_prices(TICKERS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(OUTPUT_PATH, index=False)

    print(
        f"Saved {len(prices)} rows for {prices['ticker'].nunique()} tickers "
        f"to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
