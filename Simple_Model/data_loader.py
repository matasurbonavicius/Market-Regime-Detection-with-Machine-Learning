"""
Simple data loader for market regime prediction.
Fetches SPY data and calculates technical indicators.
"""

import pandas as pd
import numpy as np
import numpy.typing as npt
import yfinance as yf
from dataclasses import dataclass
from typing import List


@dataclass
class DataSplit:
    """Container for train/validation/test split."""
    train_features: npt.NDArray[np.floating]
    train_prices: npt.NDArray[np.floating]
    train_dates: npt.NDArray[np.datetime64]

    val_features: npt.NDArray[np.floating]
    val_prices: npt.NDArray[np.floating]
    val_dates: npt.NDArray[np.datetime64]

    test_features: npt.NDArray[np.floating]
    test_prices: npt.NDArray[np.floating]
    test_dates: npt.NDArray[np.datetime64]

    feature_names: List[str]


def fetch_data(symbol: str = "SPY",
               start: str = "2000-01-01",
               end: str = "2023-12-01",
               interval: str = "1wk") -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance."""

    print(f"Fetching {symbol} data from {start} to {end}...")
    df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)

    if df is None or df.empty:
        raise ValueError("No data fetched. Check symbol and date range.")

    # Handle multi-index columns from newer yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()
    print(f"Got {len(df)} rows")
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators.
    Balanced set: 2 mean-reversion + 2 trend/momentum
    """

    close = df['Close']
    high = df['High']
    low = df['Low']

    # --- Mean Reversion Indicators ---

    # RSI (multiple windows)
    for window in [5, 10, 14]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        df[f'RSI_{window}'] = 100 - (100 / (1 + rs))

    # Stochastic (multiple windows)
    for window in [5, 10, 14]:
        lowest_low = low.rolling(window=window).min()
        highest_high = high.rolling(window=window).max()
        df[f'Stoch_{window}'] = 100 * (close - lowest_low) / (highest_high - lowest_low)

    # --- Trend/Momentum Indicators ---

    # Momentum (price change over window)
    for window in [5, 10, 14]:
        df[f'Mom_{window}'] = close.pct_change(periods=window) * 100

    # MACD
    for fast, slow in [(8, 17), (12, 26), (20, 38)]:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        df[f'MACD_{fast}_{slow}'] = ema_fast - ema_slow

    return df


def prepare_data(df: pd.DataFrame,
                 train_end: str = "2017-01-01",
                 val_end: str = "2020-01-01") -> DataSplit:
    """
    Split data temporally and prepare for modeling.

    Train: start -> train_end
    Val: train_end -> val_end
    Test: val_end -> end
    """

    # Get feature columns (everything except OHLCV)
    ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    feature_cols = [c for c in df.columns if c not in ohlcv_cols]

    # Drop NaN rows from indicator calculation
    df_clean = df.dropna()

    # Temporal splits using loc with string dates
    train = df_clean.loc[:train_end].iloc[:-1] 
    val = df_clean.loc[train_end:val_end].iloc[:-1]  
    test = df_clean.loc[val_end:]  

    print(f"Train: {train.index[0].date()} to {train.index[-1].date()} ({len(train)} rows)")
    print(f"Val:   {val.index[0].date()} to {val.index[-1].date()} ({len(val)} rows)")
    print(f"Test:  {test.index[0].date()} to {test.index[-1].date()} ({len(test)} rows)")

    return DataSplit(
        train_features=np.asarray(train[feature_cols].values, dtype=np.float64),
        train_prices=np.asarray(train['Close'].values, dtype=np.float64),
        train_dates=np.asarray(train.index.values, dtype=np.datetime64),

        val_features=np.asarray(val[feature_cols].values, dtype=np.float64),
        val_prices=np.asarray(val['Close'].values, dtype=np.float64),
        val_dates=np.asarray(val.index.values, dtype=np.datetime64),

        test_features=np.asarray(test[feature_cols].values, dtype=np.float64),
        test_prices=np.asarray(test['Close'].values, dtype=np.float64),
        test_dates=np.asarray(test.index.values, dtype=np.datetime64),

        feature_names=feature_cols
    )


def load_and_prepare(symbol: str = "SPY",
                     start: str = "2000-01-01",
                     end: str = "2023-12-01",
                     train_end: str = "2017-01-01",
                     val_end: str = "2020-01-01") -> DataSplit:
    """Main entry point: fetch, calculate indicators, split."""

    df = fetch_data(symbol, start, end)
    df = calculate_indicators(df)
    data = prepare_data(df, train_end, val_end)

    return data


if __name__ == "__main__":
    # Quick test
    data = load_and_prepare()
    print(f"\nFeatures: {data.feature_names}")
    print(f"Train shape: {data.train_features.shape}")
