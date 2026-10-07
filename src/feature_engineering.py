"""
Feature Engineering Module for Quantum Stock Predictor (Phase 4).
Calculates technical indicators and derived market features using the 'ta' library.
Guarantees NO future data leakage by strictly relying on historical OHLCV data.
"""

import os
import logging
from typing import Optional
import pandas as pd
import numpy as np
import ta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("feature_engineering")


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators and features from clean OHLCV stock data.

    Features computed:
    ------------------
    - Base OHLCV: Date, Open, High, Low, Close, Volume
    - Moving Averages: SMA_10, SMA_20, SMA_50, EMA_20
    - Momentum: RSI, MACD, MACD_Signal
    - Volatility: BB_High, BB_Low, BB_Width, ATR, Daily_Return, Volatility
    - Price Percentages: High_Low_Pct, Close_Open_Pct

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned OHLCV stock DataFrame sorted chronologically by Date.

    Returns:
    --------
    pd.DataFrame
        DataFrame with technical features added and NaN rows removed.
    """
    logger.info("Starting feature engineering calculations...")
    data = df.copy()

    # Ensure Date is sorted chronologically
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(by='Date', ascending=True, inplace=True)
    data.reset_index(drop=True, inplace=True)

    close = data['Close']
    high = data['High']
    low = data['Low']
    open_p = data['Open']

    # 1. Moving Averages
    data['SMA_10'] = ta.trend.SMAIndicator(close=close, window=10).sma_indicator()
    data['SMA_20'] = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
    data['SMA_50'] = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
    data['EMA_20'] = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()

    # 2. Momentum Indicators
    data['RSI'] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    macd_obj = ta.trend.MACD(close=close)
    data['MACD'] = macd_obj.macd()
    data['MACD_Signal'] = macd_obj.macd_signal()

    # 3. Volatility Indicators
    bb_obj = ta.volatility.BollingerBands(close=close, window=20)
    data['BB_High'] = bb_obj.bollinger_hband()
    data['BB_Low'] = bb_obj.bollinger_lband()
    data['BB_Width'] = bb_obj.bollinger_wband()
    data['ATR'] = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

    # 4. Daily Return and Volatility
    data['Daily_Return'] = close.pct_change()
    data['Volatility'] = data['Daily_Return'].rolling(window=20).std()

    # 5. Additional Derived Features
    data['High_Low_Pct'] = (high - low) / low * 100.0
    data['Close_Open_Pct'] = (close - open_p) / open_p * 100.0

    initial_count = len(data)
    # Drop rows containing NaNs introduced by indicator initialization windows (e.g. SMA_50 needs 50 rows)
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    final_count = len(data)

    logger.info(
        f"Feature engineering complete. Original records: {initial_count}, "
        f"Valid records after indicators: {final_count} (dropped {initial_count - final_count} initial window rows)."
    )

    return data


def save_featured_data(
    df: pd.DataFrame,
    ticker: str,
    processed_dir: str = "data/processed"
) -> str:
    """
    Saves featured DataFrame to CSV in processed_dir.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with technical features.
    ticker : str
        Stock ticker symbol.
    processed_dir : str
        Target output directory.

    Returns:
    --------
    str
        Path to saved CSV file.
    """
    os.makedirs(processed_dir, exist_ok=True)
    file_path = os.path.join(processed_dir, f"{ticker}_features.csv")
    df.to_csv(file_path, index=False)
    logger.info(f"Saved feature dataset to {file_path}")
    return file_path


if __name__ == "__main__":
    import argparse
    from preprocessing import preprocess_data
    from data_collection import download_stock_data

    parser = argparse.ArgumentParser(description="Generate technical features for stock data.")
    parser.add_argument("--ticker", type=str, default="RELIANCE.NS", help="Stock ticker symbol")
    args = parser.parse_args()

    raw_df = download_stock_data(ticker=args.ticker)
    cleaned_df = preprocess_data(raw_df)
    featured_df = create_features(cleaned_df)
    save_featured_data(featured_df, ticker=args.ticker)
    print(featured_df.info())
