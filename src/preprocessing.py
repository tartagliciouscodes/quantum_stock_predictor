"""
Data Preprocessing Module for Quantum Stock Predictor.
Handles data cleaning, missing-value removal, duplicate dropping,
chronological date sorting, numeric conversions, and dataset validation.
"""

import os
import logging
from typing import Tuple
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("preprocessing")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw stock market dataframe.

    Steps performed:
    ----------------
    1. Validates presence of required OHLCV columns.
    2. Converts 'Date' column to datetime and strips timezone if present.
    3. Sorts data chronologically by 'Date'.
    4. Removes duplicate rows based on 'Date'.
    5. Converts numeric columns to float/int types.
    6. Removes invalid records (e.g., negative/zero prices or volume).
    7. Drops missing/NaN values.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw stock DataFrame containing OHLCV columns.

    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame ready for feature engineering.
    """
    logger.info("Starting data preprocessing...")
    data = df.copy()

    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in data.columns:
            raise KeyError(f"Required column '{col}' missing from input DataFrame.")

    # 1. Date conversion & strip timezone
    data['Date'] = pd.to_datetime(data['Date'])
    if pd.api.types.is_datetime64_any_dtype(data['Date']):
        data['Date'] = data['Date'].dt.tz_localize(None)

    # 2. Chronological date sorting
    data.sort_values(by='Date', ascending=True, inplace=True)
    initial_rows = len(data)

    # 3. Duplicate removal based on Date
    data.drop_duplicates(subset=['Date'], keep='first', inplace=True)
    duplicates_removed = initial_rows - len(data)
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate date entries.")

    # 4. Numeric conversion
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # 5. Invalid-value handling (prices > 0, volume >= 0)
    invalid_mask = (
        (data['Open'] <= 0) |
        (data['High'] <= 0) |
        (data['Low'] <= 0) |
        (data['Close'] <= 0) |
        (data['Volume'] < 0)
    )
    invalid_count = invalid_mask.sum()
    if invalid_count > 0:
        logger.warning(f"Found and removed {invalid_count} rows with non-positive prices or negative volume.")
        data = data[~invalid_mask]

    # 6. Missing value handling
    missing_count = data.isnull().sum().sum()
    if missing_count > 0:
        logger.info(f"Found {missing_count} missing values across all columns. Dropping NaN rows...")
        data.dropna(subset=numeric_cols, inplace=True)

    data.reset_index(drop=True, inplace=True)
    logger.info(f"Preprocessing completed. Initial rows: {initial_rows}, Final clean rows: {len(data)}")

    return data


def save_processed_data(
    df: pd.DataFrame,
    ticker: str,
    processed_dir: str = "data/processed"
) -> str:
    """
    Saves cleaned DataFrame to CSV in processed_dir.

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned stock DataFrame.
    ticker : str
        Stock ticker symbol.
    processed_dir : str
        Directory to save processed dataset.

    Returns:
    --------
    str
        File path to saved processed dataset.
    """
    os.makedirs(processed_dir, exist_ok=True)
    file_path = os.path.join(processed_dir, f"{ticker}_cleaned.csv")
    df.to_csv(file_path, index=False)
    logger.info(f"Saved processed data to {file_path}")
    return file_path


if __name__ == "__main__":
    import argparse
    from data_collection import download_stock_data

    parser = argparse.ArgumentParser(description="Clean and preprocess stock OHLCV data.")
    parser.add_argument("--ticker", type=str, default="RELIANCE.NS", help="Stock ticker symbol")
    args = parser.parse_args()

    raw_df = download_stock_data(ticker=args.ticker)
    cleaned_df = preprocess_data(raw_df)
    save_processed_data(cleaned_df, ticker=args.ticker)
    print(cleaned_df.info())
