"""
Target Creation Module for Quantum Stock Predictor (Phase 5).
Creates a binary classification target for short-term stock market trend prediction:
- Target = 1 (UP) if Tomorrow's Close > Today's Close
- Target = 0 (DOWN) if Tomorrow's Close <= Today's Close

Enforces strict zero-leakage policies: Tomorrow's price is used EXCLUSIVELY to create the label
and is NEVER included as an input feature.
"""

import os
import logging
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("target_creation")


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates binary classification target and drops trailing row.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with technical features and Date column sorted chronologically.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing all technical features and binary 'Target' column.
        Trailing row with missing tomorrow's price is removed.
    """
    logger.info("Starting target creation...")
    data = df.copy()

    # Ensure chronological date order
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(by='Date', ascending=True, inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Shift Close price by -1 to get Next Trading Day's Close Price
    tomorrow_close = data['Close'].shift(-1)

    # Binary Target: 1 (UP) if Tomorrow Close > Today Close, else 0 (DOWN)
    data['Target'] = (tomorrow_close > data['Close']).astype(int)

    # Remove final row where tomorrow's price is unknown
    data = data.iloc[:-1].copy()
    data.reset_index(drop=True, inplace=True)

    logger.info(f"Target creation complete. Total model dataset records: {len(data)}.")
    return data


def validate_model_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates model dataset integrity and prints summary details.

    Validation Rules:
    -----------------
    1. Date is chronological.
    2. No duplicate dates.
    3. No NaNs in any feature or target.
    4. Target contains only 0 and 1.
    5. No future information leakage in feature columns.

    Returns:
    --------
    Dict[str, Any]
        Validation statistics dictionary.
    """
    logger.info("Validating final model dataset...")
    data = df.copy()

    # 1. Chronological order check
    if not data['Date'].is_monotonic_increasing:
        raise ValueError("Validation Failed: Dates are not in strict chronological order.")

    # 2. Duplicate dates check
    duplicates = data['Date'].duplicated().sum()
    if duplicates > 0:
        raise ValueError(f"Validation Failed: Found {duplicates} duplicate date records.")

    # 3. Missing values check
    null_count = data.isnull().sum().sum()
    if null_count > 0:
        raise ValueError(f"Validation Failed: Found {null_count} NaN values in final model dataset.")

    # 4. Target values check
    unique_targets = set(data['Target'].unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(f"Validation Failed: Target column contains invalid values {unique_targets}.")

    # 5. Future leakage check (Ensure no 'Tomorrow' or 'Next' columns exist in feature set)
    feature_cols = [c for c in data.columns if c not in ['Date', 'Target']]
    forbidden_terms = ['tomorrow', 'next', 'shift(-1)', 'future']
    for col in feature_cols:
        for term in forbidden_terms:
            if term in col.lower():
                raise ValueError(f"Validation Failed: Suspected future data leakage in feature '{col}'.")

    up_count = int((data['Target'] == 1).sum())
    down_count = int((data['Target'] == 0).sum())

    stats = {
        'total_records': len(data),
        'num_features': len(feature_cols),
        'feature_names': feature_cols,
        'up_samples': up_count,
        'down_samples': down_count,
        'up_percentage': round(up_count / len(data) * 100, 2),
        'down_percentage': round(down_count / len(data) * 100, 2)
    }

    logger.info("Model dataset validation passed successfully!")
    return stats


def save_model_data(
    df: pd.DataFrame,
    ticker: str,
    processed_dir: str = "data/processed"
) -> str:
    """
    Saves final model dataset to CSV in processed_dir.

    Parameters:
    -----------
    df : pd.DataFrame
        Final model dataset.
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
    file_path = os.path.join(processed_dir, f"{ticker}_model_data.csv")
    df.to_csv(file_path, index=False)
    logger.info(f"Saved model dataset to {file_path}")
    return file_path


if __name__ == "__main__":
    import argparse
    from data_collection import download_stock_data
    from preprocessing import preprocess_data
    from feature_engineering import create_features

    parser = argparse.ArgumentParser(description="Create binary target for stock prediction.")
    parser.add_argument("--ticker", type=str, default="RELIANCE.NS", help="Stock ticker symbol")
    args = parser.parse_args()

    raw_df = download_stock_data(ticker=args.ticker)
    cleaned_df = preprocess_data(raw_df)
    featured_df = create_features(cleaned_df)
    model_df = create_target(featured_df)

    stats = validate_model_dataset(model_df)
    save_model_data(model_df, ticker=args.ticker)

    print("\n" + "="*50)
    print(f"Dataset: {args.ticker}")
    print(f"Original records: {len(raw_df)}")
    print(f"Records after indicators: {len(featured_df)}")
    print(f"Records after target creation: {stats['total_records']}")
    print(f"Features count: {stats['num_features']}")
    print(f"UP samples: {stats['up_samples']} ({stats['up_percentage']}%)")
    print(f"DOWN samples: {stats['down_samples']} ({stats['down_percentage']}%)")
    print("="*50 + "\n")
