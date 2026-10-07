"""
Unit tests for Data Collection, Preprocessing, Feature Engineering, and Target Creation Pipeline (Phases 1-5).
Verifies data integrity, feature creation, zero future leakage, and target validation.
"""

import os
import pytest
import pandas as pd
from src.data_collection import download_stock_data, validate_real_ohlcv_data
from src.preprocessing import preprocess_data, save_processed_data
from src.feature_engineering import create_features, save_featured_data
from src.target_creation import create_target, validate_model_dataset, save_model_data

TEST_TICKER = "RELIANCE.NS"


def test_download_and_verify_real_stock_data():
    """
    Test downloading and verifying real stock data for RELIANCE.NS.
    """
    df = download_stock_data(ticker=TEST_TICKER, start_date="2022-01-01", save_raw=True)

    assert isinstance(df, pd.DataFrame), "Output must be a pandas DataFrame"
    assert not df.empty, f"Downloaded dataframe for {TEST_TICKER} should not be empty"

    expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    for col in expected_cols:
        assert col in df.columns, f"Column '{col}' must be present in real market data"

    assert validate_real_ohlcv_data(df) is True, "Dataset must pass real market data validation"


def test_feature_engineering():
    """
    Test Phase 4 feature engineering: technical indicators creation and column validation.
    """
    raw_df = download_stock_data(ticker=TEST_TICKER, start_date="2022-01-01")
    cleaned_df = preprocess_data(raw_df)
    featured_df = create_features(cleaned_df)

    assert not featured_df.empty, "Featured DataFrame should not be empty"
    assert featured_df.isnull().sum().sum() == 0, "No NaN values should remain after indicator calculation"

    # Verify expected technical indicators exist
    expected_indicators = [
        'SMA_10', 'SMA_20', 'SMA_50', 'EMA_20', 'RSI',
        'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'BB_Width',
        'ATR', 'Daily_Return', 'Volatility', 'High_Low_Pct', 'Close_Open_Pct'
    ]
    for indicator in expected_indicators:
        assert indicator in featured_df.columns, f"Indicator '{indicator}' must exist in featured dataset"

    # Verify file saving
    feat_path = save_featured_data(featured_df, ticker=TEST_TICKER)
    assert os.path.exists(feat_path), f"Featured dataset file {feat_path} should exist"


def test_target_creation_and_validation():
    """
    Test Phase 5 target creation: binary labels, final row deletion, zero leakage, and integrity validation.
    """
    raw_df = download_stock_data(ticker=TEST_TICKER, start_date="2022-01-01")
    cleaned_df = preprocess_data(raw_df)
    featured_df = create_features(cleaned_df)
    model_df = create_target(featured_df)

    # 1. Target column existence and binary values
    assert 'Target' in model_df.columns, "Target column must exist"
    unique_targets = set(model_df['Target'].unique())
    assert unique_targets.issubset({0, 1}), f"Target must contain only 0 and 1, got {unique_targets}"

    # 2. Final row removed (length should be featured_df length - 1)
    assert len(model_df) == len(featured_df) - 1, "Final row must be removed due to shift(-1)"

    # 3. No NaNs remain
    assert model_df.isnull().sum().sum() == 0, "No NaN values should exist in final model dataset"

    # 4. Chronological order and no duplicate dates
    assert model_df['Date'].is_monotonic_increasing, "Dates must be strictly chronological"
    assert model_df['Date'].duplicated().sum() == 0, "No duplicate dates allowed"

    # 5. Validation helper test
    stats = validate_model_dataset(model_df)
    assert stats['total_records'] == len(model_df)
    assert stats['up_samples'] + stats['down_samples'] == len(model_df)

    # 6. Verify file saving
    model_path = save_model_data(model_df, ticker=TEST_TICKER)
    assert os.path.exists(model_path), f"Model dataset file {model_path} should exist"


def test_validation_rejects_invalid_data():
    """
    Test that real data validation function rejects empty, malformed, or invalid datasets.
    """
    empty_df = pd.DataFrame()
    assert validate_real_ohlcv_data(empty_df) is False, "Validation must reject empty DataFrame"

    invalid_cols_df = pd.DataFrame({'Date': ['2023-01-01'], 'Price': [100.0]})
    assert validate_real_ohlcv_data(invalid_cols_df) is False, "Validation must reject missing OHLCV columns"


def test_no_synthetic_fallback_on_invalid_ticker():
    """
    Test that requesting an unresolvable ticker raises ValueError prohibiting synthetic fallback.
    """
    with pytest.raises(ValueError) as excinfo:
        download_stock_data(ticker="INVALID_UNRESOLVABLE_TICKER_XYZ_999", force_download=True)

    assert "CRITICAL ERROR: Real historical market data" in str(excinfo.value)


def test_data_source_tagging():
    """
    Test that download_stock_data returns explicit data_source string metadata.
    """
    df, data_source = download_stock_data(ticker=TEST_TICKER, return_source=True)
    assert data_source in ["Live Yahoo Finance", "Live Market Data", "Historical Archive"], f"Invalid data source: {data_source}"
    assert df.attrs.get('data_source') == data_source, "DataFrame attrs data_source must match return value"

