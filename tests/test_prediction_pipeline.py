"""
Unit tests for Prediction Pipeline Module (Phase 12).
Verifies model artifact loading, latest feature extraction, trend inference outputs,
UP/DOWN string mappings, and error handling.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.prediction import (
    load_classical_model,
    load_quantum_model_and_scaler,
    get_latest_stock_features,
    predict_trend,
    predict_all_models
)

TEST_TICKER = "RELIANCE.NS"


def test_classical_model_loading():
    """
    Test loading classical model artifacts and scaler.
    """
    model, scaler = load_classical_model("Random Forest")
    assert model is not None, "Loaded Random Forest model must not be None"
    assert scaler is not None, "Loaded scaler must not be None"


def test_quantum_model_loading():
    """
    Test loading quantum parameters and scaler.
    """
    q_params, q_scaler = load_quantum_model_and_scaler()
    assert 'weights' in q_params, "Quantum parameters dictionary must contain 'weights'"
    assert q_scaler is not None, "Quantum scaler must not be None"


def test_latest_features_extraction():
    """
    Test fetching latest feature row and overview statistics.
    """
    featured_df, overview = get_latest_stock_features(TEST_TICKER)
    assert isinstance(featured_df, pd.DataFrame), "Featured dataset must be a pandas DataFrame"
    assert not featured_df.empty, "Featured dataset should not be empty"

    expected_keys = ['ticker', 'latest_date', 'latest_close', 'daily_change', 'daily_pct', 'rsi', 'macd']
    for k in expected_keys:
        assert k in overview, f"Overview dictionary must contain key '{k}'"


def test_predict_trend_classical_and_quantum():
    """
    Test trend prediction outputs for both classical and quantum models.
    """
    featured_df, _ = get_latest_stock_features(TEST_TICKER)

    # 1. Classical Model Prediction
    rf_res = predict_trend(TEST_TICKER, "Random Forest", featured_df=featured_df)
    assert rf_res['prediction'] in [0, 1]
    assert rf_res['prediction_label'] in ["UP 📈", "DOWN 📉"]
    assert rf_res['latest_close_price'] > 0

    # 2. Quantum Model Prediction
    q_res = predict_trend(TEST_TICKER, "Quantum VQC", featured_df=featured_df)
    assert q_res['prediction'] in [0, 1]
    assert q_res['prediction_label'] in ["UP 📈", "DOWN 📉"]
    assert 0.0 <= q_res['predicted_probability'] <= 1.0


def test_predict_all_models():
    """
    Test multi-model prediction output list.
    """
    all_preds = predict_all_models(TEST_TICKER)
    assert len(all_preds) == 4, "Must return predictions for all 4 models"
    model_names = [p['model_name'] for p in all_preds]
    assert "Logistic Regression" in model_names
    assert "Random Forest" in model_names
    assert "XGBoost" in model_names
    assert "Quantum VQC" in model_names


def test_invalid_model_name_handling():
    """
    Test exception handling when requesting an un-trained model name.
    """
    with pytest.raises(FileNotFoundError):
        load_classical_model("NON_EXISTENT_MODEL_123")


def test_prediction_data_source_and_terminology():
    """
    Test that prediction output includes data_source, is_live, and uses predicted_probability terminology.
    """
    res = predict_trend(TEST_TICKER, "Quantum VQC")
    assert 'data_source' in res, "Prediction output must include 'data_source'"
    assert 'is_live' in res, "Prediction output must include 'is_live'"
    assert res['data_source'] in ["Live Yahoo Finance", "Live Market Data", "Historical Archive"]
    assert 'predicted_probability' in res, "Must use 'predicted_probability' terminology"

    if res['data_source'] == "Historical Archive":
        assert res['is_live'] is False, "Archive data must NOT be marked as live or real-time"
    else:
        assert res['is_live'] is True, "Live data must be marked as live"

