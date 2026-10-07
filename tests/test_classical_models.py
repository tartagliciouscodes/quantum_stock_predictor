"""
Unit tests for Classical ML Pipeline (Phases 6–9).
Verifies chronological splitting, zero data leakage, scaler logic, model training,
prediction constraints, metric evaluation, and saved model artifact persistence.
"""

import os
import pytest
import numpy as np
import pandas as pd
import joblib

from src.data_collection import download_stock_data
from src.preprocessing import preprocess_data
from src.feature_engineering import create_features
from src.target_creation import create_target
from src.classical_models import (
    split_time_series_data,
    scale_features,
    train_logistic_regression,
    train_random_forest,
    train_xgboost
)
from src.evaluation import evaluate_model, get_feature_importance

TEST_TICKER = "RELIANCE.NS"


@pytest.fixture(scope="module")
def model_data():
    """
    Fixture providing processed model dataset for testing.
    """
    raw_df = download_stock_data(ticker=TEST_TICKER, start_date="2022-01-01")
    cleaned_df = preprocess_data(raw_df)
    featured_df = create_features(cleaned_df)
    model_df = create_target(featured_df)
    return model_df


def test_chronological_split(model_data):
    """
    1. Test chronological train/test split.
    2. Verify training data comes before testing data.
    3. Verify Target & Date are not present in X.
    """
    train_df, test_df, X_train, y_train, X_test, y_test, feature_cols = split_time_series_data(
        model_data, train_ratio=0.80
    )

    # 1. Row counts add up
    assert len(train_df) + len(test_df) == len(model_data)
    assert len(X_train) == len(train_df)
    assert len(X_test) == len(test_df)

    # 2. Chronological guarantee (Train max date < Test min date)
    assert train_df['Date'].max() < test_df['Date'].min()

    # 3. No Target or Date in X
    assert 'Target' not in X_train.columns
    assert 'Date' not in X_train.columns
    assert 'Target' not in X_test.columns
    assert 'Date' not in X_test.columns


def test_scaler_fitting_logic(model_data):
    """
    4. Verify scaler is fitted ONLY on training data.
    """
    _, _, X_train, _, X_test, _, _ = split_time_series_data(model_data, train_ratio=0.80)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, scaler_dir="models/classical")

    # Scaler mean must match X_train mean, not X_test mean
    np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0), rtol=1e-5)
    assert os.path.exists("models/classical/scaler.pkl")


def test_model_training_and_predictions(model_data):
    """
    5. Test classical models train successfully.
    6. Verify predictions contain only 0 and 1.
    7. Verify evaluation metrics are generated.
    8. Verify saved model files exist.
    """
    _, _, X_train, y_train, X_test, y_test, feature_cols = split_time_series_data(model_data, train_ratio=0.80)
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    # 1. Train Logistic Regression
    lr = train_logistic_regression(X_train_scaled, y_train)
    lr_res = evaluate_model(lr, X_test_scaled, y_test, "Logistic Regression")
    assert os.path.exists("models/classical/logistic_regression.pkl")
    assert set(np.unique(lr_res['y_pred'])).issubset({0, 1})
    assert 0.0 <= lr_res['accuracy'] <= 1.0

    # 2. Train Random Forest
    rf = train_random_forest(X_train, y_train)
    rf_res = evaluate_model(rf, X_test, y_test, "Random Forest")
    rf_imp = get_feature_importance(rf, feature_cols)
    assert os.path.exists("models/classical/random_forest.pkl")
    assert set(np.unique(rf_res['y_pred'])).issubset({0, 1})
    assert len(rf_imp) == len(feature_cols)

    # 3. Train XGBoost
    xgb = train_xgboost(X_train, y_train)
    xgb_res = evaluate_model(xgb, X_test, y_test, "XGBoost")
    xgb_imp = get_feature_importance(xgb, feature_cols)
    assert os.path.exists("models/classical/xgboost.pkl")
    assert set(np.unique(xgb_res['y_pred'])).issubset({0, 1})
    assert len(xgb_imp) == len(feature_cols)
