"""
Classical Machine Learning Models Module (Phases 6–9).
Implements:
- Chronological time-series train/test splitting (80% train, 20% test).
- Feature scaling using StandardScaler (fitted EXCLUSIVELY on training data).
- Model training routines for:
  1. Logistic Regression
  2. Random Forest
  3. XGBoost
- Model serialization (saving trained artifacts to models/classical/).
"""

import os
import logging
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("classical_models")


def split_time_series_data(
    df: pd.DataFrame,
    train_ratio: float = 0.80
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
    """
    Performs a strict chronological time-series split (no shuffling, zero leakage).

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned model DataFrame containing 'Date', feature columns, and 'Target'.
    train_ratio : float
        Proportion of dataset to use for training (default: 0.80).

    Returns:
    --------
    Tuple: (train_df, test_df, X_train, y_train, X_test, y_test, feature_cols)
    """
    logger.info("Performing chronological train/test split...")
    data = df.copy()

    # Sort chronologically by Date
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(by='Date', ascending=True, inplace=True)
    data.reset_index(drop=True, inplace=True)

    n_samples = len(data)
    train_size = int(n_samples * train_ratio)

    train_df = data.iloc[:train_size].copy()
    test_df = data.iloc[train_size:].copy()

    # Identify feature columns (drop 'Date' and 'Target')
    feature_cols = [col for col in data.columns if col not in ['Date', 'Target']]

    X_train = train_df[feature_cols].copy()
    y_train = train_df['Target'].copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df['Target'].copy()

    train_start = train_df['Date'].min().strftime('%Y-%m-%d')
    train_end = train_df['Date'].max().strftime('%Y-%m-%d')
    test_start = test_df['Date'].min().strftime('%Y-%m-%d')
    test_end = test_df['Date'].max().strftime('%Y-%m-%d')

    logger.info(f"Chronological split completed successfully:")
    logger.info(f"  Training samples: {len(X_train)} ({train_start} -> {train_end})")
    logger.info(f"  Testing samples : {len(X_test)} ({test_start} -> {test_end})")

    # Sanity check: Ensure training date is strictly prior to testing date
    if train_df['Date'].max() >= test_df['Date'].min():
        raise ValueError("Data Leakage Error: Training dates overlap with testing dates.")

    return train_df, test_df, X_train, y_train, X_test, y_test, feature_cols


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_dir: str = "models/classical"
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fits StandardScaler ONLY on training data and applies transform to train & test sets.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training feature matrix.
    X_test : pd.DataFrame
        Testing feature matrix.
    scaler_dir : str
        Directory to save fitted scaler artifact.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, StandardScaler]
        Scaled X_train, scaled X_test, and fitted StandardScaler object.
    """
    logger.info("Scaling features using StandardScaler (Fitted ONLY on training set)...")
    os.makedirs(scaler_dir, exist_ok=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scaler_file = os.path.join(scaler_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_file)
    logger.info(f"Saved fitted StandardScaler to {scaler_file}")

    return X_train_scaled, X_test_scaled, scaler


def train_logistic_regression(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    model_dir: str = "models/classical"
) -> LogisticRegression:
    """
    Trains Logistic Regression baseline model.
    """
    logger.info("Training Logistic Regression baseline model...")
    os.makedirs(model_dir, exist_ok=True)

    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(X_train_scaled, y_train)

    model_path = os.path.join(model_dir, "logistic_regression.pkl")
    joblib.dump(clf, model_path)
    logger.info(f"Saved Logistic Regression model to {model_path}")

    return clf


def train_random_forest(
    X_train: np.ndarray,
    y_train: pd.Series,
    model_dir: str = "models/classical",
    n_estimators: int = 100,
    max_depth: int = 6
) -> RandomForestClassifier:
    """
    Trains Random Forest baseline model.
    """
    logger.info(f"Training Random Forest model (n_estimators={n_estimators}, max_depth={max_depth})...")
    os.makedirs(model_dir, exist_ok=True)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=5,
        random_state=42
    )
    clf.fit(X_train, y_train)

    model_path = os.path.join(model_dir, "random_forest.pkl")
    joblib.dump(clf, model_path)
    logger.info(f"Saved Random Forest model to {model_path}")

    return clf


def train_xgboost(
    X_train: np.ndarray,
    y_train: pd.Series,
    model_dir: str = "models/classical",
    n_estimators: int = 100,
    max_depth: int = 4,
    learning_rate: float = 0.05
) -> XGBClassifier:
    """
    Trains XGBoost baseline model.
    """
    logger.info(f"Training XGBoost model (n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate})...")
    os.makedirs(model_dir, exist_ok=True)

    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    clf.fit(X_train, y_train)

    model_path = os.path.join(model_dir, "xgboost.pkl")
    joblib.dump(clf, model_path)
    logger.info(f"Saved XGBoost model to {model_path}")

    return clf
