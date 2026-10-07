"""
Inference & Real-Time Prediction Pipeline Module (Phase 12).
Loads saved model artifacts and scalers, fetches real stock market data,
extracts active feature vectors, and computes stock trend predictions (UP / DOWN)
with predicted probabilities across Classical and Quantum ML models.
"""

import os
import logging
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
import joblib

from src.data_collection import download_stock_data
from src.preprocessing import preprocess_data
from src.feature_engineering import create_features
from src.classical_models import split_time_series_data
from src.quantum_model import load_quantum_model, predict_quantum, QUANTUM_FEATURE_NAMES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("prediction_pipeline")


def load_classical_model(
    model_name: str,
    models_dir: str = "models/classical"
) -> Tuple[Any, Optional[Any]]:
    """
    Loads trained classical model and scaler from disk.

    Parameters:
    -----------
    model_name : str
        One of 'Logistic Regression', 'Random Forest', 'XGBoost'.
    models_dir : str
        Directory where classical models are stored.

    Returns:
    --------
    Tuple[Any, Optional[Any]]
        (model_instance, scaler_instance)
    """
    clean_name = model_name.lower().replace(" ", "_")
    model_file = os.path.join(models_dir, f"{clean_name}.pkl")
    scaler_file = os.path.join(models_dir, "scaler.pkl")

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model artifact not found at {model_file}. Please run training first.")

    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file) if os.path.exists(scaler_file) else None

    logger.info(f"Successfully loaded classical model '{model_name}' from {model_file}")
    return model, scaler


def load_quantum_model_and_scaler(
    quantum_dir: str = "models/quantum"
) -> Tuple[Dict[str, Any], Any]:
    """
    Loads trained Quantum VQC parameter dictionary and quantum scaler from disk.
    """
    model_file = os.path.join(quantum_dir, "quantum_vqc.pkl")
    scaler_file = os.path.join(quantum_dir, "quantum_scaler.pkl")

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Quantum model parameters not found at {model_file}. Please run quantum training first.")
    if not os.path.exists(scaler_file):
        raise FileNotFoundError(f"Quantum feature scaler not found at {scaler_file}.")

    q_params = joblib.load(model_file)
    q_scaler = joblib.load(scaler_file)

    logger.info(f"Successfully loaded Quantum VQC model parameters from {model_file}")
    return q_params, q_scaler


def get_latest_stock_features(ticker: str = "RELIANCE.NS") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Downloads/loads real stock dataset, runs preprocessing & feature engineering,
    and extracts the latest active feature vector along with data source metadata.

    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, Any]]
        (featured_df, overview_stats_dict)
    """
    logger.info(f"Fetching latest market features for ticker '{ticker}'...")
    raw_df, data_source = download_stock_data(ticker=ticker, return_source=True)
    cleaned_df = preprocess_data(raw_df)
    featured_df = create_features(cleaned_df)
    featured_df.attrs['data_source'] = data_source

    if featured_df.empty:
        raise ValueError(f"Insufficient historical data available for '{ticker}' to calculate technical indicators.")

    latest_row = featured_df.iloc[-1]
    prev_row = featured_df.iloc[-2] if len(featured_df) > 1 else latest_row

    close_price = float(latest_row['Close'])
    prev_close = float(prev_row['Close'])
    daily_change = round(close_price - prev_close, 2)
    daily_pct = round((daily_change / prev_close) * 100.0, 2)

    trading_date = pd.to_datetime(latest_row['Date']).strftime('%Y-%m-%d')

    overview_stats = {
        'ticker': ticker,
        'latest_date': trading_date,
        'latest_close': close_price,
        'daily_change': daily_change,
        'daily_pct': daily_pct,
        'open': float(latest_row['Open']),
        'high': float(latest_row['High']),
        'low': float(latest_row['Low']),
        'volume': int(latest_row['Volume']),
        'rsi': round(float(latest_row['RSI']), 2),
        'macd': round(float(latest_row['MACD']), 2),
        'sma_20': round(float(latest_row['SMA_20']), 2),
        'sma_50': round(float(latest_row['SMA_50']), 2),
        'volatility': round(float(latest_row['Volatility']), 4),
        'atr': round(float(latest_row['ATR']), 2),
        'data_source': data_source,
        'is_live': bool(data_source and "Live" in data_source)
    }

    return featured_df, overview_stats


def predict_trend(
    ticker: str = "RELIANCE.NS",
    model_name: str = "Quantum VQC",
    featured_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Computes stock trend prediction (UP / DOWN) and predicted probability for the selected model.

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol.
    model_name : str
        One of 'Logistic Regression', 'Random Forest', 'XGBoost', 'Quantum VQC'.
    featured_df : Optional[pd.DataFrame]
        Featured stock DataFrame. If None, automatically fetches latest features.

    Returns:
    --------
    Dict[str, Any]
        Prediction result dictionary containing data_source, prediction, predicted_probability, etc.
    """
    if featured_df is None:
        featured_df, overview = get_latest_stock_features(ticker)
    else:
        latest_row = featured_df.iloc[-1]
        prev_row = featured_df.iloc[-2] if len(featured_df) > 1 else latest_row
        close_price = float(latest_row['Close'])
        prev_close = float(prev_row['Close'])
        data_src = featured_df.attrs.get('data_source', 'Historical Archive')
        overview = {
            'ticker': ticker,
            'latest_date': pd.to_datetime(latest_row['Date']).strftime('%Y-%m-%d'),
            'latest_close': close_price,
            'daily_change': round(close_price - prev_close, 2),
            'daily_pct': round(((close_price - prev_close) / prev_close) * 100.0, 2),
            'data_source': data_src,
            'is_live': bool(data_src and "Live" in data_src)
        }

    latest_row_df = featured_df.iloc[[-1]].copy()

    if model_name == "Quantum VQC":
        q_params, q_scaler = load_quantum_model_and_scaler()
        X_q_raw = latest_row_df[QUANTUM_FEATURE_NAMES]
        X_q_scaled = q_scaler.transform(X_q_raw)

        pred_arr, prob_arr = predict_quantum(q_params, X_q_scaled)
        pred_val = int(pred_arr[0])
        pred_prob = float(prob_arr[0])

    else:
        model, scaler = load_classical_model(model_name)
        feature_cols = [c for c in featured_df.columns if c not in ['Date', 'Target']]
        X_raw = latest_row_df[feature_cols]

        if model_name == "Logistic Regression" and scaler is not None:
            X_input = scaler.transform(X_raw)
        else:
            X_input = X_raw

        pred_val = int(model.predict(X_input)[0])
        if hasattr(model, "predict_proba"):
            pred_prob = float(model.predict_proba(X_input)[0, 1])
        else:
            pred_prob = None

    label = "UP 📈" if pred_val == 1 else "DOWN 📉"

    return {
        'ticker': ticker,
        'model_name': model_name,
        'prediction': pred_val,
        'prediction_label': label,
        'predicted_probability': pred_prob,
        'latest_trading_date': overview['latest_date'],
        'latest_close_price': overview['latest_close'],
        'daily_change': overview['daily_change'],
        'daily_pct': overview['daily_pct'],
        'data_source': overview['data_source'],
        'is_live': overview['is_live']
    }


def predict_all_models(ticker: str = "RELIANCE.NS", featured_df: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
    """
    Computes trend predictions across all 4 models (Logistic Regression, Random Forest, XGBoost, Quantum VQC).
    """
    if featured_df is None:
        featured_df, _ = get_latest_stock_features(ticker)

    models = ["Logistic Regression", "Random Forest", "XGBoost", "Quantum VQC"]
    results = []
    for m in models:
        try:
            res = predict_trend(ticker=ticker, model_name=m, featured_df=featured_df)
            results.append(res)
        except Exception as e:
            logger.warning(f"Prediction failed for model '{m}': {e}")
            results.append({
                'ticker': ticker,
                'model_name': m,
                'prediction': 0,
                'prediction_label': "N/A (Error)",
                'predicted_probability': None,
                'latest_trading_date': 'N/A',
                'latest_close_price': 0.0,
                'error': str(e)
            })

    return results
