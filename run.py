"""
Main CLI Execution Runner for Quantum Stock Predictor (Phases 1–11).
Executes classical ML baseline models, naive baselines, and PennyLane Quantum Neural Network (VQC).
"""

import sys
import os



import time
import argparse
import logging
import pandas as pd

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_collection import download_stock_data, SUPPORTED_TICKERS
from src.preprocessing import preprocess_data, save_processed_data
from src.feature_engineering import create_features, save_featured_data
from src.target_creation import create_target, validate_model_dataset, save_model_data
from src.classical_models import (
    split_time_series_data,
    scale_features,
    train_logistic_regression,
    train_random_forest,
    train_xgboost
)
from src.quantum_model import (
    extract_and_scale_quantum_features,
    train_quantum_vqc,
    save_quantum_artifacts,
    QUANTUM_FEATURE_NAMES
)
from src.evaluation import (
    evaluate_baselines,
    evaluate_model,
    evaluate_quantum_model,
    compare_models,
    get_feature_importance,
    save_evaluation_plots
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("run_pipeline")


def run_pipeline(
    ticker: str = "RELIANCE.NS",
    start_date: str = "2018-01-01",
    force_download: bool = False,
    run_quantum: bool = False
):
    """
    Executes Phases 1–11 experiment analysis pipeline for the given ticker.
    """
    logger.info(f"=== Starting Data & ML Pipeline for Ticker: {ticker} (Quantum: {run_quantum}) ===")

    # 1. Phase 2: Data Collection
    raw_df = download_stock_data(
        ticker=ticker,
        start_date=start_date,
        force_download=force_download
    )

    # 2. Phase 3: Data Preprocessing
    cleaned_df = preprocess_data(raw_df)
    save_processed_data(cleaned_df, ticker=ticker)

    # 3. Phase 4: Feature Engineering
    featured_df = create_features(cleaned_df)
    save_featured_data(featured_df, ticker=ticker)

    # 4. Phase 5: Target Creation & Validation
    model_df = create_target(featured_df)
    save_model_data(model_df, ticker=ticker)
    stats = validate_model_dataset(model_df)

    # 5. Phase 6: Time-Series Chronological Train/Test Split
    train_df, test_df, X_train, y_train, X_test, y_test, feature_cols = split_time_series_data(
        model_df, train_ratio=0.80
    )

    # Calculate Class Distributions
    tr_up = int((y_train == 1).sum())
    tr_down = int((y_train == 0).sum())
    te_up = int((y_test == 1).sum())
    te_down = int((y_test == 0).sum())

    # 6. Evaluate Naive Baselines
    majority_res, random_res = evaluate_baselines(y_test)

    results_dict = {
        "Majority Class": majority_res,
        "Random Baseline": random_res
    }

    # 7. Feature Scaling (Scaler fitted ONLY on training set)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # 8. Phase 7: Logistic Regression
    lr_model = train_logistic_regression(X_train_scaled, y_train)
    lr_results = evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression")
    results_dict["Logistic Regression"] = lr_results

    # 9. Phase 8: Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    rf_imp = get_feature_importance(rf_model, feature_cols)
    results_dict["Random Forest"] = rf_results

    # 10. Phase 9: XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    xgb_results = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    xgb_imp = get_feature_importance(xgb_model, feature_cols)
    results_dict["XGBoost"] = xgb_results

    # 11. Phase 10: Quantum Neural Network (VQC)
    quantum_stats = {}
    if run_quantum:
        logger.info("=== Starting Phase 10: PennyLane Quantum Machine Learning (VQC) ===")

        # Extract & scale 4 quantum features (Scaler fitted ONLY on training set)
        X_train_q_scaled, X_test_q_scaled, scaler_q = extract_and_scale_quantum_features(X_train, X_test)

        # Train 4-qubit Quantum VQC efficiently
        start_q_time = time.time()
        quantum_params, loss_history = train_quantum_vqc(
            X_train_q_scaled,
            y_train,
            n_layers=2,
            epochs=10,
            batch_size=128,
            max_train_samples=256,
            learning_rate=0.05
        )
        q_time_elapsed = time.time() - start_q_time

        save_quantum_artifacts(quantum_params, loss_history, results_dir="models/quantum/results")

        q_results = evaluate_quantum_model(quantum_params, X_test_q_scaled, y_test, "Quantum VQC")
        results_dict["Quantum VQC"] = q_results

        quantum_stats = {
            'initial_loss': loss_history[0],
            'final_loss': loss_history[-1],
            'training_time_sec': round(q_time_elapsed, 2)
        }

    # Save complete evaluation plots across all models
    save_evaluation_plots(results_dict, rf_imp=rf_imp, xgb_imp=xgb_imp, results_dir="models/classical/results")
    if run_quantum and "Quantum VQC" in results_dict:
        save_evaluation_plots(results_dict, rf_imp=rf_imp, xgb_imp=xgb_imp, results_dir="models/quantum/results")

    df_comparison = compare_models(results_dict)

    train_start = train_df['Date'].min().strftime('%Y-%m-%d')
    train_end = train_df['Date'].max().strftime('%Y-%m-%d')
    test_start = test_df['Date'].min().strftime('%Y-%m-%d')
    test_end = test_df['Date'].max().strftime('%Y-%m-%d')

    print("\n" + "="*78)
    print(f"EXPERIMENT VALIDATION & COMPARISON SUMMARY FOR TICKER: {ticker}")
    print("="*78)
    print(f"Training Period ({len(X_train)} samples): {train_start} -> {train_end}")
    print(f"  Training Class Distribution : UP (1): {tr_up} ({tr_up/len(y_train)*100:.2f}%) | DOWN (0): {tr_down} ({tr_down/len(y_train)*100:.2f}%)")
    print(f"Testing Period  ({len(X_test)} samples): {test_start} -> {test_end}")
    print(f"  Testing Class Distribution  : UP (1): {te_up} ({te_up/len(y_test)*100:.2f}%) | DOWN (0): {te_down} ({te_down/len(y_test)*100:.2f}%)")
    print("-" * 78)
    print("MODEL PERFORMANCE & PREDICTION DISTRIBUTION COMPARISON TABLE:")
    print(df_comparison.to_string(index=False))
    print("-" * 78)
    if run_quantum and quantum_stats:
        print("QUANTUM ABLATION & SANITY CHECK SUMMARY:")
        print(f"  Framework        : PennyLane (default.qubit)")
        print(f"  Qubits Count     : 4")
        print(f"  Initial BCE Loss : {quantum_stats['initial_loss']:.5f}")
        print(f"  Final BCE Loss   : {quantum_stats['final_loss']:.5f}")
        print(f"  Training Duration: {quantum_stats['training_time_sec']} seconds")
        print("-" * 78)
    print("TOP 10 FEATURE IMPORTANCE (RANDOM FOREST):")
    print(rf_imp.head(10).to_string(index=False))
    print("="*78 + "\n")

    return results_dict, df_comparison, quantum_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hybrid Quantum-Classical Stock Pipeline (Phases 1-11)")
    parser.add_argument(
        "--ticker",
        type=str,
        default="RELIANCE.NS",
        choices=SUPPORTED_TICKERS,
        help=f"Stock ticker symbol (Default: RELIANCE.NS, Supported: {', '.join(SUPPORTED_TICKERS)})"
    )
    parser.add_argument("--start-date", type=str, default="2018-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--force-download", action="store_true", help="Force re-download raw data from Yahoo Finance")
    parser.add_argument("--quantum", action="store_true", help="Run PennyLane 4-qubit Quantum Neural Network training & evaluation")
    args = parser.parse_args()

    run_pipeline(
        ticker=args.ticker,
        start_date=args.start_date,
        force_download=args.force_download,
        run_quantum=args.quantum
    )
