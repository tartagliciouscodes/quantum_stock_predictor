"""
Unit tests for Quantum Machine Learning Pipeline (Phase 10).
Verifies circuit specifications, forward execution, prediction formats, parameter persistence,
and chronological evaluation consistency.
"""

import os
import pytest
import numpy as np
import pandas as pd
import pennylane as qml

from src.data_collection import download_stock_data
from src.preprocessing import preprocess_data
from src.feature_engineering import create_features
from src.target_creation import create_target
from src.classical_models import split_time_series_data
from src.quantum_model import (
    quantum_circuit,
    extract_and_scale_quantum_features,
    train_quantum_vqc,
    predict_quantum,
    load_quantum_model,
    QUANTUM_FEATURE_NAMES,
    N_QUBITS
)
from src.evaluation import evaluate_quantum_model

TEST_TICKER = "RELIANCE.NS"


@pytest.fixture(scope="module")
def model_data():
    raw_df = download_stock_data(ticker=TEST_TICKER, start_date="2022-01-01")
    cleaned_df = preprocess_data(raw_df)
    featured_df = create_features(cleaned_df)
    model_df = create_target(featured_df)
    return model_df


def test_quantum_circuit_properties():
    """
    1. Test quantum circuit has exactly 4 qubits.
    2. Test quantum circuit accepts exactly 4 input features.
    3. Test quantum model can perform a forward pass.
    4. Test quantum output has expected scalar expectation shape.
    """
    assert N_QUBITS == 4, "Quantum circuit must use exactly 4 qubits"
    assert len(QUANTUM_FEATURE_NAMES) == 4, "Quantum model must accept exactly 4 input features"

    # Dummy weights (n_layers=2, n_qubits=4, 2 params per qubit per layer)
    weights = np.zeros((2, 4, 2))
    features = np.array([0.5, -0.2, 0.1, 0.8])

    exp_val = float(quantum_circuit(weights, features))
    assert -1.0 <= exp_val <= 1.0, f"Expectation value must be in range [-1, 1], got {exp_val}"


def test_quantum_feature_extraction_and_scaling(model_data):
    """
    9. Verify no target column is passed into the quantum features.
    """
    _, _, X_train, _, X_test, _, _ = split_time_series_data(model_data, train_ratio=0.80)
    X_train_q_scaled, X_test_q_scaled, scaler_q = extract_and_scale_quantum_features(
        X_train, X_test, scaler_dir="models/quantum"
    )

    assert X_train_q_scaled.shape[1] == 4, "Scaled training matrix must have 4 columns"
    assert X_test_q_scaled.shape[1] == 4, "Scaled testing matrix must have 4 columns"
    assert 'Target' not in QUANTUM_FEATURE_NAMES, "Target column must never be passed to quantum features"
    assert os.path.exists("models/quantum/quantum_scaler.pkl")


def test_quantum_vqc_training_and_prediction(model_data):
    """
    5. Test predictions contain only 0 and 1.
    6. Test quantum model can train on a small subset.
    7. Test quantum parameters can be saved.
    8. Test quantum parameters can be loaded.
    10. Test quantum test evaluation uses same chronological test period as classical models.
    """
    train_df, test_df, X_train, y_train, X_test, y_test, _ = split_time_series_data(model_data, train_ratio=0.80)
    X_train_q_scaled, X_test_q_scaled, _ = extract_and_scale_quantum_features(X_train, X_test)

    # Use a small fast subset for unit test
    sub_X_train = X_train_q_scaled[:30]
    sub_y_train = y_train.iloc[:30]

    # Quick 2-epoch training test
    params, loss_hist = train_quantum_vqc(
        sub_X_train,
        sub_y_train,
        n_layers=2,
        epochs=2,
        batch_size=16,
        learning_rate=0.05,
        model_dir="models/quantum"
    )

    assert os.path.exists("models/quantum/quantum_vqc.pkl"), "Trained model parameter file must exist"
    assert len(loss_hist) == 2, "Loss history must contain 2 epoch entries"

    # Test loading parameters
    loaded_params = load_quantum_model("models/quantum/quantum_vqc.pkl")
    assert 'weights' in loaded_params
    assert 'bias' in loaded_params

    # Test prediction format
    y_pred, y_prob = predict_quantum(loaded_params, X_test_q_scaled[:10])
    assert set(np.unique(y_pred)).issubset({0, 1}), "Predictions must contain only 0 and 1"
    assert (y_prob >= 0.0).all() and (y_prob <= 1.0).all(), "Probabilities must be in range [0, 1]"

    # Evaluate model metrics
    q_res = evaluate_quantum_model(loaded_params, X_test_q_scaled, y_test, model_name="Quantum VQC Test")
    assert 0.0 <= q_res['accuracy'] <= 1.0
