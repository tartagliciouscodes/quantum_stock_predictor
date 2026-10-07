"""
Quantum Machine Learning Module for Quantum Stock Predictor (Phase 10).
Implements a 4-qubit Variational Quantum Classifier (VQC) using PennyLane:
- Feature Encoding: Angle Encoding (RY rotations) of 4 quantum features:
  [RSI, MACD, Daily_Return, Volatility]
- Variational Circuit: Parameterized RY & RZ single-qubit rotations with Ring CNOT Entanglement.
- Measurement: Expectation value of PauliZ operator on Qubit 0.
- Optimization: Adam optimizer on Binary Cross-Entropy (BCE) loss.
"""

import os
import logging
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import pennylane as qml
from pennylane import numpy as pnp
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("quantum_model")

# Selected 4 Quantum Features as specified in Project Design
QUANTUM_FEATURE_NAMES = ['RSI', 'MACD', 'Daily_Return', 'Volatility']

# 4 Qubits Simulator Device
N_QUBITS = 4
dev = qml.device('default.qubit', wires=N_QUBITS)


@qml.qnode(dev, interface='autograd')
def quantum_circuit(weights: pnp.ndarray, features: pnp.ndarray) -> float:
    """
    4-qubit Variational Quantum Circuit QNode.

    Parameters:
    -----------
    weights : pnp.ndarray
        Shape (n_layers, n_qubits, 2) parameterized rotation angles.
    features : pnp.ndarray
        Shape (4,) scaled classical feature values.

    Returns:
    --------
    float
        Expectation value <Z0> in range [-1.0, 1.0].
    """
    # 1. Feature Encoding (Angle Encoding into 4 Qubits)
    for i in range(N_QUBITS):
        qml.RY(features[i], wires=i)

    # 2. Variational Layers with Entanglement
    n_layers = weights.shape[0]
    for l in range(n_layers):
        for i in range(N_QUBITS):
            qml.RY(weights[l, i, 0], wires=i)
            qml.RZ(weights[l, i, 1], wires=i)

        # Entanglement Layer (Ring CNOT topology)
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])

    # 3. Measurement: PauliZ expectation on Qubit 0
    return qml.expval(qml.PauliZ(0))


def extract_and_scale_quantum_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_dir: str = "models/quantum"
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Extracts the 4 selected quantum features [RSI, MACD, Daily_Return, Volatility]
    and applies StandardScaler fitted EXCLUSIVELY on training data.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, StandardScaler]
        Scaled X_train_q, scaled X_test_q, and fitted quantum StandardScaler.
    """
    logger.info(f"Extracting 4 quantum features: {QUANTUM_FEATURE_NAMES}")
    os.makedirs(scaler_dir, exist_ok=True)

    missing_cols = [c for c in QUANTUM_FEATURE_NAMES if c not in X_train.columns]
    if missing_cols:
        raise KeyError(f"Missing required quantum feature columns: {missing_cols}")

    X_train_q = X_train[QUANTUM_FEATURE_NAMES].copy()
    X_test_q = X_test[QUANTUM_FEATURE_NAMES].copy()

    scaler_q = StandardScaler()
    X_train_q_scaled = scaler_q.fit_transform(X_train_q)
    X_test_q_scaled = scaler_q.transform(X_test_q)

    scaler_path = os.path.join(scaler_dir, "quantum_scaler.pkl")
    joblib.dump(scaler_q, scaler_path)
    logger.info(f"Saved quantum feature scaler to {scaler_path}")

    return X_train_q_scaled, X_test_q_scaled, scaler_q


def forward_predict_proba(weights: pnp.ndarray, bias: float, x_scaled: np.ndarray) -> float:
    """
    Computes class prediction probability P(Target=1) for a single scaled sample vector.
    Expectation value <Z0> in [-1, 1] is mapped via sigmoid: sigmoid(bias + 2.0 * <Z0>).
    """
    exp_val = float(quantum_circuit(weights, x_scaled))
    prob = 1.0 / (1.0 + np.exp(-(bias + 2.0 * exp_val)))
    return float(np.clip(prob, 1e-7, 1.0 - 1e-7))


def bce_loss(weights: pnp.ndarray, bias: float, X_batch: np.ndarray, y_batch: np.ndarray) -> float:
    """
    Binary Cross-Entropy Loss function over a data batch.
    """
    loss = 0.0
    n = len(X_batch)
    for i in range(n):
        exp_val = quantum_circuit(weights, X_batch[i])
        prob = 1.0 / (1.0 + pnp.exp(-(bias + 2.0 * exp_val)))
        prob = pnp.clip(prob, 1e-7, 1.0 - 1e-7)
        y = y_batch[i]
        loss += - (y * pnp.log(prob) + (1.0 - y) * pnp.log(1.0 - prob))
    return loss / n


def train_quantum_vqc(
    X_train_q_scaled: np.ndarray,
    y_train: pd.Series,
    n_layers: int = 2,
    epochs: int = 15,
    batch_size: int = 32,
    max_train_samples: int = 320,
    learning_rate: float = 0.05,
    model_dir: str = "models/quantum"
) -> Tuple[Dict[str, Any], List[float]]:
    """
    Trains the Variational Quantum Classifier efficiently using PennyLane AdamOptimizer.

    Parameters:
    -----------
    X_train_q_scaled : np.ndarray
        Scaled training features of shape (N, 4).
    y_train : pd.Series
        Ground-truth target labels.
    n_layers : int
        Number of variational circuit layers.
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    max_train_samples : int
        Subsample size for efficient training execution on laptop CPU simulator.
    learning_rate : float
        Learning rate for Adam optimizer.
    model_dir : str
        Directory to save trained model parameters.

    Returns:
    --------
    Tuple[Dict[str, Any], List[float]]
        Trained parameters dictionary and list of loss history values.
    """
    logger.info(f"Initializing 4-Qubit Variational Quantum Classifier ({n_layers} layers)...")
    os.makedirs(model_dir, exist_ok=True)

    # Subsample most recent chronological training samples if larger than max_train_samples for fast CPU simulation
    if len(X_train_q_scaled) > max_train_samples:
        X_tr = X_train_q_scaled[-max_train_samples:]
        y_tr = np.array(y_train)[-max_train_samples:]
    else:
        X_tr = X_train_q_scaled
        y_tr = np.array(y_train)

    n_samples = len(X_tr)

    # Initialize trainable weights with reproducible random seed
    pnp.random.seed(42)
    weights = pnp.random.uniform(low=-np.pi, high=np.pi, size=(n_layers, N_QUBITS, 2), requires_grad=True)
    bias = pnp.array(0.0, requires_grad=True)

    opt = qml.AdamOptimizer(stepsize=learning_rate)
    loss_history = []

    logger.info(f"Starting QNN optimization over {epochs} epochs (Batch Size={batch_size}, Samples={n_samples})...")
    for epoch in range(1, epochs + 1):
        indices = np.random.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for start_idx in range(0, n_samples, batch_size):
            batch_idx = indices[start_idx:start_idx + batch_size]
            X_b = X_tr[batch_idx]
            y_b = y_tr[batch_idx]

            def cost_fn(w, b):
                return bce_loss(w, b, X_b, y_b)

            (weights, bias), l_val = opt.step_and_cost(cost_fn, weights, bias)
            epoch_loss += float(l_val)
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)
        logger.info(f"Epoch {epoch:02d}/{epochs:02d} - BCE Loss: {avg_loss:.5f}")

    model_params = {
        'weights': np.array(weights),
        'bias': float(bias),
        'n_layers': n_layers,
        'n_qubits': N_QUBITS,
        'quantum_features': QUANTUM_FEATURE_NAMES
    }

    model_path = os.path.join(model_dir, "quantum_vqc.pkl")
    joblib.dump(model_params, model_path)
    logger.info(f"Saved trained Quantum VQC parameters to {model_path}")

    return model_params, loss_history


def predict_quantum(model_params: Dict[str, Any], X_scaled: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates class predictions and probabilities for input features using trained quantum weights.

    Parameters:
    -----------
    model_params : Dict[str, Any]
        Dictionary containing trained 'weights' and 'bias'.
    X_scaled : np.ndarray
        Scaled 4-feature test matrix.
    threshold : float
        Classification threshold (default: 0.5).

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (y_pred, y_prob) arrays.
    """
    weights = pnp.array(model_params['weights'])
    bias = float(model_params['bias'])

    y_prob = []
    for i in range(len(X_scaled)):
        prob = forward_predict_proba(weights, bias, X_scaled[i])
        y_prob.append(prob)

    y_prob_arr = np.array(y_prob)
    y_pred_arr = (y_prob_arr >= threshold).astype(int)
    return y_pred_arr, y_prob_arr


def load_quantum_model(model_path: str = "models/quantum/quantum_vqc.pkl") -> Dict[str, Any]:
    """
    Loads saved Quantum VQC parameter dictionary from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Quantum model file not found at {model_path}")
    logger.info(f"Loading Quantum VQC model from {model_path}...")
    return joblib.load(model_path)


def save_quantum_artifacts(
    model_params: Dict[str, Any],
    loss_history: List[float],
    results_dir: str = "models/quantum/results"
):
    """
    Generates and saves circuit diagram and training loss curve plots.
    """
    os.makedirs(results_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Save Quantum Circuit Diagram (Text representation and matplotlib render)
    weights = pnp.array(model_params['weights'])
    dummy_input = pnp.array([0.5, -0.2, 0.1, 0.8])
    circuit_text = qml.draw(quantum_circuit)(weights, dummy_input)

    txt_file = os.path.join(results_dir, "quantum_circuit.txt")
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("4-QUBIT VARIATIONAL QUANTUM CLASSIFIER CIRCUIT DIAGRAM:\n")
        f.write("="*60 + "\n")
        f.write(circuit_text + "\n")
    logger.info(f"Saved text circuit diagram to {txt_file}")

    # Circuit Matplotlib Plot Render
    fig, ax = qml.draw_mpl(quantum_circuit)(weights, dummy_input)
    plt.title("4-Qubit Variational Quantum Classifier (PennyLane)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "quantum_circuit.png"), dpi=300)
    plt.close()

    # 2. Training Loss Plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', color='#8a2be2', linewidth=2, label='BCE Loss')
    plt.title("Quantum Neural Network Training Loss Curve", fontsize=14, pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Binary Cross-Entropy Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "quantum_training_loss.png"), dpi=300)
    plt.close()

    logger.info(f"Saved quantum circuit and training loss plots to {results_dir}")
